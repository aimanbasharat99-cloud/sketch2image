"""Sketch preprocessing: turn a photo/scan of a hand-drawn sketch into a ControlNet
conditioning image.

ControlNet scribble / lineart / canny models were trained on WHITE strokes on a
BLACK background, so a pencil drawing (dark lines on white paper) must be
cleaned, binarised and inverted before it is fed to the network.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _to_gray(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"))


def resize_keep_aspect(img: Image.Image, target: int = 512) -> Image.Image:
    """Resize so the longer side is `target`, both sides multiples of 8 (needed by the VAE)."""
    w, h = img.size
    s = target / max(w, h)
    nw, nh = int(round(w * s / 8) * 8), int(round(h * s / 8) * 8)
    return img.resize((max(nw, 8), max(nh, 8)), Image.LANCZOS)


def ink_mask(img: Image.Image, min_sat: int = 70, min_frac: float = 0.002):
    """Coloured-pen detection. Ballpoint ink (blue/black-blue) is saturated, while white paper
    and the GREY ruled lines of notebook paper are not, so thresholding the HSV saturation
    keeps the drawing and drops the ruled lines. Returns None for pencil / grey sketches."""
    hsv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 1] > min_sat) & (hsv[..., 2] < 230)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return m if (m > 0).mean() > min_frac else None


def crop_to_content(mask: np.ndarray, margin: float = 0.12):
    """Bounding box of the strokes plus a margin, so a small drawing fills the frame."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mx, my = int((x1 - x0) * margin) + 10, int((y1 - y0) * margin) + 10
    return (max(x0 - mx, 0), max(y0 - my, 0),
            min(x1 + mx, mask.shape[1]), min(y1 + my, mask.shape[0]))


def clean_sketch(img: Image.Image, target: int = 512, line_thickness: int = 2,
                 block_size: int = 31, c: int = 10, crop: bool = True) -> Image.Image:
    """Phone photo of paper sketch -> clean binary map (white lines, black background).

    Steps
    1. grayscale + light blur (removes paper texture / sensor noise)
    2. adaptive threshold (robust to uneven lighting / shadows on the paper)
    3. remove tiny specks (connected components < 20 px)
    4. dilate so thin pencil strokes survive downsampling to 64x64 latents
    """
    ink = ink_mask(img)
    if ink is not None:                      # pen on (possibly ruled) paper
        if crop:
            x0, y0, x1, y1 = crop_to_content(_remove_specks(ink, 40))
            img = img.crop((x0, y0, x1, y1))
        ink = ink_mask(img, min_frac=0)
        mask = Image.fromarray(ink).resize(resize_keep_aspect(img, target).size, Image.LANCZOS)
        bw = (np.array(mask) > 100).astype(np.uint8) * 255
        return _finish(bw, line_thickness)
    img = resize_keep_aspect(img, target)
    g = _to_gray(img)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    # THRESH_BINARY_INV: dark pencil strokes -> 255, paper -> 0
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, block_size, c)
    return _finish(bw, line_thickness)


def _remove_specks(bw: np.ndarray, min_area: int = 20) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    keep = np.zeros_like(bw)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep


def _finish(bw: np.ndarray, line_thickness: int) -> Image.Image:
    keep = _remove_specks(bw)
    if line_thickness > 1:
        k = np.ones((line_thickness, line_thickness), np.uint8)
        keep = cv2.dilate(keep, k, iterations=1)
    return Image.fromarray(keep).convert("RGB")


def to_canny(img: Image.Image, low: int = 100, high: int = 200, target: int = 512) -> Image.Image:
    """Canny edge map (for the canny ControlNet). On a clean sketch this gives double edges
    around each stroke, which is a useful contrast to the scribble model."""
    img = resize_keep_aspect(img, target)
    e = cv2.Canny(_to_gray(img), low, high)
    return Image.fromarray(e).convert("RGB")


def prepare_control(img: Image.Image, mode: str, target: int = 512, **kw) -> Image.Image:
    if mode in ("scribble", "lineart"):
        return clean_sketch(img, target=target, **kw)
    if mode == "canny":
        # run canny on the cleaned sketch (inverted back to dark-on-white) for stable edges
        cleaned = clean_sketch(img, target=target, line_thickness=1)
        inv = Image.fromarray(255 - np.array(cleaned.convert("L")))
        return to_canny(inv, target=target)
    raise ValueError(f"unknown mode {mode}")
