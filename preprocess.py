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


def clean_sketch(img: Image.Image, target: int = 512, line_thickness: int = 2,
                 block_size: int = 31, c: int = 10) -> Image.Image:
    """Phone photo of paper sketch -> clean binary map (white lines, black background).

    Steps
    1. grayscale + light blur (removes paper texture / sensor noise)
    2. adaptive threshold (robust to uneven lighting / shadows on the paper)
    3. remove tiny specks (connected components < 20 px)
    4. dilate so thin pencil strokes survive downsampling to 64x64 latents
    """
    img = resize_keep_aspect(img, target)
    g = _to_gray(img)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    # THRESH_BINARY_INV: dark pencil strokes -> 255, paper -> 0
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, block_size, c)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    keep = np.zeros_like(bw)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 20:
            keep[labels == i] = 255
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
