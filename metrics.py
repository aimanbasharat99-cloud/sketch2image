"""Quantitative evaluation.

1. Sketch preservation (structure): edge F1 with spatial tolerance.
   Extract edges from the generated image (Canny), compare with the sketch strokes.
   A sketch pixel counts as "recalled" if a generated edge lies within `tol` pixels,
   and vice versa for precision. Recall is the most meaningful number: did the model
   keep the lines I drew? (Precision is naturally lower because realistic images have
   extra texture edges the sketch never had.)

2. Prompt alignment (optional): CLIP score between the generated image and the prompt.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _edges(img: Image.Image, low=100, high=200) -> np.ndarray:
    g = np.array(img.convert("L"))
    g = cv2.GaussianBlur(g, (5, 5), 0)
    return cv2.Canny(g, low, high) > 0


def edge_f1(control: Image.Image, generated: Image.Image, tol: int = 4) -> dict:
    """control: white-on-black sketch map. generated: RGB output (resized to match)."""
    sk = np.array(control.convert("L")) > 127
    gen = generated.resize(control.size, Image.LANCZOS)
    ge = _edges(gen)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    sk_d = cv2.dilate(sk.astype(np.uint8), k) > 0
    ge_d = cv2.dilate(ge.astype(np.uint8), k) > 0
    recall = (sk & ge_d).sum() / max(sk.sum(), 1)
    precision = (ge & sk_d).sum() / max(ge.sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"edge_recall": float(recall), "edge_precision": float(precision), "edge_f1": float(f1)}


class ClipScorer:
    """CLIP score = 100 * cosine(image_emb, text_emb) (Hessel et al., 2021)."""

    def __init__(self, device: str = "cuda", model_id: str = "openai/clip-vit-base-patch32"):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)

    def __call__(self, image: Image.Image, text: str) -> float:
        torch = self.torch
        inp = self.proc(text=[text], images=image, return_tensors="pt", padding=True,
                        truncation=True).to(self.device)
        with torch.no_grad():
            out = self.model(**inp)
        i = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        t = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float(100 * (i * t).sum())
