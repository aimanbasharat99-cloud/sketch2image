"""Sketch-to-image generation with Stable Diffusion 1.5 + ControlNet (diffusers).

How the sketch influences the image
-----------------------------------
Stable Diffusion denoises a 64x64x4 latent over T steps; at each step a U-Net predicts
the noise, conditioned on the text embedding (CLIP) via cross-attention.
ControlNet is a trainable COPY of the U-Net encoder that receives the sketch map.
Its outputs are added to the skip connections of the frozen U-Net through
"zero convolutions" (1x1 convs initialised to 0). So:
    U-Net features  <-  U-Net features + scale * ControlNet(sketch, x_t, t, text)
`controlnet_conditioning_scale` is that `scale`; `control_guidance_start/end` choose
WHICH denoising steps receive the sketch. Early steps fix global layout, late steps
fix fine detail, so stopping control early (e.g. end=0.6) keeps the composition but
lets the model add its own texture. scale=0 recovers plain text-to-image (baseline).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                       UniPCMultistepScheduler)
from PIL import Image

from preprocess import prepare_control

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNETS = {
    "scribble": "lllyasviel/control_v11p_sd15_scribble",
    "lineart": "lllyasviel/control_v11p_sd15_lineart",
    "canny": "lllyasviel/control_v11p_sd15_canny",
}
DEFAULT_NEGATIVE = ("lowres, blurry, deformed, bad anatomy, extra limbs, watermark, text, "
                    "signature, jpeg artifacts, worst quality")


@dataclass
class GenConfig:
    prompt: str
    negative_prompt: str = DEFAULT_NEGATIVE
    mode: str = "scribble"
    steps: int = 30
    guidance_scale: float = 7.5          # classifier-free guidance (text strength)
    cond_scale: float = 1.0              # ControlNet strength (sketch strength)
    guidance_start: float = 0.0
    guidance_end: float = 1.0
    seed: int = 42
    resolution: int = 512
    num_images: int = 1
    preprocess_kwargs: dict = field(default_factory=dict)


class SketchToImage:
    def __init__(self, base_model: str = BASE_MODEL, device: str | None = None,
                 lora_path: str | None = None, lora_scale: float = 0.8):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._controlnets: dict[str, ControlNetModel] = {}
        cn = self._get_controlnet("scribble")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model, controlnet=cn, torch_dtype=self.dtype,
            safety_checker=None, requires_safety_checker=False)
        # UniPC: fast multistep solver, good quality in 20-30 steps
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
        self.lora_scale = None
        if lora_path:  # optional style LoRA (e.g. a watercolor / architecture LoRA from HF)
            self.pipe.load_lora_weights(lora_path)
            self.lora_scale = lora_scale
        self.mode = "scribble"

    def _get_controlnet(self, mode: str) -> ControlNetModel:
        if mode not in self._controlnets:
            self._controlnets[mode] = ControlNetModel.from_pretrained(
                CONTROLNETS[mode], torch_dtype=self.dtype)
        return self._controlnets[mode]

    def set_mode(self, mode: str):
        """Swap the ControlNet while reusing the same SD weights (no reload of the base)."""
        if mode == self.mode:
            return
        cn = self._get_controlnet(mode).to(self.device)
        self.pipe.controlnet = cn
        self.mode = mode

    @torch.inference_mode()
    def generate(self, sketch: Image.Image, cfg: GenConfig) -> tuple[list[Image.Image], Image.Image]:
        self.set_mode(cfg.mode)
        control = prepare_control(sketch, cfg.mode, target=cfg.resolution, **cfg.preprocess_kwargs)
        gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        extra = {}
        if self.lora_scale is not None:
            extra["cross_attention_kwargs"] = {"scale": self.lora_scale}
        out = self.pipe(
            prompt=cfg.prompt,
            negative_prompt=cfg.negative_prompt,
            image=control,
            width=control.width, height=control.height,
            num_inference_steps=cfg.steps,
            guidance_scale=float(cfg.guidance_scale),
            controlnet_conditioning_scale=float(cfg.cond_scale),
            control_guidance_start=float(cfg.guidance_start),
            control_guidance_end=float(cfg.guidance_end),
            num_images_per_prompt=cfg.num_images,
            generator=gen,
            **extra,
        )
        return out.images, control
