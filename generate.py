"""Single generation from the command line.

Example:
  python generate.py --sketch sketches/building.png \
      --prompt "a futuristic skyscraper, glass facade, golden hour, photo" \
      --mode scribble --cond-scale 1.0 --seed 42
"""
import argparse
from pathlib import Path

from PIL import Image

from src.pipeline import GenConfig, SketchToImage
from src.metrics import edge_f1
from src.viz import make_grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sketch", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default=None)
    p.add_argument("--mode", default="scribble", choices=["scribble", "lineart", "canny"])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--cond-scale", type=float, default=1.0)
    p.add_argument("--guidance-end", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--lora", default=None, help="optional LoRA (HF repo id or local path)")
    p.add_argument("--out", default="outputs/single")
    a = p.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = SketchToImage(lora_path=a.lora)
    cfg = GenConfig(prompt=a.prompt, mode=a.mode, steps=a.steps, guidance_scale=a.guidance,
                    cond_scale=a.cond_scale, guidance_end=a.guidance_end, seed=a.seed,
                    num_images=a.n)
    if a.negative is not None:
        cfg.negative_prompt = a.negative
    sketch = Image.open(a.sketch)
    imgs, control = model.generate(sketch, cfg)
    stem = Path(a.sketch).stem
    control.save(out / f"{stem}_control.png")
    for i, im in enumerate(imgs):
        im.save(out / f"{stem}_{a.mode}_s{a.seed}_{i}.png")
        print(f"[{i}]", edge_f1(control, im))
    make_grid([sketch.convert("RGB"), control, *imgs],
              ["input sketch", f"control ({a.mode})"] + [f"output {i}" for i in range(len(imgs))],
              cols=2 + len(imgs)).save(out / f"{stem}_comparison.png")
    print("saved to", out)


if __name__ == "__main__":
    main()
