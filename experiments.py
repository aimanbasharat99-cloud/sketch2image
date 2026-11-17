"""Run all experiments for the report and save grids + a metrics CSV.

  python experiments.py                 # all experiments, all sketches
  python experiments.py --only E2 E3    # subset
  python experiments.py --no-clip       # skip CLIP score (saves VRAM / time)

Experiments (each one changes ONE factor, everything else fixed at defaults):
  E1  baseline vs ControlNet     cond_scale 0 (text only) vs 1.0
  E2  sketch strength            cond_scale in {0.3, 0.6, 1.0, 1.5}
  E3  ControlNet type            scribble / lineart / canny
  E4  control schedule           control_guidance_end in {0.3, 0.6, 1.0}
  E5  prompt / style             3 prompts per sketch (prompts.json)
  E6  seed variability           seeds {1, 2, 3}
  E7  text guidance (CFG)        guidance_scale in {3, 7.5, 12}
"""
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from PIL import Image

from metrics import ClipScorer, edge_f1
from pipeline import GenConfig, SketchToImage
from viz import make_grid

ROOT = Path(__file__).parent
SWEEPS = {
    "E1": ("cond_scale", [0.0, 1.0]),
    "E2": ("cond_scale", [0.3, 0.6, 1.0, 1.5]),
    "E3": ("mode", ["scribble", "lineart", "canny"]),
    "E4": ("guidance_end", [0.3, 0.6, 1.0]),
    "E5": ("prompt", None),          # filled from prompts.json
    "E6": ("seed", [1, 2, 3]),
    "E7": ("guidance_scale", [3.0, 7.5, 12.0]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(SWEEPS))
    ap.add_argument("--sketch-dir", default=str(ROOT))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "experiments"))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--no-clip", action="store_true")
    a = ap.parse_args()

    prompts = {k: v for k, v in json.load(open(ROOT / "prompts.json")).items() if not k.startswith("_")}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = SketchToImage()
    clip = None if a.no_clip else ClipScorer(device=model.device)

    rows = []
    for fname, info in prompts.items():
        path = Path(a.sketch_dir) / fname
        if not path.exists():
            print("skip (not found):", path); continue
        sketch = Image.open(path)
        base = GenConfig(prompt=info["styles"]["photo"], steps=a.steps)
        for exp in a.only:
            key, values = SWEEPS[exp]
            if exp == "E5":
                values = list(info["styles"].values())
                labels = list(info["styles"].keys())
            else:
                labels = [f"{key}={v}" for v in values]
            imgs, controls = [], []
            for v, lab in zip(values, labels):
                cfg = replace(base, **{key: v})
                [img], control = model.generate(sketch, cfg)
                img.save(out / f"{path.stem}_{exp}_{lab.replace('=', '-')}.png")
                m = edge_f1(control, img)
                if clip:
                    m["clip_score"] = clip(img, cfg.prompt)
                rows.append({"sketch": fname, "exp": exp, "setting": lab, **m})
                print(fname, exp, lab, {k: round(x, 3) for k, x in m.items()})
                imgs.append(img); controls.append(control)
            ref_label = "control (scribble)" if exp == "E3" else "control map"
            grid = make_grid([sketch.convert("RGB"), controls[0], *imgs],
                             ["sketch", ref_label] + labels, cols=2 + len(imgs),
                             title=f"{exp} - {path.stem}")
            grid.save(out / f"grid_{path.stem}_{exp}.png")

    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("done ->", out / "metrics.csv")


if __name__ == "__main__":
    main()
