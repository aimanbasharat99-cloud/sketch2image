"""Gradio web app: upload or draw a sketch, add a prompt, tune the settings, generate.

  python app.py            # local: http://127.0.0.1:7860
  python app.py --share    # public link (useful from Colab)
"""
import argparse

import gradio as gr
import numpy as np
from PIL import Image

from metrics import edge_f1
from pipeline import DEFAULT_NEGATIVE, GenConfig, SketchToImage

STYLE_PRESETS = {
    "None": "",
    "Photorealistic": ", photorealistic, highly detailed, natural lighting, 8k photo",
    "Cyberpunk night": ", at night, neon lights, cyberpunk, cinematic, rain reflections",
    "Watercolor": ", watercolor painting, soft pastel colors, paper texture",
    "3D render": ", 3d render, octane, studio lighting, clean background",
    "Anime": ", anime style, cel shading, vibrant colors",
}

model = None


def _sketch_from_editor(value):
    """gr.ImageEditor returns {'background','layers','composite'}; gr.Image returns an array."""
    if value is None:
        return None
    if isinstance(value, dict):
        arr = value.get("composite")
        if arr is None:
            return None
        img = Image.fromarray(arr)
        if img.mode == "RGBA":  # strokes on transparent canvas -> black on white
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        return img.convert("RGB")
    return Image.fromarray(value).convert("RGB")


def run(sketch_val, prompt, style, negative, mode, cond_scale, guidance_end, cfg_scale,
        steps, seed, n):
    sketch = _sketch_from_editor(sketch_val)
    if sketch is None or np.array(sketch.convert("L")).min() > 240:
        raise gr.Error("Please upload or draw a sketch first.")
    cfg = GenConfig(prompt=prompt + STYLE_PRESETS[style], negative_prompt=negative, mode=mode,
                    cond_scale=cond_scale, guidance_end=guidance_end, guidance_scale=cfg_scale,
                    steps=int(steps), seed=int(seed), num_images=int(n))
    imgs, control = model.generate(sketch, cfg)
    scores = [edge_f1(control, im) for im in imgs]
    txt = "\n".join(f"image {i}: edge recall {s['edge_recall']:.2f}, F1 {s['edge_f1']:.2f}"
                    for i, s in enumerate(scores))
    return control, imgs, txt


def build_ui():
    with gr.Blocks(title="Sketch to Image") as demo:
        gr.Markdown("# Sketch → Image  \nStable Diffusion 1.5 + ControlNet. "
                    "Upload a photo of a pencil sketch or draw one below.")
        with gr.Row():
            with gr.Column():
                sketch = gr.ImageEditor(label="Sketch (upload or draw, dark lines on white)",
                                        type="numpy", height=420,
                                        brush=gr.Brush(colors=["#000000"], default_size=4))
                prompt = gr.Textbox(label="Prompt", value="a futuristic skyscraper, glass facade")
                style = gr.Dropdown(list(STYLE_PRESETS), value="Photorealistic", label="Style preset")
                with gr.Accordion("Advanced settings", open=False):
                    negative = gr.Textbox(label="Negative prompt", value=DEFAULT_NEGATIVE)
                    mode = gr.Radio(["scribble", "lineart", "canny"], value="scribble",
                                    label="ControlNet")
                    cond = gr.Slider(0, 2, 1.0, step=0.05, label="Sketch strength (conditioning scale)")
                    gend = gr.Slider(0.1, 1, 1.0, step=0.05, label="Apply sketch until (fraction of steps)")
                    cfg = gr.Slider(1, 15, 7.5, step=0.5, label="Prompt strength (CFG)")
                    steps = gr.Slider(10, 50, 30, step=1, label="Steps")
                    seed = gr.Number(42, label="Seed", precision=0)
                    n = gr.Slider(1, 4, 2, step=1, label="Images")
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                control = gr.Image(label="Control map fed to ControlNet")
                gallery = gr.Gallery(label="Results", columns=2, height=420)
                info = gr.Textbox(label="Sketch preservation", lines=3)
        btn.click(run, [sketch, prompt, style, negative, mode, cond, gend, cfg, steps, seed, n],
                  [control, gallery, info])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    model = SketchToImage()
    build_ui().queue().launch(share=a.share)
