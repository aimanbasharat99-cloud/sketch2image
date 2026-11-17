"""Side-by-side comparison grids for the report."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


def _font(size=18):
    for p in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_grid(images: list[Image.Image], labels: list[str], cols: int, cell: int = 320,
              title: str | None = None) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    lab_h, top = 28, (36 if title else 0)
    grid = Image.new("RGB", (cols * cell, top + rows * (cell + lab_h)), "white")
    d = ImageDraw.Draw(grid)
    if title:
        d.text((8, 8), title, fill="black", font=_font(20))
    for i, (im, lab) in enumerate(zip(images, labels)):
        r, c = divmod(i, cols)
        im = im.copy()
        im.thumbnail((cell, cell))
        x, y = c * cell, top + r * (cell + lab_h)
        grid.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
        d.text((x + 6, y + cell + 4), lab, fill="black", font=_font(16))
    return grid
