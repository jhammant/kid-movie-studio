"""Plex cover art for Robots Revenge: poster (2:3) and background art (16:9).

    python3 tools/cover_art.py <frame.png> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

IMPACT = "/System/Library/Fonts/Supplemental/Impact.ttf"
AVENIR = "/System/Library/Fonts/Avenir Next.ttc"  # index 0 bold, has ë


def chrome_text(size, text, font, top=(235, 238, 245), bottom=(200, 30, 30)):
    """Text filled with a steel-to-red gradient, black outline and red glow."""
    w, h = size
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).text((w // 2, h // 2), text, font=font, anchor="mm", fill=255)
    grad = np.zeros((h, w, 3), np.float32)
    t = np.linspace(0, 1, h)[:, None]
    band = np.clip((t - 0.48) * 6, 0, 1)  # hard horizon like chrome lettering
    for c in range(3):
        grad[..., c] = top[c] * (1 - band) + bottom[c] * band
    grad *= (1 - 0.25 * np.abs(np.sin(t * 9)))[..., None]  # subtle metal banding
    fill = Image.fromarray(grad.clip(0, 255).astype(np.uint8))
    out = Image.new("RGBA", size, (0, 0, 0, 0))
    glow = mask.filter(ImageFilter.GaussianBlur(18))
    out.paste((255, 30, 30, 255), (0, 0), glow.point(lambda v: int(v * 0.9)))
    outline = mask.filter(ImageFilter.MaxFilter(9))
    out.paste((10, 10, 14, 255), (0, 0), outline)
    out.paste(fill, (0, 0), mask)
    return out


def vignette(img, top=0.55, bottom=0.75):
    w, h = img.size
    y = np.linspace(0, 1, h)[:, None]
    dark = np.clip((0.32 - y) / 0.32, 0, 1) * top + np.clip((y - 0.52) / 0.48, 0, 1) * bottom
    arr = np.asarray(img.convert("RGB"), np.float32) * (1 - dark[..., None])
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def poster(frame, out):
    src = Image.open(frame).convert("RGB")
    # wide enough for both the robot's head and the kid's whole hat; dark bands hold the text
    shot = src.crop((520, 0, 1480, 1080)).resize((1000, 1125), Image.LANCZOS)
    canvas = Image.new("RGB", (1000, 1500), (8, 8, 12))
    canvas.paste(shot, (0, 120))
    img = vignette(canvas, top=0.35, bottom=0.85).convert("RGBA")
    d = ImageDraw.Draw(img)
    tag = ImageFont.truetype(AVENIR, 40, index=0)
    d.text((500, 92), "TONIGHT AT 9 O'CLOCK…", font=tag, fill=(255, 210, 60), anchor="mm")
    d.text((500, 146), "THE ROBOTS STRIKE BACK", font=tag, fill=(255, 255, 255), anchor="mm")
    title_font = ImageFont.truetype(IMPACT, 210)
    img.alpha_composite(chrome_text((1000, 260), "ROBOTS", title_font), (0, 1010))
    img.alpha_composite(chrome_text((1000, 260), "REVENGE", title_font), (0, 1190))
    small = ImageFont.truetype(AVENIR, 30, index=0)
    d = ImageDraw.Draw(img)
    d.text((500, 1462), "SBC NEWS PRESENTS  ·  STARRING SAM & ALEX", font=small, fill=(220, 225, 235), anchor="mm")
    img.convert("RGB").save(out, quality=92)


def background(frame, out):
    src = Image.open(frame).convert("RGB")
    img = vignette(src, top=0.25, bottom=0.6).convert("RGBA")
    title_font = ImageFont.truetype(IMPACT, 150)
    img.alpha_composite(chrome_text((1920, 200), "ROBOTS REVENGE", title_font), (0, 850))
    img.convert("RGB").save(out, quality=92)


if __name__ == "__main__":
    frame, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    poster(frame, out_dir / "poster.jpg")
    background(frame, out_dir / "fanart.jpg")
    print(out_dir / "poster.jpg", out_dir / "fanart.jpg")
