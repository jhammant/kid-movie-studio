"""Cover art for a media library (Plex, Jellyfin, ...): a 2:3 poster and 16:9 background art.

The poster crops a tall slice out of a frame from the film (centred, or wherever `crop_x` says the
subject is), darkens the top and bottom, and stacks the title in chrome-to-red lettering under a
two-line kicker, with a small credit line at the foot. The fanart is the whole frame with the
title along the bottom. Titles of one to three lines shrink to fit.

    python -m kms.render.cover_art FRAME OUT_DIR [--title T] [--kicker L1 --kicker L2] [--credit C] [--crop-x 0.5]
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from kms.fonts import fit, font
from kms.render import split_title

POSTER_W, POSTER_H = 1000, 1500
SHOT_Y, SHOT_H = 120, 1125          # where the film still sits on the poster
FAN_W, FAN_H = 1920, 1080
TITLE = "ROBOTS REVENGE"
KICKER = ("TONIGHT AT 9 O'CLOCK…", "THE ROBOTS STRIKE BACK")
CREDIT = "SBC NEWS PRESENTS · STARRING SAM & ALEX"


def chrome_text(size, text, fnt, top=(235, 238, 245), bottom=(200, 30, 30)):
    """Text filled with a steel-to-red gradient, black outline and red glow."""
    w, h = size
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).text((w // 2, h // 2), text, font=fnt, anchor="mm", fill=255)
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


def load(frame):
    """A still as an RGB image: a path, a PIL image or an (H, W, 3) uint8 array."""
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if isinstance(frame, np.ndarray):
        return Image.fromarray(np.asarray(frame, np.uint8)).convert("RGB")
    return Image.open(frame).convert("RGB")


def crop_box(size, aspect, crop_x=None):
    """The biggest box of `aspect` (w/h) in an image, centred on crop_x (0..1 across) or the middle."""
    w, h = size
    if w / h > aspect:
        cw = min(w, round(h * aspect))
        cx = (0.5 if crop_x is None else float(crop_x)) * w
        x0 = int(min(max(round(cx - cw / 2), 0), w - cw))
        return x0, 0, x0 + cw, h
    ch = min(h, round(w / aspect))
    y0 = (h - ch) // 2
    return 0, y0, w, y0 + ch


def title_lines(title, max_lines, max_w, sizes):
    """Split a title into balanced lines and the biggest Impact size where every line fits."""
    lines = split_title(" ".join(str(title).split()), max_lines=max_lines)
    cap = sizes[min(len(lines), len(sizes)) - 1]
    size = min(cap, min(fit("impact", line, max_w, start=cap).size for line in lines))
    return lines, size


def poster(frame, out, *, title=TITLE, kicker=KICKER, credit=CREDIT, crop_x=None):
    """A 1000x1500 (2:3) poster JPEG. crop_x: where the subject is, 0 (left) .. 1 (right); default centre."""
    src = load(frame)
    # a tall slice of the frame; dark bands top and bottom hold the text
    shot = src.crop(crop_box(src.size, POSTER_W / SHOT_H, crop_x)).resize((POSTER_W, SHOT_H), Image.LANCZOS)
    canvas = Image.new("RGB", (POSTER_W, POSTER_H), (8, 8, 12))
    canvas.paste(shot, (0, SHOT_Y))
    img = vignette(canvas, top=0.35, bottom=0.85).convert("RGBA")
    d = ImageDraw.Draw(img)

    if isinstance(kicker, str):
        kicker = (kicker,)
    kicker = [k for k in (kicker or ()) if k][:2]
    for k, y, colour in zip(kicker, (92, 146) if len(kicker) == 2 else (66,), ((255, 210, 60), (255, 255, 255))):
        d.text((POSTER_W // 2, y), k, font=fit("avenir-bold", k, POSTER_W - 60, start=40), fill=colour, anchor="mm")

    lines, size = title_lines(title, 3, POSTER_W - 60, sizes=(250, 210, 170))
    pitch = round(size * (180 / 210 if len(lines) < 3 else 0.93))  # three lines need a little more air
    box_h = round(size * 260 / 210)
    centre = min(1230, 1320 - (len(lines) - 1) * pitch / 2)  # never lower than the original's bottom line
    title_font = font("impact", size)
    for j, line in enumerate(lines):
        cy = centre + (j - (len(lines) - 1) / 2) * pitch
        img.alpha_composite(chrome_text((POSTER_W, box_h), line, title_font), (0, int(round(cy - box_h / 2))))

    if credit:
        credit = credit.replace(" · ", "  ·  ")  # a little air around the separators
        d = ImageDraw.Draw(img)
        d.text((POSTER_W // 2, 1462), credit, font=fit("avenir-bold", credit, POSTER_W - 40, start=30),
               fill=(220, 225, 235), anchor="mm")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out, quality=92)
    return out


def fanart(frame, out, *, title=TITLE):
    """A 1920x1080 (16:9) background JPEG: the whole frame with the title along the bottom."""
    src = load(frame)
    if src.size != (FAN_W, FAN_H):
        src = src.crop(crop_box(src.size, FAN_W / FAN_H)).resize((FAN_W, FAN_H), Image.LANCZOS)
    img = vignette(src, top=0.25, bottom=0.6).convert("RGBA")
    lines, size = title_lines(title, 1, FAN_W - 160, sizes=(150,))
    if size < 110:  # a long title reads better on two lines
        lines, size = title_lines(title, 2, FAN_W - 160, sizes=(150, 130))
    box_h = round(size * 200 / 150)
    pitch = round(size * 0.9)
    title_font = font("impact", size)
    for j, line in enumerate(lines):
        cy = 950 - (len(lines) - 1 - j) * pitch
        img.alpha_composite(chrome_text((FAN_W, box_h), line, title_font), (0, int(round(cy - box_h / 2))))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out, quality=92)
    return out


def render(frame, out_dir, *, title=TITLE, kicker=KICKER, credit=CREDIT, crop_x=None, ctx=None):
    """Write poster.jpg (2:3) and fanart.jpg (16:9) into out_dir; returns (poster path, fanart path)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return (poster(frame, out_dir / "poster.jpg", title=title, kicker=kicker, credit=credit, crop_x=crop_x),
            fanart(frame, out_dir / "fanart.jpg", title=title))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frame", help="a still from the film (any 16:9 image works best)")
    ap.add_argument("out_dir")
    ap.add_argument("--title", default=TITLE)
    ap.add_argument("--kicker", action="append", help="a line above the picture (give it twice for two lines)")
    ap.add_argument("--credit", default=CREDIT)
    ap.add_argument("--crop-x", type=float, help="where the subject is across the frame, 0..1 (default: centre)")
    a = ap.parse_args(argv)
    paths = render(a.frame, a.out_dir, title=a.title, kicker=tuple(a.kicker) if a.kicker else KICKER,
                   credit=a.credit, crop_x=a.crop_x)
    print(*paths)


if __name__ == "__main__":
    main()
