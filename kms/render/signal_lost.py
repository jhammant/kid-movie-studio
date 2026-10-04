"""Signal lost: the live feed stalls, breaks up and drops out.

The previous scene freezes on its last frame, judders, breaks up into digital garbage, drops to
analogue static and then, by style:

  glitch-beep  colour bars + 1 kHz BEEP + "We apologise for the loss of signal"   (alias A)
  hacked       a red/black "ERROR 404 - BROADCAST TAKEN OVER BY <VILLAINS>" takeover (alias B)
  static       just a burst of static                                              (alias C)

Every clip ends on a hard cut so it butts straight against the next shot. All pictures are drawn
here (PIL/numpy/OpenCV, no drawtext needed) and all sound is synthesised with numpy; the source
clip's audio is only used as a stuck-buffer stutter during the breakup. With no source (previews
are rendered before the footage exists) a dark, noisy close-up stands in for the lead-in.

    python -m kms.render.signal_lost OUT --style hacked [--source F] [--fps 25] [--frames N]
"""
import argparse
import math
import re
import subprocess
import tempfile
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from kms import media
from kms.fonts import fit, font
from kms.render import H, SR, W, Ctx

REF_FPS = 25  # the per-frame effects were tuned at 25 fps; other rates are scaled to match

# seconds in each phase after the lead-in
TIMELINE = {
    "glitch-beep": dict(glitch=0.72, static=1.48, card=2.80),
    "hacked": dict(glitch=0.72, static=1.00, card=3.20),
    "static": dict(glitch=0.40, static=1.92, card=0.0),
}
ALIASES = {"A": "glitch-beep", "B": "hacked", "C": "static"}
SEEDS = {"glitch-beep": 352, "hacked": 404, "static": 7}  # the tuned look at ctx.seed=0

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def style_name(style):
    """'glitch-beep', 'hacked' or 'static' (the old 'A'/'B'/'C' work too)."""
    s = str(style).strip()
    if s.upper() in ALIASES:
        return ALIASES[s.upper()]
    if s.lower() in TIMELINE:
        return s.lower()
    raise ValueError(f"unknown signal-lost style {style!r}; use one of {', '.join(TIMELINE)} (or A/B/C)")


# --------------------------------------------------------------------------- words


IRREGULAR = {"MAN": "MEN", "WOMAN": "WOMEN", "MOUSE": "MICE", "GOOSE": "GEESE", "CHILD": "CHILDREN",
             "PERSON": "PEOPLE", "TOOTH": "TEETH", "FOOT": "FEET", "SHEEP": "SHEEP", "FISH": "FISH",
             "DEER": "DEER", "MOOSE": "MOOSE", "SQUID": "SQUID", "CACTUS": "CACTI", "OX": "OXEN"}
NOT_MAN = {"HUMAN", "SHAMAN", "GERMAN", "ROMAN", "OTTOMAN", "CAYMAN", "TALISMAN"}


def plural(villain):
    """'ROBOT' -> 'ROBOTS', 'DINOSAUR' -> 'DINOSAURS', 'SPACE HAMSTER' -> 'SPACE HAMSTERS'."""
    v = " ".join(villain.upper().split())
    head, _, last = v.rpartition(" ")
    if not last:
        return v
    if last in IRREGULAR:
        last = IRREGULAR[last]
    elif last.endswith("MAN") and last not in NOT_MAN:
        last = last[:-3] + "MEN"  # SNOWMAN -> SNOWMEN
    elif last.endswith(("SS", "SH", "CH", "X", "Z", "US")):
        last += "ES"
    elif last.endswith("S"):
        pass  # already plural
    elif len(last) > 1 and last.endswith("Y") and last[-2] not in "AEIOU":
        last = last[:-1] + "IES"
    elif last.endswith("LF"):
        last = last[:-1] + "VES"  # WOLF -> WOLVES
    else:
        last += "S"
    return f"{head} {last}" if head else last


def initials(ident):
    """'ROBOTS REVENGE NEWS' -> 'RRN' (the channel's call sign on the hacked screen)."""
    letters = "".join(w[0] for w in re.findall(r"[A-Z0-9]+", ident.upper()))
    return letters[:5] or "TV"


# --------------------------------------------------------------------------- source


def scale_filter(fps):
    return (f"fps={fps},scale={W}:{H}:force_original_aspect_ratio=increase:flags=area,"
            f"crop={W}:{H}")


def decode(cmd):
    raw = subprocess.run([str(c) for c in cmd], capture_output=True, check=True).stdout
    n = len(raw) // (W * H * 3)
    return [np.frombuffer(raw, np.uint8, W * H * 3, i * W * H * 3).reshape(H, W, 3).copy() for i in range(n)]


def read_lead(path, start, nframes, fps):
    """`nframes` source frames at `fps` from `start`, or the clip's last ones when start is None.

    Returns (frames, start time of the first frame)."""
    nframes = max(nframes, 1)
    if start is None:
        info = media.probe(path)
        if info is None:
            raise ValueError(f"can't read {path}")
        start = max(0.0, info.duration - nframes / fps - 0.5)
        frames = decode([media.ffmpeg(), "-v", "error", "-ss", f"{start:.3f}", "-i", path,
                         "-vf", scale_filter(fps), "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
        if not frames:
            raise ValueError(f"no video decoded from the end of {path}")
        first = start + max(0, len(frames) - nframes) / fps
        frames = frames[-nframes:]
    else:
        frames = decode([media.ffmpeg(), "-v", "error", "-ss", f"{max(start, 0):.3f}", "-i", path,
                         "-frames:v", nframes, "-vf", scale_filter(fps), "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
        if not frames:
            raise ValueError(f"no video decoded from {path} at {start}s")
        first = start
    while len(frames) < nframes:  # a clip shorter than the lead-in: hold its first frame
        frames.insert(0, frames[0])
    return frames, first


def read_audio(path, start, dur, spf):
    cmd = [media.ffmpeg(), "-v", "error", "-ss", f"{max(start, 0):.3f}", "-i", str(path), "-t", f"{dur:.3f}", "-vn",
           "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    a = np.frombuffer(raw, np.float32)
    if a.size < SR // REF_FPS:
        return None
    a = a[: a.size // 2 * 2].reshape(-1, 2).astype(np.float32)
    if len(a) < 2 * spf:
        return None
    peak = np.abs(a).max()
    return a / peak if peak > 1e-4 else None


def synth_lead(n, seed):
    """A dark, noisy close-up standing in for footage that doesn't exist yet: something big and
    rim-lit filling the lower right (where the freeze punches in), studio lights out of focus
    behind it, a little handheld drift and plenty of grain."""
    rng = np.random.default_rng(seed)
    h, w = H // 2, W // 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    y = yy / h
    base = (np.float32([30, 34, 50]) * (1 - y[..., None]) + np.float32([7, 7, 11]) * y[..., None])
    lights = np.float32([[255, 170, 80], [90, 170, 255], [255, 90, 70], [210, 225, 255]])
    for _ in range(10):  # bokeh
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h * 0.55), rng.uniform(16, 44)
        disc = np.clip((r - np.hypot(xx - cx, yy - cy)) / 2.5, 0, 1)
        base += disc[..., None] * lights[rng.integers(0, len(lights))] * rng.uniform(0.10, 0.24)
    base = cv2.GaussianBlur(base, (0, 0), 4)
    # the close-up: a dark rounded mass with a cool rim light along its upper-left edge
    e = np.hypot((xx - 0.70 * w) / (0.36 * w), (yy - 0.98 * h) / (0.62 * h))
    body = np.clip((1 - e) / 0.012, 0, 1)
    light = np.clip(((0.70 * w - xx) / w + (0.98 * h - yy) / h) * 1.6, 0, 1)
    rim = np.exp(-((e - 0.985) / 0.018) ** 2) * light
    sheen = np.exp(-((e - 0.80) / 0.09) ** 2) * light * 0.35
    panel = np.exp(-((e - 0.62) / 0.006) ** 2) * 0.6 + np.exp(-((xx - 0.80 * w) / 2.0) ** 2) * (e < 0.95) * 0.5
    shape = np.float32([16, 17, 22]) * (1 + sheen[..., None] * 3) - panel[..., None] * 10
    base = base * (1 - body[..., None]) + shape * body[..., None]
    base += rim[..., None] * np.float32([150, 185, 235])
    base = cv2.resize(cv2.GaussianBlur(base, (0, 0), 1.2), (W, H), interpolation=cv2.INTER_CUBIC)
    r2 = ((XX - W / 2) / (W / 2)) ** 2 + ((YY - H / 2) / (H / 2)) ** 2
    base *= np.clip(1 - 0.35 * r2, 0.4, 1)[..., None]

    frames = []
    for i in range(max(n, 1)):
        z = 1.0 + 0.004 * i
        dx, dy = 3 * math.sin(i * 0.9), 2 * math.cos(i * 0.7)
        m = np.float32([[z, 0, (1 - z) * W / 2 + dx], [0, z, (1 - z) * H / 2 + dy]])
        f = cv2.warpAffine(base, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        grain = rng.normal(0, 7, (H // 2, W // 2, 1)).astype(np.float32)
        f = f * rng.uniform(0.96, 1.04) + cv2.resize(grain, (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return frames


# --------------------------------------------------------------------------- picture glitches


def punch(img, zoom, shift=(0, 0), centre=(0.70, 0.66)):
    """Zoom toward the subject with a jolt, as if it hit the lens."""
    cx, cy = centre[0] * W, centre[1] * H
    m = np.float32([[zoom, 0, (1 - zoom) * cx + shift[0]], [0, zoom, (1 - zoom) * cy + shift[1]]])
    return cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def dc_blocks(img, bs):
    small = cv2.resize(img, (W // bs, H // bs), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)


def rgb_split(img, dx, dy=0):
    out = img.copy()
    out[..., 0] = np.roll(img[..., 0], (dy, dx), axis=(0, 1))
    out[..., 2] = np.roll(img[..., 2], (-dy, -dx), axis=(0, 1))
    return out


def tear(img, rng, bands, maxshift, alt=None):
    """Horizontal slices knocked sideways (sometimes showing a different buffered frame)."""
    out = img.copy()
    for _ in range(bands):
        h = int(rng.integers(6, 150))
        y = int(rng.integers(0, H - h))
        src = alt if (alt is not None and rng.random() < 0.35) else img
        out[y:y + h] = np.roll(src[y:y + h], int(rng.integers(-maxshift, maxshift + 1)), axis=1)
        if rng.random() < 0.3:  # interlace combing inside the slice
            out[y:y + h:2] = np.roll(out[y:y + h:2], int(rng.integers(-40, 41)), axis=1)
    return out


def macroblock(img, prev, rng, amount):
    """H.264-style slice loss: from a random block onward, blocks are smeared from the
    previous picture, collapse to flat DC colour, go green/grey, or streak downwards."""
    bs = int(rng.choice([24, 40, 40, 60, 60]))
    gh, gw = H // bs, W // bs
    total = gh * gw
    types = np.zeros(total, np.uint8)
    start = int(rng.integers(0, int(total * 0.8)))
    end = min(total, start + int(total * amount * rng.uniform(0.6, 1.3)))
    i = start
    while i < end:
        run = int(rng.integers(4, 60))
        types[i:min(end, i + run)] = rng.choice([1, 1, 1, 2, 2, 3, 4, 4])
        i += run
    stray = (rng.random(total) < amount * 0.08) & (types == 0)
    types[stray] = rng.integers(1, 5, stray.sum())
    full = np.repeat(np.repeat(types.reshape(gh, gw), bs, 0), bs, 1)

    mv = (int(rng.integers(-2, 3)) * bs // 2, int(rng.integers(-3, 4)) * bs // 2)
    smear = np.roll(prev, mv, axis=(0, 1))
    dc = dc_blocks(img, bs)
    luma = dc.astype(np.float32).mean(axis=2, keepdims=True)
    tint = np.float32([0.35, 0.95, 0.45]) if rng.random() < 0.7 else np.float32([0.9, 0.3, 0.95])
    green = np.clip(luma * tint + np.float32([0, 30, 8]), 0, 255).astype(np.uint8)
    streak = img[(np.arange(H) // bs) * bs]  # each block row = its top line dragged down

    out = img.copy()
    for t, cand in ((1, smear), (2, dc), (3, green), (4, streak)):
        m = full == t
        out[m] = cand[m]
    return out


def posterize_band(img, rng):
    out = img.copy()
    h = int(rng.integers(60, 400))
    y = int(rng.integers(0, H - h))
    out[y:y + h] = (out[y:y + h] // 64) * 64 + 24
    return out


def wobble(img, rng, amp):
    """Analogue line-sync wobble: every scanline pushed sideways along a wavy curve."""
    k, ph = rng.uniform(1.5, 4), rng.uniform(0, 6.28)
    off = amp * np.sin(np.arange(H) / H * 6.283 * k + ph) + rng.normal(0, amp * 0.15, H)
    mapx = XX + off[:, None].astype(np.float32)
    return cv2.remap(img, mapx, YY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


# --------------------------------------------------------------------------- analogue snow


class Snow:
    def __init__(self, rng, fps=REF_FPS):
        self.rng = rng
        self.k = REF_FPS / fps  # motion per frame scales so the speed per second stays put
        self.bar = rng.uniform(0, H)
        self.roll = 0.0
        r2 = ((XX - W / 2) / (W / 2)) ** 2 + ((YY - H / 2) / (H / 2)) ** 2
        self.vign = np.clip(1 - 0.2 * r2, 0.6, 1)[..., None]
        self.rows = np.arange(H, dtype=np.float32)

    def frame(self, ghost=None, ghost_amt=0.0, roll_speed=0.0):
        rng = self.rng
        g = rng.standard_normal((360, 640), dtype=np.float32)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST)
        g = cv2.blur(g, (5, 1))  # band-limited luma: grain smeared along the scanline
        g /= g.std() + 1e-6
        luma = 0.52 + 0.27 * g + rng.normal(0, 0.035, (H, 1)).astype(np.float32)
        self.bar = (self.bar + 13 * self.k) % H  # rolling hum bar
        d = (self.rows - self.bar + H / 2) % H - H / 2
        luma *= (1 - 0.24 * np.exp(-(d / (0.15 * H)) ** 2))[:, None]
        for _ in range(int(rng.integers(0, 3))):  # bright interference streaks
            y = int(rng.integers(0, H - 8))
            luma[y:y + int(rng.integers(2, 7))] += rng.uniform(0.2, 0.45)
        if ghost is not None and ghost_amt > 0:
            luma = luma * (1 - ghost_amt) + ghost * ghost_amt
        uv = cv2.resize(rng.standard_normal((18, 32, 2), dtype=np.float32), (W, H),
                        interpolation=cv2.INTER_CUBIC) * 0.012  # faint PAL colour blotches
        rgb = np.empty((H, W, 3), np.float32)
        rgb[..., 0] = luma + 1.40 * uv[..., 1]
        rgb[..., 1] = luma - 0.34 * uv[..., 0] - 0.71 * uv[..., 1]
        rgb[..., 2] = luma + 1.77 * uv[..., 0]
        self.roll = (self.roll + roll_speed) % H  # slight vertical roll with the blanking bar
        r = int(self.roll)
        if roll_speed:
            rgb = np.roll(rgb, r, axis=0)
            rows = np.arange(r - 36, r) % H
            rgb[rows] = rgb[rows] * 0.08 + 0.02
            rgb[np.arange(r - 22, r - 14) % H, :] = 0.16  # sync pulse strip
        rgb *= self.vign
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


# --------------------------------------------------------------------------- the stalled feed


def glitch_plan(n, rng, fps):
    n_impact = max(3, round(0.12 * fps))  # the hit
    n_collapse = max(2, round(0.08 * fps))  # snow eating the picture
    plan = []
    for i in range(n):
        p = i / max(1, n - 1)
        if i < n_impact:
            kind = "impact"
        elif i >= n - n_collapse:
            kind = "collapse"
        else:
            probs = np.array([0.32, 0.33, 0.30, 0.05]) * (1 - p) + np.array([0.12, 0.2, 0.5, 0.18]) * p
            kind = str(rng.choice(["hold", "replay", "mosh", "flash"], p=probs / probs.sum()))
            if kind == "flash" and plan[-1][0] == "flash":
                kind = "mosh"
        plan.append((kind, p))
    return plan


def render_glitch(pool, plan, rng, snow):
    frozen = pool[-1]
    buf = [punch(f, 1.14) for f in pool[-5:]]
    zooms = [1.07, 1.20, 1.14]
    shakes = [(0, -46), (18, 30), (-6, -8)]
    n_impact = sum(kind == "impact" for kind, _ in plan)
    prev = frozen
    for i, (kind, p) in enumerate(plan):
        if kind == "impact":
            j = min(2, i * 3 // n_impact)  # three beats of the hit, whatever the frame rate
            f = punch(frozen, zooms[j], shakes[j])
            if j == 0:
                f = cv2.blur(f, (3, 41))  # vertical motion blur on the hit
            elif j == 1:
                f = rgb_split(f, 14, 4)
        elif kind == "hold":
            f = tear(prev if rng.random() < 0.6 else buf[-1], rng, int(rng.integers(1, 3)), 60)
        elif kind == "replay":
            f = buf[int(rng.integers(0, len(buf)))]
            f = macroblock(f, prev, rng, 0.05 + 0.15 * p)
            f = rgb_split(f, int(rng.choice([-1, 1]) * rng.integers(8, 24)))
        else:  # mosh / flash / collapse
            base = buf[-1] if rng.random() < 0.6 else prev
            f = macroblock(base, prev, rng, 0.25 + 0.6 * p)
            f = tear(f, rng, int(3 + 10 * p), int(80 + 320 * p), alt=buf[int(rng.integers(0, len(buf)))])
            f = rgb_split(f, int(rng.choice([-1, 1]) * rng.integers(16, 30 + 60 * p)), int(rng.integers(-6, 7)))
            if rng.random() < 0.4:
                f = posterize_band(f, rng)
            if kind == "flash":
                f = f[..., list(rng.permutation(3))] if rng.random() < 0.6 else macroblock(255 - f, f, rng, 0.9)
            if kind == "collapse":  # analogue snow eats the picture in bands
                s = snow.frame()
                share = 0.45 if i < len(plan) - 1 else 0.8
                y = 0
                while y < H:
                    h = int(rng.integers(20, 160))
                    if rng.random() < share:
                        f[y:y + h] = s[y:y + h]
                    y += h
        f = np.ascontiguousarray(f)
        prev = f
        yield f


def render_static(snow, n, ghost_img, rng, fps, roll_end=8.0):
    ghost = cv2.cvtColor(ghost_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    k = REF_FPS / fps
    jump = math.ceil(3 / k - 1e-9)  # 0.12 s in: lost vertical hold
    for i in range(n):
        u = i * k  # time in 25 fps frames
        amt = max(0.0, 0.55 - u * 0.08)
        g = wobble(ghost, rng, 30 + 25 * u) if amt > 0 else None
        if i < jump:
            speed = 0.0
        elif i == jump:
            speed = 240.0  # lost vertical hold: one jump...
        else:
            speed = (5.0 + (roll_end - 5.0) * i / max(1, n - 1)) * k  # ...then a slow drift
        yield snow.frame(g, amt, speed)


# --------------------------------------------------------------------------- glitch-beep: colour bars


def card_bars(n, rng, fps, ident):
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    bars = [(191, 191, 191), (191, 191, 0), (0, 191, 191), (0, 191, 0),
            (191, 0, 191), (191, 0, 0), (0, 0, 191), (12, 12, 12)]  # EBU 75% bars
    for i, c in enumerate(bars):
        d.rectangle([round(i * W / 8), 0, round((i + 1) * W / 8) - 1, H], fill=c)
    big, small = font("helvetica-medium", 70), font("helvetica", 52)
    apology = "We apologise for the loss of signal"
    ident_font = fit("avenir-bold", ident, max(d.textlength(apology, font=big), W - 420), start=40)
    lines = [(ident, ident_font, (235, 45, 45)),
             (apology, big, (255, 255, 255)),
             ("Please stand by", small, (200, 200, 200))]
    widths = [d.textlength(t, font=f) for t, f, _ in lines]
    bw, bh = max(widths) + 150, 330
    x0, y0 = (W - bw) / 2, (H - bh) / 2
    d.rectangle([x0, y0, x0 + bw, y0 + bh], fill=(8, 8, 8))
    d.rectangle([x0 + 10, y0 + 10, x0 + bw - 10, y0 + bh - 10], outline=(90, 90, 90), width=2)
    for (t, f, c), y in zip(lines, (y0 + 62, y0 + 168, y0 + 262)):
        d.text((W / 2, y), t, font=f, fill=c, anchor="mm")
    base = np.asarray(img).astype(np.float32)

    settle = [0.52, 0.27, 0.11, 0.035, 0.0]  # vertical hold catching the signal
    k = REF_FPS / fps
    for i in range(n):
        u = i * k
        f = base.copy()
        noise = float(np.interp(u, [0, 1, 2, 3], [0.45, 0.3, 0.12, 0.025]))
        if u < len(settle) - 1:
            r = int(np.interp(u, range(len(settle)), settle) * H)
            f = np.roll(f, r, axis=0)
            f[np.arange(r - 40, r) % H] = 8
        f *= 1 + rng.normal(0, 0.012)
        g = cv2.resize(rng.standard_normal((540, 960), dtype=np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        f += g[..., None] * (255 * noise)
        yield np.clip(f, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- hacked: the takeover

ROBOT = [
    ".........LL.........",
    ".........SS.........",
    "...##############...",
    "..#ffffffffffffff#..",
    "..#ffffffffffffff#..",
    "..#ffEEEEffEEEEff#..",
    "BB#ffEWEEffEWEEff#BB",
    "BB#ffEEEEffEEEEff#BB",
    "..#ffEEEEffEEEEff#..",
    "..#ffffffffffffff#..",
    "..#ffMMMMMMMMMMff#..",
    "..#ffMMMMMMMMMMff#..",
    "..#ffffffffffffff#..",
    "...##############...",
    "........NNNN........",
]
CELL = 30
RED = (255, 45, 45)
TYPE_X = 760  # where the typed message starts (right of the face)


def robot_sprite(talk, eyes_open, wink, antenna_on, smile):
    """LED-matrix robot face as (rgb, eye-glow mask). talk: 10 mouth column heights (0-2)."""
    rows, cols = len(ROBOT), len(ROBOT[0])
    rgb = np.zeros((rows * CELL, cols * CELL, 3), np.float32)
    glow = np.zeros(rgb.shape[:2], np.float32)
    colours = {"#": (255, 55, 55), "f": (70, 4, 4), "B": (190, 30, 30), "S": (170, 40, 40),
               "N": (130, 20, 20), "E": (255, 214, 70), "W": (255, 255, 235)}
    for r, line in enumerate(ROBOT):
        for c, ch in enumerate(line):
            if ch == ".":
                continue
            col, lit = colours.get(ch, (70, 4, 4)), False
            if ch in "EW":
                left = c < cols // 2
                closed = (not eyes_open) or (wink and left)
                if closed:
                    col, lit = ((255, 214, 70), True) if r == 7 else ((70, 4, 4), False)
                else:
                    lit = True
            elif ch == "L":
                col, lit = ((255, 255, 190), True) if antenna_on else ((120, 10, 10), False)
            elif ch == "M":
                k = c - 5
                if smile:  # a little LED grin
                    on = (r == 10 and k in (0, 9)) or (r == 11 and 1 <= k <= 8)
                else:
                    on = talk[k] >= (2 if r == 10 else 1)
                col, lit = ((255, 120, 50), True) if on else ((95, 12, 12), False)
            y, x = r * CELL, c * CELL
            rgb[y + 2:y + CELL - 2, x + 2:x + CELL - 2] = col
            if lit:
                glow[y:y + CELL, x:x + CELL] = 1.0
    return rgb, glow


def warning_triangle(size, colour):
    """Hand-drawn warning sign (the fonts here have no U+26A0)."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = size * 0.04
    pts = [(size / 2, pad), (size - pad, size - pad), (pad, size - pad)]
    d.polygon(pts, fill=colour)
    inner = [(size / 2, size * 0.24), (size * 0.79, size * 0.86), (size * 0.21, size * 0.86)]
    d.polygon(inner, fill=(0, 0, 0, 255))
    d.rounded_rectangle([size * 0.465, size * 0.40, size * 0.535, size * 0.70], radius=size * 0.03, fill=colour)
    d.ellipse([size * 0.455, size * 0.73, size * 0.545, size * 0.82], fill=colour)
    return img


def glow_layer(img, radius, strength):
    a = np.asarray(img).astype(np.float32)
    blur = cv2.GaussianBlur(a, (0, 0), radius)
    return a + blur * strength


def wrap(words, fnt, max_w):
    lines, cur = [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if cur and fnt.getlength(trial) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    return lines + [cur]


def typed_lines(message, max_w=W - TYPE_X - 60, max_lines=4):
    """Word-wrap the typed message; shrink the typewriter font if it needs more than 4 lines."""
    size = 84
    while True:
        typer = font("courier-bold", size)
        lines = wrap(message.split(), typer, max_w)
        if (len(lines) <= max_lines and max(typer.getlength(t) for t in lines) <= max_w) or size <= 36:
            return typer, size, lines
        size -= 4


def hacked_words(villain, ident):
    v = " ".join(villain.upper().split()) or "ROBOT"
    many = plural(v)
    robots = "ROBOT" in v
    return dict(
        status=f"{initials(ident)}://LIVE-FEED  >>  STATUS:",
        control=f"{v} CONTROL 100%",
        message=f"BROADCAST TAKEN OVER BY {many}",
        ticker=["BEEP BOOP" if robots else f"{many} RULE!", f"THE {many} ARE IN CHARGE NOW",
                "HUMANS PLEASE STAND BY", f"ALL NEWS IS NOW {v} NEWS"],
    )


def card_hacked(n, rng, notes, snow, fps, words):
    k = REF_FPS / fps
    # ---- static layers
    bg = np.zeros((H, W, 3), np.float32)
    bg[::60, :] = (38, 0, 0)
    bg[:, ::60] = (38, 0, 0)
    text = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(text)
    tri = warning_triangle(200, (255, 200, 0, 255))
    text.paste(tri, (760, 175), tri)
    err = font("impact", 200)
    d.text((990, 290), "ERROR 404", font=err, fill=RED, anchor="lm")
    mono_s = font("menlo-bold", 30)
    status = words["status"]
    if d.textlength(status, font=mono_s) > 640:
        status = "LIVE-FEED  >>  STATUS:"
    d.text((60, 40), status, font=mono_s, fill=(200, 40, 40))
    hacked_x = int(60 + d.textlength(status, font=mono_s) + 18)
    room = W - 60 - (hacked_x + 150 + 40)
    d.text((W - 60, 40), words["control"], font=fit("menlo-bold", words["control"], room, start=30),
           fill=(200, 40, 40), anchor="ra")
    static_layer = glow_layer(text, 14, 1.1)
    lbl = Image.new("RGB", (150, 46), (230, 30, 30))
    ImageDraw.Draw(lbl).text((75, 23), "HACKED", font=mono_s, fill=(255, 255, 255), anchor="mm")
    hacked = np.asarray(lbl).astype(np.float32)

    typer, tsize, lines = typed_lines(words["message"])
    s = tsize / 84
    line_y = [510 + round(105 * s) * j for j in range(len(lines))]
    type_layers = []  # one glow layer per line so their halos never double up
    for t, y in zip(lines, line_y):
        img = Image.new("RGB", (W, H))
        ImageDraw.Draw(img).text((TYPE_X, y), t, font=typer, fill=(255, 225, 225))
        layer = glow_layer(img, 12, 1.4) * np.float32([1.0, 0.55, 0.55])
        rows = np.nonzero(layer.max(axis=(1, 2)) > 0.5)[0]
        type_layers.append((rows[0], rows[-1] + 1, layer[rows[0]:rows[-1] + 1]))
    td = ImageDraw.Draw(text)
    prefix_x = [[TYPE_X + td.textlength(t[:c], font=typer) for c in range(len(t) + 1)] for t in lines]
    char_w = td.textlength("M", font=typer)

    # ticker strip
    tk_font = font("impact", 56)
    strip = Image.new("RGB", (6000, 84), (215, 25, 25))
    sd = ImageDraw.Draw(strip)
    x, small_tri = 0, warning_triangle(60, (0, 0, 0, 255))
    while x < 6000:
        for m in words["ticker"]:
            sd.text((x, 42), m, font=tk_font, fill=(10, 0, 0), anchor="lm")
            x += sd.textlength(m, font=tk_font) + 40
            strip.paste(small_tri, (int(x), 12), small_tri)
            x += 100
    strip = np.asarray(strip).astype(np.float32)
    loop = int(x)  # not exact loop; strip is long enough for the card

    scan = np.ones((H, 1, 1), np.float32)
    scan[::3] = 0.68
    r2 = ((XX - W / 2) / (W / 2)) ** 2 + ((YY - H / 2) / (H / 2)) ** 2
    vign = np.clip(1 - 0.28 * r2, 0.5, 1)[..., None]

    talk_until = notes["talk_until"]
    for i in range(n):
        t = i / fps
        u = i * k  # time in 25 fps frames
        f = bg.copy()
        # robot face (talks while beeping, blinks once, winks at the evil boop)
        sounding = any(a <= t < b for a, b in notes["beeps"])
        talk = rng.integers(0, 3, 10) if sounding else np.zeros(10, int)
        eyes_open = not (1.3 <= t < 1.42)
        wink = t >= notes["boop_start"]
        rob, rglow = robot_sprite(talk, eyes_open, wink, int(u // 3) % 2 == 0, smile=t >= talk_until)
        ry, rx = 225, 105
        pulse = 1.0 + (0.5 if sounding else 0.0) + 0.15 * np.sin(u * 0.9)
        f[ry:ry + rob.shape[0], rx:rx + rob.shape[1]] = rob
        glow = cv2.GaussianBlur(rglow, (0, 0), 16)[..., None] * np.float32([255, 120, 40]) * 0.9 * pulse
        f[ry:ry + rob.shape[0], rx:rx + rob.shape[1]] += glow
        # text
        if u >= 2:
            f += static_layer
            if int(u // 6) % 2 == 0:
                f[34:80, hacked_x - 8:hacked_x + 142] = hacked
        chars = max(0, int((u - 5) * 2))
        last_x, last_y = None, None
        for li, (tline, y) in enumerate(zip(lines, line_y)):
            c = min(len(tline), max(0, chars))
            chars -= len(tline) + 3  # small pause between lines
            if c > 0:
                xe = int(prefix_x[li][c])
                y0, y1, layer = type_layers[li]
                f[y0:y1, :xe + 20] += layer[:, :xe + 20]
                last_x, last_y = xe, y
            if c < len(tline):
                break
        if last_x is not None and int(u // 4) % 2 == 0:  # blinking block cursor
            cx = min(last_x + round(8 * s), W - int(char_w * 0.8))
            f[last_y + round(12 * s):last_y + round(86 * s), cx:cx + int(char_w * 0.8)] = (255, 70, 70)
        # ticker
        off = min(int(u * 14), min(loop, strip.shape[1]) - W)
        f[968:1052] = strip[:, off:off + W]
        f *= scan * vign
        f *= rng.uniform(0.86, 1.04) if rng.random() < 0.85 else rng.uniform(0.55, 0.75)  # flicker
        f += rng.normal(0, 7, (H // 4, W // 4, 1)).repeat(4, 0).repeat(4, 1).astype(np.float32)
        out = np.clip(f, 0, 255).astype(np.uint8)
        # glitching in, plus the odd hiccup later
        if u < 4:
            out = tear(out, rng, int(10 - 2 * u), int(300 - 60 * u), alt=snow.frame())
            out = rgb_split(out, int(40 - 8 * u), 3)
        elif rng.random() < 0.12:
            out = tear(out, rng, 3, 90)
            out = rgb_split(out, int(rng.integers(8, 26)))
        yield np.ascontiguousarray(out)


# --------------------------------------------------------------------------- audio


def db(x):
    return 10 ** (x / 20)


def limit(x, ceiling_db):
    c = db(ceiling_db)
    return c * np.tanh(x / c)


def norm(x, peak_db):
    m = np.abs(x).max()
    return x * (db(peak_db) / m) if m > 0 else x


def fft_filter(x, response):
    spec = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(x.shape[0], 1 / SR)
    return np.fft.irfft(spec * response(f)[:, None], n=x.shape[0], axis=0).astype(np.float32)


def crush(x, bits, hold):
    y = np.repeat(x[::hold], hold, axis=0)[: len(x)]
    q = 2 ** (bits - 1)
    return np.round(y * q) / q


def tone(freqs, shape="square"):
    ph = np.cumsum(np.asarray(freqs, np.float64)) / SR
    s = np.sin(2 * np.pi * ph)
    return (np.sign(s) if shape == "square" else s).astype(np.float32)


def stereo(x, pan=0.0):
    return np.stack([x * (1 - max(pan, 0)), x * (1 + min(pan, 0))], axis=1).astype(np.float32)


def thud(rng, n):
    t = np.arange(n) / SR
    body = tone(38 + 75 * np.exp(-t * 16), "sine") * np.exp(-t * 7)
    click = rng.standard_normal(n).astype(np.float32) * np.exp(-t * 90) * 0.8
    return np.tanh(3 * (body + click)).astype(np.float32)


def glitch_audio(plan, rng, grain, spf):
    """One segment per picture frame, so the crunch always lines up with the breakup."""
    n = len(plan) * spf
    out = np.zeros((n, 2), np.float32)
    if grain is None:
        grain = stereo(rng.standard_normal(SR // 4).astype(np.float32) * 0.5)
    for i, (kind, p) in enumerate(plan):
        a, b = i * spf, (i + 1) * spf
        if kind == "impact":
            seg = crush(grain[-spf:], 5, 3) * 0.35
        elif kind == "hold":  # stuck audio buffer: a tiny grain looped into a buzz
            g = int(rng.integers(SR // 160, SR // 60))
            st = int(rng.integers(0, len(grain) - g))
            seg = np.tile(grain[st:st + g], (spf // g + 1, 1))[:spf] * 0.8
        elif kind == "replay":
            st = int(rng.integers(0, len(grain) - spf))
            seg = crush(grain[st:st + spf], 4, int(rng.integers(2, 6)))
        elif kind == "mosh":
            sh = int(rng.integers(8, 60))
            noise = crush(rng.uniform(-1, 1, (spf, 1)).astype(np.float32), 3, sh)
            buzz = tone(np.full(spf, rng.uniform(70, 900)))[:, None] * 0.5
            seg = np.repeat((noise * 0.8 + buzz) * 0.8, 2, axis=1)
        elif kind == "flash":
            seg = np.repeat(rng.uniform(-1, 1, (spf, 1)).astype(np.float32) + tone(np.full(spf, 60.0))[:, None] * 0.6,
                            2, axis=1)
        else:  # collapse: hiss swallowing the crunch
            seg = rng.uniform(-1, 1, (spf, 2)).astype(np.float32) * 0.9
        pan = rng.uniform(-0.4, 0.4)
        seg = seg * np.float32([1 - max(pan, 0), 1 + min(pan, 0)]) * (0.55 + 0.45 * p)
        if rng.random() < 0.2 and kind not in ("impact", "collapse"):  # dropout
            seg[int(rng.integers(0, spf // 2)):] *= 0.05
        out[a:b] += seg
    th = thud(rng, min(n, int(0.4 * SR)))
    out[: len(th)] += th[:, None] * 1.2
    return limit(norm(out, -2.0), -3.0)  # AAC overshoots, so leave headroom


def static_audio(n, rng):
    """The BZZZZ: hiss + 50 Hz field buzz, with a KSSSH as the picture drops out."""
    t = np.arange(n) / SR
    hiss = rng.standard_normal((n, 2)).astype(np.float32)
    hiss = fft_filter(hiss, lambda f: np.where(f < 80, 0.2, 1.0) / np.sqrt(1 + (f / 11000) ** 4))
    buzz = np.zeros(n, np.float32)
    for k in range(1, 30):
        buzz += np.sin(2 * np.pi * 50 * k * t + k).astype(np.float32) / k
    buzz = np.tanh(buzz * 1.5)
    am = (1 + 0.18 * np.sin(2 * np.pi * 0.9 * t) + 0.08 * np.sin(2 * np.pi * 7 * t)).astype(np.float32)
    x = hiss * 0.20 * am[:, None] + (buzz * 0.07)[:, None]
    burst = np.ones(n, np.float32)
    k = min(n, int(round(0.12 * SR)))
    burst[:k] = np.linspace(1.7, 1.0, k)
    x *= burst[:, None]
    return limit(x, -2.5)


def beep_audio(n, start_s=0.08):
    """The 1 kHz line-up tone under the colour bars."""
    x = np.zeros(n, np.float32)
    s = min(n, int(round(start_s * SR)))
    t = np.arange(n - s) / SR
    x[s:] = np.sin(2 * np.pi * 1000 * t) * db(-12)
    x[s:s + 192] *= np.linspace(0, 1, 192)[: len(x[s:s + 192])]
    return stereo(x)


def robot_score(rng, card_s):
    """Beep/bloop schedule for the takeover, then a descending evil 'boop-boop-booooop'."""
    beeps, t = [], 0.32
    talk_until = card_s - 1.05
    while t < talk_until - 0.12:
        d = float(rng.choice([0.07, 0.09, 0.12, 0.16]))
        beeps.append((t, t + d))
        t += d + float(rng.choice([0.03, 0.05, 0.08]))
    boop_start = card_s - 0.98
    return dict(beeps=beeps, talk_until=talk_until, boop_start=boop_start)


def robot_audio(n, rng, score):
    x = np.zeros(n, np.float32)
    # takeover "zwoop"
    k = int(0.28 * SR)
    sweep = tone(np.geomspace(140, 2200, k)) * np.linspace(1, 0.3, k)
    x[:k] += sweep * 0.6
    scale = [392, 440, 523, 587, 659, 784, 880, 1047, 1175, 1319]
    for a, b in score["beeps"]:
        m = int((b - a) * SR)
        f0 = float(rng.choice(scale))
        kind = rng.random()
        if kind < 0.35:  # bloop up
            fr = np.geomspace(f0 * 0.6, f0 * 1.4, m)
        elif kind < 0.6:  # bloop down
            fr = np.geomspace(f0 * 1.5, f0 * 0.7, m)
        else:
            fr = np.full(m, f0)
        env = np.minimum(1, np.minimum(np.arange(m), m - np.arange(m)) / 120)
        s = int(a * SR)
        x[s:s + m] += tone(fr)[: n - s] * env[: n - s] * 0.5
    # evil boop-boop-booooop powering down
    t0 = score["boop_start"]
    for f0, d in ((523, 0.14), (392, 0.14)):
        m, s = int(d * SR), int(t0 * SR)
        env = np.minimum(1, np.minimum(np.arange(m), m - np.arange(m)) / 150)
        x[s:s + m] += tone(np.full(m, f0)) * env * 0.55
        t0 += d + 0.05
    s = int(t0 * SR)
    m = n - s
    tt = np.arange(m) / SR
    fr = np.geomspace(330, 55, m) * (1 + 0.04 * np.sin(2 * np.pi * 7 * tt))
    x[s:] += tone(fr) * np.minimum(1, np.arange(m) / 150) * 0.6
    x = crush(x[:, None], 5, 4)[:, 0]  # 8-bit-ish crunch
    x = fft_filter(x[:, None], lambda f: 1 / np.sqrt(1 + (f / 7000) ** 4))[:, 0]
    return stereo(limit(norm(x, -8.5), -8.0))


# --------------------------------------------------------------------------- encode


def encode(out, frames, audio, fps, nframes, workdir):
    """Pipe RGB frames + float stereo audio into a house-format H.264/AAC file (BT.709, like the originals)."""
    a = np.asarray(audio, np.float32).copy()
    k = min(96, len(a))
    a[len(a) - k:] *= np.linspace(1, 0, k)[:, None]  # 2 ms de-click, not a fade
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        wav = Path(tmp) / "audio.wav"
        media.write_wav(wav, a)
        cmd = [media.ffmpeg(), "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
               "-i", wav, "-map", "0:v", "-map", "1:a",
               "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
               *media.video_args(18),
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
               *media.audio_args(), "-t", f"{nframes / fps:.3f}", "-movflags", "+faststart", out]
        proc = subprocess.Popen([str(c) for c in cmd], stdin=subprocess.PIPE)
        count = 0
        try:
            for f in frames:
                proc.stdin.write(np.ascontiguousarray(f, np.uint8).tobytes())
                count += 1
        except BrokenPipeError:
            pass
        finally:
            proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {out}")
    return count


# --------------------------------------------------------------------------- main


def render(out, *, style="glitch-beep", source=None, source_start=None, lead_in=0.24,
           ident="ROBOTS REVENGE NEWS", villain="ROBOT", ctx=None):
    """Render a signal-lost transition to `out` and return its Path.

    style         'glitch-beep' (colour bars + BEEP), 'hacked' (the villains took over) or 'static'
    source        the previous scene; its last frames freeze and break up. None draws a stand-in.
    source_start  where the lead-in starts in the source (None: the last `lead_in` seconds)
    ident         the channel name on the colour-bars card (and its initials on the hacked screen)
    villain       who took over: 'ROBOT' -> "ROBOT CONTROL 100%", "BROADCAST TAKEN OVER BY ROBOTS"
    """
    ctx = ctx or Ctx()
    style = style_name(style)
    fps = int(ctx.fps)
    spf = SR // fps  # audio samples per video frame (48000 divides by 25 and 30)
    tl = TIMELINE[style]
    seed = SEEDS[style] + ctx.seed
    rng = np.random.default_rng(seed)
    nlead = int(round(lead_in * fps))
    n_glitch, n_static, n_card = (int(round(tl[p] * fps)) for p in ("glitch", "static", "card"))

    if source:
        pool, first = read_lead(source, source_start, nlead, fps)
        grain = read_audio(source, first + nlead / fps - 0.3, 0.3, spf)
    else:
        pool, grain = synth_lead(nlead, seed + 1000), None
    lead = pool[:nlead]
    plan = glitch_plan(n_glitch, rng, fps)
    snow = Snow(rng, fps)

    # ---- sound (made first, in the same order as the originals, so the look stays seeded the same)
    parts = [np.zeros((len(lead) * spf, 2), np.float32), glitch_audio(plan, rng, grain, spf),
             static_audio(n_static * spf, rng)]
    score = None
    if style == "glitch-beep":
        parts.append(beep_audio(n_card * spf))
    elif style == "hacked":
        score = robot_score(rng, n_card / fps)
        parts.append(robot_audio(n_card * spf, rng, score))
    audio = np.concatenate(parts)
    total = len(lead) + n_glitch + n_static + n_card
    nframes = min(total, ctx.limit_frames) if ctx.limit_frames else total

    def frames():
        yield from lead
        feed = pool[:max(nlead, 1)]  # the last of these is the frame that freezes
        yield from render_glitch(feed, plan, rng, snow)
        yield from render_static(snow, n_static, punch(feed[-1], 1.14), rng, fps,
                                 roll_end=14.0 if style == "glitch-beep" else 8.0)
        if style == "glitch-beep":
            yield from card_bars(n_card, rng, fps, ident)
        elif style == "hacked":
            yield from card_hacked(n_card, rng, score, snow, fps, hacked_words(villain, ident))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    encode(out, islice(frames(), nframes), audio[: nframes * spf], fps, nframes, ctx.scratch("signal_lost"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--style", default="glitch-beep", help="glitch-beep | hacked | static (or A/B/C)")
    ap.add_argument("--source", help="lead-in clip (default: a drawn stand-in)")
    ap.add_argument("--source-start", type=float, help="lead-in start in the source, seconds (default: its end)")
    ap.add_argument("--lead-in", type=float, default=0.24,
                    help="seconds of source before the freeze (the last of them is the frame that freezes)")
    ap.add_argument("--ident", default="ROBOTS REVENGE NEWS", help="channel name on the colour-bars card")
    ap.add_argument("--villain", default="ROBOT", help="who takes over the broadcast (singular)")
    ap.add_argument("--fps", type=int, default=25, choices=(25, 30))
    ap.add_argument("--frames", type=int, help="render only the first N frames")
    ap.add_argument("--seed", type=int, default=0, help="added to the per-style seed")
    a = ap.parse_args(argv)
    ctx = Ctx(fps=a.fps, seed=a.seed, limit_frames=a.frames)
    out = render(a.out, style=a.style, source=a.source, source_start=a.source_start, lead_in=a.lead_in,
                 ident=a.ident, villain=a.villain, ctx=ctx)
    info = media.probe(out)
    print(f"{out}: {info.duration:.2f}s" if info else out)


if __name__ == "__main__":
    main()
