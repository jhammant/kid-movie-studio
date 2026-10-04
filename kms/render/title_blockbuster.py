"""Blockbuster: a chrome movie title on black (6 s).

The title in polished chrome with a slow push-in, a light sweep across the metal, a lens
glint, drifting embers and a spark burst when it lands. The last line gets the hot red
finish (ROBOTS / REVENGE), a lone "THE" or "OF" becomes a small spaced-out kicker line.
Audio: a riser, a deep detuned "BWAAAM" braam with a sub boom, a metallic clang when the
title lands, a shimmer that follows the sweep and a long ring-out.

Every frame is drawn with PIL/numpy/OpenCV and all audio is synthesised with numpy, then
muxed with ffmpeg.

    python -m kms.render.title_blockbuster OUT.mp4 --title "THE DINOSAUR DISCO"
    python -m kms.render.title_blockbuster OUT.mp4 --stills DIR      # a few PNGs, no video
"""
import argparse
import itertools
import subprocess
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import signal

from kms import media
from kms.fonts import font
from kms.render import H, SR, W, Ctx

DUR = 6.0
T_HIT = 0.92            # title lands: clang + braam (frame 23 at 25 fps)
SWEEP = (1.55, 3.35)    # light sweep across the metal
T_GLINT = 3.30          # star glint on the top-right corner of the hero line
T_GLINT2 = 2.05         # smaller glint on its top-left corner
FADE_OUT = 0.5
SS = 1.15               # title canvas supersampling (push-in stays below this)
CW, CH = int(W * SS), int(H * SS)
FONT = "arial-black"
STILLS = (0.5, 1.0, 2.4, 3.4, 5.0)

# Layout. Lines are fitted to the same width (a justified block), then evened out.
TARGET_W = 0.70 * W * SS      # every main line is fitted to this width
TRACK_BASE, TRACK_HOT, TRACK_KICKER = 0.04, 0.10, 0.25   # letter spacing, in ems
MAX_CAP_RATIO = 1.35          # no main line's capitals taller than this x the smallest
MAX_CAP = 0.26 * CH           # absolute cap-height limit (a single short word)
KICKER_CAP = 0.45             # kicker capitals vs the smallest main line
MAX_BLOCK = 0.54 * CH         # whole title block, top of first line to bottom of last
BEVEL_REF = 240               # font size the bevel and extrusion were tuned at (and above)
KICKER_WORDS = {"THE", "A", "AN", "OF", "AND", "&", "IN", "ON", "AT", "TO", "FOR", "VS", "VS.",
                "LA", "LE", "LES", "EL", "LOS", "LAS", "DE", "DU", "DES", "DER", "DIE", "DAS"}


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, dtype=np.float32) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def gradient_lut(stops, n=1024):
    pos = np.array([s[0] for s in stops], np.float32)
    col = np.array([s[1] for s in stops], np.float32)
    x = np.linspace(0, 1, n, dtype=np.float32)
    return np.stack([np.interp(x, pos, col[:, c]) for c in range(3)], -1)


# Classic chrome: sky above a hard dark horizon, warm ground below.
STEEL = gradient_lut([
    (0.00, (250, 253, 255)), (0.22, (196, 214, 236)), (0.44, (92, 108, 134)),
    (0.50, (18, 20, 28)), (0.55, (58, 52, 50)), (0.72, (160, 150, 140)),
    (0.88, (236, 232, 226)), (1.00, (255, 255, 255))])
# Hot red chrome for the last line.
RED = gradient_lut([
    (0.00, (255, 236, 220)), (0.22, (246, 150, 120)), (0.44, (170, 32, 22)),
    (0.50, (40, 4, 4)), (0.56, (110, 18, 12)), (0.74, (226, 70, 40)),
    (0.90, (255, 196, 150)), (1.00, (255, 246, 230))])
FINISHES = {"steel": STEEL, "red": RED}


# ----------------------------------------------------------------- layout
@dataclass
class Line:
    text: str
    finish: str
    track_em: float
    kicker: bool = False
    size: int = 0
    track: float = 0.0      # letter spacing, px (from the unrounded fitted size, as tuned)
    cap: float = 0.0
    over: float = 0.0       # glyphs poking above the capitals (accents), px
    desc: float = 0.0       # glyphs hanging below the baseline (Q, commas), px
    base: float = 0.0
    boxes: list = field(default_factory=list)


@lru_cache(maxsize=None)
def _ink(role, ch):
    """Size and bitmap of a character in a font, or None when it leaves no ink."""
    f = font(role, 48)
    x0, y0, x1, y1 = f.getbbox(ch)
    if x1 <= x0 or y1 <= y0:
        return None
    img = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(img).text((-x0, -y0), ch, font=f, fill=255)
    return (x1 - x0, y1 - y0), img.tobytes()


def drawable(text, role):
    """Drop characters the font can't draw (emoji, other scripts) rather than show empty boxes."""
    missing = _ink(role, "\U0010FFFD")  # a private-use code point: the font's "no glyph" box
    kept = "".join(c for c in text if c.isspace() or _ink(role, c) not in (None, missing))
    return " ".join(kept.split()) or " ".join(text.split())


def clean_title(title, uppercase=True):
    words = str(title).split()
    if not words:
        raise ValueError("title is empty")
    text = " ".join(words)
    return drawable(text.upper() if uppercase else text, FONT)


def partitions(words, max_lines=3):
    """Every way to break words into 1..max_lines lines, keeping their order."""
    n = len(words)
    for k in range(1, min(max_lines, n) + 1):
        for cuts in itertools.combinations(range(1, n), k - 1):
            b = (0,) + cuts + (n,)
            yield [" ".join(words[i:j]) for i, j in zip(b, b[1:])]


def _metrics(text, size):
    """Cap height, overshoot above it and descent below the baseline, at a font size."""
    f = font(FONT, size)
    cap = -f.getbbox("H", anchor="ls")[1]
    tops, bots = [0], [0]
    for c in text:
        if not c.isspace():
            x0, y0, x1, y1 = f.getbbox(c, anchor="ls")
            if x1 > x0 or y1 > y0:
                tops.append(-y0 - cap)
                bots.append(y1)
    tol = 0.05 * cap  # round letters overshoot a little; that's not an accent
    over, desc = max(tops), max(bots)
    return cap, (over if over > tol else 0.0), (desc if desc > tol else 0.0)


def _fit_width(text, track_em):
    """Font size at which the tracked line is exactly TARGET_W wide."""
    probe = font(FONT, 100)
    return 100 * TARGET_W / (probe.getlength(text) + track_em * 100 * max(len(text) - 1, 0) + 1e-6)


def plan(lines_text):
    """Lay out one way of breaking the title. Returns (lines, score)."""
    k = len(lines_text)
    is_kicker = [k > 1 and all(w in KICKER_WORDS for w in t.split()) for t in lines_text]
    if all(is_kicker):
        is_kicker = [False] * k
    lines = []
    for i, t in enumerate(lines_text):
        hot = k > 1 and i == k - 1
        track = TRACK_KICKER if is_kicker[i] else (TRACK_HOT if hot else TRACK_BASE)
        lines.append(Line(t, "red" if hot else "steel", track, is_kicker[i]))

    cap100 = font(FONT, 100).getbbox("H", anchor="ls")
    cap_per_size = -cap100[1] / 100
    sizes = [_fit_width(ln.text, ln.track_em) for ln in lines]
    main = [s for s, ln in zip(sizes, lines) if not ln.kicker]
    smallest = min(main)
    for i, ln in enumerate(lines):
        if ln.kicker:
            sizes[i] = min(sizes[i], KICKER_CAP * smallest)
        else:
            sizes[i] = min(sizes[i], MAX_CAP_RATIO * smallest, MAX_CAP / cap_per_size)

    # Shrink everything together until the block fits (accents and gaps included).
    for _ in range(8):
        for ln, s in zip(lines, sizes):
            ln.size, ln.track = max(8, int(s)), ln.track_em * s
            ln.cap, ln.over, ln.desc = _metrics(ln.text, ln.size)
        total = block_height(lines)
        if total <= MAX_BLOCK:
            break
        sizes = [s * MAX_BLOCK / total * 0.995 for s in sizes]
    score = min(ln.cap for ln in lines if not ln.kicker) * (1 - 0.1 * (k - 1))
    return lines, score


def gaps(lines):
    return [0.20 * max(a.cap, b.cap) + a.desc + b.over for a, b in zip(lines, lines[1:])]


def block_height(lines):
    return lines[0].over + sum(ln.cap for ln in lines) + sum(gaps(lines)) + lines[-1].desc


def layout(title, uppercase=True):
    """The best line breaks for a title, sized and placed on the canvas."""
    words = clean_title(title, uppercase).split()
    best = max((plan(p) for p in partitions(words)), key=lambda r: r[1])[0]
    top = CH / 2 - block_height(best) / 2 - 0.02 * CH + best[0].over
    y = top
    for ln, gap in zip(best, gaps(best) + [0]):
        ln.base = y + ln.cap
        y = ln.base + gap
    return best


# ----------------------------------------------------------------- title art
def draw_tracked(draw, text, fnt, cx, baseline, track):
    widths = [fnt.getlength(c) for c in text]
    total = sum(widths) + track * (len(text) - 1)
    x = cx - total / 2
    boxes = []
    for c, w in zip(text, widths):
        if not c.isspace():
            draw.text((x, baseline), c, font=fnt, fill=255, anchor="ls")
            boxes.append((x, x + w))
        x += w + track
    return boxes


def build_title(lines, rng):
    """Return (rgba, aux, anchors) on the CW x CH canvas.

    rgba: premultiplied chrome (0..1) + alpha. aux: face alpha, face luminance,
    soft glow. anchors: canvas points used for glints/sparks.
    """
    alphas = []
    for ln in lines:
        m = Image.new("L", (CW, CH), 0)
        ln.boxes = draw_tracked(ImageDraw.Draw(m), ln.text, font(FONT, ln.size), CW / 2, ln.base, ln.track)
        alphas.append(np.asarray(m, np.float32) / 255)
    alpha = alphas[0] if len(alphas) == 1 else np.maximum.reduce(alphas)
    owner = np.argmax(np.stack(alphas), axis=0) if len(alphas) > 1 else np.zeros(alpha.shape, np.int64)

    # Bevel height field from the distance to the glyph edge (quarter round).
    # Small text gets a proportionally narrower bevel so its letters keep a flat face.
    scale = [float(np.clip(ln.size / BEVEL_REF, 0.55, 1.0)) for ln in lines]
    dist = cv2.distanceTransform((alpha > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    if min(scale) == 1.0:
        bw = 13 * SS
    else:
        bw = np.choose(owner, [13 * SS * s for s in scale]).astype(np.float32)
    d = np.clip(dist / bw, 0, 1)
    hgt = cv2.GaussianBlur(1 - (1 - d) ** 2, (0, 0), 1.0)
    gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=3) / 8
    k = bw * 0.75
    nx, ny = -gx * k, -gy * k
    inv = 1 / np.sqrt(nx * nx + ny * ny + 1)
    nx, ny, nz = nx * inv, ny * inv, inv

    # Environment reflection coordinate: position within the line + normal tilt.
    yy, xx = np.mgrid[0:CH, 0:CW].astype(np.float32)
    wobble = 0.035 * np.sin(xx / (170 * SS)) + 0.02 * np.sin(xx / (53 * SS) + 1.3)
    del xx
    col = np.zeros((CH, CW, 3), np.float64)
    n = len(STEEL) - 1
    for i, ln in enumerate(lines):
        sel = owner == i
        v = (yy - (ln.base - ln.cap)) / ln.cap
        r = np.clip(v + 0.55 * ny + 0.12 * nx + wobble, 0, 1)
        col[sel] = FINISHES[ln.finish][(r[sel] * n).astype(np.int32)]
    del yy, v, r, wobble

    # Key light from upper left: lambert on the bevel + hot specular.
    L = np.array([-0.45, -0.65, 0.62], np.float32)
    L /= np.linalg.norm(L)
    ndl = nx * L[0] + ny * L[1] + nz * L[2]
    shade = np.clip(0.85 + 0.9 * (ndl - L[2]), 0.30, 1.35)
    hv = L + np.array([0, 0, 1], np.float32)
    hv /= np.linalg.norm(hv)
    spec = np.clip(nx * hv[0] + ny * hv[1] + nz * hv[2], 0, 1) ** 70
    # Brushed-metal streaks.
    brush = rng.standard_normal((CH, CW)).astype(np.float32)
    brush = cv2.blur(cv2.blur(brush, (151, 1)), (151, 1))
    brush /= brush.std() + 1e-6
    col = col * shade[..., None] * (1 + 0.045 * brush[..., None]) + 330 * spec[..., None]
    col = np.clip(col / 255, 0, 1.6)

    # Dark steel extrusion straight down gives the letters some thickness.
    depth = max(2, int(9 * SS * min(scale)))
    ext = np.zeros_like(alpha)
    for i in range(1, depth + 1):
        ext[i:] = np.maximum(ext[i:], alpha[:-i] * (1 - 0.5 * i / depth))
    ext_only = ext * (1 - alpha)
    ext_col = np.array([46, 52, 64], np.float32) / 255

    rgb = col * alpha[..., None] + ext_col * ext_only[..., None]
    a_all = np.clip(alpha + ext_only, 0, 1)
    lum = (0.3 * col[..., 0] + 0.59 * col[..., 1] + 0.11 * col[..., 2])
    glow = cv2.GaussianBlur(alpha, (0, 0), 28 * SS)
    rgba = np.dstack([rgb, a_all]).astype(np.float32)
    aux = np.dstack([alpha, lum * alpha, glow, np.zeros_like(alpha)]).astype(np.float32)
    rgba = cv2.GaussianBlur(rgba, (0, 0), 0.45)   # pre-filter before downscale warp

    # Glints sit on the top corners of the hero line: the first one (nearly) as wide as the widest.
    widths = [(b[-1][1] - b[0][0]) if b else 0 for b in (ln.boxes for ln in lines)]
    hero = next(ln for ln, w in zip(lines, widths) if w >= 0.9 * max(widths))
    lefts = [ln.boxes[0][0] for ln in lines if ln.boxes]
    rights = [ln.boxes[-1][1] for ln in lines if ln.boxes]
    hb = hero.boxes or [(CW / 2, CW / 2)]
    anchors = {
        "s_corner": (hb[-1][1] - 0.12 * hero.cap, hero.base - 0.93 * hero.cap),
        "r_corner": (hb[0][0] + 0.08 * hero.cap, hero.base - 0.98 * hero.cap),
        "center": (CW / 2, (lines[0].base - lines[0].cap + lines[-1].base) / 2),
        "x_span": (min(lefts, default=CW / 2), max(rights, default=CW / 2)),
    }
    return rgba, aux, anchors


# ---------------------------------------------------------------- particles
class Embers:
    def __init__(self, rng, n=170):
        self.x0 = rng.uniform(-50, W + 50, n)
        self.y0 = rng.uniform(0, H + 120, n)
        depth = rng.uniform(0, 1, n) ** 1.6
        self.vy = -(25 + 110 * depth)
        self.r = 1.0 + 2.6 * depth
        self.amp = rng.uniform(8, 45, n)
        self.fx = rng.uniform(0.15, 0.6, n)
        self.ph = rng.uniform(0, 2 * np.pi, n)
        self.b = rng.uniform(0.35, 1.0, n)
        self.ff = rng.uniform(3, 9, n)
        self.warm = rng.uniform(0, 1, n)

    def pos(self, t):
        y = (self.y0 + self.vy * t) % (H + 120) - 60
        x = self.x0 + self.amp * np.sin(2 * np.pi * self.fx * t + self.ph)
        return x, y

    def draw(self, layer, t, gain):
        if gain <= 0:
            return
        x1, y1 = self.pos(t)
        x0, y0 = self.pos(t - 0.03)
        flick = 0.6 + 0.4 * np.sin(2 * np.pi * self.ff * t + self.ph * 3)
        for i in range(len(x1)):
            if abs(y1[i] - y0[i]) > 200:
                continue  # wrapped this frame
            b = self.b[i] * flick[i] * gain
            c = (255 * b, (120 + 90 * self.warm[i]) * b, (30 + 40 * self.warm[i]) * b)
            cv2.line(layer, (int(x0[i] * 4), int(y0[i] * 4)), (int(x1[i] * 4), int(y1[i] * 4)),
                     c, max(2, int(round(self.r[i] * 1.3))), cv2.LINE_AA, shift=2)
        # A few big out-of-focus embers drifting in front.
        for i in range(0, len(x1), 9):
            if abs(y1[i] - y0[i]) > 200:
                continue
            b = 0.35 * self.b[i] * flick[i] * gain
            cv2.circle(layer, (int(x1[i] * 1.07 * 4) % (W * 4), int(y1[i] * 4)), int(4 * (7 + 9 * self.warm[i])),
                       (255 * b, 130 * b, 40 * b), -1, cv2.LINE_AA, shift=2)


class Sparks:
    """Burst of hot sparks from the letters when the title lands."""

    def __init__(self, rng, points, t0, n=160):
        idx = rng.integers(0, len(points), n)
        p = points[idx].astype(np.float64)
        ang = rng.uniform(0, 2 * np.pi, n)
        spd = rng.uniform(250, 1150, n) * rng.uniform(0.4, 1.0, n) ** 0.5
        v = np.stack([np.cos(ang) * spd, np.sin(ang) * spd - 380], -1)
        self.t0 = t0
        self.life = rng.uniform(0.35, 1.4, n)
        self.thick = rng.choice([1, 2, 2, 3], n)
        dt = 1 / 400
        steps = int(1.6 / dt) + 2
        self.traj = np.zeros((steps, n, 2))
        for s in range(steps):
            self.traj[s] = p
            v[:, 1] += 1500 * dt
            v *= np.exp(-1.6 * dt)
            p = p + v * dt
        self.dt = dt

    def at(self, age):
        i = np.clip(age / self.dt, 0, len(self.traj) - 1)
        return self.traj[int(i)]

    def draw(self, layer, t):
        age = t - self.t0
        if age < 0 or age > 1.6:
            return
        p1 = self.at(age)
        p0 = self.at(max(0.0, age - 0.045))
        cols = np.array([[255, 255, 235], [255, 214, 100], [255, 120, 30], [170, 35, 8]])
        for i in range(p1.shape[0]):
            a = age / self.life[i]
            if a >= 1:
                continue
            heat = np.interp(a, [0, 0.25, 0.6, 1], [0, 1, 2, 3])
            k = int(min(heat, 2.999))
            c = cols[k] + (cols[k + 1] - cols[k]) * (heat - k)
            c = c * (1 - a) ** 0.8
            cv2.line(layer, (int(p0[i, 0] * 4), int(p0[i, 1] * 4)), (int(p1[i, 0] * 4), int(p1[i, 1] * 4)),
                     tuple(float(x) for x in c), int(self.thick[i]), cv2.LINE_AA, shift=2)


def add_glint(img, cx, cy, inten, size, rot=0.0, tint=(1.0, 0.97, 0.9)):
    """Four-point star + soft core + small diagonal spikes, additive."""
    if inten <= 0.002:
        return
    R = int(size * 1.6)
    x0, x1 = max(0, int(cx - R)), min(W, int(cx + R))
    y0, y1 = max(0, int(cy - R)), min(H, int(cy + R))
    if x0 >= x1 or y0 >= y1:
        return
    xs = (np.arange(x0, x1, dtype=np.float32) - cx)[None, :]
    ys = (np.arange(y0, y1, dtype=np.float32) - cy)[:, None]
    c, s = np.cos(rot), np.sin(rot)
    u = xs * c + ys * s
    v = -xs * s + ys * c
    r2 = u * u + v * v
    val = (np.exp(-r2 / (2 * (0.05 * size) ** 2)) + 0.35 * np.exp(-r2 / (2 * (0.2 * size) ** 2))
           + np.exp(-np.abs(v) / 1.4) * np.exp(-np.abs(u) / (0.55 * size))
           + 0.8 * np.exp(-np.abs(u) / 1.4) * np.exp(-np.abs(v) / (0.35 * size)))
    d1, d2 = (u + v) * 0.7071, (u - v) * 0.7071
    val += 0.35 * (np.exp(-np.abs(d2) / 1.2) * np.exp(-np.abs(d1) / (0.15 * size))
                   + np.exp(-np.abs(d1) / 1.2) * np.exp(-np.abs(d2) / (0.15 * size)))
    img[y0:y1, x0:x1] += (255 * inten * val)[..., None] * np.array(tint, np.float32)


def add_streak(img, cy, cx, inten):
    """Anamorphic blue lens streak across the frame."""
    if inten <= 0.002:
        return
    y0, y1 = max(0, int(cy - 40)), min(H, int(cy + 40))
    if y0 >= y1:
        return
    ys = (np.arange(y0, y1, dtype=np.float32) - cy)[:, None]
    xs = (np.arange(W, dtype=np.float32) - cx)[None, :]
    val = (np.exp(-np.abs(ys) / 1.8) * np.exp(-np.abs(xs) / 650)
           + 0.25 * np.exp(-np.abs(ys) / 12) * np.exp(-np.abs(xs) / 320))
    img[y0:y1] += (255 * inten * val)[..., None] * np.array([0.45, 0.72, 1.0], np.float32)


def fbm(rng, h, w, octaves=5):
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gh, gw = max(2, h // (160 >> o)), max(2, w // (160 >> o))
        g = rng.standard_normal((gh, gw)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.55
    out /= tot
    out = (out - out.min()) / (out.max() - out.min())
    return out


# ---------------------------------------------------------------- renderer
class Renderer:
    def __init__(self, title="ROBOTS REVENGE", uppercase=True, fps=25, seed=0):
        self.fps = fps
        self.lines = layout(title, uppercase)
        self.rng = np.random.default_rng(42 + seed)
        self.rgba, self.aux, self.anch = build_title(self.lines, np.random.default_rng(3 + seed))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        self.vignette = (1 - 0.6 * smoothstep(0.55, 1.45, r))[..., None]
        self.xx, self.yy = xx, yy
        # Background: teal haze up top, warm ember glow from below.
        self.haze = fbm(self.rng, H + 60, W + 260)
        center = np.exp(-(((xx - W / 2) / 760) ** 2 + ((yy - H * 0.47) / 330) ** 2))
        floor = smoothstep(0.55, 1.0, yy / H)
        self.bg_static = (center[..., None] * np.array([8, 16, 22], np.float32)
                          + floor[..., None] * np.array([34, 12, 3], np.float32))
        self.grain = [self.rng.standard_normal((H, W)).astype(np.float32) * 1.4 for _ in range(6)]
        self.embers = Embers(self.rng)
        pts = np.argwhere(self.aux[..., 0] > 0.6)[:, ::-1].astype(np.float64)  # (x, y) canvas
        if len(pts) == 0:  # nothing drawable (e.g. glyphs the font lacks): spark from the middle
            pts = np.array([[CW / 2, CH / 2]])
        pts = pts[self.rng.integers(0, len(pts), 4000)]
        self.sparks = Sparks(self.rng, self.to_screen(pts, T_HIT), T_HIT)

    @staticmethod
    def cam(t):
        s = 0.955 + 0.095 * (t / DUR) ** 0.9
        dx = dy = 0.0
        if t >= T_HIT:
            e = 9 * np.exp(-(t - T_HIT) / 0.11)
            dx = e * np.sin(2 * np.pi * 23 * t)
            dy = e * np.cos(2 * np.pi * 19 * t + 0.7)
        return s, dx, dy

    def matrix(self, t):
        s, dx, dy = self.cam(t)
        k = s / SS
        return np.float32([[k, 0, W / 2 - k * CW / 2 + dx], [0, k, H / 2 - k * CH / 2 + dy]])

    def to_screen(self, pts, t):
        M = self.matrix(t)
        return pts @ M[:, :2].T + M[:, 2]

    def render(self, t, fi):
        M = self.matrix(t)
        rgba = cv2.warpAffine(self.rgba, M, (W, H), flags=cv2.INTER_LINEAR)
        aux = cv2.warpAffine(self.aux, M, (W, H), flags=cv2.INTER_LINEAR)
        a_all, face, lum, glow = rgba[..., 3:4], aux[..., 0:1], aux[..., 1:2], aux[..., 2:3]

        # Title visibility: dim ghost before the hit, full chrome after.
        vis = 1.0 if t >= T_HIT else 0.13 * float(smoothstep(0.35, 0.88, t))
        flash = float(np.exp(-(t - T_HIT) / 0.2)) if t >= T_HIT else 0.0
        amb = float(smoothstep(0.0, 0.8, t))

        ox = int(t * 14) % 200
        oy = int(t * 6) % 50
        haze = self.haze[oy:oy + H, ox:ox + W][..., None]
        img = self.bg_static * amb + haze * np.array([6, 13, 18], np.float32) * amb
        img += glow * np.array([20, 44, 62], np.float32) * (0.35 * amb + 0.9 * vis)

        layer = np.zeros((H, W, 3), np.uint8)
        self.embers.draw(layer, t, 0.85 * amb)
        emb = layer.astype(np.float32)
        small = cv2.resize(emb, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        img += cv2.GaussianBlur(emb, (0, 0), 1.2) + 3.0 * cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.5), (W, H))

        # Title over.
        img = img * (1 - a_all * min(1.0, vis * 3)) + rgba[..., :3] * 255 * vis
        if flash > 0:
            img += face * (255 * 0.9 * flash) * np.array([1.0, 0.97, 0.9], np.float32)
            img += 40 * float(np.exp(-(t - T_HIT) / 0.07))

        # Light sweep across the metal.
        if SWEEP[0] <= t <= SWEEP[1]:
            p = float(smoothstep(SWEEP[0], SWEEP[1], t))
            xs0, xs1 = self.to_screen(np.array([[self.anch["x_span"][0], 0], [self.anch["x_span"][1], 0]]), t)[:, 0]
            c = xs0 - 260 + (xs1 - xs0 + 520) * p
            u = self.xx + 0.42 * (self.yy - H / 2) - c
            band = (np.exp(-(u / 48) ** 2) + 0.3 * np.exp(-(u / 170) ** 2)
                    + 0.55 * np.exp(-((u + 95) / 14) ** 2))
            img += (band[..., None] * face * (0.3 + 0.8 * lum)) * 255 * np.array([1.0, 0.98, 0.94], np.float32)

        # Sparks (with glow).
        layer = np.zeros((H, W, 3), np.uint8)
        self.sparks.draw(layer, t)
        if layer.any():
            sp = layer.astype(np.float32)
            small = cv2.resize(sp, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
            img += sp + 2.5 * cv2.resize(cv2.GaussianBlur(small, (0, 0), 3), (W, H))

        # Lens glints: big one on impact, star on the hero line's corner after the sweep.
        if t >= T_HIT:
            e = float(np.exp(-(t - T_HIT) / 0.32))
            cx, cy = self.to_screen(np.array([self.anch["center"]]), t)[0]
            add_glint(img, cx, cy, 1.1 * e, 520, rot=0.0)
            add_streak(img, cy, cx, 0.9 * e)
        for t0, key, peak, size in ((T_GLINT, "s_corner", 1.25, 420), (T_GLINT2, "r_corner", 0.5, 200)):
            a = t - t0
            if -0.12 < a < 0.9:
                env = float(smoothstep(-0.12, 0.0, a)) if a < 0 else float(np.exp(-a / 0.22))
                gx, gy = self.to_screen(np.array([self.anch[key]]), t)[0]
                add_glint(img, gx, gy, peak * env, size * (0.7 + 0.3 * env), rot=0.35 * a)
                add_streak(img, gy, gx, 0.55 * peak * env)

        # Bloom: let the hot highlights bleed a little light.
        small = cv2.resize(np.maximum(img - 190, 0), (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        img += 0.9 * cv2.resize(cv2.GaussianBlur(small, (0, 0), 5), (W, H))
        img *= self.vignette
        img += self.grain[fi % len(self.grain)][..., None]
        img *= 1 - float(smoothstep(DUR - FADE_OUT, DUR, t + 0.5 / self.fps))
        return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- audio
def pan_gains(p):
    return np.sqrt(0.5 * (1 - p)), np.sqrt(0.5 * (1 + p))


def norm(x):
    return x / (np.max(np.abs(x)) + 1e-12)


def synth_audio(rng):
    """The full 6 s soundtrack, (2, n) floats. Hits are placed in seconds, so any fps lines up."""
    n = int(round(DUR * SR))
    t = np.arange(n) / SR
    hit = int(round(T_HIT * SR))
    dry = np.zeros((2, n))
    send = np.zeros((2, n))

    # 1) Riser: reverse-cymbal hiss + rumble that sucks into the hit.
    tr = t[:hit]
    env = (tr / T_HIT) ** 3
    hiss = signal.sosfilt(signal.butter(2, 3500, "hp", fs=SR, output="sos"), rng.standard_normal((2, hit)))
    rumble = signal.sosfilt(signal.butter(2, 160, "lp", fs=SR, output="sos"), rng.standard_normal((2, hit)))
    riser = 0.55 * norm(hiss) * env + 0.6 * norm(rumble) * env ** 0.7
    dry[:, :hit] += riser
    send[:, :hit] += 0.5 * riser

    # 2) Braam: detuned brassy saws (additive) with an opening/closing filter.
    nb = n - hit
    tb = np.arange(nb) / SR
    bd = nb / SR
    scoop = 1 - 0.055 * np.exp(-tb / 0.05)
    dive = 1 - 0.03 * smoothstep(1.6, bd, tb)
    fc = 170 + 2300 * (1 - np.exp(-tb / 0.035)) * np.exp(-tb / 0.8) + 480 * np.exp(-tb / 2.3)
    amp = (1 - np.exp(-tb / 0.02)) * (0.55 * np.exp(-tb / 1.1) + 0.45 * np.exp(-tb / 2.6))
    amp *= 1 - smoothstep(bd - 1.0, bd - 0.05, tb)
    braam = np.zeros((2, nb))
    for f0, g in ((55.0, 1.0), (82.41, 0.5), (110.0, 0.75), (130.81, 0.36)):
        for cents, pan in ((-9, -0.75), (0, 0.0), (9, 0.75)):
            f = f0 * 2 ** (cents / 1200) * scoop * dive
            ph = 2 * np.pi * np.cumsum(f) / SR
            v = np.zeros(nb)
            for k in range(1, int(min(70, 5200 / f0)) + 1):
                fk = k * f0
                lp = 1 / np.sqrt(1 + (fk / fc) ** 4)
                res = 1 + 1.3 * np.exp(-(np.log2(fk / fc) / 0.35) ** 2)
                v += (lp * res / k) * np.sin(k * ph + rng.uniform(0, 2 * np.pi))
            gl, gr = pan_gains(pan)
            braam[0] += g * gl * v
            braam[1] += g * gr * v
    braam = np.tanh(1.8 * norm(braam)) * amp
    braam = norm(braam)

    # 3) Sub boom + punch under the hit.
    fb = 34 + 62 * np.exp(-tb / 0.09)
    boom = np.sin(2 * np.pi * np.cumsum(fb) / SR) * np.exp(-tb / 0.75) * (1 - np.exp(-tb / 0.003))
    punch = signal.sosfilt(signal.butter(2, 400, "lp", fs=SR, output="sos"), rng.standard_normal(nb))
    boom = norm(boom + 1.2 * norm(punch) * np.exp(-tb / 0.018))

    # 4) Metallic clang: inharmonic partials with a slightly detuned stereo pair.
    ratios = np.array([1.0, 1.47, 2.09, 2.56, 2.98, 3.61, 4.23, 5.17, 5.93, 6.85, 8.12, 9.5, 11.2, 13.4])
    f0 = 196.0
    clang = np.zeros((2, nb))
    for ch in range(2):
        for i, r in enumerate(ratios):
            f = f0 * r * (1 + rng.normal(0, 0.0018))
            tau = 3.2 / r ** 0.85
            a = rng.uniform(0.6, 1.0) / r ** 0.45
            clang[ch] += a * np.sin(2 * np.pi * f * tb + rng.uniform(0, 6.28)) * np.exp(-tb / tau)
    clang *= 1 - np.exp(-tb / 0.0007)
    tick = signal.sosfilt(signal.butter(2, [1800, 9000], "bp", fs=SR, output="sos"), rng.standard_normal((2, nb)))
    thunk = signal.sosfilt(signal.butter(2, [500, 1400], "bp", fs=SR, output="sos"), rng.standard_normal((2, nb)))
    clang = norm(norm(clang) + 0.9 * norm(tick) * np.exp(-tb / 0.02) + 0.7 * norm(thunk) * np.exp(-tb / 0.05))

    # 5) Spark crackle after the hit.
    crack = np.zeros((2, nb))
    rate = 260 * np.exp(-tb / 0.35)
    hits = rng.uniform(0, 1, nb) < rate / SR
    crack[rng.integers(0, 2, nb), np.arange(nb)] = hits * rng.uniform(-1, 1, nb)
    crack = signal.sosfilt(signal.butter(2, [2500, 11000], "bp", fs=SR, output="sos"), crack)
    crack = norm(crack)

    # Lift the brassy mids so the BWAAAM reads on small speakers too.
    growl = signal.sosfilt(signal.butter(2, [220, 1800], "bp", fs=SR, output="sos"), braam)
    mix_b = 1.0 * braam + 0.55 * norm(growl) + 0.7 * boom[None] + 0.75 * clang + 0.10 * crack
    dry[:, hit:] += mix_b
    send[:, hit:] += 0.35 * braam + 0.65 * clang + 0.1 * crack

    # 6) Shimmer that follows the light sweep (panned L -> R) and the glint "ting".
    s0, s1 = int(SWEEP[0] * SR), int(SWEEP[1] * SR)
    ts = np.arange(s1 - s0) / SR
    pos = ts / (ts[-1] + 1e-9)
    sh = signal.sosfilt(signal.butter(2, [5000, 12000], "bp", fs=SR, output="sos"), rng.standard_normal(s1 - s0))
    sh = norm(sh) * np.sin(np.pi * pos) ** 2
    gl, gr = pan_gains(-0.8 + 1.6 * pos)
    dry[0, s0:s1] += 0.08 * sh * gl
    dry[1, s0:s1] += 0.08 * sh * gr
    send[0, s0:s1] += 0.08 * sh * gl
    send[1, s0:s1] += 0.08 * sh * gr
    g0 = int(T_GLINT * SR)
    tg = np.arange(n - g0) / SR
    ting = sum(a * np.sin(2 * np.pi * 2350 * r * tg) * np.exp(-tg / tau)
               for r, a, tau in ((1.0, 1.0, 0.45), (1.5, 0.45, 0.3), (2.76, 0.4, 0.2), (5.4, 0.2, 0.1)))
    ting = norm(ting) * (1 - np.exp(-tg / 0.002))
    dry[0, g0:] += 0.06 * ting
    dry[1, g0:] += 0.09 * ting
    send[:, g0:] += 0.12 * ting

    # Reverb: decaying stereo noise impulse response.
    nir = int(3.0 * SR)
    ti = np.arange(nir) / SR
    ir = rng.standard_normal((2, nir)) * np.exp(-6.91 * ti / 2.6)
    ir = signal.sosfilt(signal.butter(1, 4200, "lp", fs=SR, output="sos"), ir)
    ir[:, : int(0.025 * SR)] = 0
    ir /= np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True))
    wet = np.stack([signal.fftconvolve(send[c], ir[c])[:n] for c in range(2)])
    mix = dry + 0.55 * wet

    mix = signal.sosfilt(signal.butter(2, 28, "hp", fs=SR, output="sos"), mix)
    mix = np.tanh(1.25 * norm(mix)) / np.tanh(1.25)
    mix *= 10 ** (-1.8 / 20) / np.max(np.abs(mix))
    mix *= 1 - smoothstep(DUR - 0.4, DUR, t)
    mix[:, :240] *= np.linspace(0, 1, 240)
    return mix


def trim_audio(mix, seconds):
    """Cut the soundtrack to a shorter render, with a 10 ms fade so it doesn't click."""
    n = int(round(seconds * SR))
    if n >= mix.shape[1]:
        return mix
    out = mix[:, :n].copy()
    f = min(n, int(0.01 * SR))
    if f:
        out[:, n - f:] *= np.linspace(1, 0, f)
    return out


# ---------------------------------------------------------------- encode
def frames(renderer, n, fps, workers=1):
    """Frames 0..n-1 in order; a few threads in flight when workers > 1 (numpy/OpenCV release the GIL)."""
    if workers <= 1 or n < 8:
        for i in range(n):
            yield renderer.render(i / fps, i)
        return
    with ThreadPoolExecutor(workers) as ex:
        pending = deque()
        for i in range(n):
            pending.append(ex.submit(renderer.render, i / fps, i))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def encode(out, wav, frame_iter, fps, seconds, crf=18):
    """Raw RGB frames + a wav -> H.264/AAC mp4 in the house format (BT.709, faststart)."""
    cmd = [media.ffmpeg(), "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-i", str(wav), "-map", "0:v", "-map", "1:a",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
           *media.video_args(crf),
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           *media.audio_args(), "-t", f"{seconds:.3f}", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for fr in frame_iter:
            proc.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
        proc.stdin.close()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {out}")


# ---------------------------------------------------------------- api
def render(out, *, title="ROBOTS REVENGE", uppercase=True, ctx=None) -> Path:
    """Render the blockbuster title card to `out` (mp4). Returns the path."""
    ctx = ctx or Ctx()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = ctx.nframes(DUR)
    seconds = n / ctx.fps
    r = Renderer(title, uppercase, ctx.fps, ctx.seed)
    with tempfile.TemporaryDirectory(dir=ctx.scratch("title_blockbuster")) as td:
        wav = Path(td) / "title_blockbuster.wav"
        media.write_wav(wav, trim_audio(synth_audio(np.random.default_rng(2024 + ctx.seed)), seconds).T)
        encode(out, wav, frames(r, n, ctx.fps, ctx.workers), ctx.fps, seconds)
    return out


def stills(out_dir, times=STILLS, *, title="ROBOTS REVENGE", uppercase=True, ctx=None, stem="blockbuster"):
    """PNG frames at a few moments, for a quick look without encoding. Returns the paths."""
    ctx = ctx or Ctx()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r = Renderer(title, uppercase, ctx.fps, ctx.seed)
    paths = []
    for tt in times:
        fi = int(round(tt * ctx.fps))
        p = out_dir / f"{stem}_t{tt:.2f}.png"
        Image.fromarray(r.render(fi / ctx.fps, fi)).save(p)
        paths.append(p)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kms.render.title_blockbuster",
                                 description="Chrome blockbuster title card (6 s).")
    ap.add_argument("out", help="output .mp4")
    ap.add_argument("--title", default="ROBOTS REVENGE")
    ap.add_argument("--keep-case", action="store_true", help="don't capitalise the title")
    ap.add_argument("--fps", type=int, default=25, choices=(25, 30))
    ap.add_argument("--frames", type=int, default=None, help="render only the first N frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--stills", metavar="DIR", help="write a few PNG frames to DIR instead of the video")
    ap.add_argument("--at", default=",".join(f"{t:g}" for t in STILLS), help="still times in seconds")
    a = ap.parse_args(argv)
    ctx = Ctx(fps=a.fps, seed=a.seed, limit_frames=a.frames)
    if a.workers:
        ctx.workers = a.workers
    if a.stills:
        times = [float(x) for x in a.at.split(",") if x.strip()]
        for p in stills(a.stills, times, title=a.title, uppercase=not a.keep_case, ctx=ctx, stem=Path(a.out).stem):
            print(p)
        return
    media.require_ffmpeg()
    print(render(a.out, title=a.title, uppercase=not a.keep_case, ctx=ctx))


if __name__ == "__main__":
    main()
