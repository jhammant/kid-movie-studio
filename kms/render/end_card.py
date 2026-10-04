"""End card: "To Be Continued....." (or "The End") with the villain's eyes in the dark.

Black and silent for a beat, then two red robot eyes (round lenses joined like sunglasses) glow
up out of the dark, take a slow blink and look around. The words flicker on like a dodgy tube in
chrome and red, the dots land one at a time on a low synth "dun" plus a digital blip (the eyes
glance at each one), the eyes narrow into a cheeky-evil squint with one last pulse, and the picture
zaps off to black on a low boom.

Dots inside the text ("To Be Continued... Maybe!") land one at a time too, and whatever follows
them lands as the final beat. With dots=False the words land once with a single "dun" while the
eyes read across them. Long texts shrink to fit, or break onto two lines.

All pictures are drawn here with PIL/numpy/OpenCV (no drawtext needed) and all sound is
synthesised with numpy.

    python -m kms.render.end_card OUT [--text "The End" --no-dots] [--fps 25] [--frames N] [--still PNG]
"""
import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from kms import media
from kms.fonts import font
from kms.render import H, SR, W, Ctx, split_title

REF_FPS = 25  # per-frame effects were tuned at 25 fps; other rates are scaled to match
SEED = 9  # the tuned look at ctx.seed=0

# ---- timeline, in seconds
T_EYES_ON = 0.48            # black silence first
T_EYES_FULL = 1.28          # eyes fully lit
T_BLINK = (1.48, 1.68, 1.80, 2.04)  # start closing, shut, start opening, open
T_TEXT_ON = 2.16            # tube-flicker of the words starts
T_SHEEN = (2.40, 2.84)      # glint across the chrome once the words settle
T_GLITCH_SMALL = (2.76, 2.80)
T_LAND = 2.96               # the first dot lands...
LAND_GAP = 0.36             # ...and the rest follow this far apart
PULSE_AFTER = 0.16          # final squint + pulse starts this long after the last landing
PULSE_LEN = 0.56            # then a glitch burst, a one-frame CRT collapse and black
BLACK_AFTER = 0.64

# the words' tube flicker, one value per 25 fps frame from T_TEXT_ON
TEXT_FLICKER = [0.0, 0.85, 0.0, 0.0, 0.0, 0.45, 1.0, 0.15, 0.0, 1.0, 1.0, 0.55, 1.0, 0.8, 1.0]

# ---- layout
EYE_CY = 352
EYE_CX = (W // 2 - 178, W // 2 + 178)
LENS_R = 132
GLOW_R = 104
FONT_SIZE = 192
TWO_LINE_SIZE = 150         # biggest size when the text breaks onto two lines
DOT_TRACK = 24              # extra space between the dots so the trail reads (at FONT_SIZE)
TEXT_CY = 790
MAX_TEXT_W = 1720

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class Timeline:
    """Frame numbers for every event at a frame rate, for `n_land` landings (0: the words land once)."""

    def __init__(self, fps, n_land):
        self.fps = fps
        F = self.F
        self.reading = n_land == 0
        beats = max(n_land, 1)
        self.eyes_on, self.eyes_full = F(T_EYES_ON), F(T_EYES_FULL)
        self.stutter = (self.eyes_on + F(0.08), self.eyes_on + F(0.24))
        self.flare = self.eyes_on + F(0.12)
        self.blink = tuple(F(s) for s in T_BLINK)
        self.text_on = F(T_TEXT_ON)
        self.sheen = tuple(F(s) for s in T_SHEEN)
        self.glitch_small = tuple(F(s) for s in T_GLITCH_SMALL)
        self.lands = [F(T_LAND + j * LAND_GAP) for j in range(beats)]
        # glance further right with each landing: 0.15, 0.29, 0.43, 0.57, 0.71 for five dots
        self.looks = [round(0.15 + 0.56 * j / (beats - 1), 4) for j in range(beats)] if beats > 1 else [0.43]
        last = T_LAND + (2 if self.reading else beats - 1) * LAND_GAP
        self.read_end = F(last)  # reading mode: the eyes sweep across the words until here
        self.tension = F(T_LAND + 2 * LAND_GAP)
        self.pulse = (F(last + PULSE_AFTER), F(last + PULSE_AFTER + PULSE_LEN))
        self.cut = self.pulse[1]
        self.total = self.cut + F(BLACK_AFTER)
        self.still = self.pulse[0] - 1  # fully revealed, eyes round and steady

    def F(self, seconds):
        return int(round(seconds * self.fps))


# --------------------------------------------------------------------------- eyes

EY0, EY1, EX0, EX1 = 0, 760, 300, 1620  # crop that holds the eyes and their bloom
ey, ex = YY[EY0:EY1, EX0:EX1], XX[EY0:EY1, EX0:EX1]


def lens_shape():
    """The robot's black 'sunglasses': two round lenses joined by a bridge."""
    m = np.zeros(ey.shape, np.float32)
    for cx in EYE_CX:
        r = np.hypot(ex - cx, ey - EYE_CY)
        m = np.maximum(m, smooth(LENS_R + 1.5, LENS_R - 1.5, r))
    bridge = smooth(30, 27, np.abs(ey - (EYE_CY - 34))) * smooth(EYE_CX[1] + 5, EYE_CX[1] - 5, ex) \
        * smooth(EYE_CX[0] - 5, EYE_CX[0] + 5, ex)
    return np.maximum(m, bridge)


LENS = lens_shape()
LENS_EDGE = np.clip(LENS - cv2.erode(LENS, np.ones((9, 9), np.uint8)), 0, 1)


def eyes(level, openness, squint, look, pulse):
    """Returns (rgb crop, glow luminance crop) for the two eyes."""
    rgb = np.zeros(ey.shape + (3,), np.float32)
    lum = np.zeros(ey.shape, np.float32)
    # glossy black lenses, rim faintly lit by their own glow
    rgb += (LENS * 6 + LENS_EDGE * 70 * level * pulse)[..., None] * np.float32([1.0, 0.16, 0.14])
    for k, cx in enumerate(EYE_CX):
        dx, dy = ex - cx, ey - EYE_CY
        r = np.hypot(dx, dy)
        disc = smooth(GLOW_R + 3, GLOW_R - 3, r)
        inner = 1 if k == 0 else -1  # +x is towards the nose for the left eye
        top = -GLOW_R * openness + squint * GLOW_R * (0.62 + 0.62 * inner * dx / GLOW_R)
        if openness < 0.02:
            lid = np.zeros_like(r)
        else:  # lids close as an almond, not a letterbox
            b = GLOW_R * openness
            ell = np.hypot(dx / (GLOW_R * 1.04), dy / b)
            lid = smooth(1 + 2.5 / b, 1 - 2.5 / b, ell)
        lid = lid * smooth(top - 2, top + 2, dy)
        px, py = look[0] * GLOW_R * 0.32, look[1] * GLOW_R * 0.32
        core = np.exp(-(np.hypot(dx - px, dy - py) / (GLOW_R * 0.36)) ** 2)
        ring = 1 - 0.32 * np.exp(-((r - GLOW_R * 0.66) / 5.0) ** 2)  # robot iris ring
        shade = 0.55 + 0.45 * smooth(GLOW_R, GLOW_R * 0.3, r)
        a = disc * lid * level
        col = (np.float32([235, 22, 14]) * (shade * ring)[..., None]
               + np.float32([255, 205, 175]) * (core * 0.95 * pulse)[..., None])
        rgb = rgb * (1 - a[..., None]) + col * (a * pulse)[..., None]
        lum += a * (0.6 + core) * pulse
        # tiny glossy glint on the lens (top-left), so they read as the robot's sunglasses
        rot = np.exp(-((dx + dy + LENS_R * 1.05) / 9) ** 2) * smooth(LENS_R * 0.95, LENS_R * 0.75, r) \
            * smooth(LENS_R * 0.45, LENS_R * 0.7, r)
        rgb += (rot * 60 * level)[..., None] * np.float32([1, 0.85, 0.85])
    return rgb, lum


def bloom(lum, level):
    small = cv2.resize(lum, (lum.shape[1] // 4, lum.shape[0] // 4), interpolation=cv2.INTER_AREA)
    b = 0.9 * cv2.GaussianBlur(small, (0, 0), 6) + 0.75 * cv2.GaussianBlur(small, (0, 0), 22)
    b = cv2.resize(b, (lum.shape[1], lum.shape[0]), interpolation=cv2.INTER_LINEAR)
    return b[..., None] * np.float32([255, 38, 22]) * level


# --------------------------------------------------------------------------- text


def parse(text, dots):
    """(words, number of dots that land, tail that lands after them)."""
    t = " ".join(str(text).replace("…", "...").split())
    if not t:
        raise ValueError("the end card needs some text")
    if not dots:
        return t, 0, ""
    m = re.search(r"\.{2,}", t)
    if m is None:
        return t.rstrip(".") or t, 5, ""
    words, tail = t[:m.start()].rstrip(), t[m.end():].strip()
    if not words:
        return t, 0, ""
    return words, min(len(m.group()), 8), tail


def measure(lines, size):
    """Lay out lines of (kind, text) pieces at a size: (font, [(width, [(kind, text, x)])])."""
    f = font("impact", size)
    track = DOT_TRACK * size / FONT_SIZE
    dot_w, space = f.getlength("."), f.getlength(" ")
    rows = []
    for spec in lines:
        x, items = 0.0, []
        for kind, s in spec:
            if kind == "dot":
                items.append((kind, s, x))
                x += dot_w + track
            else:
                if kind == "tail" and items:
                    x += space - (track if items[-1][0] == "dot" else 0)
                items.append((kind, s, x))
                x += f.getlength(s)
        if items[-1][0] == "dot":
            x -= track * 0.5
        rows.append((x, items))
    return f, rows


def fit_lines(lines, start):
    size = start
    while size > 24:
        _, rows = measure(lines, size)
        widest = max(w for w, _ in rows)
        if widest <= MAX_TEXT_W:
            break
        size = min(size - 1, int(size * MAX_TEXT_W / widest))
    return max(size, 24)


def layout(text, dots):
    """Choose one line (shrunk to fit) or two, whichever reads bigger."""
    words, ndots, tail = parse(text, dots)
    trail = [("dot", ".")] * ndots + ([("tail", tail)] if tail else [])
    options = [[[("words", words)] + trail]]
    if tail:
        options.append([[("words", words)] + [("dot", ".")] * ndots, [("tail", tail)]])
    halves = split_title(words, max_lines=2)
    if len(halves) == 2:
        options.append([[("words", halves[0])], [("words", halves[1])] + trail])
    best = (fit_lines(options[0], FONT_SIZE), options[0])
    if best[0] < 140:
        for lines in options[1:]:
            size = fit_lines(lines, TWO_LINE_SIZE)
            if size > best[0] * 1.2 and size > best[0] + 12:
                best = max(best, (size, lines), key=lambda b: b[0])
    size, lines = best
    f, rows = measure(lines, size)
    return f, size, rows, ndots + (1 if tail else 0)


def chrome_gradient(h):
    """Chrome on top, red-hot reflection below the horizon line."""
    stops = [(0.00, (255, 255, 255)), (0.30, (200, 202, 214)), (0.49, (96, 96, 112)),
             (0.51, (40, 8, 12)), (0.60, (215, 28, 22)), (0.82, (255, 128, 104)), (1.00, (160, 18, 18))]
    t = np.linspace(0, 1, h)
    out = np.zeros((h, 3), np.float32)
    for c in range(3):
        out[:, c] = np.interp(t, [s[0] for s in stops], [s[1][c] for s in stops])
    return out


def text_layers(rng, text, dots):
    """Pre-renders the words and each landing piece as (rgb, alpha, glow) bands."""
    f, size, rows, n_land = layout(text, dots)
    cap_top, base = f.getbbox("T")[1], f.getbbox("T")[3]
    pitch = round(size * 1.15)
    centres = [TEXT_CY + (j - (len(rows) - 1) / 2) * pitch for j in range(len(rows))]
    band0 = int(max(0, min(TEXT_CY - 190, centres[0] - 190)))
    band1 = int(min(H, max(TEXT_CY + 190, centres[-1] + 190)))
    bh = band1 - band0
    y_texts = [cy - (cap_top + base) / 2 for cy in centres]  # centre on the cap height
    x0s = [(W - w) / 2 for w, _ in rows]

    # chrome face: each line gets its own horizon
    grad = chrome_gradient(base - cap_top + 4)
    gfull = np.zeros((bh, 3), np.float32)
    starts = [int(y + cap_top - band0) - 2 for y in y_texts]
    edges = [0] + [(starts[j] + len(grad) + starts[j + 1]) // 2 for j in range(len(rows) - 1)] + [bh]
    for j, gy0 in enumerate(starts):
        r0, r1 = edges[j], edges[j + 1]
        gfull[r0:r1] = grad[-1]
        gfull[r0:max(r0, gy0)] = grad[0]
        a, b = max(gy0, r0), min(gy0 + len(grad), r1)
        if b > a:
            gfull[a:b] = grad[a - gy0:b - gy0]

    # worn-chrome grunge + a few scratches, shared by all pieces
    n = rng.standard_normal((bh // 3 + 1, W // 3 + 1)).astype(np.float32)
    n = cv2.resize(cv2.GaussianBlur(n, (0, 0), 2.2), (W, bh), interpolation=cv2.INTER_CUBIC)
    n /= n.std()
    wear = np.clip((n - 1.9) * 1.4, 0, 1) * 0.7
    scratch = Image.new("L", (W, bh))
    sd = ImageDraw.Draw(scratch)
    left, right = min(x0s), max(x0 + w for x0, (w, _) in zip(x0s, rows))
    for _ in range(22):
        x, y = rng.uniform(left, right), rng.uniform(40, bh - 40)
        ang, ln = rng.uniform(-0.5, 0.5), rng.uniform(25, 110)
        sd.line([(x, y), (x + ln * np.cos(ang), y + ln * np.sin(ang))], fill=int(rng.uniform(120, 220)),
                width=int(rng.integers(1, 3)))
    wear = np.maximum(wear, np.asarray(scratch, np.float32) / 255)
    distress = 1 - 0.55 * wear

    keyline, outline = round(3 * size / FONT_SIZE), round(10 * size / FONT_SIZE)
    words, lands = [], []
    for (_, items), x0, y_text in zip(rows, x0s, y_texts):
        for kind, s, x in items:
            def mask(stroke):
                im = Image.new("L", (W, bh))
                ImageDraw.Draw(im).text((x0 + x, y_text - band0), s, font=f, fill=255,
                                        stroke_width=stroke, stroke_fill=255)
                return np.asarray(im, np.float32) / 255

            fill, dark, red = mask(0), mask(keyline), mask(outline)
            rgb = (red - dark)[..., None] * np.float32([235, 24, 20])          # red outer outline
            rgb += (dark - fill)[..., None] * np.float32([18, 4, 6])           # thin dark keyline
            rgb += fill[..., None] * gfull[:, None, :] * distress[..., None]   # chrome face
            glow = cv2.GaussianBlur(red, (0, 0), 16) * 1.2 + cv2.GaussianBlur(red, (0, 0), 45) * 0.9
            glow = glow[..., None] * np.float32([255, 28, 18])
            (words if kind == "words" else lands).append(dict(rgb=rgb, a=red, fill=fill, glow=glow))
    sheen_fill = np.maximum.reduce([w["fill"] for w in words])
    mid = (centres[0] + centres[-1]) / 2 - band0
    return dict(band=(band0, band1), words=words, lands=lands, sheen=sheen_fill, mid=mid, n_land=n_land)


# --------------------------------------------------------------------------- CRT + glitches


def rgb_split(img, dx, dy=0):
    out = img.copy()
    out[..., 0] = np.roll(img[..., 0], (dy, dx), axis=(0, 1))
    out[..., 2] = np.roll(img[..., 2], (-dy, -dx), axis=(0, 1))
    return out


def tear(img, rng, bands, maxshift):
    out = img.copy()
    for _ in range(bands):
        h = int(rng.integers(6, 120))
        y = int(rng.integers(0, H - h))
        out[y:y + h] = np.roll(img[y:y + h], int(rng.integers(-maxshift, maxshift + 1)), axis=1)
    return out


SCAN = np.ones((H, 1, 1), np.float32)
SCAN[::3] = 0.62
SCAN[1::3] = 0.92
_r2 = ((XX - W / 2) / (W / 2)) ** 2 + ((YY - H / 2) / (H / 2)) ** 2
VIGN = np.clip(1 - 0.32 * _r2, 0.45, 1)[..., None]
ROWS = np.arange(H, dtype=np.float32)


def crt(f, i, rng, fps):
    bar = (i * (REF_FPS / fps) * 9.0 + 300) % H  # slow rolling hum bar
    d = (ROWS - bar + H / 2) % H - H / 2
    hum = (1 - 0.10 * np.exp(-(d / (0.10 * H)) ** 2))[:, None, None]
    f = f * SCAN * VIGN * hum
    f += rng.normal(0, 4.5, (H // 2, W // 2, 1)).repeat(2, 0).repeat(2, 1).astype(np.float32)
    f += 4.0  # lifted CRT black
    out = np.clip(f, 0, 255).astype(np.uint8)
    out[..., 0] = np.roll(out[..., 0], -1, axis=1)  # a touch of convergence fringe
    out[..., 2] = np.roll(out[..., 2], 1, axis=1)
    return out


def crt_collapse(rng):
    """The tube switching off: everything squeezed into one white-hot line."""
    f = np.zeros((H, W, 3), np.float32)
    prof = np.exp(-((ROWS - H / 2) / 3.2) ** 2)[:, None] * smooth(0, 380, XX[0])[None, :] \
        * smooth(W, W - 380, XX[0])[None, :]
    f += prof[..., None] * np.float32([255, 235, 230])
    f += cv2.GaussianBlur(prof, (0, 0), 18)[..., None] * np.float32([255, 40, 30]) * 3
    return np.clip(f + rng.normal(0, 3, (H, 1, 1)), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- per-frame state


def landing_boost(age, fps):
    """Extra brightness on a piece as it lands, fading over three 25 fps frames."""
    return float(np.interp(age * REF_FPS / fps, [0, 1, 2, 3], [0.9, 0.5, 0.2, 0.0]))


def state(i, tl):
    s = dict(level=0.0, open=1.0, squint=0.0, look=(0.0, 0.0), pulse=1.0, text=0.0, lands=[], stamp=None, glitch=0)
    if i < tl.eyes_on:
        return s
    # eyes power on with a couple of stutters
    p = (i - tl.eyes_on) / (tl.eyes_full - tl.eyes_on)
    lvl = ease(p) ** 1.3
    if i in tl.stutter:
        lvl *= 0.25
    if i == tl.flare:
        lvl = min(1.0, lvl * 2.2)
    s["level"] = lvl
    # slow blink
    b0, b1, b2, b3 = tl.blink
    if b0 <= i < b1:
        s["open"] = 1 - ease((i - b0) / (b1 - b0))
    elif b1 <= i < b2:
        s["open"] = 0.0
    elif b2 <= i < b3:
        s["open"] = ease((i - b2) / (b3 - b2))
    # look down at the words as they flicker on, glance at each dot as it lands (or read along
    # the words when there are no dots), then snap back to centre to stare at you for the finale
    look = (0.0, 0.0)
    if i >= tl.text_on:
        q = ease((i - tl.text_on) / tl.F(0.24))
        look = (-0.25 * q, 0.6 * q)
    if tl.reading:
        if i >= tl.lands[0]:
            q = ease((i - tl.lands[0]) / max(1, tl.read_end - tl.lands[0]))
            look = (look[0] + (0.45 - look[0]) * q, look[1] + (0.7 - look[1]) * q)
    else:
        for k, fd in enumerate(tl.lands):
            if i >= fd:
                q = ease((i - fd + 1) / tl.F(0.12))
                look = (look[0] + (tl.looks[k] - look[0]) * q, look[1] + (0.7 - look[1]) * q)
    if i >= tl.pulse[0]:
        look = tuple(v * (1 - ease((i - tl.pulse[0]) / tl.F(0.16))) for v in look)
    s["look"] = look
    # words + dots
    j = i - tl.text_on
    if j >= 0:
        jj = int(j * REF_FPS / tl.fps)
        s["text"] = TEXT_FLICKER[jj] if jj < len(TEXT_FLICKER) else 1.0
    ages = [i - fd for fd in tl.lands if i >= fd]
    if tl.reading:
        s["stamp"] = ages[0] if ages else None
    else:
        s["lands"] = ages
    # final squint + pulse
    if i >= tl.pulse[0]:
        q = (i - tl.pulse[0]) / (tl.pulse[1] - tl.pulse[0])
        s["squint"] = 0.62 * ease(q * 1.6)
        s["pulse"] = 1.0 + 0.55 * ease(q) + 0.25 * np.sin(q * np.pi * 3) * q
    if i in tl.glitch_small or i == tl.cut - 1:
        s["glitch"] = 1
    return s


def draw(i, s, eye_cache, text, rng, tl, still=False):
    if i >= tl.cut and not still:
        return crt_collapse(rng) if i == tl.cut else np.zeros((H, W, 3), np.uint8)
    f = np.zeros((H, W, 3), np.float32)
    f[:] = (5, 3, 4)
    if s["level"] > 0:
        key = (round(s["level"], 3), round(s["open"], 3), round(s["squint"], 3),
               tuple(round(v, 3) for v in s["look"]), round(s["pulse"], 3))
        if key not in eye_cache:
            rgb, lum = eyes(s["level"], s["open"], s["squint"], s["look"], s["pulse"])
            eye_cache.clear()
            eye_cache[key] = rgb + bloom(lum, s["level"] * min(s["pulse"], 1.4))
        crop = f[EY0:EY1, EX0:EX1]
        crop[:] = np.maximum(crop, 0) + eye_cache[key]
    band0, band1 = text["band"]
    band = f[band0:band1]
    glow_gain = 0.5 * (1 + 0.8 * (s["pulse"] - 1))
    vis = []
    if s["text"] > 0:
        amt = s["text"] + (landing_boost(s["stamp"], tl.fps) if s["stamp"] is not None else 0.0)
        vis += [(lay, amt) for lay in text["words"]]
    for k, age in enumerate(s["lands"]):
        vis.append((text["lands"][k], 1.0 + landing_boost(age, tl.fps)))
    for lay, amt in vis:  # glow goes underneath so the chrome stays clean
        band += lay["glow"] * glow_gain * amt
    for lay, amt in vis:
        a = lay["a"][..., None] * min(amt, 1.0)
        band[:] = band * (1 - a) + lay["rgb"] * min(amt, 1.0) * (1 + 0.6 * max(amt - 1, 0))
    s0, s1 = tl.sheen
    if s0 <= i < s1 and not still:  # one glint sweeping across the chrome
        xs = 100 + (i - s0) / (s1 - s0 - 1) * 1500
        bx = XX[: band.shape[0]] - xs + (np.arange(band.shape[0], dtype=np.float32)[:, None] - text["mid"]) * 0.45
        band += (np.exp(-(bx / 34) ** 2) * text["sheen"] * 150)[..., None]
    if still:
        return crt(f, i, rng, tl.fps)
    out = crt(f, i, rng, tl.fps)
    warm = tl.F(0.2)
    if i < tl.eyes_on:  # true black first, then the tube's faint grain warms up
        out = (out * max(0.0, (i - warm) / (tl.eyes_on - warm))).astype(np.uint8)
    if s["glitch"]:
        heavy = i == tl.cut - 1
        out = tear(out, rng, 9 if heavy else 4, 260 if heavy else 70)
        out = rgb_split(out, 34 if heavy else 12, 4 if heavy else 0)
        if heavy:
            out = np.clip(out.astype(np.int16) * 14 // 10, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


# --------------------------------------------------------------------------- audio


def db(x):
    return 10 ** (x / 20)


def norm(x, peak_db):
    m = np.abs(x).max()
    return x * (db(peak_db) / m) if m > 0 else x


def phase(freq, n):
    f = np.broadcast_to(np.asarray(freq, np.float64), (n,))
    return 2 * np.pi * np.cumsum(f) / SR


def saw(freq, n, bright, nharm=48):
    """Band-limited additive saw; 'bright' (scalar or per-sample) sets the harmonic roll-off,
    so it doubles as a sweepable low-pass filter."""
    ph = phase(freq, n)
    f0 = float(np.max(freq))
    out = np.zeros(n, np.float64)
    for k in range(1, nharm + 1):
        if k * f0 > SR / 2.2:
            break
        out += np.sin(k * ph) / k * np.exp(-(k - 1) / np.asarray(bright, np.float64))
    return out.astype(np.float32)


def square(freq, n, nharm=15):
    ph = phase(freq, n)
    f0 = float(np.max(freq))
    out = np.zeros(n, np.float64)
    for k in range(1, 2 * nharm, 2):
        if k * f0 > SR / 2.2:
            break
        out += np.sin(k * ph) / k
    return (out * 4 / np.pi).astype(np.float32)


def crush(x, bits, hold):
    y = np.repeat(x[::hold], hold, axis=0)[: len(x)]
    q = 2 ** (bits - 1)
    return np.round(y * q) / q


def lowpass(x, cutoff):
    spec = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(x.shape[0], 1 / SR)
    resp = 1 / np.sqrt(1 + (f / cutoff) ** 4)
    return np.fft.irfft(spec * (resp[:, None] if x.ndim == 2 else resp), n=x.shape[0], axis=0).astype(np.float32)


def place(buf, start_s, sig, pan=0.0, gain=1.0):
    s = int(round(start_s * SR))
    if s >= len(buf):
        return
    sig = sig[: len(buf) - s]
    if sig.ndim == 1:
        sig = np.stack([sig * (1 - max(pan, 0)), sig * (1 + min(pan, 0))], axis=1)
    buf[s:s + len(sig)] += sig * gain


def env_ar(n, attack, decay_rate):
    t = np.arange(n) / SR
    a = np.minimum(1, t / max(attack, 1e-4))
    return (a * np.exp(-t * decay_rate)).astype(np.float32)


DUN, DUN_LAST = 55.0, 58.27  # the last dot's dun goes up a semitone
BLIPS = [880.0, 1046.5, 1318.5, 1568.0, 1760.0]  # blips walking left to right


def audio(rng, tl):
    fps = tl.fps
    spf = SR // fps
    n = tl.total * spf
    t = np.arange(n) / SR
    mix = np.zeros((n, 2), np.float32)
    cut_s = tl.cut / fps

    # -- ominous drone: detuned saws on A, a fifth above, swelling and opening up to the cut
    d0 = (tl.eyes_on - tl.F(0.08)) / fps
    swell = np.clip((t - d0) / 1.6, 0, 1) ** 1.5 * 0.55 + np.clip((t - 2.2) / (cut_s - 2.2), 0, 1) ** 2 * 0.45
    swell *= t >= d0
    bright = 1.6 + 6.5 * np.clip((t - d0) / (cut_s - d0), 0, 1) ** 1.6
    lfo = 1 + 0.12 * np.sin(2 * np.pi * 0.31 * t)
    left = saw(55.0, n, bright) + saw(82.6, n, bright) * 0.5 + saw(110.35, n, bright) * 0.35
    right = saw(55.18, n, bright) + saw(82.3, n, bright) * 0.5 + saw(109.8, n, bright) * 0.35
    tension = saw(116.54, n, bright * 0.8) * np.clip((t - tl.tension / fps) / 1.4, 0, 1) * 0.28  # creeping B-flat
    left += tension
    right += tension
    sub = np.sin(2 * np.pi * 27.5 * t).astype(np.float32) * 0.5
    drone = np.stack([left + sub, right + sub], axis=1) * (swell * lfo)[:, None]
    drone[t >= cut_s] = 0
    k = int(0.004 * SR)
    c = int(cut_s * SR)
    drone[c - k:c] *= np.linspace(1, 0, k)[:, None]
    mix += drone * 0.10

    # -- eerie shimmer while the eyes are on
    sh = (np.sin(phase(1318.5 * (1 + 0.006 * np.sin(2 * np.pi * 5.3 * t)), n))
          + 0.7 * np.sin(phase(1396.9 * (1 + 0.006 * np.sin(2 * np.pi * 4.7 * t + 1)), n))).astype(np.float32)
    sh_env = np.clip((t - tl.eyes_full / fps) / 1.5, 0, 1) * (t < cut_s) * 0.010
    mix += np.stack([sh * sh_env, np.roll(sh, 900) * sh_env], axis=1)

    # -- eyes powering up: a rising "vwoooom"
    m = int(1.05 * SR)
    tt = np.arange(m) / SR
    fr = 70 * (9.5 ** (np.clip(tt / 0.8, 0, 1) ** 1.4))
    v = saw(fr, m, 3.0, nharm=12) * np.minimum(1, tt / 0.6) * np.exp(-np.clip(tt - 0.75, 0, None) * 9)
    place(mix, tl.eyes_on / fps, v, gain=0.12)
    for fs in tl.stutter:  # the two stutters
        place(mix, fs / fps, crush(rng.uniform(-1, 1, spf).astype(np.float32), 3, 30), gain=0.05)

    # -- blink: two little servo whirrs
    for fdx, f0, f1 in ((tl.blink[0], 160, 120), (tl.blink[2], 120, 175)):
        m = int(0.20 * SR)
        z = square(np.geomspace(f0, f1, m), m, nharm=6) * np.sin(np.pi * np.arange(m) / m) ** 0.6
        place(mix, fdx / fps, lowpass(z, 1400), gain=0.05)

    # -- tube flicker buzz while the words stutter on (in step with the 25 fps flicker table)
    seg_n = SR // REF_FPS
    for j, lv in enumerate(TEXT_FLICKER):
        if lv > 0 and j < len(TEXT_FLICKER) - 3:
            seg = square(100.0, seg_n, nharm=12) * 0.6 + rng.uniform(-1, 1, seg_n).astype(np.float32) * 0.4
            seg *= np.minimum(1, np.minimum(np.arange(seg_n), seg_n - np.arange(seg_n)) / 60)
            place(mix, tl.text_on / fps + j / REF_FPS, seg, pan=rng.uniform(-0.2, 0.2), gain=0.05 * lv)

    # -- small glitch chirp
    m = len(tl.glitch_small) * spf
    g = crush(rng.uniform(-1, 1, m).astype(np.float32), 3, 18) * 0.5 + square(np.geomspace(1800, 300, m), m) * 0.4
    place(mix, tl.glitch_small[0] / fps, g, pan=0.3, gain=0.07)

    # -- the landings: low synth "dun" + digital blip (one big stamp when there are no dots)
    for j, fd in enumerate(tl.lands):
        nb = len(tl.lands)
        idx = round(j * 4 / (nb - 1)) if nb > 1 else 0
        f0 = DUN_LAST if (nb > 1 and j == nb - 1) else DUN
        m = int(0.85 * SR)
        tt = np.arange(m) / SR
        body = saw(f0, m, 2.0 + 14 * np.exp(-tt * 14), nharm=40) * env_ar(m, 0.003, 4.2)
        thump = np.sin(phase(f0 * (1 + 1.6 * np.exp(-tt * 40)), m)).astype(np.float32) * env_ar(m, 0.002, 6.0)
        dun = np.tanh(1.6 * (body * 0.8 + thump * 0.9))
        place(mix, fd / fps, dun, gain=0.28 if tl.reading else 0.22 + 0.03 * idx)
        mb = int(0.075 * SR)
        bl = crush(square(BLIPS[idx], mb, nharm=8), 5, 2) * env_ar(mb, 0.002, 30)
        bl = np.concatenate([bl[: mb // 2], bl[: mb // 2] * 0.6])  # bi-bip
        place(mix, fd / fps + 0.01, bl, pan=0.0 if tl.reading else -0.45 + 0.225 * idx, gain=0.085)

    # -- final squint: a charging whine + rising swell into the cut
    p0 = tl.pulse[0] / fps
    m = int((cut_s - p0) * SR)
    tt = np.arange(m) / SR
    q = tt / tt[-1]
    whine = np.sin(phase(380 * (3.2 ** q) * (1 + 0.02 * np.sin(2 * np.pi * 11 * tt)), m)).astype(np.float32)
    rise = saw(55 * (1 + 0.5 * q ** 2), m, 4 + 8 * q, nharm=30)
    noise = lowpass(rng.standard_normal(m).astype(np.float32), 900)
    sw = (whine * 0.25 + rise * 0.5 + noise * 0.5) * q ** 2.2
    place(mix, p0, sw, gain=0.55)

    # -- the cut: glitch zap + low boom, decaying into the black
    m = n - c
    tt = np.arange(m) / SR
    boom = np.sin(phase(32 + 58 * np.exp(-tt * 9), m)).astype(np.float32)
    boom = np.tanh(2.4 * boom) * (np.minimum(1, tt / 0.003) * np.exp(-tt * 5.2))
    click = lowpass(rng.standard_normal(m).astype(np.float32), 2500) * np.exp(-tt * 45) * 0.9
    place(mix, cut_s, boom + click, gain=0.85)
    mz = int(0.16 * SR)
    zap = (square(np.geomspace(2600, 70, mz), mz, nharm=10) * 0.6
           + crush(rng.uniform(-1, 1, mz).astype(np.float32), 3, 12) * 0.5) * env_ar(mz, 0.001, 18)
    place(mix, (tl.cut - 1) / fps, np.stack([zap, np.roll(zap, 200)], axis=1), gain=0.32)

    mix = np.tanh(mix * 1.15) / 1.15  # gentle glue on the peaks
    mix = norm(mix, -2.0)
    fade = int(0.12 * SR)
    mix[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 2
    return mix


# --------------------------------------------------------------------------- main


def encode(out, frames, audio_, fps, nframes, workdir):
    """Pipe RGB frames + float stereo audio into a house-format H.264/AAC file (BT.709, like the original)."""
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        wav = Path(tmp) / "audio.wav"
        media.write_wav(wav, audio_)
        cmd = [media.ffmpeg(), "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
               "-i", wav, "-map", "0:v", "-map", "1:a",
               "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
               *media.video_args(18),
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
               *media.audio_args(), "-t", f"{nframes / fps:.3f}", "-movflags", "+faststart", out]
        proc = subprocess.Popen([str(c) for c in cmd], stdin=subprocess.PIPE)
        try:
            for f in frames:
                proc.stdin.write(np.ascontiguousarray(f, np.uint8).tobytes())
        except BrokenPipeError:
            pass
        finally:
            proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {out}")


def _setup(text, dots, ctx):
    seed = SEED + ctx.seed
    text_ = text_layers(np.random.default_rng(seed + 1), text, dots)
    tl = Timeline(int(ctx.fps), text_["n_land"])
    return seed, text_, tl


def render(out, *, text="To Be Continued", dots=True, ctx=None):
    """Render the end card to `out` and return its Path.

    text  the words ("To Be Continued", "The End", ...); dots inside it ("...") land one at a time
    dots  True: dots land one at a time after the words (five unless the text has its own);
          False: the words land once and the eyes read across them
    """
    ctx = ctx or Ctx()
    seed, text_, tl = _setup(text, dots, ctx)
    fps = tl.fps
    spf = SR // fps
    rng = np.random.default_rng(seed)
    mix = audio(np.random.default_rng(seed + 3), tl)
    nframes = min(tl.total, ctx.limit_frames) if ctx.limit_frames else tl.total
    if nframes < tl.total:
        mix = mix[: nframes * spf].copy()
        k = min(96, len(mix))
        mix[len(mix) - k:] *= np.linspace(1, 0, k)[:, None]  # de-click the trimmed end
    cache = {}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = (draw(i, state(i, tl), cache, text_, rng, tl) for i in range(nframes))
    encode(out, frames, mix, fps, nframes, ctx.scratch("end_card"))
    return out


def still(out, *, text="To Be Continued", dots=True, ctx=None):
    """The fully revealed card as a 1920x1080 PNG, the eyes staring straight at you, a bit sly."""
    ctx = ctx or Ctx()
    seed, text_, tl = _setup(text, dots, ctx)
    st = state(tl.still, tl)
    st.update(look=(0.0, 0.12), squint=0.3, pulse=1.08, text=1.0)
    img = draw(tl.still, st, {}, text_, np.random.default_rng(seed + 2), tl, still=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output .mp4 (use '' to only write the still)")
    ap.add_argument("--text", default="To Be Continued")
    ap.add_argument("--no-dots", dest="dots", action="store_false", help="no dots: the words land once")
    ap.add_argument("--still", help="also write the fully revealed frame as a 1920x1080 PNG")
    ap.add_argument("--fps", type=int, default=25, choices=(25, 30))
    ap.add_argument("--frames", type=int, help="render only the first N frames")
    ap.add_argument("--seed", type=int, default=0, help="added to the tuned seed")
    a = ap.parse_args(argv)
    ctx = Ctx(fps=a.fps, seed=a.seed, limit_frames=a.frames)
    if a.still:
        print(still(a.still, text=a.text, dots=a.dots, ctx=ctx))
    if a.out:
        out = render(a.out, text=a.text, dots=a.dots, ctx=ctx)
        info = media.probe(out)
        print(f"{out}: {info.duration:.2f}s" if info else out)


if __name__ == "__main__":
    main()
