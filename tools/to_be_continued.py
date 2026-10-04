#!/usr/bin/env python3
"""ROBOTS REVENGE: post-credits "To Be Continued....." card.

Black and silent for a beat, then two red robot eyes (round lenses joined like
sunglasses, the little white robot's eyes) glow up out of the dark, take a slow
blink and look around. The words flicker on like a dodgy tube in chrome and red,
the five dots land one at a time on a low synth "dun" plus a digital blip (the
eyes glance at each one), the eyes narrow into a cheeky-evil squint with one
last pulse, and the picture zaps off to black on a low boom.

All pictures are drawn here with PIL/numpy/OpenCV (no drawtext needed) and all
sound is synthesised with numpy.

    python3 to_be_continued.py "09 To Be Continued.mp4"
    python3 to_be_continued.py out.mp4 --still "extras/To Be Continued.png"
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORDS = "To Be Continued"
DOTS = "....."
TEXT = WORDS + DOTS  # exactly "To Be Continued....."

W, H, FPS, SR = 1920, 1080, 25, 48000
SPF = SR // FPS  # audio samples per video frame
FFMPEG = "ffmpeg"
SUPP = "/System/Library/Fonts/Supplemental"

# ---- timeline, in frames (25 fps)
EYES_ON = 12             # 0.48 s of black silence first
EYES_FULL = 32           # eyes fully lit
BLINK = (37, 42, 45, 51)  # start closing, shut, start opening, open
TEXT_ON = 54             # tube-flicker of the words starts
SHEEN = (60, 71)          # glint across the chrome once the words settle
GLITCH_SMALL = (69, 70)
DOT_FRAMES = [74, 83, 92, 101, 110]
PULSE = (114, 128)       # final squint + pulse
CUT = 128                # glitch burst, then a one-frame CRT collapse, then black
TOTAL = 144              # 5.76 s; 14 frames (0.56 s) of black at the end
STILL_FRAME = 113        # fully revealed, eyes round and steady

TEXT_FLICKER = [0.0, 0.85, 0.0, 0.0, 0.0, 0.45, 1.0, 0.15, 0.0, 1.0, 1.0, 0.55, 1.0, 0.8, 1.0]

# ---- layout
EYE_CY = 352
EYE_CX = (W // 2 - 178, W // 2 + 178)
LENS_R = 132
GLOW_R = 104
FONT_SIZE = 192
DOT_TRACK = 24           # extra space between the dots so the trail reads
TEXT_CY = 790

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def font(name, size, index=0):
    return ImageFont.truetype(f"{SUPP}/{name}", size, index=index)


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


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

def chrome_gradient(h):
    """Chrome on top, red-hot reflection below the horizon line."""
    stops = [(0.00, (255, 255, 255)), (0.30, (200, 202, 214)), (0.49, (96, 96, 112)),
             (0.51, (40, 8, 12)), (0.60, (215, 28, 22)), (0.82, (255, 128, 104)), (1.00, (160, 18, 18))]
    t = np.linspace(0, 1, h)
    out = np.zeros((h, 3), np.float32)
    for c in range(3):
        out[:, c] = np.interp(t, [s[0] for s in stops], [s[1][c] for s in stops])
    return out


def text_layers(rng):
    """Pre-renders the words and each dot as (rgb, alpha, glow) bands."""
    f = font("Impact.ttf", FONT_SIZE)
    d = ImageDraw.Draw(Image.new("L", (8, 8)))
    word_w = d.textlength(WORDS, font=f)
    dot_w = d.textlength(".", font=f)
    total = word_w + len(DOTS) * (dot_w + DOT_TRACK) - DOT_TRACK * 0.5
    x0 = (W - total) / 2
    cap_top, base = f.getbbox("T")[1], f.getbbox("T")[3]
    y_text = TEXT_CY - (cap_top + base) / 2  # centre on the cap height
    band0, band1 = int(TEXT_CY - 190), int(TEXT_CY + 190)
    bh = band1 - band0

    pieces = [(WORDS, x0)] + [(".", x0 + word_w + i * (dot_w + DOT_TRACK)) for i in range(len(DOTS))]
    grad = chrome_gradient(base - cap_top + 4)
    gfull = np.zeros((bh, 3), np.float32)
    gy0 = int(y_text + cap_top - band0) - 2
    gfull[:] = grad[-1]
    gfull[:gy0] = grad[0]
    gfull[gy0:gy0 + len(grad)] = grad

    # worn-chrome grunge + a few scratches, shared by all pieces
    n = rng.standard_normal((bh // 3 + 1, W // 3 + 1)).astype(np.float32)
    n = cv2.resize(cv2.GaussianBlur(n, (0, 0), 2.2), (W, bh), interpolation=cv2.INTER_CUBIC)
    n /= n.std()
    wear = np.clip((n - 1.9) * 1.4, 0, 1) * 0.7
    scratch = Image.new("L", (W, bh))
    sd = ImageDraw.Draw(scratch)
    for _ in range(22):
        x, y = rng.uniform(x0, x0 + total), rng.uniform(40, bh - 40)
        ang, ln = rng.uniform(-0.5, 0.5), rng.uniform(25, 110)
        sd.line([(x, y), (x + ln * np.cos(ang), y + ln * np.sin(ang))], fill=int(rng.uniform(120, 220)),
                width=int(rng.integers(1, 3)))
    wear = np.maximum(wear, np.asarray(scratch, np.float32) / 255)
    distress = 1 - 0.55 * wear

    layers = []
    for text, x in pieces:
        def mask(stroke):
            im = Image.new("L", (W, bh))
            ImageDraw.Draw(im).text((x, y_text - band0), text, font=f, fill=255,
                                    stroke_width=stroke, stroke_fill=255)
            return np.asarray(im, np.float32) / 255

        fill, dark, red = mask(0), mask(3), mask(10)
        rgb = (red - dark)[..., None] * np.float32([235, 24, 20])          # red outer outline
        rgb += (dark - fill)[..., None] * np.float32([18, 4, 6])           # thin dark keyline
        rgb += fill[..., None] * gfull[:, None, :] * distress[..., None]   # chrome face
        glow = cv2.GaussianBlur(red, (0, 0), 16) * 1.2 + cv2.GaussianBlur(red, (0, 0), 45) * 0.9
        glow = glow[..., None] * np.float32([255, 28, 18])
        cols = np.nonzero(red.max(axis=0) > 0)[0]
        layers.append(dict(rgb=rgb, a=red, fill=fill, glow=glow, x=(cols[0] + cols[-1]) / 2))
    return band0, band1, layers


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


def crt(f, i, rng):
    bar = (i * 9.0 + 300) % H  # slow rolling hum bar
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

def state(i, dot_look=(0.15, 0.29, 0.43, 0.57, 0.71)):
    s = dict(level=0.0, open=1.0, squint=0.0, look=(0.0, 0.0), pulse=1.0, text=0.0, dots=[], glitch=0)
    if i < EYES_ON:
        return s
    # eyes power on with a couple of stutters
    p = (i - EYES_ON) / (EYES_FULL - EYES_ON)
    lvl = ease(p) ** 1.3
    if i in (EYES_ON + 2, EYES_ON + 6):
        lvl *= 0.25
    if i == EYES_ON + 3:
        lvl = min(1.0, lvl * 2.2)
    s["level"] = lvl
    # slow blink
    b0, b1, b2, b3 = BLINK
    if b0 <= i < b1:
        s["open"] = 1 - ease((i - b0) / (b1 - b0))
    elif b1 <= i < b2:
        s["open"] = 0.0
    elif b2 <= i < b3:
        s["open"] = ease((i - b2) / (b3 - b2))
    # look down at the words as they flicker on, glance at each dot as it lands,
    # then snap back to centre to stare at you for the finale
    look = (0.0, 0.0)
    if i >= TEXT_ON:
        q = ease((i - TEXT_ON) / 6)
        look = (-0.25 * q, 0.6 * q)
    for k, fd in enumerate(DOT_FRAMES):
        if i >= fd:
            q = ease((i - fd + 1) / 3)
            look = (look[0] + (dot_look[k] - look[0]) * q, look[1] + (0.7 - look[1]) * q)
    if i >= PULSE[0]:
        look = tuple(v * (1 - ease((i - PULSE[0]) / 4)) for v in look)
    s["look"] = look
    # words + dots
    j = i - TEXT_ON
    if j >= 0:
        s["text"] = TEXT_FLICKER[j] if j < len(TEXT_FLICKER) else 1.0
    s["dots"] = [max(0, i - fd) for fd in DOT_FRAMES if i >= fd]  # age of each landed dot
    # final squint + pulse
    if i >= PULSE[0]:
        q = (i - PULSE[0]) / (PULSE[1] - PULSE[0])
        s["squint"] = 0.62 * ease(q * 1.6)
        s["pulse"] = 1.0 + 0.55 * ease(q) + 0.25 * np.sin(q * np.pi * 3) * q
    if i in GLITCH_SMALL:
        s["glitch"] = 1
    if i == CUT - 1:
        s["glitch"] = 1
    return s


def draw(i, s, eye_cache, text, rng, still=False):
    if i >= CUT and not still:
        return crt_collapse(rng) if i == CUT else np.zeros((H, W, 3), np.uint8)
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
    band0, band1, layers = text
    band = f[band0:band1]
    glow_gain = 0.5 * (1 + 0.8 * (s["pulse"] - 1))
    vis = []
    if s["text"] > 0:
        vis.append((layers[0], s["text"]))
    for k, age in enumerate(s["dots"]):
        vis.append((layers[k + 1], 1.0 + (0.9 if age == 0 else 0.5 if age == 1 else 0.2 if age == 2 else 0.0)))
    for lay, amt in vis:  # glow goes underneath so the chrome stays clean
        band += lay["glow"] * glow_gain * amt
    for lay, amt in vis:
        a = lay["a"][..., None] * min(amt, 1.0)
        band[:] = band * (1 - a) + lay["rgb"] * min(amt, 1.0) * (1 + 0.6 * max(amt - 1, 0))
    if SHEEN[0] <= i < SHEEN[1] and not still:  # one glint sweeping across the chrome
        xs = 100 + (i - SHEEN[0]) / (SHEEN[1] - SHEEN[0] - 1) * 1500
        bx = XX[: band.shape[0]] - xs + (np.arange(band.shape[0], dtype=np.float32)[:, None] - 190) * 0.45
        band += (np.exp(-(bx / 34) ** 2) * layers[0]["fill"] * 150)[..., None]
    if still:
        return crt(f, i, rng)
    out = crt(f, i, rng)
    if i < EYES_ON:  # true black first, then the tube's faint grain warms up
        out = (out * max(0.0, (i - 5) / (EYES_ON - 5))).astype(np.uint8)
    if s["glitch"]:
        heavy = i == CUT - 1
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


def audio(rng):
    n = TOTAL * SPF
    t = np.arange(n) / SR
    mix = np.zeros((n, 2), np.float32)
    cut_s = CUT / FPS

    # -- ominous drone: detuned saws on A, a fifth above, swelling and opening up to the cut
    d0 = (EYES_ON - 2) / FPS
    swell = np.clip((t - d0) / 1.6, 0, 1) ** 1.5 * 0.55 + np.clip((t - 2.2) / (cut_s - 2.2), 0, 1) ** 2 * 0.45
    swell *= t >= d0
    bright = 1.6 + 6.5 * np.clip((t - d0) / (cut_s - d0), 0, 1) ** 1.6
    lfo = 1 + 0.12 * np.sin(2 * np.pi * 0.31 * t)
    left = saw(55.0, n, bright) + saw(82.6, n, bright) * 0.5 + saw(110.35, n, bright) * 0.35
    right = saw(55.18, n, bright) + saw(82.3, n, bright) * 0.5 + saw(109.8, n, bright) * 0.35
    tension = saw(116.54, n, bright * 0.8) * np.clip((t - DOT_FRAMES[2] / FPS) / 1.4, 0, 1) * 0.28  # creeping B-flat
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
    sh_env = np.clip((t - EYES_FULL / FPS) / 1.5, 0, 1) * (t < cut_s) * 0.010
    mix += np.stack([sh * sh_env, np.roll(sh, 900) * sh_env], axis=1)

    # -- eyes powering up: a rising "vwoooom"
    m = int(1.05 * SR)
    tt = np.arange(m) / SR
    fr = 70 * (9.5 ** (np.clip(tt / 0.8, 0, 1) ** 1.4))
    v = saw(fr, m, 3.0, nharm=12) * np.minimum(1, tt / 0.6) * np.exp(-np.clip(tt - 0.75, 0, None) * 9)
    place(mix, EYES_ON / FPS, v, gain=0.12)
    for k2 in (2, 6):  # the two stutters
        place(mix, (EYES_ON + k2) / FPS, crush(rng.uniform(-1, 1, SPF).astype(np.float32), 3, 30), gain=0.05)

    # -- blink: two little servo whirrs
    for fdx, f0, f1 in ((BLINK[0], 160, 120), (BLINK[2], 120, 175)):
        m = int(0.20 * SR)
        z = square(np.geomspace(f0, f1, m), m, nharm=6) * np.sin(np.pi * np.arange(m) / m) ** 0.6
        place(mix, fdx / FPS, lowpass(z, 1400), gain=0.05)

    # -- tube flicker buzz while the words stutter on
    for j, lv in enumerate(TEXT_FLICKER):
        if lv > 0 and j < len(TEXT_FLICKER) - 3:
            seg = square(100.0, SPF, nharm=12) * 0.6 + rng.uniform(-1, 1, SPF).astype(np.float32) * 0.4
            seg *= np.minimum(1, np.minimum(np.arange(SPF), SPF - np.arange(SPF)) / 60)
            place(mix, (TEXT_ON + j) / FPS, seg, pan=rng.uniform(-0.2, 0.2), gain=0.05 * lv)

    # -- small glitch chirp
    m = len(GLITCH_SMALL) * SPF
    g = crush(rng.uniform(-1, 1, m).astype(np.float32), 3, 18) * 0.5 + square(np.geomspace(1800, 300, m), m) * 0.4
    place(mix, GLITCH_SMALL[0] / FPS, g, pan=0.3, gain=0.07)

    # -- the five dots: low synth "dun" + digital blip, blips walking left to right
    duns = [55.0, 55.0, 55.0, 55.0, 58.27]
    blips = [880.0, 1046.5, 1318.5, 1568.0, 1760.0]
    for k2, fd in enumerate(DOT_FRAMES):
        m = int(0.85 * SR)
        tt = np.arange(m) / SR
        f0 = duns[k2]
        body = saw(f0, m, 2.0 + 14 * np.exp(-tt * 14), nharm=40) * env_ar(m, 0.003, 4.2)
        thump = np.sin(phase(f0 * (1 + 1.6 * np.exp(-tt * 40)), m)).astype(np.float32) * env_ar(m, 0.002, 6.0)
        dun = np.tanh(1.6 * (body * 0.8 + thump * 0.9))
        place(mix, fd / FPS, dun, gain=0.22 + 0.03 * k2)
        mb = int(0.075 * SR)
        bl = crush(square(blips[k2], mb, nharm=8), 5, 2) * env_ar(mb, 0.002, 30)
        bl = np.concatenate([bl[: mb // 2], bl[: mb // 2] * 0.6])  # bi-bip
        place(mix, fd / FPS + 0.01, bl, pan=-0.45 + 0.225 * k2, gain=0.085)

    # -- final squint: a charging whine + rising swell into the cut
    p0 = PULSE[0] / FPS
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
    place(mix, cut_s - 1 / FPS, np.stack([zap, np.roll(zap, 200)], axis=1), gain=0.32)

    mix = np.tanh(mix * 1.15) / 1.15  # gentle glue on the peaks
    mix = norm(mix, -2.0)
    fade = int(0.12 * SR)
    mix[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 2
    return mix


def write_wav(path, a):
    pcm = (np.clip(a, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# --------------------------------------------------------------------------- main

def render(out, still_path, seed):
    rng = np.random.default_rng(seed)
    text = text_layers(np.random.default_rng(seed + 1))
    cache = {}
    if still_path:
        os.makedirs(os.path.dirname(os.path.abspath(still_path)), exist_ok=True)
        st = state(STILL_FRAME)
        st.update(look=(0.0, 0.12), squint=0.3, pulse=1.08)  # staring straight at you, a bit sly
        img = draw(STILL_FRAME, st, {}, text, np.random.default_rng(seed + 2), still=True)
        Image.fromarray(img).save(still_path)
        print(f"{still_path}: still")
    if not out:
        return
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "audio.wav")
        write_wav(wav, audio(np.random.default_rng(seed + 3)))
        cmd = ["nice", "-n", "10", FFMPEG, "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
               "-i", wav, "-map", "0:v", "-map", "1:a",
               "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
               "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
               "-c:a", "aac", "-ar", str(SR), "-ac", "2", "-b:a", "192k",
               "-t", f"{TOTAL / FPS:.3f}", "-movflags", "+faststart", out]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i in range(TOTAL):
            proc.stdin.write(draw(i, state(i), cache, text, rng).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            sys.exit("ffmpeg failed")
    print(f"{out}: {TOTAL} frames, {TOTAL / FPS:.2f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output .mp4 (use '' to only write the still)")
    ap.add_argument("--still", help="also write the fully revealed frame as a 1920x1080 PNG")
    ap.add_argument("--seed", type=int, default=9)
    a = ap.parse_args()
    render(a.out, a.still, a.seed)


if __name__ == "__main__":
    main()
