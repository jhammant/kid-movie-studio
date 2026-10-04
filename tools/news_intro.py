#!/usr/bin/env python3
"""ROBOTS REVENGE: "The Nine O'Clock News" opening titles and lower thirds.

Everything is synthesised. Frames are drawn with PIL + OpenCV + numpy and piped
to ffmpeg as raw RGB (this ffmpeg has no drawtext filter); the audio (six time
pips, then a news sting) is built with numpy. 

    python3 tools/news_intro.py --channel "SBC NEWS" out.mp4
    python3 tools/news_intro.py --channel "" out.mp4          # programme title only
    python3 tools/news_intro.py --lower-thirds OUT_DIR        # transparent PNG overlays

Timeline (10 s, 25 fps): the clock sweeps up towards nine, five short pips at
1-5 s, the long pip at exactly 6.0 s as the clock reads 21:00:00, then the sting,
title and channel logo, the BREAKING NEWS banner at 8 s, and a 0.32 s fade.
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile
import wave
from functools import lru_cache
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, fftconvolve, sosfilt

W, H, FPS = 1920, 1080, 25
SR = 48000
DUR = 10.0
NF = int(round(DUR * FPS))
T_PIP = 1.0            # first short pip: clock reads 20:59:55
T_HOUR = 6.0           # long pip starts: clock reads 21:00:00
T_BANNER = 8.0
FADE_FRAMES = 8        # 0.32 s fade to black so it cuts to the studio
SAFE_X, SAFE_Y = 96, 54  # 5% title-safe margins
CX, CY = 960, 520      # clock and globe centre
R0 = 300               # clock radius at scale 1
HOUR_SECS = 21 * 3600  # 21:00:00

FONT_AV = "/System/Library/Fonts/Avenir Next.ttc"
FONT_AVC = "/System/Library/Fonts/Avenir Next Condensed.ttc"
FONT_DIN = "/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf"
BOLD, DEMI, MEDIUM, HEAVY = 0, 2, 5, 8

TITLE = "THE NINE O’CLOCK NEWS"
DIRECTOR_STRAPLINE = "Sam Broadcasting Corporation"
CRAWL = ["ROBOT INVASION CONTINUES", "PEOPLE TOLD TO STAY IN THEIR HOMES",
         "RESIDENTS FIGHT BACK WITH WATER", "MORE ON THIS STORY AT NINE"]


def rgb(r, g, b):
    return np.array([r, g, b], np.float32)


NAVY = rgb(0.035, 0.085, 0.22)
NAVY_DARK = rgb(0.012, 0.032, 0.09)
NAVY_INK = rgb(0.03, 0.07, 0.19)
RED = rgb(0.90, 0.06, 0.14)
RED_LIGHT = rgb(1.0, 0.30, 0.34)
RED_DARK = rgb(0.50, 0.0, 0.05)
WHITE = rgb(1, 1, 1)
BLUE = rgb(0.25, 0.58, 1.0)
ICE = rgb(0.62, 0.86, 1.0)


# ---------------------------------------------------------------- helpers

def clamp01(x):
    return min(max(x, 0.0), 1.0)


def ramp(t, a, b):
    return clamp01((t - a) / (b - a))


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_out_cubic(x):
    return 1 - (1 - clamp01(x)) ** 3


def ease_out_expo(x):
    x = clamp01(x)
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def ease_out_back(x, s=1.9):
    x = clamp01(x) - 1
    return 1 + x * x * ((s + 1) * x + s)


def lerp(a, b, u):
    return a + (b - a) * u


def fx(v):
    """Coordinates for OpenCV drawing with 4 bits of sub-pixel precision."""
    return np.round(np.asarray(v, np.float64) * 16).astype(np.int32)


@lru_cache(maxsize=None)
def font(path, size, index=0):
    return ImageFont.truetype(path, int(round(size)), index=index)


def fit_font(path, index, cap_px):
    """Font whose capital height is cap_px pixels."""
    probe = font(path, 200, index)
    cap = -probe.getbbox("H", anchor="ls")[1]
    return font(path, 200 * cap_px / cap, index)


class Text:
    """A rendered text mask (float 0..1) with its baseline and per-glyph positions."""

    def __init__(self, text, fnt, tracking=0.0):
        asc, desc = fnt.getmetrics()
        self.pad = pad = int(fnt.size * 0.3) + 4
        xs, x = [], 0.0
        for ch in text:
            xs.append(x)
            x += fnt.getlength(ch) + tracking
        self.width = fnt.getlength(text) if tracking == 0 else x - tracking
        w, h = int(math.ceil(self.width)) + 2 * pad, asc + desc + 2 * pad
        im = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(im)
        if tracking == 0:
            d.text((pad, pad + asc), text, font=fnt, fill=255, anchor="ls")
        else:
            for ch, x0 in zip(text, xs):
                d.text((pad + x0, pad + asc), ch, font=fnt, fill=255, anchor="ls")
        self.mask = np.asarray(im, np.float32) / 255.0
        self.base = pad + asc
        self.cap = -fnt.getbbox("H", anchor="ls")[1]
        self.xs = [pad + x0 for x0 in xs]
        self.text = text
        self.tracking = tracking

    def origin(self, x, baseline, align="l"):
        """Top-left of the mask so the text starts (l), centres (c) or ends (r) at x."""
        x = x - {"l": 0, "c": self.width / 2, "r": self.width}[align]
        return int(round(x - self.pad)), int(round(baseline - self.base))


def _crop(a, ys, ye, xs, xe):
    if not isinstance(a, np.ndarray) or a.ndim < 3:
        return a
    if a.shape[0] > 1:
        a = a[ys:ye]
    if a.shape[1] > 1:
        a = a[:, xs:xe]
    return a


def _clip(img, x, y, h, w):
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    return x0, y0, x1, y1


def paint(img, x, y, mask, color, opacity=1.0):
    """Paint `color` (rgb, or an (h,1,3)/(h,w,3) gradient) through a float coverage mask."""
    if opacity <= 0:
        return
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = _clip(img, x, y, h, w)
    if x1 <= x0 or y1 <= y0:
        return
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x, None] * opacity
    c = _crop(color, y0 - y, y1 - y, x0 - x, x1 - x)
    region = img[y0:y1, x0:x1]
    region += m * (c - region)


def add(img, x, y, mask, color, gain=1.0):
    if gain <= 0:
        return
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = _clip(img, x, y, h, w)
    if x1 <= x0 or y1 <= y0:
        return
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x, None] * gain
    img[y0:y1, x0:x1] += m * _crop(color, y0 - y, y1 - y, x0 - x, x1 - x)


def over(img, x, y, rgba, opacity=1.0):
    """Composite a premultiplied float RGBA image at (x, y)."""
    if opacity <= 0:
        return
    h, w = rgba.shape[:2]
    x0, y0, x1, y1 = _clip(img, x, y, h, w)
    if x1 <= x0 or y1 <= y0:
        return
    s = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    region = img[y0:y1, x0:x1]
    region *= 1 - s[..., 3:4] * opacity
    region += s[..., :3] * opacity


def warp_over(img, rgba, cx, cy, scale, opacity=1.0):
    """Composite premultiplied RGBA centred on (cx, cy) at a sub-pixel scale."""
    if opacity <= 0 or scale <= 0.01:
        return
    n_h, n_w = rgba.shape[:2]
    half_w, half_h = n_w * scale / 2 + 2, n_h * scale / 2 + 2
    x0, y0 = max(int(cx - half_w), 0), max(int(cy - half_h), 0)
    x1, y1 = min(int(cx + half_w) + 1, W), min(int(cy + half_h) + 1, H)
    if x1 <= x0 or y1 <= y0:
        return
    m = np.float32([[scale, 0, cx - x0 - scale * (n_w - 1) / 2],
                    [0, scale, cy - y0 - scale * (n_h - 1) / 2]])
    interp = cv2.INTER_AREA if scale < 0.75 else cv2.INTER_LINEAR
    out = cv2.warpAffine(rgba, m, (x1 - x0, y1 - y0), flags=interp,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    over(img, x0, y0, out, opacity)


def blur(mask, sigma, down=4):
    """Cheap wide Gaussian blur: downsample, blur, upsample."""
    h, w = mask.shape[:2]
    if min(h, w) < down * 4:
        return cv2.GaussianBlur(mask, (0, 0), sigma)
    small = cv2.resize(mask, (max(w // down, 1), max(h // down, 1)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(sigma / down, 0.5))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def vgrad(h, top, bottom, gamma=1.0):
    u = (np.linspace(0, 1, h, dtype=np.float32) ** gamma)[:, None, None]
    return (top[None, None, :] * (1 - u) + bottom[None, None, :] * u).astype(np.float32)


def poly_mask(h, w, pts):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [fx(pts)], 255, cv2.LINE_AA, shift=4)
    return m.astype(np.float32) / 255.0


def gloss_panel_color(h, top, bottom, sheen=0.10):
    """Glassy vertical gradient: brighter upper half with a soft specular step."""
    g = vgrad(h, top, bottom)
    u = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    g += sheen * np.clip(1 - u / 0.48, 0, 1) ** 1.5
    return g


# ---------------------------------------------------------------- shared artwork

def build_emblem(d, ss=4):
    """Original channel emblem: glossy red disc, wireframe globe, orbit ring. Premult RGBA."""
    D = d * ss
    S = D + 4 * ss
    c = (S - 1) / 2
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    dx, dy = xx - c, yy - c
    r = np.hypot(dx, dy)
    R = D / 2
    disc = np.clip(R - r + 0.5, 0, 1)
    g = np.clip(0.5 - 0.5 * (0.45 * dx + 0.9 * dy) / R, 0, 1) ** 1.3
    col = RED_DARK[None, None] * (1 - g[..., None]) + RED_LIGHT[None, None] * g[..., None]
    gloss = np.clip(1 - ((dx / (0.80 * R)) ** 2 + ((dy + 0.42 * R) / (0.50 * R)) ** 2), 0, 1) ** 0.6
    col = col + 0.16 * gloss[..., None] * np.clip(-dy / R + 0.2, 0, 1)[..., None]
    P = col * disc[..., None]
    A = disc.copy()

    im = Image.new("L", (S, S), 0)
    dr = ImageDraw.Draw(im)
    gr, lw = 0.50 * R, max(2, int(0.07 * R))
    dr.ellipse([c - gr, c - gr, c + gr, c + gr], outline=255, width=lw)
    dr.ellipse([c - gr * 0.42, c - gr, c + gr * 0.42, c + gr], outline=255, width=lw)
    dr.line([c - gr, c, c + gr, c], fill=255, width=lw)
    for lat in (-0.52, 0.52):
        yl = c + lat * gr
        half = gr * math.sqrt(1 - lat * lat)
        dr.line([c - half, yl, c + half, yl], fill=255, width=max(2, lw * 2 // 3))
    a = np.linspace(0, 2 * np.pi, 720)
    rx, ry, rot = 0.88 * R, 0.30 * R, math.radians(-22)
    ex, ey = rx * np.cos(a), ry * np.sin(a)
    X = c + ex * math.cos(rot) - ey * math.sin(rot)
    Y = c + ex * math.sin(rot) + ey * math.cos(rot)
    behind = (np.sin(a) < 0) & (np.hypot(X - c, Y - c) < gr + lw)
    # knock a dark gap into the globe where the ring passes in front, then draw the ring
    for i in range(len(a) - 1):
        if not behind[i] and not behind[i + 1]:
            dr.line([X[i], Y[i], X[i + 1], Y[i + 1]], fill=0, width=int(lw * 2.6))
    for i in range(len(a) - 1):
        if not behind[i] and not behind[i + 1]:
            dr.line([X[i], Y[i], X[i + 1], Y[i + 1]], fill=255, width=int(lw * 1.15))
    sa = 0.12 * np.pi
    sx = c + rx * math.cos(sa) * math.cos(rot) - ry * math.sin(sa) * math.sin(rot)
    sy = c + rx * math.cos(sa) * math.sin(rot) + ry * math.sin(sa) * math.cos(rot)
    sr = 0.085 * R
    dr.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=255)
    m = np.asarray(im, np.float32)[..., None] / 255.0 * disc[..., None]
    P = P * (1 - m) + m * WHITE
    rgba = np.dstack([P, A]).astype(np.float32)
    return cv2.resize(rgba, (S // ss, S // ss), interpolation=cv2.INTER_AREA)


def text_rgba(txt, color):
    m = txt.mask
    return np.dstack([m[..., None] * color, m]).astype(np.float32)


def stack_over(dst, x, y, src):
    """Premultiplied src over premultiplied dst (both float RGBA), in place."""
    h, w = src.shape[:2]
    x0, y0, x1, y1 = _clip(dst, x, y, h, w)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    dst[y0:y1, x0:x1] = s + dst[y0:y1, x0:x1] * (1 - s[..., 3:4])


def build_wordmark(channel, cap_px):
    """Channel wordmark: MAIN in heavy white, a trailing NEWS/LIVE/number in a red box."""
    words = channel.split()
    tag = None
    if len(words) > 1 and (words[-1] in ("NEWS", "LIVE", "TV", "HD") or words[-1].isdigit()):
        tag = words.pop()
    main = Text(" ".join(words), fit_font(FONT_AV, HEAVY, cap_px), tracking=cap_px * 0.05)
    box_h = int(round(cap_px * 1.62))
    h = box_h + 8
    base = int(round((h + main.cap) / 2))
    if tag:
        tt = Text(tag, fit_font(FONT_AV, HEAVY if tag.isdigit() else BOLD, cap_px * 0.78),
                  tracking=cap_px * 0.06)
        box_w = int(round(max(tt.width + cap_px * 0.95, box_h)))
        gap = int(round(cap_px * 0.42))
    else:
        tt, box_w, gap = None, 0, 0
    w = int(math.ceil(main.width)) + (gap + box_w if tag else 0) + 8
    out = np.zeros((h, w, 4), np.float32)
    ox, oy = main.origin(4, base)
    stack_over(out, ox, oy, text_rgba(main, WHITE))
    if tag:
        bx = int(math.ceil(main.width)) + 4 + gap
        by = (h - box_h) // 2
        col = gloss_panel_color(box_h, RED * 1.08, RED_DARK * 1.3, sheen=0.12)
        box = np.zeros((box_h, box_w, 4), np.float32)
        box[..., :3] = col
        box[..., 3] = 1
        box[:2, :, :3] = np.clip(col[:2] + 0.25, 0, 1)
        stack_over(out, bx, by, box)
        tx, ty = tt.origin(bx + box_w / 2, by + (box_h + tt.cap) / 2, "c")
        stack_over(out, tx, ty, text_rgba(tt, WHITE))
    return out


# ---------------------------------------------------------------- intro assets

class Assets:
    pass


A = None


def build_background():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.hypot((xx - 960) / 1150, (yy - 470) / 820)
    u = np.clip(d, 0, 1.4) / 1.4
    centre, edge = rgb(0.06, 0.13, 0.32), rgb(0.004, 0.010, 0.035)
    bg = centre[None, None] * (1 - u[..., None] ** 0.9) + edge[None, None] * (u[..., None] ** 0.9)
    red_glow = np.exp(-(((xx - 960) / 1100) ** 2 + ((yy - 1180) / 420) ** 2))[..., None] * rgb(0.9, 0.05, 0.12)
    vig = 1 - 0.45 * np.clip(np.hypot((xx - 960) / 1100, (yy - 540) / 760) - 0.35, 0, 1) ** 1.6
    # slow diagonal light beams, wide enough to slide sideways
    bx = np.arange(W * 2, dtype=np.float32)[None, :]
    by = np.arange(H, dtype=np.float32)[:, None]
    beams = np.zeros((H, W * 2), np.float32)
    rng = np.random.default_rng(4)
    for _ in range(7):
        pos, wid, amp = rng.uniform(0, W * 2), rng.uniform(30, 160), rng.uniform(0.25, 1.0)
        beams += amp * np.exp(-(((bx + by * 0.55) - pos) / wid) ** 2)
    beams *= np.clip(1 - by / H, 0.15, 1) ** 1.5
    return bg.astype(np.float32), red_glow.astype(np.float32), vig[..., None].astype(np.float32), beams


def build_globe():
    lines = []
    lon = np.linspace(0, 2 * np.pi, 181)
    for lat in np.radians([-60, -30, 0, 30, 60]):
        lines.append(np.stack([np.cos(lat) * np.cos(lon), np.full_like(lon, np.sin(lat)),
                               np.cos(lat) * np.sin(lon)], 1))
    a = np.linspace(0, 2 * np.pi, 241)
    for lo in np.radians(np.arange(0, 180, 20)):
        lines.append(np.stack([np.cos(a) * np.cos(lo), np.sin(a), np.cos(a) * np.sin(lo)], 1))
    n = 7000
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    pts = np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)
    rng = np.random.default_rng(11)
    val = np.zeros(n)
    for _ in range(8):
        k = rng.normal(size=3)
        k = k / np.linalg.norm(k) * rng.uniform(1.6, 4.2)
        val += np.sin(pts @ k + rng.uniform(0, 2 * np.pi)) / np.linalg.norm(k) ** 0.5
    land = pts[val > np.quantile(val, 0.60)]
    rings = []
    for tilt, roll in ((math.radians(72), math.radians(18)), (math.radians(-64), math.radians(-30))):
        ct, st, cr, sr = math.cos(tilt), math.sin(tilt), math.cos(roll), math.sin(roll)
        rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
        rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        rings.append(rz @ rx)
    return lines, land, rings


def build_clock_face():
    """Static clock face, 2x supersampled, premultiplied RGBA."""
    ss = 2
    R = R0 * ss
    S = int(2 * R * 1.14) // 2 * 2
    c = (S - 1) / 2
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    dx, dy = xx - c, yy - c
    r = np.hypot(dx, dy)
    ang = np.arctan2(dx, -dy)
    P = np.zeros((S, S, 3), np.float32)
    Aa = np.zeros((S, S), np.float32)

    def lay(cov, col, a=1.0):
        cov = (cov * a).astype(np.float32)
        P[:] = P * (1 - cov[..., None]) + cov[..., None] * col
        Aa[:] = cov + Aa * (1 - cov)

    def ring(rin, rout):
        return np.clip(rout - r + 0.5, 0, 1) * np.clip(r - rin + 0.5, 0, 1)

    lay(np.exp(-((r - R) / (0.07 * R)) ** 2), BLUE, 0.40)
    u = np.clip(r / (0.94 * R), 0, 1) ** 1.5
    face_col = rgb(0.08, 0.17, 0.38)[None, None] * (1 - u[..., None]) + NAVY_DARK[None, None] * u[..., None]
    lay(np.clip(0.94 * R - r + 0.5, 0, 1), face_col, 0.96)
    for rr in (0.30, 0.60):
        lay(ring(rr * R - 1, rr * R + 1), BLUE, 0.16)
    cross = np.clip(1.6 - np.minimum(np.abs(dx), np.abs(dy)), 0, 1) * (r < 0.66 * R)
    lay(cross, BLUE, 0.07)
    m = np.zeros((S, S), np.uint8)
    for k in range(12):
        a = 2 * np.pi * k / 12
        major = k % 3 == 0
        r1, r2 = (0.64 if major else 0.69) * R, 0.80 * R
        half = (0.026 if major else 0.016) * R
        ux, uy = math.sin(a), -math.cos(a)
        px, py = -uy, ux
        pts = [(c + ux * r1 + px * half, c + uy * r1 + py * half), (c + ux * r2 + px * half, c + uy * r2 + py * half),
               (c + ux * r2 - px * half, c + uy * r2 - py * half), (c + ux * r1 - px * half, c + uy * r1 - py * half)]
        cv2.fillPoly(m, [fx(pts)], 255, cv2.LINE_AA, shift=4)
    lay(m.astype(np.float32) / 255, rgb(0.95, 0.97, 1.0))
    metal = 0.62 - 0.30 * (dy / R) + 0.10 * np.cos(4 * ang + 0.6)
    lay(ring(0.948 * R, 1.0 * R), np.clip(metal, 0, 1)[..., None] * rgb(0.88, 0.93, 1.0))
    lay(ring(0.928 * R, 0.950 * R), RED)
    gloss = np.clip(1 - ((dx / (0.86 * R)) ** 2 + ((dy + 0.36 * R) / (0.58 * R)) ** 2), 0, 1) ** 0.5
    gloss *= np.clip(-dy / R + 0.15, 0, 1) * (r < 0.93 * R)
    lay(gloss, WHITE, 0.09)
    face2 = np.dstack([P, Aa]).astype(np.float32)
    emb = build_emblem(int(0.24 * R))
    eh = emb.shape[0]
    stack_over(face2, int(round(c - (eh - 1) / 2)), int(round(c - 0.40 * R - (eh - 1) / 2)), emb)
    face1 = cv2.resize(face2, (S // 2, S // 2), interpolation=cv2.INTER_AREA)
    return face1, face2


def init_assets(channel, strapline):
    global A
    A = Assets()
    A.bg, A.red_glow, A.vig, A.beams = build_background()
    A.lines, A.land, A.rings = build_globe()
    A.face1, A.face2 = build_clock_face()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    A.rad = np.hypot(xx - CX, yy - CY)
    A.ang = np.mod(np.arctan2(xx - CX, -(yy - CY)), 2 * np.pi).astype(np.float32)
    A.flash = np.exp(-(A.rad / 560) ** 2)[..., None].astype(np.float32)
    rng = np.random.default_rng(5)
    A.dither = [rng.uniform(-0.5, 0.5, (H, W, 1)).astype(np.float32) for _ in range(3)]

    # title
    size = 128
    while True:
        f = font(FONT_AV, size, HEAVY)
        t = Text(TITLE, f, tracking=size * 0.05)
        if t.width <= 1560:
            break
        size -= 2
    A.title = t
    A.channel = channel.strip()
    A.strap = Text(strapline, font(FONT_AV, 40, MEDIUM), tracking=3.0) if strapline else None
    if A.channel:
        A.emblem = build_emblem(124)
        A.wordmark = build_wordmark(A.channel, 64)
    else:
        A.emblem = build_emblem(150)
        A.wordmark = None
    # breaking banner
    A.tab = Text("BREAKING NEWS", fit_font(FONT_AVC, HEAVY, 31), tracking=3.0)
    cf = fit_font(FONT_AVC, BOLD, 36)
    A.crawl, A.crawl_dots = [], []
    x = 0.0
    for seg in CRAWL * 2:
        tx = Text(seg, cf, tracking=1.0)
        A.crawl.append((x, tx))
        x += tx.width + 70
        A.crawl_dots.append(x - 35)
    A.time_txt = Text("21:00", font(FONT_DIN, 52), tracking=1.0)
    A.banner_emblem = build_emblem(70)


# ---------------------------------------------------------------- intro drawing

def clock_time(t):
    """Seconds since midnight shown on the clock: whizzes up from 20:49:54, real time from 1 s."""
    if t >= T_PIP:
        return HOUR_SECS - (T_HOUR - t)
    u = (T_PIP - t) / T_PIP
    return HOUR_SECS - 5 - (T_PIP - t) - 600 * u ** 4


def draw_globe(img, t, cx, cy, R, gain):
    if gain <= 0.003:
        return
    spin = 0.9 + 0.32 * t
    tilt = math.radians(-22)
    cs, ss = math.cos(spin), math.sin(spin)
    ct, st = math.cos(tilt), math.sin(tilt)
    M = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]]) @ np.array([[cs, 0, ss], [0, 1, 0], [-ss, 0, cs]])
    half = int(R * 1.45) + 8
    x0, y0, x1, y1 = max(int(cx) - half, 0), max(int(cy) - half, 0), min(int(cx) + half, W), min(int(cy) + half, H)
    lw, lh = x1 - x0, y1 - y0
    ox, oy = cx - x0, cy - y0

    def proj(P):
        k = 4.0 / (4.0 - P[:, 2])
        return np.stack([ox + R * P[:, 0] * k, oy - R * P[:, 1] * k], 1), P[:, 2]

    lines = np.zeros((lh, lw), np.uint8)
    projected = [proj(L @ M.T) for L in A.lines]
    for want_front, val, thick in ((False, 70, 1), (True, 255, 2)):
        for xy, z in projected:
            sel = np.where((z > -0.02) == want_front)[0]
            if len(sel) < 2:
                continue
            for run in np.split(sel, np.where(np.diff(sel) != 1)[0] + 1):
                if len(run) >= 2:
                    cv2.polylines(lines, [fx(xy[run])], False, val, thick, cv2.LINE_AA, shift=4)
    # orbit rings with satellites; the half behind the globe is dimmed
    rings = np.zeros((lh, lw), np.uint8)
    sats = []
    aa = np.linspace(0, 2 * np.pi, 361)
    for j, RR in enumerate(A.rings):
        pts = np.stack([1.32 * np.cos(aa), np.zeros_like(aa), 1.32 * np.sin(aa)], 1) @ RR.T
        xy, z = proj(pts)
        occl = (z < 0) & (np.hypot(xy[:, 0] - ox, xy[:, 1] - oy) < R * 1.02)
        for want, val in ((True, 45), (False, 200)):
            sel = np.where(occl == want)[0]
            if len(sel) < 2:
                continue
            for run in np.split(sel, np.where(np.diff(sel) != 1)[0] + 1):
                if len(run) >= 2:
                    cv2.polylines(rings, [fx(xy[run])], False, val, 1, cv2.LINE_AA, shift=4)
        sa = (0.55 + 0.21 * j) * t * (1 if j == 0 else -1) + j * 2.1
        sp = np.array([[1.32 * math.cos(sa), 0, 1.32 * math.sin(sa)]]) @ RR.T
        sxy, sz = proj(sp)
        hidden = sz[0] < 0 and math.hypot(sxy[0, 0] - ox, sxy[0, 1] - oy) < R
        if not hidden:
            sats.append(sxy[0])
    # land dots: splat the front hemisphere
    P = A.land @ M.T
    keep = P[:, 2] > 0.04
    xy, z = proj(P[keep])
    dots = np.zeros((lh, lw), np.float32)
    ix, iy = np.floor(xy[:, 0]).astype(int), np.floor(xy[:, 1]).astype(int)
    fxx, fyy = xy[:, 0] - ix, xy[:, 1] - iy
    wgt = np.clip(z, 0, 1) ** 0.8
    ok = (ix >= 0) & (iy >= 0) & (ix < lw - 1) & (iy < lh - 1)
    ix, iy, fxx, fyy, wgt = ix[ok], iy[ok], fxx[ok], fyy[ok], wgt[ok]
    for ddx, ddy, ww in ((0, 0, (1 - fxx) * (1 - fyy)), (1, 0, fxx * (1 - fyy)), (0, 1, (1 - fxx) * fyy), (1, 1, fxx * fyy)):
        np.add.at(dots, (iy + ddy, ix + ddx), wgt * ww)
    dots = cv2.GaussianBlur(dots, (0, 0), 0.9 * max(R / 440, 0.8)) * 3.2

    region = img[y0:y1, x0:x1]
    lf = lines.astype(np.float32) / 255
    rf = rings.astype(np.float32) / 255
    region += (lf[..., None] * BLUE * 0.85 + rf[..., None] * ICE * 0.55 + np.minimum(dots, 1)[..., None] * ICE * 0.55) * gain
    region += blur(lf + 0.5 * rf, 10)[..., None] * BLUE * 0.75 * gain
    # soft atmospheric rim
    rr = A.rad[y0:y1, x0:x1] if (cx, cy) == (CX, CY) else np.hypot(*np.meshgrid(np.arange(x0, x1) - cx, np.arange(y0, y1) - cy))
    region += (np.exp(-((rr - R) / (0.05 * R)) ** 2) * 0.22 * gain)[..., None] * BLUE
    for sx, sy in sats:
        sz = 26
        gx = np.arange(-sz, sz + 1, dtype=np.float32)
        g = np.exp(-(gx[None, :] ** 2 + gx[:, None] ** 2) / 18.0)
        g += 0.25 * np.exp(-(gx[None, :] ** 2 + gx[:, None] ** 2) / 160.0)
        add(region, int(round(sx)) - sz, int(round(sy)) - sz, g, ICE, 0.9 * gain)


def draw_radar_rings(img, t, gain):
    m = np.zeros((H, W), np.uint8)
    c = (int(CX * 16), int(CY * 16))
    for k in range(36):
        a0 = k * 10 + 14 * t
        cv2.ellipse(m, c, (530 * 16, 530 * 16), 0, a0, a0 + 5, 255, 2, cv2.LINE_AA, shift=4)
    for k in range(4):
        a0 = k * 90 - 9 * t + 20
        cv2.ellipse(m, c, (600 * 16, 600 * 16), 0, a0, a0 + 62, 200, 1, cv2.LINE_AA, shift=4)
    for k in range(120):
        a = math.radians(k * 3 + 5 * t)
        r1 = 668 if k % 5 else 650
        p1 = (CX + r1 * math.cos(a), CY + r1 * math.sin(a))
        p2 = (CX + 682 * math.cos(a), CY + 682 * math.sin(a))
        cv2.line(m, tuple(fx(p1)), tuple(fx(p2)), 150, 1, cv2.LINE_AA, shift=4)
    img += (m.astype(np.float32) / 255 * gain)[..., None] * BLUE


def draw_glass_band(img, x_left, y_top, height, width, slant, opacity, tint):
    y0, y1 = int(y_top) - 2, int(y_top + height) + 3
    x0 = int(min(x_left, x_left + slant)) - 2
    x1 = int(max(x_left + width, x_left + width + slant)) + 3
    if x1 < 0 or x0 > W or opacity <= 0:
        return
    pts = [(x_left + slant - x0, y_top - y0), (x_left + width + slant - x0, y_top - y0),
           (x_left + width - x0, y_top + height - y0), (x_left - x0, y_top + height - y0)]
    m = poly_mask(y1 - y0, x1 - x0, pts)
    col = gloss_panel_color(y1 - y0, tint * 1.6, tint * 0.6, sheen=0.08)
    paint(img, x0, y0, m, col, opacity)
    edge = np.zeros_like(m)
    edge[max(int(y_top - y0) - 1, 0):int(y_top - y0) + 2] = 1
    add(img, x0, y0, edge * m.max(axis=0, keepdims=True), ICE, 0.35 * opacity / max(opacity, 1e-3) * min(opacity * 3, 1))


def streak(img, x, y, sx, sy, color, gain):
    if gain <= 0.003:
        return
    r0, r1 = max(int(y - 4 * sy), 0), min(int(y + 4 * sy) + 1, H)
    if r1 <= r0:
        return
    gy = np.exp(-((np.arange(r0, r1, dtype=np.float32) - y) / sy) ** 2)
    gx = np.exp(-((np.arange(W, dtype=np.float32) - x) / sx) ** 2)
    img[r0:r1] += (gy[:, None] * gx[None, :] * gain)[..., None] * color


def clock_scale_opacity(t):
    if t < T_PIP:
        s = 0.80 + 0.20 * ease_out_cubic(ramp(t, 0.0, 0.85))
        op = smooth(ramp(t, 0.0, 0.45))
    else:
        s = 1.0 + 0.06 * smooth(ramp(t, T_PIP, T_HOUR))
        op = 1.0
    z = ramp(t, 6.18, 6.55)
    if z > 0:
        s *= 1 + 1.7 * z * z
        op *= 1 - smooth(min(z * 1.5, 1.0))
    return s, op


def lit_dots(t):
    if t < T_PIP:
        return 56 * ease_out_cubic(ramp(t, 0.05, T_PIP)), 0.0
    sec = 55 + (t - T_PIP)
    return min(math.floor(sec) + 1, 60), sec - math.floor(sec)


def draw_clock(img, t):
    s, op = clock_scale_opacity(t)
    if op <= 0.003:
        return
    R = R0 * s
    clk = clock_time(t)
    half = int(R * 1.55) + 6
    bx0, by0, bx1, by1 = max(CX - half, 0), max(CY - half, 0), min(CX + half, W), min(CY + half, H)
    region = img[by0:by1, bx0:bx1]
    bh, bw = by1 - by0, bx1 - bx0
    ox, oy = CX - bx0, CY - by0
    RAD, ANG = A.rad[by0:by1, bx0:bx1], A.ang[by0:by1, bx0:bx1]

    face, k = (A.face1, s) if s <= 1.25 else (A.face2, s / 2)
    warp_over(region, face, ox, oy, k, op)

    sec_alpha = smooth(ramp(t, 0.45, 0.85))
    th_s = 2 * np.pi * (clk % 60) / 60
    th_m = 2 * np.pi * ((clk / 60) % 60) / 60
    th_h = 2 * np.pi * ((clk / 3600) % 12) / 12

    # radar trail behind the seconds hand
    d = np.mod(th_s - ANG, 2 * np.pi)
    wedge = np.clip(1 - d / 1.1, 0, 1) ** 2.2 * np.clip((0.84 * R - RAD) / 2, 0, 1) * np.clip((RAD - 0.06 * R) / 4, 0, 1)
    region += (wedge * 0.30 * sec_alpha * op)[..., None] * RED

    # digital readout
    hh, mm, sc = int(clk // 3600) % 24, int(clk // 60) % 60, int(clk % 60)
    txt = Text(f"{hh:02d}:{mm:02d}:{sc:02d}", font(FONT_DIN, max(int(0.15 * R), 8)), tracking=0.012 * R)
    hit = math.exp(-max(t - T_HOUR, 0) / 0.5) if t >= T_HOUR else 0.0
    tx, ty = txt.origin(ox, oy + 0.44 * R, "c")
    col = rgb(0.80, 0.88, 1.0) * (1 - hit) + rgb(1.0, 0.35, 0.40) * hit
    paint(region, tx, ty, txt.mask, col, 0.92 * op)

    # seconds dots
    n_lit, frac = lit_dots(t)
    unlit = np.zeros((bh, bw), np.uint8)
    lit = np.zeros((bh, bw), np.uint8)
    pop = np.zeros((bh, bw), np.uint8)
    newest = int(math.floor(n_lit)) - 1
    pop_amt = math.exp(-frac / 0.12) if t >= T_PIP else 0.6
    for kk in range(60):
        a = 2 * np.pi * kk / 60
        p = (ox + math.sin(a) * 0.865 * R, oy - math.cos(a) * 0.865 * R)
        if kk < math.floor(n_lit):
            rad = 0.020 * R * (1 + 0.6 * pop_amt * (kk == newest))
            cv2.circle(lit, tuple(fx(p)), int(rad * 16), 255, -1, cv2.LINE_AA, shift=4)
            if kk == newest:
                cv2.circle(pop, tuple(fx(p)), int(rad * 16), int(255 * pop_amt), -1, cv2.LINE_AA, shift=4)
        else:
            cv2.circle(unlit, tuple(fx(p)), int(0.013 * R * 16), 255, -1, cv2.LINE_AA, shift=4)
    lit_f = lit.astype(np.float32) / 255
    flash_white = math.exp(-(t - T_HOUR) / 0.25) if t >= T_HOUR else 0.0
    paint(region, 0, 0, unlit.astype(np.float32) / 255, rgb(0.30, 0.38, 0.55), op)
    paint(region, 0, 0, lit_f, RED_LIGHT * (1 - flash_white) + WHITE * flash_white, op)
    add(region, 0, 0, pop.astype(np.float32) / 255, WHITE, 0.8 * op)

    # hands (with soft shadow)
    def hand(theta, r_tail, r_tip, w_base, w_tip, dx=0.0, dy=0.0):
        ux, uy = math.sin(theta), -math.cos(theta)
        px, py = -uy, ux
        cx_, cy_ = ox + dx, oy + dy
        return [(cx_ + ux * r_tail + px * w_base, cy_ + uy * r_tail + py * w_base),
                (cx_ + ux * r_tip + px * w_tip, cy_ + uy * r_tip + py * w_tip),
                (cx_ + ux * (r_tip + w_tip), cy_ + uy * (r_tip + w_tip)),
                (cx_ + ux * r_tip - px * w_tip, cy_ + uy * r_tip - py * w_tip),
                (cx_ + ux * r_tail - px * w_base, cy_ + uy * r_tail - py * w_base)]

    shadow = np.zeros((bh, bw), np.uint8)
    hands = np.zeros((bh, bw), np.uint8)
    sd = (0.018 * R, 0.026 * R)
    for th, tail, tip, wb, wt in ((th_h, -0.10, 0.50, 0.030, 0.018), (th_m, -0.12, 0.80, 0.022, 0.012)):
        cv2.fillPoly(shadow, [fx(hand(th, tail * R, tip * R, wb * R, wt * R, *sd))], 255, cv2.LINE_AA, shift=4)
        cv2.fillPoly(hands, [fx(hand(th, tail * R, tip * R, wb * R, wt * R))], 255, cv2.LINE_AA, shift=4)
    sec = np.zeros((bh, bw), np.uint8)
    if sec_alpha > 0:
        cv2.fillPoly(sec, [fx(hand(th_s, -0.20 * R, 0.86 * R, 0.008 * R, 0.005 * R))], 255, cv2.LINE_AA, shift=4)
        tail_p = (ox - math.sin(th_s) * 0.13 * R, oy + math.cos(th_s) * 0.13 * R)
        cv2.circle(sec, tuple(fx(tail_p)), int(0.030 * R * 16), 255, -1, cv2.LINE_AA, shift=4)
        cv2.fillPoly(shadow, [fx(hand(th_s, -0.20 * R, 0.86 * R, 0.008 * R, 0.005 * R, *sd))], 200, cv2.LINE_AA, shift=4)
    paint(region, 0, 0, blur(shadow.astype(np.float32) / 255, 0.02 * R, 2), rgb(0, 0, 0.02), 0.55 * op)
    paint(region, 0, 0, hands.astype(np.float32) / 255, rgb(0.97, 0.98, 1.0), op)
    sec_f = sec.astype(np.float32) / 255
    paint(region, 0, 0, sec_f, rgb(1.0, 0.12, 0.18), op * sec_alpha)
    cap = np.zeros((bh, bw), np.uint8)
    cv2.circle(cap, tuple(fx((ox, oy))), int(0.045 * R * 16), 255, -1, cv2.LINE_AA, shift=4)
    paint(region, 0, 0, cap.astype(np.float32) / 255, rgb(0.92, 0.10, 0.16), op)
    cap[:] = 0
    cv2.circle(cap, tuple(fx((ox, oy))), int(0.016 * R * 16), 255, -1, cv2.LINE_AA, shift=4)
    paint(region, 0, 0, cap.astype(np.float32) / 255, WHITE, op)
    region += (blur(lit_f + sec_f * sec_alpha, 0.03 * R, 4) * 0.55 * op)[..., None] * RED

    # bezel draw-on light at the start
    if t < 1.1:
        phi = 2 * np.pi * ease_out_cubic(ramp(t, 0.05, 0.95))
        da = np.mod(phi - ANG, 2 * np.pi)
        tail = np.exp(-da / 0.6) * (da < np.pi * 1.5)
        amp = (1 - ramp(t, 0.75, 1.1)) * smooth(ramp(t, 0.0, 0.15))
        band = np.exp(-((RAD - 0.975 * R) / (0.02 * R)) ** 2)
        region += (tail * band * amp * 0.9)[..., None] * ICE
    # pip shockwaves and rim flashes
    for tp in (1, 2, 3, 4, 5, 6):
        dt = t - tp
        if not 0 <= dt < 0.8:
            continue
        long_pip = tp == 6
        u = dt / 0.8
        rr = R * (1.0 + (0.95 if long_pip else 0.30) * ease_out_cubic(u))
        amp = (1 - u) ** 2 * (1.0 if long_pip else 0.55)
        wid = 0.012 * R + 0.06 * R * u
        region += (np.exp(-((RAD - rr) / wid) ** 2) * amp)[..., None] * (WHITE if long_pip else ICE)
        rim = math.exp(-dt / 0.12) * (0.9 if long_pip else 0.6)
        region += (np.exp(-((RAD - 0.975 * R) / (0.018 * R)) ** 2) * rim * op)[..., None] * WHITE


def draw_title(img, t):
    T = A.title
    # glass title panel slides in from the right
    pt, pb = 452, 612
    e = ease_out_expo(ramp(t, 6.12, 6.62))
    if e > 0:
        left = lerp(W + 60, -160, e)
        draw_glass_band(img, left, pt, pb - pt, W + 400, 54, 0.86, rgb(0.05, 0.12, 0.30))
        edge = np.zeros((4, W), np.float32)
        x_start = max(int(left), 0)
        edge[1:3, x_start:] = 1
        add(img, 0, pb - 2, edge, ICE, 0.5)
    # red bar from the left
    e2 = ease_out_expo(ramp(t, 6.22, 6.70))
    if e2 > 0:
        right = lerp(-80, W + 80, e2)
        x1 = int(right)
        bar_h = 14
        pts = [(0, 0), (max(x1, 1), 0), (max(x1 - 10, 1), bar_h), (0, bar_h)]
        m = poly_mask(bar_h, W, pts)
        paint(img, 0, pb, m, gloss_panel_color(bar_h, RED_LIGHT, RED_DARK, 0.1), 1.0)
    # letters rise out of the panel one by one
    if t >= 6.30:
        mh, mw = T.mask.shape
        x0, y0 = T.origin(960, (pt + pb) / 2 + T.cap / 2, "c")
        layer = np.zeros((pb - pt, mw), np.float32)
        edges = [x - T.tracking / 2 for x in T.xs] + [mw]
        edges[0] = 0
        for i, ch in enumerate(T.text):
            if ch == " ":
                continue
            st = 6.30 + i * 0.022
            ee = ease_out_cubic(ramp(t, st, st + 0.34))
            al = ramp(t, st, st + 0.16)
            if al <= 0:
                continue
            dy = int(round(70 * (1 - ee)))
            c0, c1 = int(edges[i]), int(edges[i + 1])
            src = T.mask[:, c0:c1] * al
            top = y0 + dy - pt
            s0, s1 = max(0, -top), min(mh, pb - pt - top)
            if s1 > s0:
                layer[top + s0:top + s1, c0:c1] = np.maximum(layer[top + s0:top + s1, c0:c1], src[s0:s1])
        sh = cv2.GaussianBlur(layer, (0, 0), 4)
        paint(img, x0 + 2, pt + 5, sh, rgb(0, 0, 0.03), 0.65)
        col = vgrad(pb - pt, rgb(1, 1, 1), rgb(0.74, 0.82, 0.96), 1.2)
        paint(img, x0, pt, layer, col, 1.0)
        # glint sweeping across the settled title
        g = ramp(t, 7.25, 7.85)
        if 0 < g < 1:
            xx = np.arange(mw, dtype=np.float32)[None, :]
            yy = np.arange(pb - pt, dtype=np.float32)[:, None]
            pos = lerp(-300, mw + 300, g)
            band = np.exp(-((xx - pos + 0.5 * yy) / 55) ** 2)
            add(img, x0, pt, layer * band, WHITE, 0.9)
    # emblem + wordmark above the panel
    ecy = 352
    if A.wordmark is not None:
        ew, ww = A.emblem.shape[1], A.wordmark.shape[1]
        gap = 30
        total = ew + gap + ww
        ex = 960 - total / 2 + ew / 2
        wx = int(round(960 - total / 2 + ew + gap))
    else:
        ex, wx = 960, None
    pe = ease_out_back(ramp(t, 6.50, 6.92))
    if pe > 0:
        warp_over(img, A.emblem, ex, ecy, max(pe, 0.02), ramp(t, 6.50, 6.62))
        spin_glow = math.exp(-max(t - 6.62, 0) / 0.25) * (t > 6.5)
        streak(img, ex, ecy, 260, 3, ICE, 0.6 * spin_glow)
    if wx is not None:
        r = ease_out_cubic(ramp(t, 6.66, 7.10))
        if r > 0:
            wm = A.wordmark
            wh, ww_ = wm.shape[:2]
            vis = int(ww_ * r)
            if vis > 0:
                piece = wm[:, :vis].copy()
                over(img, wx - int(30 * (1 - r)), int(round(ecy - (wh - 1) / 2)), piece, 1.0)
    # strapline
    if A.strap is not None:
        a = ramp(t, 7.0, 7.45)
        if a > 0:
            sx, sy = A.strap.origin(960, pb + 14 + 62 - int(12 * (1 - ease_out_cubic(a))), "c")
            paint(img, sx, sy, A.strap.mask, rgb(0.78, 0.85, 0.97), a)


BAN_TAB_TOP, BAN_TOP, BAN_BOT = 806, 866, 966


def draw_banner(img, t):
    x_l, x_r = SAFE_X, W - SAFE_X
    bar_h = BAN_BOT - BAN_TOP
    # white bar wipes on from the left
    e = ease_out_expo(ramp(t, T_BANNER, T_BANNER + 0.40))
    if e <= 0:
        return
    right = lerp(x_l, x_r, e)
    # tab rises from behind the bar
    te = ease_out_cubic(ramp(t, T_BANNER + 0.18, T_BANNER + 0.48))
    tab_w = int(A.tab.width + 64)
    tab_h = BAN_TOP - BAN_TAB_TOP
    if te > 0:
        top = BAN_TOP - tab_h * te
        h = int(BAN_TOP - top) + 1
        pts = [(0, 0), (tab_w, 0), (tab_w + 24, tab_h), (0, tab_h)]
        m = poly_mask(tab_h, tab_w + 26, pts)
        cut = tab_h - h
        m = m[max(cut, 0):]
        col = gloss_panel_color(tab_h, RED * 1.1, RED_DARK * 1.2, 0.12)[max(cut, 0):]
        paint(img, x_l, BAN_TOP - m.shape[0], m, col, 1.0)
        tx, ty = A.tab.origin(x_l + 30, BAN_TAB_TOP + (tab_h + A.tab.cap) / 2, "l")
        tm = A.tab.mask.copy()
        clip_rows = int(BAN_TOP - ty)
        tm[max(clip_rows, 0):] = 0
        shift = int(round(tab_h * (1 - te)))
        paint(img, tx, ty + shift, tm[:max(tm.shape[0] - shift, 0)] if shift else tm, WHITE, 1.0)
        # breathing light on the tab
        pulse = 0.5 + 0.5 * math.sin((t - T_BANNER) * 2 * math.pi * 1.2)
        streak(img, x_l + tab_w / 2, BAN_TAB_TOP + 4, tab_w * 0.4, 2, WHITE, 0.25 * pulse * te)
    # bar
    bw_ = int(right - x_l)
    if bw_ > 2:
        col = gloss_panel_color(bar_h, rgb(1, 1, 1), rgb(0.82, 0.86, 0.92), 0.0)
        paint(img, x_l, BAN_TOP, np.ones((bar_h, bw_), np.float32), col, 1.0)
        paint(img, x_l, BAN_BOT, np.ones((6, bw_), np.float32), RED, 1.0)
        # navy emblem block
        sq = np.ones((bar_h, min(bar_h, bw_)), np.float32)
        paint(img, x_l, BAN_TOP, sq, gloss_panel_color(bar_h, NAVY * 1.6, NAVY_DARK, 0.08), 1.0)
        over(img, int(x_l + bar_h / 2 - (A.banner_emblem.shape[1] - 1) / 2),
             int(BAN_TOP + bar_h / 2 - (A.banner_emblem.shape[0] - 1) / 2), A.banner_emblem, ease_out_cubic(ramp(t, 8.15, 8.4)))
        # time box on the right
        ti = ease_out_expo(ramp(t, T_BANNER + 0.25, T_BANNER + 0.55))
        time_w = 170
        clip_r = right
        if ti > 0:
            bx = int(lerp(x_r + 40, x_r - time_w, ti))
            if bx < right:
                w_box = int(min(right, x_r) - bx)
                if w_box > 0:
                    paint(img, bx, BAN_TOP, np.ones((bar_h, w_box), np.float32),
                          gloss_panel_color(bar_h, NAVY * 1.5, NAVY_DARK, 0.08), 1.0)
                    tx, ty = A.time_txt.origin(bx + time_w / 2, BAN_TOP + (bar_h + A.time_txt.cap) / 2, "c")
                    paint(img, tx, ty, A.time_txt.mask, WHITE, 1.0)
                clip_r = min(clip_r, bx)
        # crawl
        ca = ramp(t, T_BANNER + 0.30, T_BANNER + 0.48)
        if ca > 0:
            cx0 = x_l + bar_h + 4
            cx1 = int(min(clip_r, x_r)) - 4
            if cx1 > cx0:
                strip = np.zeros((bar_h, cx1 - cx0), np.float32)
                dots = np.zeros((bar_h, cx1 - cx0), np.uint8)
                scroll = 140 * max(t - (T_BANNER + 0.85), 0)
                base_x = 30 - scroll
                for x, tx_ in A.crawl:
                    px, py = tx_.origin(base_x + x, bar_h / 2 + tx_.cap / 2, "l")
                    if px > strip.shape[1] or px + tx_.mask.shape[1] < 0:
                        continue
                    _paint_mask(strip, px, py, tx_.mask)
                for dxp in A.crawl_dots:
                    p = (base_x + dxp, bar_h / 2)
                    if -20 < p[0] < strip.shape[1] + 20:
                        cv2.circle(dots, tuple(fx(p)), int(9 * 16), 255, -1, cv2.LINE_AA, shift=4)
                paint(img, cx0, BAN_TOP, strip, NAVY_INK, ca)
                paint(img, cx0, BAN_TOP, dots.astype(np.float32) / 255, RED, ca)
    # wipe light on the leading edge
    if e < 1:
        streak(img, right, (BAN_TOP + BAN_BOT) / 2, 40, 50, WHITE, 0.8 * (1 - e))


def _paint_mask(dst, x, y, m):
    h, w = m.shape
    x0, y0, x1, y1 = _clip(dst, x, y, h, w)
    if x1 > x0 and y1 > y0:
        dst[y0:y1, x0:x1] = np.maximum(dst[y0:y1, x0:x1], m[y0 - y:y1 - y, x0 - x:x1 - x])


def render_frame(i):
    t = i / FPS
    img = A.bg.copy()
    hit = ramp(t, T_HOUR - 0.05, T_HOUR + 0.6)
    img += A.red_glow * (0.10 + 0.22 * hit + 0.12 * ramp(t, T_BANNER, T_BANNER + 0.6))
    off = int(38 * t)
    img += (A.beams[:, off:off + W] * 0.035)[..., None] * rgb(0.45, 0.65, 1.0)
    draw_radar_rings(img, t, 0.10 * smooth(ramp(t, 0.2, 1.2)) * (1 - 0.6 * hit))

    # globe: behind the clock, then grows behind the title
    g = smooth(ramp(t, 6.0, 7.3))
    pulse = math.exp(-(t - T_HOUR) / 0.35) if t >= T_HOUR else 0.0
    draw_globe(img, t, CX, lerp(CY, 548, g), lerp(445, 640, g),
               (0.62 * smooth(ramp(t, 0.0, 0.9)) * (1 - 0.5 * g)) + 0.5 * pulse)

    # glossy bands gliding behind the clock during the countdown
    bands = (1 - ramp(t, 6.05, 6.35)) * smooth(ramp(t, 0.1, 0.8))
    if bands > 0:
        draw_glass_band(img, -1200 + 150 * t, 300, 64, 1250, 40, 0.30 * bands, rgb(0.10, 0.22, 0.48))
        draw_glass_band(img, 1500 - 165 * t, 702, 46, 1150, -34, 0.26 * bands, rgb(0.10, 0.22, 0.48))
        draw_glass_band(img, -900 + 95 * t, 760, 18, 700, 18, 0.45 * bands, rgb(0.45, 0.04, 0.10))

    if t < 6.6:
        draw_clock(img, t)
    if t >= 6.0:
        draw_title(img, t)
    if t >= T_BANNER:
        draw_banner(img, t)

    # light streaks and the hit flash
    if t < 0.9:
        streak(img, lerp(-400, W + 400, ease_out_cubic(ramp(t, 0.05, 0.85))), CY, 520, 2.5, ICE, 1.1 * (1 - ramp(t, 0.6, 0.9)))
        streak(img, lerp(-400, W + 400, ease_out_cubic(ramp(t, 0.05, 0.85))), CY, 380, 40, BLUE, 0.15 * (1 - ramp(t, 0.6, 0.9)))
    if t >= T_HOUR:
        dt = t - T_HOUR
        img += A.flash * (0.42 * math.exp(-dt / 0.08))
        streak(img, CX, CY, 1100, 3.5, WHITE, 1.1 * math.exp(-dt / 0.18))
        streak(img, CX, CY, 800, 45, ICE, 0.25 * math.exp(-dt / 0.2))
    for tp in (1, 2, 3, 4, 5):
        dt = t - tp
        if 0 <= dt < 0.3:
            streak(img, CX, CY - R0 * 1.03, 380, 2.0, ICE, 0.45 * math.exp(-dt / 0.08))

    img *= A.vig
    fade = smooth(ramp(t, 0.0, 0.28))
    if i >= NF - FADE_FRAMES:
        fade *= (NF - 1 - i) / FADE_FRAMES
    img *= fade
    out = np.clip(img * 255 + A.dither[i % 3], 0, 255).astype(np.uint8)
    return out.tobytes()


# ---------------------------------------------------------------- audio

def synth_audio(path):
    rng = np.random.default_rng(21)
    n = int(DUR * SR)
    dry = np.zeros((n, 2))
    send = np.zeros((n, 2))

    def tt(d):
        return np.arange(int(d * SR)) / SR

    def put(buf, sig, t0, gain=1.0, pan=0.0):
        if sig.ndim == 1:
            a = (pan + 1) * math.pi / 4
            sig = np.stack([sig * math.cos(a), sig * math.sin(a)], 1) * math.sqrt(2)
        i0 = int(round(t0 * SR))
        if i0 < 0:
            sig, i0 = sig[-i0:], 0
        i1 = min(n, i0 + len(sig))
        if i1 > i0:
            buf[i0:i1] += sig[:i1 - i0] * gain

    def midi(m):
        return 440.0 * 2 ** ((m - 69) / 12)

    def saw(f, d, bright, kmax=40):
        t = tt(d)
        ph = 2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)
        K = int(max(1, min(kmax, 15000 / f)))
        b = bright(t) if callable(bright) else np.full_like(t, bright)
        out = np.zeros_like(t)
        bk = np.ones_like(t)
        for k in range(1, K + 1):
            out += np.sin(k * ph) * bk / k
            bk = bk * b
        return out

    def env(t, attack, tau, release_at=None):
        e = (1 - np.exp(-t / attack)) * np.exp(-t / tau)
        return e

    def lp(x, fc, order=2):
        return sosfilt(butter(order, fc, "low", fs=SR, output="sos"), x)

    def hp(x, fc, order=2):
        return sosfilt(butter(order, fc, "high", fs=SR, output="sos"), x)

    def bp(x, lo, hi, order=2):
        return sosfilt(butter(order, [lo, hi], "band", fs=SR, output="sos"), x)

    CH = {"Dm": [38, 50, 57, 62, 65, 69], "Bb": [34, 46, 53, 58, 62, 65],
          "C": [36, 48, 55, 60, 64, 67], "Gm": [43, 50, 58, 62, 67, 70]}

    def stab(t0, chord, gain=1.0, tau=0.45, d=1.6):
        t = tt(d)
        e = env(t, 0.005, tau)
        bright = lambda tt_: 0.84 + 0.11 * np.exp(-tt_ / 0.12)
        sig_l = np.zeros_like(t)
        sig_r = np.zeros_like(t)
        notes = CH[chord][1:] + [m + 12 for m in CH[chord][-3:]]
        for j, m in enumerate(notes):
            for cents, pan in ((-9, -0.6), (0, 0.0), (9, 0.6)):
                v = saw(midi(m) * 2 ** (cents / 1200), d, bright) * (0.7 if j == 0 else 0.55 if j >= 5 else 0.9)
                a = (pan + 1) * math.pi / 4
                sig_l += v * math.cos(a)
                sig_r += v * math.sin(a)
        sig = np.stack([sig_l, sig_r], 1) * e[:, None] / (len(notes) * 1.6)
        snap = bp(rng.normal(size=len(t)), 1800, 6000) * np.exp(-t / 0.010) * 0.6
        sig += snap[:, None]
        put(dry, sig, t0, gain)
        put(send, sig, t0, gain * 0.55)

    def timpani(t0, gain=1.0, f0=73.42, d=1.8):
        t = tt(d)
        pitch = f0 * (1 + 0.07 * np.exp(-t / 0.05))
        ph = 2 * np.pi * np.cumsum(pitch) / SR
        sig = np.zeros_like(t)
        for ratio, amp, tau in ((1.0, 1.0, 0.75), (1.504, 0.55, 0.45), (1.742, 0.35, 0.35), (2.0, 0.28, 0.30), (2.245, 0.18, 0.22)):
            sig += amp * np.sin(ratio * ph) * np.exp(-t / tau)
        sig *= 1 - np.exp(-t / 0.002)
        sig += lp(rng.normal(size=len(t)), 1400) * np.exp(-t / 0.018) * 0.9
        put(dry, sig * 0.55, t0, gain, -0.15)
        put(send, sig * 0.55, t0, gain * 0.5, -0.15)

    def boom(t0, gain=1.0, d=1.6):
        t = tt(d)
        f = 38 + 85 * np.exp(-t / 0.07)
        sig = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.55) * (1 - np.exp(-t / 0.003))
        put(dry, sig, t0, gain)

    def crash(t0, gain=1.0, d=2.5, tau=0.9):
        t = tt(d)
        e = np.exp(-t / tau) * (1 - np.exp(-t / 0.002))
        sig = np.stack([hp(rng.normal(size=len(t)), 3500) * e, hp(rng.normal(size=len(t)), 3500) * e], 1)
        sig += np.stack([bp(rng.normal(size=len(t)), 6000, 11000)] * 2, 1) * e[:, None] * 0.4
        put(dry, sig * 0.35, t0, gain)
        put(send, sig * 0.35, t0, gain * 0.5)

    def reverse_cymbal(t_end, d, gain=1.0):
        t = tt(d)
        e = np.exp(-t / (d * 0.35))[::-1]
        sig = np.stack([hp(rng.normal(size=len(t)), 2500), hp(rng.normal(size=len(t)), 2500)], 1) * e[:, None]
        put(dry, sig * 0.3, t_end - d, gain)

    def pluck(t0, m, gain, tau=0.09, bright0=0.75, pan=0.0, d=0.35):
        t = tt(d)
        v = saw(midi(m), d, lambda x: 0.35 + bright0 * 0.5 * np.exp(-x / 0.05), kmax=24) * env(t, 0.002, tau)
        put(dry, v, t0, gain, pan)
        return v

    def bass(t0, m, gain, bright=0.6, tau=0.07, d=0.22):
        t = tt(d)
        v = saw(midi(m), d, bright, kmax=30) * env(t, 0.002, tau)
        v += 0.45 * np.sin(2 * np.pi * midi(m) * t) * env(t, 0.002, tau * 1.6)
        put(dry, v, t0, gain)

    def hat(t0, gain):
        t = tt(0.06)
        v = hp(rng.normal(size=len(t)), 7000) * np.exp(-t / 0.012)
        put(dry, v, t0, gain, 0.3)

    # --- pre-roll whoosh as the clock arrives
    t = tt(1.2)
    noise = rng.normal(size=len(t))
    lo, mid, hi = lp(noise, 700), bp(noise, 700, 3000), hp(noise, 3000)
    u = np.clip(t / 1.2, 0, 1)
    wh = (lo * (1 - u) + mid * np.sin(np.pi * u) + hi * u * 0.6) * np.sin(np.pi * np.clip(t / 0.9, 0, 1)) ** 1.5
    put(dry, np.stack([wh * (1 - u * 0.6), wh * (0.4 + u * 0.6)], 1) * 0.18, 0.0)
    boom(0.02, 0.35)
    crash(0.02, 0.25, tau=0.6)

    # --- countdown bed: drone, 8th-note pulse that slowly opens, ticks, riser
    t = tt(5.2)
    sw = np.clip(t / 5.0, 0, 1)
    drone = (saw(midi(38), 5.2, lambda x: 0.25 + 0.3 * np.clip(x / 5.0, 0, 1), kmax=20)
             + 0.6 * saw(midi(45), 5.2, 0.3, kmax=16) + 0.35 * np.sin(2 * np.pi * midi(26) * t))
    drone *= (0.25 + 0.75 * sw) * (1 - np.exp(-t / 0.4)) * (1 + 0.15 * np.sin(2 * np.pi * 4 * t))
    put(dry, drone, 0.85, 0.05)
    put(send, drone, 0.85, 0.03)
    for k in range(20):
        tk = T_PIP + k * 0.25
        prog = k / 19
        bass(tk, 38 if k % 2 == 0 else 50, 0.08 + 0.14 * prog, bright=0.35 + 0.40 * prog)
        if k % 2 == 1:
            hat(tk, 0.05 + 0.08 * prog)
    t = tt(3.0)
    riser = bp(rng.normal(size=len(t)), 400, 9000) * (t / 3.0) ** 2.5
    put(dry, np.stack([riser, bp(rng.normal(size=len(t)), 400, 9000) * (t / 3.0) ** 2.5], 1) * 0.10, T_HOUR - 3.0)
    reverse_cymbal(T_HOUR, 1.6, 0.8)
    t = tt(2.5)
    tone = np.sin(2 * np.pi * np.cumsum(midi(50) * 2 ** (np.clip(t / 2.5, 0, 1) ** 2)) / SR) * (t / 2.5) ** 2
    put(dry, tone, T_HOUR - 2.5, 0.05)

    # --- the sting: 120 bpm from the hour, D minor
    s16 = 0.125
    hour = T_HOUR
    hits = [(0, "Dm", 1.0, 0.55, True), (6, "Dm", 0.55, 0.16, False), (7, "Dm", 0.6, 0.16, False),
            (8, "Bb", 0.9, 0.40, True), (14, "C", 0.7, 0.22, False),
            (16, "Dm", 1.0, 0.50, True), (19, "Dm", 0.55, 0.14, False), (22, "Bb", 0.65, 0.18, False),
            (24, "Gm", 0.8, 0.30, True), (27, "C", 0.6, 0.15, False), (28, "Dm", 1.05, 0.60, True)]
    for step, chord, g, tau, big in hits:
        t0 = hour + step * s16
        stab(t0, chord, 1.0 * g, tau=tau)
        if big:
            timpani(t0, 0.7 * g, f0=midi(CH[chord][0] + 12) / 1.0 if chord != "Dm" else 73.42)
    boom(hour, 0.55)
    crash(hour, 0.9)
    crash(hour + 16 * s16, 0.6)
    crash(hour + 28 * s16, 0.75)
    boom(hour + 16 * s16, 0.4)
    boom(hour + 28 * s16, 0.5)
    for k, step in enumerate(range(20, 28)):
        timpani(hour + step * s16, 0.25 + 0.06 * k, f0=73.42)
    roots = {0: 38, 8: 34, 14: 36, 16: 38, 22: 34, 24: 43, 27: 36, 28: 38}
    root = 38
    arp_pat = [0, 7, 12, 15, 12, 7, 19, 15]
    for step in range(0, 30):
        root = roots.get(step, root)
        t0 = hour + step * s16
        acc = 1.0 if step % 4 == 0 else 0.7
        bass(t0, root + (12 if step % 4 == 2 else 0), 0.16 * acc, bright=0.72, tau=0.06)
        if step >= 2:
            m = root + 24 + arp_pat[step % 8]
            v = pluck(t0, m, 0.09, pan=-0.35 if step % 2 else 0.35)
            put(dry, v, t0 + 3 * s16, 0.035, 0.6 if step % 2 else -0.6)
            put(send, v, t0, 0.03)
        if step % 2 == 1:
            hat(t0, 0.10)

    # --- the six pips: 1 kHz, five 0.1 s then a 0.5 s long pip on the hour (dry, centred)
    pips = np.zeros(n)
    for k in range(6):
        t0 = T_PIP + k
        d = 0.5 if k == 5 else 0.1
        t = tt(d)
        e = np.minimum(1, np.minimum(t / 0.003, (d - t) / 0.003))
        i0 = int(round(t0 * SR))
        pips[i0:i0 + len(t)] += np.sin(2 * np.pi * 1000 * t) * e
    pip_gain = 0.42

    # --- reverb (synthetic hall) and master
    ir_len = int(2.2 * SR)
    ti = np.arange(ir_len) / SR
    ir = np.stack([lp(rng.normal(size=ir_len), 7000) * np.exp(-ti / 0.32),
                   lp(rng.normal(size=ir_len), 7000) * np.exp(-ti / 0.32)], 1)
    ir[:int(0.02 * SR)] = 0
    ir /= np.sqrt((ir ** 2).sum(0))
    wet = np.stack([fftconvolve(send[:, c], ir[:, c])[:n] for c in range(2)], 1)
    mix = dry + 0.55 * wet
    mix = hp(mix.T, 32).T
    mix = mix - 0.6 * lp(mix.T, 120).T  # low shelf: keep it punchy on small speakers
    mix /= np.abs(mix).max()
    mix = np.tanh(1.25 * mix) / np.tanh(1.25)
    mix *= 0.80
    mix += (pips * pip_gain)[:, None]
    fade_n = int(0.36 * SR)
    mix[-fade_n:] *= np.cos(np.linspace(0, np.pi / 2, fade_n))[:, None] ** 2
    mix[:240] *= np.linspace(0, 1, 240)[:, None]
    mix *= 10 ** (-1.8 / 20) / np.abs(mix).max()
    pcm = np.round(mix * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


# ---------------------------------------------------------------- render intro

def render_intro(channel, strapline, out, workers):
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        synth_audio(wav)
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
               "-i", wav, "-map", "0:v", "-map", "1:a",
               "-vf", "scale=out_color_matrix=bt709:out_range=tv",
               "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
               "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
               "-movflags", "+faststart", "-shortest", out]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        with Pool(workers, initializer=init_assets, initargs=(channel, strapline)) as pool:
            for frame in pool.imap(render_frame, range(NF), chunksize=2):
                proc.stdin.write(frame)
        proc.stdin.close()
        if proc.wait() != 0:
            sys.exit("ffmpeg failed")
    finally:
        os.unlink(wav)


# ---------------------------------------------------------------- lower thirds

class Canvas:
    """Full-frame premultiplied RGBA canvas for overlay PNGs."""

    def __init__(self):
        self.rgba = np.zeros((H, W, 4), np.float32)

    def fill(self, x, y, mask, color, opacity=1.0):
        h, w = mask.shape
        layer = np.zeros((h, w, 4), np.float32)
        a = mask * opacity
        c = np.broadcast_to(color, (h, w, 3)) if np.ndim(color) == 3 else color
        layer[..., :3] = a[..., None] * c
        layer[..., 3] = a
        stack_over(self.rgba, int(x), int(y), layer)

    def rect(self, x, y, w, h, color, slant_right=0, opacity=1.0):
        pts = [(0, 0), (w + max(slant_right, 0), 0), (w + min(slant_right, 0), h), (0, h)]
        if slant_right < 0:
            pts = [(0, 0), (w, 0), (w - slant_right, h), (0, h)]
        m = poly_mask(h, int(w + abs(slant_right)) + 2, pts)
        self.fill(x, y, m, color[:h] if np.ndim(color) == 3 else color, opacity)

    def text(self, txt, x, baseline, color, align="l"):
        ox, oy = txt.origin(x, baseline, align)
        self.fill(ox, oy, txt.mask, color)

    def save(self, path, shadow=True):
        rgba = self.rgba
        if shadow:
            a = rgba[..., 3]
            sh = np.zeros_like(a)
            sh[6:, 3:] = a[:-6, :-3]
            sh = cv2.GaussianBlur(sh, (0, 0), 7) * 0.45
            rgba = rgba.copy()
            rgba[..., 3] = rgba[..., 3] + sh * (1 - rgba[..., 3])
        a = rgba[..., 3]
        col = rgba[..., :3] / np.maximum(a, 1e-6)[..., None]
        col[a < 1e-6] = 0
        out = np.dstack([np.clip(col, 0, 1), np.clip(a, 0, 1)])
        out8 = np.round(out * 255).astype(np.uint8)
        out8[out8[..., 3] == 0] = 0
        Image.fromarray(out8, "RGBA").save(path, optimize=True)


def lower_third(c, name, role, x=150, top=826):
    """Navy glass name bar + red rule + white role bar + navy emblem block (intro language)."""
    name_h, rule_h, role_h = 86, 6, 52
    total = name_h + rule_h + role_h
    nt = Text(name, fit_font(FONT_AV, HEAVY, 40), tracking=2.5)
    rt = Text(role, fit_font(FONT_AV, DEMI, 23.5), tracking=0.8)
    sq = total
    c.rect(x, top, sq, total, gloss_panel_color(total, NAVY * 1.7, NAVY_DARK, 0.09))
    emb = build_emblem(int(sq * 0.72))
    eh = emb.shape[0]
    stack_over(c.rgba, int(x + sq / 2 - (eh - 1) / 2), int(top + total / 2 - (eh - 1) / 2), emb)
    bx = x + sq
    name_w = int(nt.width + 44 + 50)
    role_w = int(max(rt.width + 44 + 40, name_w * 0.55))
    c.rect(bx, top, name_w, name_h, gloss_panel_color(name_h, rgb(0.07, 0.16, 0.38), rgb(0.02, 0.05, 0.14), 0.10),
           slant_right=0)
    c.rect(bx, top, name_w, 2, ICE[None, None, :] * np.ones((2, 1, 1), np.float32), opacity=0.7)
    c.text(nt, bx + 40, top + (name_h + nt.cap) / 2, vgrad(nt.mask.shape[0], WHITE, rgb(0.82, 0.88, 0.98)))
    c.rect(bx, top + name_h, name_w, rule_h, gloss_panel_color(rule_h, RED_LIGHT, RED_DARK, 0.0))
    ry = top + name_h + rule_h
    c.rect(bx, ry, role_w, role_h, gloss_panel_color(role_h, rgb(1, 1, 1), rgb(0.84, 0.87, 0.93), 0.0))
    c.text(rt, bx + 40, ry + (role_h + rt.cap) / 2, NAVY_INK)
    return x + sq + name_w


def live_bug(c, x=SAFE_X + 14, y=SAFE_Y + 22):
    h = 66
    c.rect(x, y, h, h, gloss_panel_color(h, NAVY * 1.7, NAVY_DARK, 0.09))
    emb = build_emblem(int(h * 0.74))
    eh = emb.shape[0]
    stack_over(c.rgba, int(x + h / 2 - (eh - 1) / 2), int(y + h / 2 - (eh - 1) / 2), emb)
    lt = Text("LIVE", fit_font(FONT_AV, HEAVY, 30), tracking=3.0)
    bw = int(lt.width + 34 + 40 + 30)
    c.rect(x + h, y, bw, h, gloss_panel_color(h, RED * 1.1, RED_DARK * 1.25, 0.12))
    c.rect(x + h, y, bw, 2, rgb(1, 0.6, 0.6)[None, None] * np.ones((2, 1, 1), np.float32), opacity=0.6)
    dot = np.zeros((h, 40), np.uint8)
    cv2.circle(dot, tuple(fx((20, h / 2))), 9 * 16, 255, -1, cv2.LINE_AA, shift=4)
    c.fill(x + h + 14, y, dot.astype(np.float32) / 255, WHITE)
    c.text(lt, x + h + 14 + 40, y + (h + lt.cap) / 2, WHITE)


def breaking_banner(c, headline):
    x_l, x_r = SAFE_X, W - SAFE_X
    tab_h, bar_h = BAN_TOP - BAN_TAB_TOP, BAN_BOT - BAN_TOP
    tab = Text("BREAKING NEWS", fit_font(FONT_AVC, HEAVY, 31), tracking=3.0)
    tab_w = int(tab.width + 64)
    c.rect(x_l, BAN_TAB_TOP, tab_w, tab_h, gloss_panel_color(tab_h, RED * 1.1, RED_DARK * 1.2, 0.12), slant_right=-24)
    c.text(tab, x_l + 30, BAN_TAB_TOP + (tab_h + tab.cap) / 2, WHITE)
    c.rect(x_l, BAN_TOP, x_r - x_l, bar_h, gloss_panel_color(bar_h, rgb(1, 1, 1), rgb(0.82, 0.86, 0.92), 0.0))
    c.rect(x_l, BAN_BOT, x_r - x_l, 6, RED)
    c.rect(x_l, BAN_TOP, bar_h, bar_h, gloss_panel_color(bar_h, NAVY * 1.6, NAVY_DARK, 0.08))
    emb = build_emblem(70)
    eh = emb.shape[0]
    stack_over(c.rgba, int(x_l + bar_h / 2 - (eh - 1) / 2), int(BAN_TOP + bar_h / 2 - (eh - 1) / 2), emb)
    avail = x_r - (x_l + bar_h) - 2 * 30
    cap = 36
    while True:
        ht = Text(headline, fit_font(FONT_AVC, BOLD, cap), tracking=0.8)
        if ht.width <= avail:
            break
        cap -= 1
    c.text(ht, x_l + bar_h + 30, BAN_TOP + (bar_h + ht.cap) / 2, NAVY_INK)


def render_lower_thirds(outdir):
    os.makedirs(outdir, exist_ok=True)
    c = Canvas()
    lower_third(c, "REX NEWSOME", "Newsreader")
    c.save(os.path.join(outdir, "Lower third - Rex Newsome.png"))
    c = Canvas()
    lower_third(c, "PENNY SPARKS", "Roving Reporter · The Battlefield")
    live_bug(c)
    c.save(os.path.join(outdir, "Lower third - Penny Sparks LIVE.png"))
    c = Canvas()
    breaking_banner(c, "ROBOT INVASION CONTINUES — PEOPLE TOLD TO STAY IN THEIR HOMES")
    c.save(os.path.join(outdir, "Breaking banner - Robot Invasion.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", help="output .mp4 for the intro")
    ap.add_argument("--channel", default="", help='channel name, e.g. "SBC NEWS"; empty for title only')
    ap.add_argument("--strapline", default=None,
                    help="small line under the title (default: Sam Broadcasting Corporation for SBC)")
    ap.add_argument("--lower-thirds", metavar="DIR", help="write the transparent lower-third PNGs here")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.lower_thirds:
        render_lower_thirds(args.lower_thirds)
    if args.out:
        strap = args.strapline
        if strap is None:
            strap = DIRECTOR_STRAPLINE if args.channel.strip().upper().startswith("SBC") else ""
        render_intro(args.channel, strap, args.out, args.workers)
    if not args.out and not args.lower_thirds:
        ap.error("give an output .mp4 and/or --lower-thirds DIR")


if __name__ == "__main__":
    main()
