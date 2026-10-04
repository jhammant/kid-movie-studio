#!/usr/bin/env python3
"""ROBOTS REVENGE - Title A: "Blockbuster" chrome title card (6 s).

Chrome ROBOTS / REVENGE on black with a slow push-in, a light sweep across the
metal, a lens glint, embers and a spark burst. Audio is a deep detuned "BWAAAM"
braam with sub boom, a metallic clang when the title lands and a long ring-out.

Every frame is drawn with PIL/numpy/OpenCV (this ffmpeg build has no drawtext)
and all audio is synthesised with numpy, then muxed with ffmpeg.

Usage:
    python3 title_blockbuster.py OUT.mp4
    python3 title_blockbuster.py OUT.mp4 --stills 1.0,2.4,3.4   # PNG stills only
"""
import os
import subprocess
import sys
import tempfile
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import signal

W, H, FPS = 1920, 1080, 25
DUR = 6.0
NF = int(round(DUR * FPS))
SR = 48000
T_HIT = 0.92            # title lands (frame 23): clang + braam
SWEEP = (1.55, 3.35)    # light sweep across the metal
T_GLINT = 3.30          # star glint on the corner of the S
FADE_OUT = 0.5
FFMPEG = "ffmpeg"
FONT = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
SS = 1.15               # title canvas supersampling (push-in stays below this)
CW, CH = int(W * SS), int(H * SS)


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
# Hot red chrome for REVENGE.
RED = gradient_lut([
    (0.00, (255, 236, 220)), (0.22, (246, 150, 120)), (0.44, (170, 32, 22)),
    (0.50, (40, 4, 4)), (0.56, (110, 18, 12)), (0.74, (226, 70, 40)),
    (0.90, (255, 196, 150)), (1.00, (255, 246, 230))])


# ----------------------------------------------------------------- title art
def draw_tracked(draw, text, font, cx, baseline, track):
    widths = [font.getlength(c) for c in text]
    total = sum(widths) + track * (len(text) - 1)
    x = cx - total / 2
    boxes = []
    for c, w in zip(text, widths):
        draw.text((x, baseline), c, font=font, fill=255, anchor="ls")
        boxes.append((x, x + w))
        x += w + track
    return boxes


def build_title():
    """Return (rgba, aux, anchors) on the CW x CH canvas.

    rgba: premultiplied chrome (0..1) + alpha. aux: face alpha, face luminance,
    soft glow. anchors: canvas points used for glints/sparks.
    """
    target_w = 0.70 * W * SS
    f1 = ImageFont.truetype(FONT, 100)
    s1 = 100 * target_w / (f1.getlength("ROBOTS") + 0.04 * 100 * 5)
    f1 = ImageFont.truetype(FONT, int(s1))
    track1 = 0.04 * s1
    f2 = ImageFont.truetype(FONT, 100)
    s2 = 100 * target_w / (f2.getlength("REVENGE") + 0.10 * 100 * 6)
    f2 = ImageFont.truetype(FONT, int(s2))
    track2 = 0.10 * s2
    cap1 = -f1.getbbox("H", anchor="ls")[1]
    cap2 = -f2.getbbox("H", anchor="ls")[1]
    gap = 0.20 * cap1
    top = CH / 2 - (cap1 + gap + cap2) / 2 - 0.02 * CH
    base1 = top + cap1
    base2 = base1 + gap + cap2

    m1 = Image.new("L", (CW, CH), 0)
    m2 = Image.new("L", (CW, CH), 0)
    b1 = draw_tracked(ImageDraw.Draw(m1), "ROBOTS", f1, CW / 2, base1, track1)
    draw_tracked(ImageDraw.Draw(m2), "REVENGE", f2, CW / 2, base2, track2)
    a1 = np.asarray(m1, np.float32) / 255
    a2 = np.asarray(m2, np.float32) / 255
    alpha = np.maximum(a1, a2)

    # Bevel height field from the distance to the glyph edge (quarter round).
    dist = cv2.distanceTransform((alpha > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    bw = 13 * SS
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
    v1 = (yy - (base1 - cap1)) / cap1
    v2 = (yy - (base2 - cap2)) / cap2
    wobble = 0.035 * np.sin(xx / (170 * SS)) + 0.02 * np.sin(xx / (53 * SS) + 1.3)
    r1 = np.clip(v1 + 0.55 * ny + 0.12 * nx + wobble, 0, 1)
    r2 = np.clip(v2 + 0.55 * ny + 0.12 * nx + wobble, 0, 1)
    n = len(STEEL) - 1
    col = np.where((a1 >= a2)[..., None], STEEL[(r1 * n).astype(np.int32)],
                   RED[(r2 * n).astype(np.int32)])

    # Key light from upper left: lambert on the bevel + hot specular.
    L = np.array([-0.45, -0.65, 0.62], np.float32)
    L /= np.linalg.norm(L)
    ndl = nx * L[0] + ny * L[1] + nz * L[2]
    shade = np.clip(0.85 + 0.9 * (ndl - L[2]), 0.30, 1.35)
    hv = L + np.array([0, 0, 1], np.float32)
    hv /= np.linalg.norm(hv)
    spec = np.clip(nx * hv[0] + ny * hv[1] + nz * hv[2], 0, 1) ** 70
    # Brushed-metal streaks.
    rng = np.random.default_rng(3)
    brush = rng.standard_normal((CH, CW)).astype(np.float32)
    brush = cv2.blur(cv2.blur(brush, (151, 1)), (151, 1))
    brush /= brush.std() + 1e-6
    col = col * shade[..., None] * (1 + 0.045 * brush[..., None]) + 330 * spec[..., None]
    col = np.clip(col / 255, 0, 1.6)

    # Dark steel extrusion straight down gives the letters some thickness.
    depth = int(9 * SS)
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
    anchors = {
        "s_corner": (b1[-1][1] - 0.12 * cap1, base1 - 0.93 * cap1),
        "r_corner": (b1[0][0] + 0.08 * cap1, base1 - 0.98 * cap1),
        "center": (CW / 2, (base1 - cap1 + base2) / 2),
        "x_span": (b1[0][0], b1[-1][1]),
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
        for i in range(p1.shape[0]):
            a = age / self.life[i]
            if a >= 1:
                continue
            heat = np.interp(a, [0, 0.25, 0.6, 1], [0, 1, 2, 3])
            cols = np.array([[255, 255, 235], [255, 214, 100], [255, 120, 30], [170, 35, 8]])
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
    def __init__(self):
        self.rng = np.random.default_rng(42)
        self.rgba, self.aux, self.anch = build_title()
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

        # Lens glints: big one on impact, star on the S after the sweep.
        if t >= T_HIT:
            e = float(np.exp(-(t - T_HIT) / 0.32))
            cx, cy = self.to_screen(np.array([self.anch["center"]]), t)[0]
            add_glint(img, cx, cy, 1.1 * e, 520, rot=0.0)
            add_streak(img, cy, cx, 0.9 * e)
        for t0, key, peak, size in ((T_GLINT, "s_corner", 1.25, 420), (2.05, "r_corner", 0.5, 200)):
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
        img *= 1 - float(smoothstep(DUR - FADE_OUT, DUR, t + 0.5 / FPS))
        return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- audio
def pan_gains(p):
    return np.sqrt(0.5 * (1 - p)), np.sqrt(0.5 * (1 + p))


def norm(x):
    return x / (np.max(np.abs(x)) + 1e-12)


def synth_audio(rng):
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


def write_wav(path, stereo):
    pcm = (np.clip(stereo.T, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------- main
def encode(out, wav, frames):
    cmd = [FFMPEG, "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", wav, "-map", "0:v", "-map", "1:a",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
           "-t", f"{DUR:.3f}", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    out = sys.argv[1]
    r = Renderer()
    if "--stills" in sys.argv:
        times = [float(x) for x in sys.argv[sys.argv.index("--stills") + 1].split(",")]
        stem = os.path.splitext(out)[0]
        for tt in times:
            fi = int(round(tt * FPS))
            Image.fromarray(r.render(fi / FPS, fi)).save(f"{stem}_t{tt:.2f}.png")
        return
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "title_a.wav")
        write_wav(wav, synth_audio(np.random.default_rng(2024)))
        encode(out, wav, (r.render(i / FPS, i) for i in range(NF)))
    print(out)


if __name__ == "__main__":
    main()
