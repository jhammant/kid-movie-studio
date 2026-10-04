"""Opening title D for ROBOTS REVENGE: red alert! A siren wails, then the title SLAMS in.

Hazard stripes scroll top and bottom, a flashing WARNING sign and a pulsing
red alarm light lead up to a chrome ROBOTS REVENGE that crashes onto the screen
with a camera shake, shockwave and sparks. Pictures are drawn with PIL/numpy
and the sound is synthesised with numpy (ffmpeg here has no drawtext).

Usage:  python3 title_warning.py OUT.mp4 [--stills DIR f1,f2,...]
"""
import os
import subprocess
import sys
import tempfile
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.signal import butter, fftconvolve, sosfilt

W, H, FPS = 1920, 1080, 25
DUR = 5.0
NF = int(round(DUR * FPS))
SR = 48000
FFMPEG = "ffmpeg"
FONTS = "/System/Library/Fonts/Supplemental/"
IMPACT = FONTS + "Impact.ttf"
DIN = FONTS + "DIN Condensed Bold.ttf"

# ---------------------------------------------------------------- timeline (s)
SIREN_T0 = 0.08
SIREN_P = 0.6          # one "wee-woo"
SLAM_T = 2.40          # the title lands (a frame boundary: frame 60)
FLY = 0.20             # how long the title takes to fall in
SUB_T = 2.95           # "tonight on the 9 o'clock news"
SHINE_T = 3.45
YELLOW = (255, 208, 0)


def siren_phase(t):
    return (t - SIREN_T0) / SIREN_P


def alarm_pulse(t):
    """0..1 light pulse, peaking in the middle of every high 'wee'."""
    if t < SIREN_T0:
        return 0.0
    p = 0.5 + 0.5 * np.cos(2 * np.pi * (siren_phase(t) - 0.25))
    return float(p ** 1.5)


def ease_out(u):
    u = np.clip(u, 0, 1)
    return 1 - (1 - u) ** 3


# ---------------------------------------------------------------- static layers
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
CX, CY = W / 2, H / 2
RN = np.sqrt(((xx - CX) / CX) ** 2 + ((yy - CY) / CY) ** 2)
EDGE = np.clip(RN / 1.25, 0, 1) ** 1.8                       # red glow lives at the edges
VIGNETTE = np.clip(1 - 0.33 * RN ** 2.2, 0, 1)
THETA = np.arctan2(yy - (CY - 40), xx - CX)
RADIAL = np.clip(np.hypot(xx - CX, yy - CY + 40) / 760, 0, 1) ** 0.8

BAND = 120
P = 150  # stripe period (px, along x)


def stripe_tile():
    w = W + 2 * P
    sx, sy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(BAND, dtype=np.float32))
    s = (sx + sy) % P
    a = np.clip(np.minimum(s, P / 2 - s) + 0.5, 0, 1) * (s < P / 2 + 1)  # anti-aliased diagonal edge
    a = np.clip(a, 0, 1)
    yellow = np.array(YELLOW, np.float32) / 255
    black = np.array([0.04, 0.035, 0.03], np.float32)
    tile = black + a[..., None] * (yellow - black)
    # a little bevel: bright lip on the top edge, shade on the bottom
    shade = 1 - 0.28 * (np.arange(BAND, dtype=np.float32) / BAND)[:, None, None]
    return tile * shade


STRIPES = stripe_tile()


def band_frame(offset):
    o = int(offset) % P
    return STRIPES[:, o:o + W]


def big_font(size):
    return ImageFont.truetype(IMPACT, size)


def warn_triangle(size, bang=True):
    """A yellow warning sign drawn by hand (the font has no usable ⚠)."""
    ss = 4
    S = size * ss
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = 10 * ss
    pts = [(S / 2, pad), (S - pad, S - pad * 1.4), (pad, S - pad * 1.4)]
    d.polygon(pts, fill=(15, 8, 5, 255))
    d.line(pts + [pts[0]], fill=(15, 8, 5, 255), width=int(18 * ss), joint="curve")
    inner = [(S / 2, pad + 26 * ss), (S - pad - 20 * ss, S - pad * 1.4 - 9 * ss), (pad + 20 * ss, S - pad * 1.4 - 9 * ss)]
    d.polygon(inner, fill=YELLOW + (255,))
    d.line(inner + [inner[0]], fill=YELLOW + (255,), width=int(10 * ss), joint="curve")
    if bang:
        f = ImageFont.truetype(DIN, int(S * 0.52))
        d.text((S / 2, S * 0.64), "!", font=f, fill=(15, 8, 5, 255), anchor="mm")
    return img.resize((size, size), Image.LANCZOS)


def text_layer(text, font, fill, stroke, stroke_fill, center, spacing=0):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if spacing:
        widths = [font.getlength(c) for c in text]
        total = sum(widths) + spacing * (len(text) - 1)
        x = center[0] - total / 2
        for c, w in zip(text, widths):
            d.text((x, center[1]), c, font=font, fill=fill, anchor="lm", stroke_width=stroke, stroke_fill=stroke_fill)
            x += w + spacing
    else:
        d.text(center, text, font=font, fill=fill, anchor="mm", stroke_width=stroke, stroke_fill=stroke_fill)
    return img


def to_premul(img):
    a = np.asarray(img, np.float32) / 255
    return a[..., :3] * a[..., 3:4], a[..., 3]


def build_warning_group():
    """Big flashing WARNING with signs either side (pre-slam)."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    f = ImageFont.truetype(DIN, 270)
    tw = f.getlength("WARNING")
    tri = warn_triangle(230)
    gap = 56
    total = tw + 2 * (230 + gap)
    x0 = W / 2 - total / 2
    cy = 455
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.text((W / 2, cy + 14), "WARNING", font=f, fill=YELLOW + (255,), anchor="mm", stroke_width=26, stroke_fill=YELLOW + (255,))
    glow.alpha_composite(tri, (int(x0), int(cy - 125)))
    glow.alpha_composite(tri, (int(x0 + total - 230), int(cy - 125)))
    glow = glow.filter(ImageFilter.GaussianBlur(28))
    d = ImageDraw.Draw(img)
    d.text((W / 2, cy + 14), "WARNING", font=f, fill=YELLOW + (255,), anchor="mm", stroke_width=9, stroke_fill=(15, 8, 5, 255))
    img.alpha_composite(tri, (int(x0), int(cy - 125)))
    img.alpha_composite(tri, (int(x0 + total - 230), int(cy - 125)))
    sub = text_layer("ROBOT INVASION DETECTED", ImageFont.truetype(DIN, 104), (255, 255, 255, 255), 6,
                     (15, 8, 5, 255), (W / 2, 705), spacing=10)
    return to_premul(img), np.asarray(glow, np.float32)[..., :3] / 255 * (np.asarray(glow, np.float32)[..., 3:4] / 255), to_premul(sub)


def build_title():
    """Chrome ROBOTS / REVENGE with black outline, red 3D extrusion and red glow."""
    f = big_font(300)
    lines = [("ROBOTS", 405), ("REVENGE", 678)]
    fill_m = Image.new("L", (W, H), 0)
    stroke_m = Image.new("L", (W, H), 0)
    df, ds = ImageDraw.Draw(fill_m), ImageDraw.Draw(stroke_m)
    boxes = []
    for word, cy in lines:
        df.text((W / 2, cy), word, font=f, fill=255, anchor="mm")
        ds.text((W / 2, cy), word, font=f, fill=255, anchor="mm", stroke_width=13, stroke_fill=255)
        boxes.append(df.textbbox((W / 2, cy), word, font=f, anchor="mm"))
    M = np.asarray(fill_m, np.float32) / 255
    S = np.asarray(stroke_m, np.float32) / 255
    # chrome gradient per line: bright sky, hard horizon, darker ground, bright lip
    grad = np.zeros((H, 3), np.float32)
    for (x0, y0, x1, y1) in boxes:
        ys = np.arange(y0, y1)
        u = (ys - y0) / max(1, (y1 - y0))
        stops = [0.0, 0.48, 0.5, 0.82, 1.0]
        cols = np.array([[1.00, 1.00, 1.00], [0.80, 0.83, 0.90], [0.42, 0.44, 0.52], [0.66, 0.68, 0.76], [0.95, 0.96, 1.00]])
        for c in range(3):
            grad[y0:y1, c] = np.interp(u, stops, cols[:, c])
    fill_rgb = np.broadcast_to(grad[:, None, :], (H, W, 3))
    # extrusion: stack the outline down-right
    ext = np.zeros_like(S)
    for k in range(1, 17):
        ext = np.maximum(ext, np.roll(np.roll(S, k, axis=0), k, axis=1) * (1 - 0.02 * k))
    ext_rgb = np.array([0.45, 0.02, 0.02], np.float32)
    # layer: extrusion under black outline under chrome
    rgb = ext[..., None] * ext_rgb * (1 - S[..., None])
    a = np.maximum(ext, S)
    rgb = rgb * (1 - S[..., None]) + S[..., None] * np.array([0.03, 0.0, 0.0], np.float32)
    rgb = rgb * (1 - M[..., None]) + M[..., None] * fill_rgb
    glow = cv2.GaussianBlur(np.maximum(S, ext), (0, 0), 30)
    glow_rgb = glow[..., None] * np.array([1.0, 0.08, 0.04], np.float32)
    x0 = min(b[0] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y0 = boxes[0][1]
    y1 = boxes[1][3]
    return rgb.astype(np.float32), a.astype(np.float32), glow_rgb.astype(np.float32), M, (x0, y0, x1, y1)


def build_pill():
    """Small flashing ⚠ WARNING ⚠ tag that sits above the title after the slam."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(DIN, 78)
    label = "WARNING"
    sp = 14
    tw = sum(f.getlength(c) for c in label) + sp * (len(label) - 1)
    tri = warn_triangle(84)
    inner = tw + 2 * (84 + 26)
    x0, x1 = W / 2 - inner / 2 - 34, W / 2 + inner / 2 + 34
    cy = 196
    d.rounded_rectangle([x0, cy - 50, x1, cy + 50], radius=50, fill=(18, 4, 4, 235), outline=YELLOW + (255,), width=6)
    x = W / 2 - tw / 2
    for c in label:
        d.text((x, cy + 4), c, font=f, fill=YELLOW + (255,), anchor="lm")
        x += f.getlength(c) + sp
    img.alpha_composite(tri, (int(W / 2 - inner / 2), int(cy - 46)))
    img.alpha_composite(tri, (int(W / 2 + inner / 2 - 84), int(cy - 46)))
    return to_premul(img)


def build_subtitle():
    return to_premul(text_layer("TONIGHT ON THE 9 O'CLOCK NEWS", ImageFont.truetype(DIN, 80),
                                (255, 255, 255, 255), 5, (15, 8, 5, 255), (W / 2, 893), spacing=6))


WARN, WARN_GLOW, WARN_SUB = build_warning_group()
T_RGB, T_A, T_GLOW, T_FILL, T_BOX = build_title()
T_CX, T_CY = (T_BOX[0] + T_BOX[2]) / 2, (T_BOX[1] + T_BOX[3]) / 2
PILL = build_pill()
SUBT = build_subtitle()


def scaled(layer, s, cx=None, cy=None):
    """Scale a full-frame layer about (cx, cy)."""
    if abs(s - 1) < 1e-4:
        return layer
    cx = T_CX if cx is None else cx
    cy = T_CY if cy is None else cy
    M = np.float32([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]])
    return cv2.warpAffine(layer, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def over(dst, rgb, a, alpha=1.0):
    """Premultiplied-alpha over."""
    a = a[..., None] * alpha
    return dst * (1 - a) + rgb * alpha


# ---------------------------------------------------------------- sparks & debris
rng = np.random.default_rng(11)
SPARKS = []
for i in range(70):
    side = rng.random()
    x = rng.uniform(T_BOX[0] - 40, T_BOX[2] + 40)
    y = T_BOX[3] - rng.uniform(0, 60) if side < 0.6 else rng.uniform(T_BOX[1], T_BOX[3])
    ang = rng.uniform(-np.pi * 0.95, -np.pi * 0.05) if side < 0.6 else rng.uniform(-np.pi, np.pi)
    sp = rng.uniform(500, 1700)
    SPARKS.append((x, y, np.cos(ang) * sp, np.sin(ang) * sp, rng.uniform(0.35, 0.9), rng.random() < 0.3))


def draw_sparks(frame_rgb, dt):
    if dt < 0 or dt > 1.0:
        return frame_rgb
    layer = np.zeros((H // 2, W // 2, 3), np.float32)
    chunks = []
    for (x, y, vx, vy, life, chunk) in SPARKS:
        if dt > life:
            continue
        g = 2600
        px, py = x + vx * dt, y + vy * dt + 0.5 * g * dt * dt
        qx, qy = px - vx * 0.03, py - (vy + g * dt) * 0.03
        fade = 1 - dt / life
        if chunk:
            chunks.append((px, py, fade, vx))
        else:
            cv2.line(layer, (int(qx / 2), int(qy / 2)), (int(px / 2), int(py / 2)),
                     (1.0 * fade, 0.75 * fade, 0.3 * fade), 2, cv2.LINE_AA)
    glow = cv2.GaussianBlur(layer, (0, 0), 4)
    layer = cv2.resize(layer + glow * 1.5, (W, H), interpolation=cv2.INTER_LINEAR)
    out = frame_rgb + layer
    for (px, py, fade, vx) in chunks:
        r = 9
        rot = (px * 0.05 + vx * 0.001) % np.pi
        pts = np.array([[px + r * np.cos(rot + k * np.pi / 2), py + r * np.sin(rot + k * np.pi / 2)] for k in range(4)], np.int32)
        cv2.fillConvexPoly(out, pts, (0.30, 0.30, 0.33), cv2.LINE_AA)
    return out


# ---------------------------------------------------------------- frame
def background(t):
    p = alarm_pulse(t)
    after = t >= SLAM_T
    amp = 0.75 if not after else 0.45
    base = np.array([0.055, 0.006, 0.010], np.float32)
    red = np.array([0.95, 0.06, 0.04], np.float32)
    k = (0.30 + amp * p) * EDGE
    img = base + k[..., None] * red
    # rotating beacon beams
    phi = t * 2 * np.pi * 0.55
    beams = np.zeros((H, W), np.float32)
    for off in (0.0, np.pi):
        d = np.angle(np.exp(1j * (THETA - phi - off)))
        beams += np.exp(-(d / 0.15) ** 2)
    img += (beams * RADIAL * (0.13 + 0.12 * p))[..., None] * np.array([1.0, 0.42, 0.18], np.float32)
    return img


def compose(t, frame):
    img = background(t)
    p = alarm_pulse(t)
    dt = t - SLAM_T
    # --- pre-slam: flashing WARNING
    if t < SLAM_T:
        on = ((t - SIREN_T0) % (SIREN_P / 2)) < 0.21 and t >= SIREN_T0
        intro = ease_out((t - 0.12) / 0.25)
        k = 1.0
        if t > SLAM_T - FLY:
            k = 1 - (t - (SLAM_T - FLY)) / FLY
        if on and intro > 0:
            s = 0.6 + 0.4 * intro + 0.25 * (1 - k)
            rgb, a = scaled(WARN[0], s, W / 2, 470), scaled(WARN[1], s, W / 2, 470)
            img += scaled(WARN_GLOW, s, W / 2, 470) * 0.55 * k
            img = over(img, rgb, a, k * min(1, intro * 1.5))
        if t > 0.35:
            sa = min(1, (t - 0.35) / 0.2) * k
            img = over(img, WARN_SUB[0], WARN_SUB[1], sa * (0.75 + 0.25 * p))
    # --- title flying in (motion-blurred) and landed
    if dt > -FLY:
        if dt < 0:
            acc_rgb = np.zeros_like(img)
            acc_a = np.zeros((H, W), np.float32)
            subs = 4
            for i in range(subs):
                u = 1 + (dt + i * (1 / FPS) / subs) / FLY       # 0..1 over the fall
                u = np.clip(u, 0, 1)
                s = 1 + 5.0 * (1 - u) ** 2.2
                acc_rgb += scaled(T_RGB, s)
                acc_a += scaled(T_A, s)
            alpha = np.clip((dt + FLY) / (FLY * 0.5), 0, 1)
            img = over(img, acc_rgb / subs, acc_a / subs, alpha)
        else:
            bounce = 1 - 0.075 * np.exp(-dt / 0.09) * np.cos(2 * np.pi * dt / 0.24)
            glow_k = 0.55 + 0.35 * p + 1.6 * np.exp(-dt / 0.15)
            img += scaled(T_GLOW, bounce) * glow_k
            rgb = scaled(T_RGB, bounce)
            a = scaled(T_A, bounce)
            # chrome shine sweeping across the letters
            if SHINE_T <= t < SHINE_T + 0.6:
                pos = T_BOX[0] - 300 + (T_BOX[2] - T_BOX[0] + 600) * (t - SHINE_T) / 0.6
                band = np.exp(-(((xx + (yy - T_CY) * 0.45) - pos) / 60) ** 2)
                rgb = rgb + (band * scaled(T_FILL, bounce))[..., None] * 0.55
            img = over(img, rgb, a)
            # shockwave ring
            if dt < 0.6:
                ring = np.zeros((H // 2, W // 2), np.float32)
                r = 90 + 1500 * dt ** 0.7
                cv2.ellipse(ring, (int(T_CX / 2), int(T_CY / 2)), (int(r * 1.35), int(r)), 0, 0, 360,
                            1.0, max(2, int(26 * (1 - dt / 0.6))), cv2.LINE_AA)
                ring = cv2.GaussianBlur(ring, (0, 0), 3)
                ring = cv2.resize(ring, (W, H)) * (1 - dt / 0.6) ** 1.5
                img += ring[..., None] * np.array([1.0, 0.85, 0.6], np.float32) * 0.8
            img = draw_sparks(img, dt)
            # after the slam: WARNING tag + subtitle
            if dt > 0.25:
                on = ((t - SIREN_T0) % (SIREN_P / 2)) < 0.21
                pa = min(1, (dt - 0.25) / 0.15) * (1.0 if on else 0.35)
                img = over(img, PILL[0], PILL[1], pa)
            if t > SUB_T:
                u = ease_out((t - SUB_T) / 0.35)
                sh = int(40 * (1 - u))
                rgb_s = np.roll(SUBT[0], sh, axis=0)
                a_s = np.roll(SUBT[1], sh, axis=0)
                img = over(img, rgb_s, a_s, u)
    # bands of hazard stripes (slide in at the start)
    drop = ease_out(t / 0.28)
    top = band_frame(t * 210)
    bot = band_frame(-t * 210 + 37)
    ty = int(-BAND + BAND * drop)
    by = int(H - BAND * drop)
    shadow = np.exp(-np.arange(40, dtype=np.float32) / 14)[:, None, None] * 0.6
    if ty + BAND > 0:
        img[max(0, ty):ty + BAND] = top[max(0, -ty):]
        img[ty + BAND:ty + BAND + 40] *= (1 - shadow[: max(0, min(40, H - ty - BAND))])
        img[ty + BAND - 6:ty + BAND] = 0.03
    if by < H:
        img[by:H] = bot[: H - by]
        img[max(0, by - 40):by] *= (1 - shadow[::-1][-(by - max(0, by - 40)):])
        img[by:by + 6] = 0.03
    # impact: white flash, punch-in zoom + shake, chromatic split
    if 0 <= dt < 0.2:
        img += (0.7 * np.exp(-dt / 0.05)) * np.array([1.0, 0.95, 0.9], np.float32)
    img *= VIGNETTE[..., None]
    if 0 <= dt < 0.9:
        amp = 46 * np.exp(-dt / 0.17)
        sx = amp * np.sin(dt * 71.0) * 0.9
        sy = amp * np.cos(dt * 57.0)
        rot = 1.2 * np.exp(-dt / 0.17) * np.sin(dt * 43.0)
        zoom = 1 + 0.055 * np.exp(-dt / 0.3)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rot, zoom)
        M[0, 2] += sx
        M[1, 2] += sy
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        ca = int(round(14 * np.exp(-dt / 0.14)))
        if ca:
            img[..., 0] = np.roll(img[..., 0], ca, axis=1)
            img[..., 2] = np.roll(img[..., 2], -ca, axis=1)
    grain = np.random.default_rng(frame).normal(0, 0.012, (H // 2, W // 2)).astype(np.float32)
    img += cv2.resize(grain, (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def render_frame(frame):
    return compose(frame / FPS, frame)


# ---------------------------------------------------------------- audio
N = int(round(DUR * SR))


def lp(x, hz, order=2):
    return sosfilt(butter(order, hz, "low", fs=SR, output="sos"), x)


def hp(x, hz, order=2):
    return sosfilt(butter(order, hz, "high", fs=SR, output="sos"), x)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "band", fs=SR, output="sos"), x)


def phase_of(f):
    return 2 * np.pi * np.cumsum(f) / SR


def env(n, attack=0.002, decay=0.1):
    t = np.arange(n) / SR
    return np.minimum(1, t / max(attack, 1e-4)) * np.exp(-t / decay)


def fade(sig, a=0.01, b=0.01):
    na, nb = int(a * SR), int(b * SR)
    if na:
        sig[:na] *= np.linspace(0, 1, na)
    if nb:
        sig[-nb:] *= np.linspace(1, 0, nb)
    return sig


class Mix:
    def __init__(self):
        self.dry = np.zeros((2, N))
        self.wet = np.zeros((2, N))

    def add(self, sig, t0, db=0.0, pan=0.0, send=0.2):
        i0 = int(round(t0 * SR))
        if i0 >= N:
            return
        if i0 < 0:
            sig, i0 = sig[-i0:], 0
        sig = sig[: N - i0] * 10 ** (db / 20)
        for ch, g in ((0, np.sqrt(1 - pan)), (1, np.sqrt(1 + pan))):
            self.dry[ch, i0:i0 + len(sig)] += sig * g
            self.wet[ch, i0:i0 + len(sig)] += sig * g * send

    def render(self, room=0.9):
        r = np.random.default_rng(5)
        n = int(room * 2 * SR)
        tt = np.arange(n) / SR
        out = self.dry.copy()
        for ch in range(2):
            ir = lp(r.normal(0, 1, n), 4200) * np.exp(-tt / room * 3.2)
            ir[: int(0.02 * SR)] = 0
            ir /= np.sqrt(np.sum(ir ** 2))
            out[ch] += fftconvolve(self.wet[ch], ir)[:N]
        return out


def siren(t0, t1, hi=960.0, lo=720.0):
    n = int((t1 - t0) * SR)
    tt = t0 + np.arange(n) / SR
    high = ((tt - SIREN_T0) % SIREN_P) < SIREN_P / 2
    f = np.where(high, hi, lo).astype(float)
    k = int(0.035 * SR)
    f = np.convolve(np.pad(f, (k, k), mode="edge"), np.ones(k) / k, mode="same")[k:-k]  # little glide
    f *= 1 + 0.006 * np.sin(2 * np.pi * 5.5 * tt)
    voice = np.zeros(n)
    for det in (1.0, 1.006):
        ph = phase_of(f * det)
        voice += sum(np.sin(ph * h) / h for h in range(1, 11)) * 0.5 + 0.35 * np.tanh(2.5 * np.sin(ph))
    voice = lp(voice, 3800)
    voice = hp(voice, 250)
    return voice / np.max(np.abs(voice))


def build_audio():
    r = np.random.default_rng(8)
    m = Mix()
    # siren until just before the slam (cut leaves a gap of whoosh, then BOOM)
    cut = SLAM_T - 0.14
    s = fade(siren(SIREN_T0, cut), 0.03, 0.04)
    m.add(s, SIREN_T0, -9, pan=-0.12, send=0.35)
    m.add(np.roll(s, int(0.011 * SR)), SIREN_T0, -13, pan=0.5, send=0.35)
    # low alarm buzz on each "woo" for a bit of menace (still cartoony)
    n = int((cut - SIREN_T0) * SR)
    tt = SIREN_T0 + np.arange(n) / SR
    woo = (((tt - SIREN_T0) % SIREN_P) >= SIREN_P / 2).astype(float)
    woo = np.convolve(woo, np.ones(400) / 400, mode="same")
    buzz = lp(np.sign(np.sin(2 * np.pi * 110 * tt)) * 0.6 + np.sin(2 * np.pi * 55 * tt), 900) * woo
    m.add(fade(buzz, 0.03, 0.04), SIREN_T0, -17, send=0.2)
    # whoosh into the slam
    n = int(0.42 * SR)
    noise = r.normal(0, 1, n)
    u = np.arange(n) / n
    wh = np.zeros(n)
    for i in range(0, n, 2400):
        seg = noise[i:i + 2400 + 400]
        fc = 300 + 4200 * (i / n) ** 2
        part = bp(seg, fc * 0.6, min(fc * 1.6, 20000))[:2400]
        wh[i:i + len(part)] = part
    wh = wh * u ** 2.5
    wh[-200:] *= np.linspace(1, 0, 200)
    m.add(wh / np.max(np.abs(wh)), SLAM_T - 0.42, -9, pan=0.0, send=0.25)
    # ---- SLAM
    n = int(0.03 * SR)
    m.add(hp(r.normal(0, 1, n), 2500) * env(n, 0.0003, 0.008), SLAM_T, -3, send=0.4)
    n = int(0.5 * SR)
    f = 48 + 140 * np.exp(-np.arange(n) / SR / 0.04)
    punch = np.tanh(2.8 * np.sin(phase_of(f))) * env(n, 0.001, 0.16)
    m.add(punch, SLAM_T, -1, send=0.3)
    n = int(1.9 * SR)
    f = 31 + 24 * np.exp(-np.arange(n) / SR / 0.25)
    sub = np.tanh(1.6 * np.sin(phase_of(f))) * env(n, 0.004, 0.65)
    m.add(sub, SLAM_T, -2, send=0.05)
    n = int(1.4 * SR)
    tt = np.arange(n) / SR
    clang = sum(a * np.sin(2 * np.pi * fr * tt) * np.exp(-tt / d)
                for fr, a, d in ((187, 1.0, 0.55), (433, 0.7, 0.42), (761, 0.55, 0.3), (1129, 0.4, 0.22), (1693, 0.3, 0.15)))
    m.add(clang * env(n, 0.001, 10) / 2.5, SLAM_T, -10, send=0.6)
    n = int(1.2 * SR)
    m.add(lp(r.normal(0, 1, n), 260, 4) * env(n, 0.002, 0.35) * 4, SLAM_T, -6, send=0.4)
    # tinkly debris after the hit
    for k in range(9):
        n = int(0.25 * SR)
        tt = np.arange(n) / SR
        fr = r.uniform(2200, 5200)
        ping = (np.sin(2 * np.pi * fr * tt) + 0.5 * np.sin(2 * np.pi * fr * 2.7 * tt)) * np.exp(-tt / 0.05)
        m.add(ping, SLAM_T + 0.15 + r.uniform(0, 0.7), -26 + r.uniform(-4, 2), pan=r.uniform(-0.8, 0.8), send=0.3)
    # siren comes back, further away, and fades out
    back = SLAM_T + 0.85
    s2 = siren(back, DUR, hi=900, lo=675)
    ramp = np.minimum(1, np.arange(len(s2)) / (0.4 * SR))
    tail = np.linspace(1, 0, len(s2)) ** 0.5
    m.add(lp(s2, 2500) * ramp * tail, back, -16, pan=0.1, send=0.6)
    out = m.render()
    out /= np.max(np.abs(out))
    out = np.tanh(1.8 * out) / np.tanh(1.8)
    tail = int(0.15 * SR)
    out[:, -tail:] *= np.linspace(1, 0, tail)
    out *= 10 ** (-2.5 / 20) / np.max(np.abs(out))
    return out


def write_wav(path, stereo):
    pcm = (np.clip(stereo.T, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def encode(out, wav):
    cmd = [FFMPEG, "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", wav, "-map", "0:v", "-map", "1:a",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-ac", "2",
           "-movflags", "+faststart", "-t", f"{DUR}", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(NF):
        p.stdin.write(render_frame(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        sys.exit("ffmpeg failed")


def main():
    out = sys.argv[1]
    if "--stills" in sys.argv:
        d = sys.argv[sys.argv.index("--stills") + 1]
        frames = [int(x) for x in sys.argv[sys.argv.index("--stills") + 2].split(",")]
        os.makedirs(d, exist_ok=True)
        for f in frames:
            Image.fromarray(render_frame(f)).save(os.path.join(d, f"D_{f:03d}.png"))
        return
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "title_warning.wav")
        write_wav(wav, build_audio())
        encode(out, wav)
    print("wrote", out)


if __name__ == "__main__":
    main()
