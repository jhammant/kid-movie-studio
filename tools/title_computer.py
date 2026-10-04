"""Opening title C for ROBOTS REVENGE: a retro green-phosphor robot computer boots up.

A CRT terminal powers on, types its boot log, fills a progress bar, spots the
humans, glitches, then flickers a giant dot-matrix ROBOTS REVENGE onto the
screen before switching itself off. All pictures are drawn with PIL/numpy and
all sound is synthesised with numpy (ffmpeg here has no drawtext).

Usage:  python3 title_computer.py OUT.mp4 [--stills DIR f1,f2,...]
"""
import os
import subprocess
import sys
import tempfile
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, fftconvolve, sosfilt

W, H, FPS = 1920, 1080, 25
DUR = 6.0
NF = int(round(DUR * FPS))
SR = 48000
FFMPEG = "ffmpeg"
MENLO = "/System/Library/Fonts/Menlo.ttc"  # index 1 = Bold

# ---------------------------------------------------------------- timeline (s)
ON_END = 0.32
L1, L1_T, L1_CPS = "> BOOTING ROBOT_PROTOCOL.EXE ...", 0.40, 46
L2, L2_T, L2_CPS = "> LOADING EVIL PLANS ...", 1.14, 55
BAR_T0, BAR_T1 = 1.62, 2.42
L4, L4_T, L4_CPS = "> HUMANS DETECTED!", 2.48, 45
ALERT_T0, ALERT_T1 = 2.90, 3.36
GLITCH_T0, GLITCH_T1 = 3.36, 3.52
TITLE_T0, TITLE_FILL = 3.52, 0.56
CATCH_T = 4.08  # whole title dips for one frame, like a tube catching
BEEP_T = 4.12
SUB, SUB_T, SUB_CPS = "> LIVE: THE 9 O'CLOCK NEWS", 4.30, 42
OFF_T0, OFF_T1 = 5.60, 6.00

# ---------------------------------------------------------------- layout (px)
X0 = 200
LINE_Y = {"l1": 245, "l2": 340, "bar": 440, "l4": 560}
FONT = ImageFont.truetype(MENLO, 56, index=1)
HDR_FONT = ImageFont.truetype(MENLO, 40, index=1)
SUB_FONT = ImageFont.truetype(MENLO, 52, index=1)
CW = FONT.getlength("M")


def type_times(text, start, cps, seed):
    """When each character of a line appears (slightly uneven, like real typing)."""
    r = np.random.default_rng(seed)
    out, t = [], start
    for _ in text:
        out.append(t)
        t += (1.0 / cps) * (1 + 0.35 * (r.random() * 2 - 1))
    return np.array(out)


T1 = type_times(L1, L1_T, L1_CPS, 1)
T2 = type_times(L2, L2_T, L2_CPS, 2)
T4 = type_times(L4, L4_T, L4_CPS, 4)
TS = type_times(SUB, SUB_T, SUB_CPS, 5)


def shown(times, t):
    return int(np.searchsorted(times, t, side="right"))


def bar_progress(t):
    """0..1 with a couple of dramatic stalls."""
    u = np.clip((t - BAR_T0) / (BAR_T1 - BAR_T0), 0, 1)
    return float(np.interp(u, [0, 0.28, 0.42, 0.72, 0.86, 1.0], [0, 0.44, 0.47, 0.89, 0.91, 1.0]))


# ---------------------------------------------------------------- dot-matrix title
GLYPHS = {
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
}
CELL, BLOCK = 36, 30
TITLE_LINES = ["ROBOTS", "REVENGE"]
TITLE_TOP = 205


def build_title_cells():
    """List of (x, y, lit, t_on) for every cell of the matrix sign."""
    rng = np.random.default_rng(42)
    cells = []
    widest = max(len(s) for s in TITLE_LINES) * 6 - 1
    for li, word in enumerate(TITLE_LINES):
        ncols = len(word) * 6 - 1
        x_left = W / 2 - ncols * CELL / 2
        y_top = TITLE_TOP + li * 9 * CELL
        for gi, ch in enumerate(word):
            for r, row in enumerate(GLYPHS[ch]):
                for c, bit in enumerate(row):
                    col = gi * 6 + c
                    x = x_left + col * CELL
                    y = y_top + r * CELL
                    sweep = (x - (W / 2 - widest * CELL / 2)) / (widest * CELL)
                    t_on = TITLE_T0 + 0.36 * sweep + 0.18 * rng.random()
                    cells.append((x, y, bit == "1", t_on, rng.random()))
    return cells


CELLS = build_title_cells()


def flick(seed, frame):
    """Deterministic pseudo-random 0..1 per (cell, frame)."""
    v = np.sin(seed * 12.9898 + frame * 78.233) * 43758.5453
    return v - np.floor(v)


# ---------------------------------------------------------------- drawing
def draw_cursor(d, x, y, font_px, on):
    if on:
        d.rectangle([x + 2, y - 2, x + font_px * 0.6, y + font_px * 1.02], fill=235)


def draw_header(d, t):
    d.rectangle([0, 70, W, 140], fill=205)
    d.text((X0, 105), "ROBO-OS 9000", font=HDR_FONT, fill=0, anchor="lm")
    d.text((W / 2, 105), "MODE: WORLD DOMINATION", font=HDR_FONT, fill=0, anchor="mm")
    sec = 57 + max(0, int(np.floor(t - (TITLE_T0 - 3.0))))
    clock = "21:00:00" if sec >= 60 else f"20:59:{sec:02d}"
    d.text((W - X0, 105), clock, font=HDR_FONT, fill=0, anchor="rm")


def draw_terminal(d, t, frame):
    blink = (t * 2.2) % 1 < 0.55
    n1, n2, n4 = shown(T1, t), shown(T2, t), shown(T4, t)
    d.text((X0, LINE_Y["l1"]), L1[:n1], font=FONT, fill=215)
    d.text((X0, LINE_Y["l2"]), L2[:n2], font=FONT, fill=215)
    cursor = None
    if t < L1_T:
        cursor = (X0, LINE_Y["l1"], blink)
    elif n1 < len(L1) or t < T2[0]:
        cursor = (X0 + n1 * CW, LINE_Y["l1"], True if n1 < len(L1) else blink)
    elif n2 < len(L2) or t < BAR_T0:
        cursor = (X0 + n2 * CW, LINE_Y["l2"], True if n2 < len(L2) else blink)

    if t >= BAR_T0 - 0.04:
        p = bar_progress(t)
        bx0, by0, bx1, by1 = X0 + 2 * CW, LINE_Y["bar"], 1480, LINE_Y["bar"] + 62
        d.rectangle([bx0, by0, bx1, by1], outline=215, width=5)
        nseg = 20
        segw = (bx1 - bx0 - 20) / nseg
        for s in range(int(round(p * nseg))):
            sx = bx0 + 12 + s * segw
            d.rectangle([sx, by0 + 12, sx + segw - 7, by1 - 12], fill=225)
        pct = int(round(p * 100))
        d.text((bx1 + 36, by0 + 31), f"{pct:3d}%", font=FONT, fill=230 if pct == 100 else 200, anchor="lm")

    if t >= L4_T - 0.02:
        y = LINE_Y["l4"]
        inverse = ALERT_T0 <= t < ALERT_T1 and int((t - ALERT_T0) / 0.06) % 2 == 0
        txt = L4[:n4]
        if inverse:
            x1 = X0 + len(L4) * CW
            d.rectangle([X0 - 14, y - 10, x1 + 14, y + 70], fill=240)
            d.text((X0, y), txt, font=FONT, fill=0)
        else:
            d.text((X0, y), txt, font=FONT, fill=235)
        if n4 < len(L4):
            cursor = (X0 + n4 * CW, y, True)
        elif t < ALERT_T0:
            cursor = (X0 + n4 * CW, y, blink)
    if cursor:
        draw_cursor(d, cursor[0], cursor[1], 56, cursor[2])
    draw_robot(d, t)


ROBOT_HEAD = [
    "......#......",
    ".....###.....",
    "......#......",
    "..#########..",
    ".#.........#.",
    ".#.........#.",
    "##.........##",
    "##.........##",
    ".#.........#.",
    ".#..#####..#.",
    ".#.........#.",
    "..#########..",
]
RCELL, RX, RY = 24, 1300, 600


def draw_robot(d, t):
    """Little pixel robot in the corner: eyes scan about, then go angry at HUMANS DETECTED."""
    if t < L2_T:
        return
    reveal = int(np.clip((t - L2_T) / 0.25, 0, 1) * len(ROBOT_HEAD))
    angry = t >= T4[-1]
    cells = [(c, r) for r, row in enumerate(ROBOT_HEAD[:reveal]) for c, bit in enumerate(row) if bit == "#"]
    if angry:
        cells += [(3, 5), (4, 5), (3, 6), (4, 6), (5, 6), (8, 5), (9, 5), (7, 6), (8, 6), (9, 6)]
        cells = [cc for cc in cells if not (cc[1] == 9 and 3 < cc[0] < 9)]  # grille -> frown
        cells += [(5, 8), (6, 8), (7, 8), (4, 9), (8, 9)]
    elif reveal > 7:
        dx = int(np.round(np.sin((t - L2_T) * 4.5) * 1.2))
        for ex in (4, 7):
            cells += [(ex + dx + i, 5 + j) for i in (0, 1) for j in (0, 1)]
    flash = angry and ALERT_T0 <= t < ALERT_T1 and int((t - ALERT_T0) / 0.06) % 2 == 0
    val = 255 if flash else 200
    for c, r in cells:
        x, y = RX + c * RCELL, RY + r * RCELL
        d.rounded_rectangle([x + 2, y + 2, x + RCELL - 2, y + RCELL - 2], radius=3, fill=val)


def draw_title(d, t, frame):
    lit_level = 1.0
    if abs(t - CATCH_T) < 0.5 / FPS:
        lit_level = 0.35
    elif t > CATCH_T:
        lit_level = 0.93 + 0.07 * np.sin(t * 9.0)
    for i, (x, y, lit, t_on, seed) in enumerate(CELLS):
        box = [x + 3, y + 3, x + 3 + BLOCK, y + 3 + BLOCK]
        val = 0.075
        if lit and t >= t_on:
            on = True
            if t < t_on + 0.14 and flick(seed * 97 + i, frame) < 0.35:
                on = False
            if on:
                val = lit_level
        d.rounded_rectangle(box, radius=5, fill=int(val * 255))
    n = shown(TS, t)
    if t >= SUB_T - 0.3:
        x = W / 2 - len(SUB) * SUB_FONT.getlength("M") / 2
        y = 860
        d.text((x, y), SUB[:n], font=SUB_FONT, fill=215)
        blink = (t * 2.2) % 1 < 0.55
        draw_cursor(d, x + n * SUB_FONT.getlength("M"), y, 52, n < len(SUB) or blink)


def glitch(I, frame, k):
    """k: 0..1 glitch strength. Tears horizontal slices and sprays noise rows."""
    r = np.random.default_rng(1000 + frame)
    out = I.copy()
    y = 0
    while y < H:
        h = int(r.integers(8, 90))
        if r.random() < 0.3 + 0.6 * k:
            out[y:y + h] = np.roll(out[y:y + h], int(r.normal(0, 140 * k + 20)), axis=1)
        y += h
    rows = r.random(H) < 0.05 + 0.35 * k
    out[rows] = np.clip(out[rows] + r.random((rows.sum(), W)) * 0.9 * k, 0, 1)
    if k > 0.8:
        out = out * 0.4 + (r.random((H, W)) > 0.82) * 0.5
    return out


# ---------------------------------------------------------------- CRT model
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
NX, NY = (xx + 0.5) / W * 2 - 1, (yy + 0.5) / H * 2 - 1
CURVE, ZOOM = 3.6, 1.075
SRC_X = NX * (1 + (NY / CURVE) ** 2) * ZOOM
SRC_Y = NY * (1 + (NX / CURVE) ** 2) * ZOOM
MAP_X = ((SRC_X + 1) / 2 * W - 0.5).astype(np.float32)
MAP_Y = ((SRC_Y + 1) / 2 * H - 0.5).astype(np.float32)
# soft-edged screen mask with rounded corners
edge = np.maximum(np.abs(SRC_X), np.abs(SRC_Y))
MASK = np.clip((1.0 - edge) * 260, 0, 1).astype(np.float32)
MASK = cv2.GaussianBlur(MASK, (0, 0), 1.2)
RIM = (MASK * (1 - MASK) * 4) ** 2
R2 = (NX * 1.0) ** 2 + (NY * 1.1) ** 2
VIGNETTE = np.clip(1 - 0.42 * R2 ** 1.4, 0, 1).astype(np.float32)
SHEEN = (0.045 * np.exp(-(((xx - 0.30 * W) / 620) ** 2 + ((yy - 0.18 * H) / 260) ** 2))).astype(np.float32)
SCAN = np.tile(np.array([1.0, 1.0, 0.80, 0.56], np.float32), H // 4 + 1)[:H, None]
NOISE = [np.random.default_rng(s).normal(0, 0.018, (H, W)).astype(np.float32) for s in range(6)]
GLASS = np.array([0.010, 0.030, 0.018], np.float32)
BEZEL = np.array([0.030, 0.032, 0.030], np.float32)


def crt(I, t, frame):
    small = cv2.resize(I, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    big = cv2.resize(cv2.GaussianBlur(small, (0, 0), 7), (W, H), interpolation=cv2.INTER_LINEAR)
    mid = cv2.GaussianBlur(I, (0, 0), 3.0)
    soft = cv2.GaussianBlur(I, (0, 0), 1.1)
    beam = 0.75 * soft + 0.45 * mid + 0.75 * big
    y0 = (t * 380) % (H + 500) - 250
    band = 1 + 0.06 * np.exp(-(((np.arange(H, dtype=np.float32) - y0) / 110) ** 2))[:, None]
    r = np.random.default_rng(frame)
    flicker = 1 + 0.022 * np.sin(t * 2 * np.pi * 13.7) + 0.015 * r.normal()
    beam = beam * SCAN * band * flicker + NOISE[frame % len(NOISE)] * (0.6 + beam)
    beam = cv2.remap(beam, MAP_X, MAP_Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    beam = np.maximum(beam, 0) * VIGNETTE
    hot = np.clip(beam - 0.72, 0, 1.5)
    rgb = np.empty((H, W, 3), np.float32)
    rgb[..., 0] = 0.24 * beam + 0.65 * hot
    rgb[..., 1] = 1.0 * beam
    rgb[..., 2] = 0.42 * beam + 0.45 * hot
    rgb += GLASS + SHEEN[..., None]
    m = MASK[..., None]
    rgb = rgb * m + (BEZEL + 0.05 * RIM[..., None]) * (1 - m) + 0.05 * RIM[..., None]
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def ease(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def squash(I, sy, sx, boost):
    """Squeeze the picture into a horizontal line / dot (CRT on/off)."""
    h = max(2, int(H * sy))
    w = max(4, int(W * sx))
    small = cv2.resize(I, (w, h), interpolation=cv2.INTER_AREA)
    small = np.clip(small + boost, 0, 3)
    out = np.zeros_like(I)
    y0, x0 = (H - h) // 2, (W - w) // 2
    out[y0:y0 + h, x0:x0 + w] = small
    return out


def content(t, frame):
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    draw_header(d, t)
    if t < GLITCH_T0:
        draw_terminal(d, t, frame)
    elif t < GLITCH_T1:
        draw_terminal(d, GLITCH_T0 - 0.01, frame)
    else:
        draw_title(d, t, frame)
    I = np.asarray(img, np.float32) / 255.0
    if GLITCH_T0 <= t < GLITCH_T1:
        I = glitch(I, frame, (t - GLITCH_T0) / (GLITCH_T1 - GLITCH_T0) + 0.25)
    if t < ON_END:
        u = t / ON_END
        sy = 0.004 + 0.996 * ease(u) ** 1.6
        sx = min(1.0, 0.25 + u * 3)
        I = squash(I, sy, sx, 1.4 * (1 - u) ** 1.5)
    elif t >= OFF_T0:
        u = (t - OFF_T0) / (OFF_T1 - OFF_T0)
        if u < 0.5:
            v = ease(u / 0.5)
            I = squash(I * (1 - v) + v * 0.9, 1 - 0.996 * v, 1.0, 0.8 * v)
        elif u < 0.8:
            v = (u - 0.5) / 0.3
            I = squash(np.ones_like(I), 0.004, max(0.003, (1 - v) ** 2), 1.0)
        else:
            v = (u - 0.8) / 0.2
            I = squash(np.ones_like(I), 0.004, 0.003, 0.6) * max(0.0, 1 - v) * 1.4
    return I


def render_frame(frame):
    t = frame / FPS
    return crt(content(t, frame), t, frame)


# ---------------------------------------------------------------- audio
N = int(round(DUR * SR))


def osc(freq, n, kind="sine"):
    f = np.full(n, float(freq)) if np.ndim(freq) == 0 else np.asarray(freq, float)[:n]
    ph = 2 * np.pi * np.cumsum(f) / SR
    if kind == "sine":
        return np.sin(ph)
    if kind == "square":
        return np.tanh(3.5 * np.sin(ph))
    if kind == "tri":
        return 2 / np.pi * np.arcsin(np.sin(ph))
    if kind == "saw":
        return sum(np.sin(ph * k) / k for k in range(1, 9)) * 0.6
    raise ValueError(kind)


def env(n, attack=0.002, decay=0.05, release=None):
    t = np.arange(n) / SR
    e = np.minimum(1, t / max(attack, 1e-4)) * np.exp(-t / decay)
    if release:
        r = int(release * SR)
        e[-r:] *= np.linspace(1, 0, r)
    return e


def lp(x, hz, order=2):
    return sosfilt(butter(order, hz, "low", fs=SR, output="sos"), x)


def hp(x, hz, order=2):
    return sosfilt(butter(order, hz, "high", fs=SR, output="sos"), x)


class Mix:
    def __init__(self):
        self.dry = np.zeros((2, N))
        self.wet = np.zeros((2, N))

    def add(self, sig, t0, db=0.0, pan=0.0, send=0.15):
        i0 = int(t0 * SR)
        if i0 >= N:
            return
        sig = sig[: N - i0] * 10 ** (db / 20)
        gl, gr = np.sqrt(1 - pan), np.sqrt(1 + pan)
        for ch, g in ((0, gl), (1, gr)):
            self.dry[ch, i0:i0 + len(sig)] += sig * g
            self.wet[ch, i0:i0 + len(sig)] += sig * g * send

    def render(self, room=0.35):
        r = np.random.default_rng(77)
        n = int(room * 2.2 * SR)
        tt = np.arange(n) / SR
        out = self.dry.copy()
        for ch in range(2):
            ir = r.normal(0, 1, n) * np.exp(-tt / room * 3)
            ir = lp(ir, 5000)
            ir[: int(0.012 * SR)] = 0
            ir /= np.sqrt(np.sum(ir ** 2))
            out[ch] += fftconvolve(self.wet[ch], ir)[:N] * 0.9
        return out


def bleep(freq, dur=0.032, kind="square", decay=0.013):
    n = int(dur * SR)
    return osc(freq, n, kind) * env(n, 0.001, decay)


def build_audio():
    r = np.random.default_rng(3)
    m = Mix()
    # power-on: relay click, degauss thrum, rising tube whoop
    n = int(0.01 * SR)
    m.add(hp(r.normal(0, 1, n), 1500) * env(n, 0.0005, 0.003), 0.0, -15, send=0.3)
    n = int(0.8 * SR)
    wob = 1 + 0.5 * np.sin(2 * np.pi * 9 * np.arange(n) / SR)
    m.add(lp(osc(58, n, "saw"), 900) * env(n, 0.03, 0.22) * wob, 0.0, -9, send=0.1)
    n = int(0.34 * SR)
    sweep = 140 * (1100 / 140) ** (np.arange(n) / n)
    m.add(osc(sweep, n, "tri") * env(n, 0.02, 0.2, 0.06), 0.02, -17, send=0.3)
    # mains hum + fan bed while the set is on
    a, b = int(0.05 * SR), int(OFF_T0 * SR) + int(0.2 * SR)
    n = b - a
    hum = osc(60, n) + 0.5 * osc(120, n) + 0.3 * osc(180, n)
    fan = lp(r.normal(0, 1, n), 700)
    bed = (hum * 0.02 + fan * 0.03) * np.minimum(1, np.arange(n) / (0.15 * SR))
    bed[-int(0.25 * SR):] *= np.linspace(1, 0, int(0.25 * SR))
    m.add(bed, 0.05, 0, send=0)
    # the little robot pops up: "bloo-beep"
    for f0, dt in ((523, 0.0), (1046, 0.07)):
        m.add(bleep(f0, 0.07, "tri", 0.05), L2_T + dt, -15, pan=0.4, send=0.2)
    # key bleeps
    scale = [784, 880, 1047, 1175, 1319, 1568]
    for text, times, db in ((L1, T1, -15), (L2, T2, -15), (L4, T4, -14)):
        for ch, tc in zip(text, times):
            if ch == " ":
                continue
            m.add(bleep(r.choice(scale)), tc, db, pan=r.uniform(-0.35, 0.35), send=0.12)
    # progress bar: pitch follows progress, chirpy data underneath, ding at 100%
    a, b = BAR_T0, BAR_T1
    n = int((b - a) * SR)
    tt = a + np.arange(n) / SR
    prog = np.array([bar_progress(x) for x in tt[:: SR // 200]])
    prog = np.interp(np.arange(n), np.arange(len(prog)) * (SR // 200), prog)
    f = 190 * 7.5 ** prog * (1 + 0.025 * np.sin(2 * np.pi * 7 * (tt - a)))
    whoop = osc(f, n, "tri") * (0.45 + 0.55 * prog)
    whoop[:480] *= np.linspace(0, 1, 480)
    whoop[-960:] *= np.linspace(1, 0, 960)
    m.add(whoop, a, -13, send=0.2)
    tc = a
    while tc < b:
        m.add(bleep(r.uniform(1800, 4200), 0.014, "sine", 0.006), tc, -21, pan=r.uniform(-0.6, 0.6), send=0.1)
        tc += r.uniform(0.022, 0.05)
    n = int(0.5 * SR)
    ding = (osc(1760, n) + 0.35 * osc(3520, n) + 0.2 * osc(5274, n)) * env(n, 0.002, 0.16)
    m.add(ding, b, -11, send=0.35)
    # alert: four quick bips in time with the inverse flashes
    k = 0
    while ALERT_T0 + k * 0.12 < ALERT_T1:
        m.add(bleep(1397, 0.07, "square", 0.08) * env(int(0.07 * SR), 0.002, 1, 0.01), ALERT_T0 + k * 0.12, -12, send=0.2)
        k += 1
    # glitch: bit-crushed static
    n = int((GLITCH_T1 - GLITCH_T0) * SR)
    sh = np.repeat(r.uniform(-1, 1, n // 24 + 1), 24)[:n]
    m.add(sh * np.sign(np.sin(2 * np.pi * 37 * np.arange(n) / SR)) * env(n, 0.003, 0.4, 0.02), GLITCH_T0, -16, send=0.05)
    # title fill: one tiny blip per lit cell, pitch rising with time; electric buzz underneath
    for (x, y, lit, t_on, seed) in CELLS:
        if lit:
            u = (t_on - TITLE_T0) / TITLE_FILL
            m.add(bleep(600 * 4 ** u * r.uniform(0.95, 1.05), 0.012, "sine", 0.005), t_on, -22,
                  pan=(x / W - 0.5) * 1.2, send=0.1)
    n = int(TITLE_FILL * SR)
    m.add(lp(osc(100, n, "saw"), 1600) * env(n, 0.05, 1.0, 0.08), TITLE_T0, -24, send=0.1)
    n = int(0.06 * SR)
    m.add(lp(osc(120, n, "saw"), 3000) * env(n, 0.002, 0.04), CATCH_T, -14, send=0.1)
    # the confident robot two-tone "ba-DEEP" + a weighty thump
    for f0, t0, d in ((880, BEEP_T, 0.15), (1318.5, BEEP_T + 0.18, 0.32)):
        n = int(d * SR)
        tone = 0.65 * osc(f0, n, "square") + 0.35 * osc(f0 / 2, n)
        ring = 0.75 + 0.25 * np.sin(2 * np.pi * 33 * np.arange(n) / SR)
        m.add(lp(tone * ring, 7000) * env(n, 0.004, 3.0, 0.04), t0, -8, send=0.3)
    n = int(0.4 * SR)
    m.add(osc(72 * (0.62 ** (np.arange(n) / n)), n) * env(n, 0.003, 0.12), BEEP_T, -5, send=0.05)
    # subtitle: softer keys
    for ch, tc in zip(SUB, TS):
        if ch != " ":
            m.add(bleep(r.choice(scale) * 1.5, 0.025, "tri", 0.01), tc, -20, pan=r.uniform(-0.3, 0.3))
    # switch-off: falling "tiuuu" and a click
    n = int(0.42 * SR)
    fall = 1900 * (60 / 1900) ** (np.arange(n) / n)
    m.add(osc(fall, n, "tri") * env(n, 0.004, 0.25, 0.05), OFF_T0, -11, send=0.35)
    n = int(0.008 * SR)
    m.add(hp(r.normal(0, 1, n), 2000) * env(n, 0.0005, 0.002), OFF_T0, -10)
    out = m.render()
    out /= np.max(np.abs(out))
    out = np.tanh(1.6 * out) / np.tanh(1.6)
    out *= 10 ** (-1.5 / 20) / np.max(np.abs(out))
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
            Image.fromarray(render_frame(f)).save(os.path.join(d, f"C_{f:03d}.png"))
        return
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "title_computer.wav")
        write_wav(wav, build_audio())
        encode(out, wav)
    print("wrote", out)


if __name__ == "__main__":
    main()
