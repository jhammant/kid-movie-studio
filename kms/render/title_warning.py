"""Opening title: red alert! A siren wails, then the title SLAMS in.

Hazard stripes scroll top and bottom, a flashing WARNING sign and a pulsing red alarm
light lead up to a chrome title that crashes onto the screen with a camera shake,
shockwave and sparks. Pictures are drawn with PIL/numpy and the sound is synthesised
with numpy.

    python -m kms.render.title_warning OUT.mp4 [--title T] [--villain V] [--fps 25]
                                               [--frames N] [--seed S] [--stills DIR]
"""
import argparse
import itertools
import multiprocessing as mp
import subprocess
import tempfile
import unicodedata
import wave
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.signal import butter, fftconvolve, sosfilt

from kms.fonts import font
from kms.media import audio_args, ffmpeg, video_args
from kms.render import SR, H, W, Ctx, split_title

DUR = 5.0
IMPACT = "impact"
DIN = "din-condensed-bold"

DEFAULT_TITLE = "ROBOTS REVENGE"
DEFAULT_VILLAIN = "ROBOT"
DEFAULT_WARNING = "WARNING"
DEFAULT_SUBTITLE = "TONIGHT ON THE 9 O'CLOCK NEWS"

# ---------------------------------------------------------------- timeline (s)
SIREN_T0 = 0.08
SIREN_P = 0.6          # one "wee-woo"
SLAM_T = 2.40          # the title lands (a frame boundary at 25 and 30 fps)
FLY = 0.20             # how long the title takes to fall in
SUB_T = 2.95           # "tonight on the 9 o'clock news"
SHINE_T = 3.45
YELLOW = (255, 208, 0)
INK = (15, 8, 5, 255)

# ---------------------------------------------------------------- layout (px)
SAFE_W = 1728                       # 5% title-safe width
TITLE_PX, TITLE_PX_ONE = 300, 400   # chrome title size (a single short line may go bigger)
TITLE_CY = 541.5                    # centre between the WARNING tag and the subtitle
TITLE_PITCH = 273                   # line spacing at 300 px
TITLE_MAX_W = 1560                  # widest line, leaving room for outline, extrusion and glow
TITLE_TOP, TITLE_BOTTOM = 280, 806  # glyph boxes stay clear of the tag and the subtitle
WARN_CY, ALERT_CY, PILL_CY, SUB_CY = 455, 705, 196, 893
BAND = 120
P = 150  # stripe period (px, along x)


_NO_GLYPH = "\U000F0000"  # a private-use character no font draws: it shows the "missing glyph" box


@lru_cache(maxsize=None)
def _ink(role, ch):
    img = Image.new("L", (96, 96), 0)
    ImageDraw.Draw(img).text((16, 8), ch, font=font(role, 48), fill=255)
    return img.tobytes()


def has_glyph(role, ch):
    """True if the role's font really draws ch, rather than its missing-glyph box."""
    return ch.isspace() or _ink(role, ch) != _ink(role, _NO_GLYPH)


def _clean(text, role=DIN):
    """Upper-case one line of text for a font, collapsing whitespace.

    Accented letters the font lacks fall back to their base letter; anything else it can't
    draw (emoji, symbols, control characters) is dropped, so no tofu boxes reach the screen.
    """
    out = []
    for ch in unicodedata.normalize("NFC", str(text)).upper():
        if ch.isspace() or unicodedata.category(ch)[0] == "C":
            out.append(" ")
        elif unicodedata.category(ch)[0] == "M":
            continue
        elif has_glyph(role, ch):
            out.append(ch)
        else:
            out.extend(b for b in unicodedata.normalize("NFKD", ch)
                       if not unicodedata.combining(b) and has_glyph(role, b))
    return " ".join("".join(out).split())


def default_alert(villain=DEFAULT_VILLAIN):
    v = _clean(villain) or DEFAULT_VILLAIN
    return f"{v} INVASION DETECTED"


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
@lru_cache(maxsize=1)
def _static():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W / 2, H / 2
    rn = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    return {
        "xx": xx, "yy": yy,
        "edge": np.clip(rn / 1.25, 0, 1) ** 1.8,                  # red glow lives at the edges
        "vignette": np.clip(1 - 0.33 * rn ** 2.2, 0, 1),
        "theta": np.arctan2(yy - (cy - 40), xx - cx),
        "radial": np.clip(np.hypot(xx - cx, yy - cy + 40) / 760, 0, 1) ** 0.8,
        "stripes": _stripe_tile(),
    }


def _stripe_tile():
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


def band_frame(offset):
    o = int(offset) % P
    return _static()["stripes"][:, o:o + W]


@lru_cache(maxsize=8)
def warn_triangle(size, bang=True):
    """A yellow warning sign drawn by hand (the fonts have no usable ⚠)."""
    ss = 4
    S = size * ss
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = 10 * ss
    pts = [(S / 2, pad), (S - pad, S - pad * 1.4), (pad, S - pad * 1.4)]
    d.polygon(pts, fill=INK)
    d.line(pts + [pts[0]], fill=INK, width=int(18 * ss), joint="curve")
    inner = [(S / 2, pad + 26 * ss), (S - pad - 20 * ss, S - pad * 1.4 - 9 * ss), (pad + 20 * ss, S - pad * 1.4 - 9 * ss)]
    d.polygon(inner, fill=YELLOW + (255,))
    d.line(inner + [inner[0]], fill=YELLOW + (255,), width=int(10 * ss), joint="curve")
    if bang:
        d.text((S / 2, S * 0.64), "!", font=font(DIN, int(S * 0.52)), fill=INK, anchor="mm")
    return img.resize((size, size), Image.LANCZOS)


def _spaced_width(text, f, spacing):
    return sum(f.getlength(c) for c in text) + spacing * max(0, len(text) - 1)


def text_layer(text, f, fill, stroke, stroke_fill, center, spacing=0, img=None):
    if img is None:
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if spacing:
        widths = [f.getlength(c) for c in text]
        total = sum(widths) + spacing * (len(text) - 1)
        x = center[0] - total / 2
        for c, w in zip(text, widths):
            d.text((x, center[1]), c, font=f, fill=fill, anchor="lm", stroke_width=stroke, stroke_fill=stroke_fill)
            x += w + spacing
    else:
        d.text(center, text, font=f, fill=fill, anchor="mm", stroke_width=stroke, stroke_fill=stroke_fill)
    return img


def spaced_lines(text, px, spacing, stroke, cy, max_w=SAFE_W - 40, min_px=None, max_lines=2):
    """A letter-spaced white caption, shrunk (or broken over two lines) to fit max_w."""
    def fit(lines, top_px):
        widest = max(_spaced_width(s, font(DIN, top_px), spacing * top_px / px) + 2 * stroke for s in lines)
        k = min(1.0, max_w / max(widest, 1))
        return int(top_px * k)

    lines = [text]
    size = fit(lines, px)
    if min_px and size < min_px and len(text.split()) > 1 and max_lines > 1:
        lines = split_title(text, max_lines)
        size = fit(lines, int(px * 0.85))
    size = max(size, 12)
    k = size / px
    f = font(DIN, size)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    pitch = size * 1.02
    for i, s in enumerate(lines):
        y = cy + (i - (len(lines) - 1) / 2) * pitch
        text_layer(s, f, (255, 255, 255, 255), max(1, int(round(stroke * k))), INK, (W / 2, y),
                   spacing=spacing * k if s else 0, img=img)
    return img


def to_premul(img):
    a = np.asarray(img, np.float32) / 255
    return a[..., :3] * a[..., 3:4], a[..., 3]


# ---------------------------------------------------------------- title layout
def _splits(words, max_lines=3):
    n = len(words)
    for k in range(1, min(max_lines, n) + 1):
        for cuts in itertools.combinations(range(1, n), k - 1):
            bounds = (0, *cuts, n)
            yield [" ".join(words[a:b]) for a, b in zip(bounds, bounds[1:])]


def _title_fit(lines, size):
    """(fits, width, top, bottom) of the title set at `size` px around TITLE_CY."""
    f = font(IMPACT, size)
    d = ImageDraw.Draw(Image.new("L", (1, 1)))
    top, bottom, width = H, 0, 0
    for i, s in enumerate(lines):
        cy = TITLE_CY + (i - (len(lines) - 1) / 2) * TITLE_PITCH * size / 300
        b = d.textbbox((W / 2, cy), s, font=f, anchor="mm")
        top, bottom, width = min(top, b[1]), max(bottom, b[3]), max(width, b[2] - b[0])
    return width <= TITLE_MAX_W and top >= TITLE_TOP and bottom <= TITLE_BOTTOM, width, top, bottom


def layout_title(title, max_lines=3):
    """(lines, px): the 1-3 line break that sets the chrome title biggest."""
    words = _clean(title, IMPACT).split()
    if not words:
        return [], TITLE_PX
    f100 = font(IMPACT, 100)
    options = []
    for lines in _splits(words, max_lines):
        cap = TITLE_PX_ONE if len(lines) == 1 else TITLE_PX
        _, width, top, bottom = _title_fit(lines, 100)
        size = min(cap, 100 * TITLE_MAX_W / max(width, 1),
                   100 * (TITLE_CY - TITLE_TOP) / max(TITLE_CY - top, 1),
                   100 * (TITLE_BOTTOM - TITLE_CY) / max(bottom - TITLE_CY, 1))
        widths = [f100.getlength(s) for s in lines]
        options.append((size, len(lines), max(widths) - min(widths), lines))
    # nearly as big counts as a tie: then fewer lines, then the most even line lengths
    biggest = max(o[0] for o in options)
    size, _, _, lines = min((o for o in options if o[0] >= biggest * 0.95), key=lambda o: (o[1], o[2], -o[0]))
    size = int(size)
    while size > 20 and not _title_fit(lines, size)[0]:
        size -= 2
    return lines, size


# ---------------------------------------------------------------- the scene
class Scene:
    """Everything about one render that doesn't change frame to frame."""

    def __init__(self, title=DEFAULT_TITLE, villain=DEFAULT_VILLAIN, alert=None, warning=DEFAULT_WARNING,
                 subtitle=DEFAULT_SUBTITLE, fps=25, seed=0):
        self.fps, self.seed = fps, seed
        self.off = 100003 * seed  # seed 0 keeps the original look exactly
        self.warning = _clean(warning) or DEFAULT_WARNING
        self.alert = _clean(default_alert(villain) if alert is None else alert)
        self.subtitle = _clean(subtitle)
        self.title_lines, self.title_px = layout_title(title)
        self.WARN, self.WARN_GLOW, self.WARN_SUB = self.build_warning_group()
        self.T_RGB, self.T_A, self.T_GLOW, self.T_FILL, self.T_BOX = self.build_title()
        self.T_CX, self.T_CY = (self.T_BOX[0] + self.T_BOX[2]) / 2, (self.T_BOX[1] + self.T_BOX[3]) / 2
        self.PILL = self.build_pill()
        self.SUBT = to_premul(spaced_lines(self.subtitle, 80, 6, 5, SUB_CY, max_lines=1))
        self.sparks = self.build_sparks()

    # -- layers
    def build_warning_group(self):
        """Big flashing WARNING with signs either side (pre-slam), and the alert line under it."""
        word = self.warning
        tw = font(DIN, 270).getlength(word)
        k = min(1.0, (SAFE_W - 40) / (tw + 2 * (230 + 56)))
        f = font(DIN, int(270 * k)) if k < 1 else font(DIN, 270)
        tw = f.getlength(word)
        ts, gap = int(230 * k), 56 * k
        tri = warn_triangle(ts)
        total = tw + 2 * (ts + gap)
        x0 = W / 2 - total / 2
        cy = WARN_CY
        ty = int(cy - 10 * k - ts / 2) if k < 1 else int(cy - 125)
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.text((W / 2, cy + 14 * k), word, font=f, fill=YELLOW + (255,), anchor="mm",
                stroke_width=int(round(26 * k)), stroke_fill=YELLOW + (255,))
        glow.alpha_composite(tri, (int(x0), ty))
        glow.alpha_composite(tri, (int(x0 + total - ts), ty))
        glow = glow.filter(ImageFilter.GaussianBlur(28))
        d = ImageDraw.Draw(img)
        d.text((W / 2, cy + 14 * k), word, font=f, fill=YELLOW + (255,), anchor="mm",
               stroke_width=max(1, int(round(9 * k))), stroke_fill=INK)
        img.alpha_composite(tri, (int(x0), ty))
        img.alpha_composite(tri, (int(x0 + total - ts), ty))
        sub = spaced_lines(self.alert, 104, 10, 6, ALERT_CY, min_px=70)
        g = np.asarray(glow, np.float32)
        return to_premul(img), g[..., :3] / 255 * (g[..., 3:4] / 255), to_premul(sub)

    def build_title(self):
        """Chrome title lines with black outline, red 3D extrusion and red glow."""
        lines, px = self.title_lines, self.title_px
        k = px / 300
        f = font(IMPACT, px)
        stroke, depth = max(1, int(round(13 * k))), max(1, int(round(16 * k)))
        fill_m = Image.new("L", (W, H), 0)
        stroke_m = Image.new("L", (W, H), 0)
        df, ds = ImageDraw.Draw(fill_m), ImageDraw.Draw(stroke_m)
        boxes = []
        for i, word in enumerate(lines):
            cy = TITLE_CY + (i - (len(lines) - 1) / 2) * TITLE_PITCH * px / 300
            df.text((W / 2, cy), word, font=f, fill=255, anchor="mm")
            ds.text((W / 2, cy), word, font=f, fill=255, anchor="mm", stroke_width=stroke, stroke_fill=255)
            boxes.append(df.textbbox((W / 2, cy), word, font=f, anchor="mm"))
        M = np.asarray(fill_m, np.float32) / 255
        S = np.asarray(stroke_m, np.float32) / 255
        # chrome gradient per line: bright sky, hard horizon, darker ground, bright lip
        grad = np.zeros((H, 3), np.float32)
        stops = [0.0, 0.48, 0.5, 0.82, 1.0]
        cols = np.array([[1.00, 1.00, 1.00], [0.80, 0.83, 0.90], [0.42, 0.44, 0.52], [0.66, 0.68, 0.76], [0.95, 0.96, 1.00]])
        for (x0, y0, x1, y1) in boxes:
            y0, y1 = max(0, int(y0)), min(H, int(y1))
            ys = np.arange(y0, y1)
            u = (ys - y0) / max(1, (y1 - y0))
            for c in range(3):
                grad[y0:y1, c] = np.interp(u, stops, cols[:, c])
        fill_rgb = np.broadcast_to(grad[:, None, :], (H, W, 3))
        # extrusion: stack the outline down-right
        ext = np.zeros_like(S)
        for j in range(1, depth + 1):
            ext = np.maximum(ext, np.roll(np.roll(S, j, axis=0), j, axis=1) * (1 - 0.02 * j * 16 / depth))
        ext_rgb = np.array([0.45, 0.02, 0.02], np.float32)
        # layer: extrusion under black outline under chrome
        rgb = ext[..., None] * ext_rgb * (1 - S[..., None])
        a = np.maximum(ext, S)
        rgb = rgb * (1 - S[..., None]) + S[..., None] * np.array([0.03, 0.0, 0.0], np.float32)
        rgb = rgb * (1 - M[..., None]) + M[..., None] * fill_rgb
        glow = cv2.GaussianBlur(np.maximum(S, ext), (0, 0), 30)
        glow_rgb = glow[..., None] * np.array([1.0, 0.08, 0.04], np.float32)
        if boxes:
            box = (min(b[0] for b in boxes), boxes[0][1], max(b[2] for b in boxes), boxes[-1][3])
        else:
            box = (W / 2 - 1, TITLE_CY - 1, W / 2 + 1, TITLE_CY + 1)
        return rgb.astype(np.float32), a.astype(np.float32), glow_rgb.astype(np.float32), M, box

    def build_pill(self):
        """Small flashing ⚠ WARNING ⚠ tag that sits above the title after the slam."""
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        label = self.warning
        px, sp = 78, 14
        tw = _spaced_width(label, font(DIN, px), sp)
        k = min(1.0, (1400 - 2 * (84 + 26 + 34)) / max(tw, 1))
        if k < 1:
            px, sp = int(px * k), sp * k
        f = font(DIN, px)
        tw = _spaced_width(label, f, sp)
        tri = warn_triangle(84)
        inner = tw + 2 * (84 + 26)
        x0, x1 = W / 2 - inner / 2 - 34, W / 2 + inner / 2 + 34
        cy = PILL_CY
        d.rounded_rectangle([x0, cy - 50, x1, cy + 50], radius=50, fill=(18, 4, 4, 235), outline=YELLOW + (255,), width=6)
        x = W / 2 - tw / 2
        for c in label:
            d.text((x, cy + 4), c, font=f, fill=YELLOW + (255,), anchor="lm")
            x += f.getlength(c) + sp
        img.alpha_composite(tri, (int(W / 2 - inner / 2), int(cy - 46)))
        img.alpha_composite(tri, (int(W / 2 + inner / 2 - 84), int(cy - 46)))
        return to_premul(img)

    def build_sparks(self):
        rng = np.random.default_rng(11 + self.off)
        box, sparks = self.T_BOX, []
        for _ in range(70):
            side = rng.random()
            x = rng.uniform(box[0] - 40, box[2] + 40)
            y = box[3] - rng.uniform(0, 60) if side < 0.6 else rng.uniform(box[1], box[3])
            ang = rng.uniform(-np.pi * 0.95, -np.pi * 0.05) if side < 0.6 else rng.uniform(-np.pi, np.pi)
            sp = rng.uniform(500, 1700)
            sparks.append((x, y, np.cos(ang) * sp, np.sin(ang) * sp, rng.uniform(0.35, 0.9), rng.random() < 0.3))
        return sparks

    # -- per frame
    def scaled(self, layer, s, cx=None, cy=None):
        """Scale a full-frame layer about (cx, cy)."""
        if abs(s - 1) < 1e-4:
            return layer
        cx = self.T_CX if cx is None else cx
        cy = self.T_CY if cy is None else cy
        M = np.float32([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]])
        return cv2.warpAffine(layer, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def draw_sparks(self, frame_rgb, dt):
        if dt < 0 or dt > 1.0:
            return frame_rgb
        layer = np.zeros((H // 2, W // 2, 3), np.float32)
        chunks = []
        for (x, y, vx, vy, life, chunk) in self.sparks:
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

    def background(self, t):
        st = _static()
        p = alarm_pulse(t)
        after = t >= SLAM_T
        amp = 0.75 if not after else 0.45
        base = np.array([0.055, 0.006, 0.010], np.float32)
        red = np.array([0.95, 0.06, 0.04], np.float32)
        k = (0.30 + amp * p) * st["edge"]
        img = base + k[..., None] * red
        # rotating beacon beams
        phi = t * 2 * np.pi * 0.55
        beams = np.zeros((H, W), np.float32)
        for off in (0.0, np.pi):
            d = np.angle(np.exp(1j * (st["theta"] - phi - off)))
            beams += np.exp(-(d / 0.15) ** 2)
        img += (beams * st["radial"] * (0.13 + 0.12 * p))[..., None] * np.array([1.0, 0.42, 0.18], np.float32)
        return img

    def compose(self, t, frame):
        st = _static()
        xx, yy = st["xx"], st["yy"]
        img = self.background(t)
        p = alarm_pulse(t)
        dt = t - SLAM_T
        WARN, WARN_GLOW, WARN_SUB = self.WARN, self.WARN_GLOW, self.WARN_SUB
        T_RGB, T_A, T_BOX = self.T_RGB, self.T_A, self.T_BOX
        # --- pre-slam: flashing WARNING
        if t < SLAM_T:
            on = ((t - SIREN_T0) % (SIREN_P / 2)) < 0.21 and t >= SIREN_T0
            intro = ease_out((t - 0.12) / 0.25)
            k = 1.0
            if t > SLAM_T - FLY:
                k = 1 - (t - (SLAM_T - FLY)) / FLY
            if on and intro > 0:
                s = 0.6 + 0.4 * intro + 0.25 * (1 - k)
                rgb, a = self.scaled(WARN[0], s, W / 2, 470), self.scaled(WARN[1], s, W / 2, 470)
                img += self.scaled(WARN_GLOW, s, W / 2, 470) * 0.55 * k
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
                    u = 1 + (dt + i * (1 / self.fps) / subs) / FLY       # 0..1 over the fall
                    u = np.clip(u, 0, 1)
                    s = 1 + 5.0 * (1 - u) ** 2.2
                    acc_rgb += self.scaled(T_RGB, s)
                    acc_a += self.scaled(T_A, s)
                alpha = np.clip((dt + FLY) / (FLY * 0.5), 0, 1)
                img = over(img, acc_rgb / subs, acc_a / subs, alpha)
            else:
                bounce = 1 - 0.075 * np.exp(-dt / 0.09) * np.cos(2 * np.pi * dt / 0.24)
                glow_k = 0.55 + 0.35 * p + 1.6 * np.exp(-dt / 0.15)
                img += self.scaled(self.T_GLOW, bounce) * glow_k
                rgb = self.scaled(T_RGB, bounce)
                a = self.scaled(T_A, bounce)
                # chrome shine sweeping across the letters
                if SHINE_T <= t < SHINE_T + 0.6:
                    pos = T_BOX[0] - 300 + (T_BOX[2] - T_BOX[0] + 600) * (t - SHINE_T) / 0.6
                    band = np.exp(-(((xx + (yy - self.T_CY) * 0.45) - pos) / 60) ** 2)
                    rgb = rgb + (band * self.scaled(self.T_FILL, bounce))[..., None] * 0.55
                img = over(img, rgb, a)
                # shockwave ring
                if dt < 0.6:
                    ring = np.zeros((H // 2, W // 2), np.float32)
                    r = 90 + 1500 * dt ** 0.7
                    cv2.ellipse(ring, (int(self.T_CX / 2), int(self.T_CY / 2)), (int(r * 1.35), int(r)), 0, 0, 360,
                                1.0, max(2, int(26 * (1 - dt / 0.6))), cv2.LINE_AA)
                    ring = cv2.GaussianBlur(ring, (0, 0), 3)
                    ring = cv2.resize(ring, (W, H)) * (1 - dt / 0.6) ** 1.5
                    img += ring[..., None] * np.array([1.0, 0.85, 0.6], np.float32) * 0.8
                img = self.draw_sparks(img, dt)
                # after the slam: WARNING tag + subtitle
                if dt > 0.25:
                    on = ((t - SIREN_T0) % (SIREN_P / 2)) < 0.21
                    pa = min(1, (dt - 0.25) / 0.15) * (1.0 if on else 0.35)
                    img = over(img, self.PILL[0], self.PILL[1], pa)
                if t > SUB_T:
                    u = ease_out((t - SUB_T) / 0.35)
                    sh = int(40 * (1 - u))
                    rgb_s = np.roll(self.SUBT[0], sh, axis=0)
                    a_s = np.roll(self.SUBT[1], sh, axis=0)
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
        img *= st["vignette"][..., None]
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
        grain = np.random.default_rng(frame + self.off).normal(0, 0.012, (H // 2, W // 2)).astype(np.float32)
        img += cv2.resize(grain, (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
        return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)

    def frame(self, frame):
        return self.compose(frame / self.fps, frame)


def over(dst, rgb, a, alpha=1.0):
    """Premultiplied-alpha over."""
    a = a[..., None] * alpha
    return dst * (1 - a) + rgb * alpha


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
    def __init__(self, seed_off=0):
        self.dry = np.zeros((2, N))
        self.wet = np.zeros((2, N))
        self.off = seed_off

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
        r = np.random.default_rng(5 + self.off)
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


def build_audio(seed=0):
    """The whole soundtrack, (2, N) floats peaking at -2.5 dBFS. Nothing in it depends on the words."""
    off = 100003 * seed
    r = np.random.default_rng(8 + off)
    m = Mix(off)
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
    for _ in range(9):
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


# ---------------------------------------------------------------- rendering
_SCENE = None


def _init_worker(params):
    global _SCENE
    cv2.setNumThreads(1)
    _SCENE = Scene(**params)


def _render_worker(i):
    return _SCENE.frame(i)


def _frames(params, n, workers):
    """Yield frames 0..n-1 in order, in parallel when it's worth it."""
    if workers > 1 and n >= 4 * workers:
        with mp.get_context("spawn").Pool(workers, _init_worker, (params,)) as pool:
            yield from pool.imap(_render_worker, range(n), chunksize=2)
    else:
        scene = Scene(**params)
        for i in range(n):
            yield scene.frame(i)


def _write_wav(path, stereo):
    """16-bit stereo wav from (2, n) floats, quantised in float64 exactly as the tuned original."""
    pcm = (np.clip(stereo.T, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def _encode(out, frames, wav, fps, seconds):
    cmd = [ffmpeg(), "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-i", str(wav), "-map", "0:v", "-map", "1:a",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
           *video_args(18),
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           *audio_args(), "-movflags", "+faststart", "-t", f"{seconds:.6f}", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for img in frames:
            p.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())
    except BaseException:
        p.kill()
        p.wait()
        raise
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {out}")


def _params(title, villain, alert, warning, subtitle, ctx):
    return dict(title=title, villain=villain, alert=alert, warning=warning, subtitle=subtitle,
                fps=ctx.fps, seed=ctx.seed)


def render(out, *, title=DEFAULT_TITLE, villain=DEFAULT_VILLAIN, alert=None, warning=DEFAULT_WARNING,
           subtitle=DEFAULT_SUBTITLE, ctx=None) -> Path:
    """Render the 5-second red-alert title to `out` (mp4) and return its path.

    title     the film's name, slammed in as chrome letters (1-3 lines, auto-sized)
    villain   builds the default alert line: "<VILLAIN> INVASION DETECTED"
    alert     the line under the flashing sign, replacing the villain default
    warning   the flashing word (and the tag above the title)
    subtitle  rises in under the title after the slam
    """
    ctx = ctx or Ctx()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    params = _params(title, villain, alert, warning, subtitle, ctx)
    n = ctx.nframes(DUR)
    seconds = n / ctx.fps
    audio = build_audio(ctx.seed)[:, :int(round(seconds * SR))]
    with tempfile.TemporaryDirectory(dir=ctx.scratch("title_warning")) as tmp:
        wav = Path(tmp) / "title_warning.wav"
        _write_wav(wav, audio)
        _encode(out, _frames(params, n, ctx.workers), wav, ctx.fps, seconds)
    return out


STILL_TIMES = (0.5, 1.5, 2.3, 2.44, 2.7, 3.2, 3.6, 4.5)


def stills(out_dir, times=STILL_TIMES, *, title=DEFAULT_TITLE, villain=DEFAULT_VILLAIN, alert=None,
           warning=DEFAULT_WARNING, subtitle=DEFAULT_SUBTITLE, ctx=None):
    """PNG frames at a few key moments, for checking a look without encoding the video."""
    ctx = ctx or Ctx()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = Scene(**_params(title, villain, alert, warning, subtitle, ctx))
    paths = []
    for t in times:
        f = min(int(round(t * ctx.fps)), int(round(DUR * ctx.fps)) - 1)
        p = out_dir / f"D_{f:03d}.png"
        Image.fromarray(scene.frame(f)).save(p)
        paths.append(p)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kms.render.title_warning",
                                 description="Opening title: red alert, a siren, then the title slams in.")
    ap.add_argument("out", help="output .mp4")
    ap.add_argument("--title", default=DEFAULT_TITLE)
    ap.add_argument("--villain", default=DEFAULT_VILLAIN)
    ap.add_argument("--alert", default=None, help='default: "<VILLAIN> INVASION DETECTED"')
    ap.add_argument("--warning", default=DEFAULT_WARNING)
    ap.add_argument("--subtitle", default=DEFAULT_SUBTITLE)
    ap.add_argument("--fps", type=int, default=25, choices=(25, 30))
    ap.add_argument("--frames", type=int, default=None, help="render only the first N frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stills", metavar="DIR", help="write PNG stills of key moments instead of the video")
    a = ap.parse_args(argv)
    ctx = Ctx(fps=a.fps, seed=a.seed, limit_frames=a.frames)
    kw = dict(title=a.title, villain=a.villain, alert=a.alert, warning=a.warning, subtitle=a.subtitle, ctx=ctx)
    if a.stills:
        for p in stills(a.stills, **kw):
            print("wrote", p)
        return
    print("wrote", render(a.out, **kw))


if __name__ == "__main__":
    main()
