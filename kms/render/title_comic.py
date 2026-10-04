"""Comic Book: a pop-art title card (4.4 s).

Sky-blue sunburst with halftone dots, chunky letters with a thick black outline and a block
shadow that pop in one by one with overshoot (lines alternate yellow / red / yellow), a
"KA-POW!" starburst that slams in with a screen shake, a squash-and-stretch bounce and a
cartoon iris-out. Audio: a whoosh, a rising "bloop" per letter, a punchy POW, a cartoon
BOING and a slide whistle for the iris-out.

Every frame is drawn with PIL/numpy/OpenCV and all audio is synthesised with numpy, then
muxed with ffmpeg.

    python -m kms.render.title_comic OUT.mp4 --title "THE DINOSAUR DISCO" --pow "BOOM!"
    python -m kms.render.title_comic OUT.mp4 --stills DIR      # a few PNGs, no video
"""
import argparse
import itertools
import math
import subprocess
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import signal

from kms import media
from kms.fonts import font
from kms.render import H, SR, W, Ctx

DUR = 4.4
FONT = "impact"
POW_FONT = "arial-black"
STILLS = (0.8, 1.3, 2.3, 3.0, 4.0)

T_L1, STAG1, STAG2 = 0.16, 0.065, 0.060   # first letter, stagger on line 1 / later lines
LINE_GAP = 0.095              # pause between the last letter of a line and the next line
T_LAST = 0.94                 # the last letter starts popping by here
T_POW = 1.20                  # KA-POW slams in (frame 30 at 25 fps)
T_BOING = 2.20                # squash-and-stretch bounce
T_IRIS = 3.85                 # iris-out starts
T_IRIS_END = 4.30
SPR = 1.3                     # sprites are drawn 1.3x so the overshoot stays crisp
TILT = -6.0                   # whole title rises to the right (degrees)

SIZE = 282                    # letter size (px) the look was tuned at; titles only shrink from here
SOLO = 1.4                    # ...except a title on one line, which may grow up to this x SIZE
TRACK, STROKE, DEPTH, PITCH = 14, 15, 13, 300   # at SIZE: letter gap, outline, block shadow, line pitch
CENTER = (W / 2 + 85, H / 2 + 65)               # the title sits right and low, leaving room for the POW
POW_AT = (365, 235, 0.68)                       # KA-POW centre and scale
# Other placements tried, in order of preference, when a big title would crowd the POW.
PLACEMENTS = (((CENTER[0], CENTER[1]), POW_AT, 1.0),
              ((CENTER[0], CENTER[1]), (330, 210, 0.58), 0.95),
              ((CENTER[0], CENTER[1] - 25), (300, 190, 0.50), 0.90))
SAFE_X, SAFE_Y = 96, 54       # 5% title-safe margins

YELLOW = ((255, 240, 90), (255, 160, 20))
RED = ((255, 92, 70), (200, 16, 36))
LINE_COLS = (YELLOW, RED)
INK = (0, 0, 0)
SHADOW = (24, 22, 72)
PENTATONIC = (261.6, 293.7, 329.6, 392.0, 440.0, 523.3, 587.3, 659.3, 784.0, 880.0)


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, dtype=np.float32) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- sprites
def halftone(h, w, spacing, radius_fn):
    """Anti-aliased 45-degree halftone dots; radius_fn(y) -> dot radius (px)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx + yy) / math.sqrt(2)
    v = (xx - yy) / math.sqrt(2)
    du = (u % spacing) - spacing / 2
    dv = (v % spacing) - spacing / 2
    d = np.sqrt(du * du + dv * dv)
    return np.clip(radius_fn(yy) - d + 0.5, 0, 1)


def cap_like(box, cap):
    """True for a glyph that fills the capital height (R, O, S...), not an accent, comma or quote."""
    x0, y0, x1, y1 = box
    return y0 >= -1.08 * cap and y1 <= 0.08 * cap and (y1 - y0) >= 0.85 * cap


def letter_sprite(ch, fnt, cols, stroke, depth):
    """RGBA sprite of one chunky comic letter + its anchor (centre of the capitals)."""
    x0, y0, x1, y1 = fnt.getbbox(ch, anchor="ls")
    cap = -fnt.getbbox("H", anchor="ls")[1]
    pad = stroke + depth + 40
    top_y = min(y0, -cap)
    w, h = int(max(x1 - x0, 1) + 2 * pad), int(max(y1, 0) - top_y + 2 * pad)
    ox, oy = pad - x0, pad - top_y                 # baseline-left origin
    if cap_like((x0, y0, x1, y1), cap):            # (the look was tuned on these)
        cap_top, base, anchor_y = oy + y0, oy + y1, oy + (y0 + y1) / 2
    else:                                          # accents, quotes, commas: sit on the line
        cap_top, base, anchor_y = oy - cap, oy, oy - cap / 2

    def mask(dx=0.0, dy=0.0, sw=0):
        m = Image.new("L", (w, h), 0)
        ImageDraw.Draw(m).text((ox + dx, oy + dy), ch, font=fnt, fill=255, anchor="ls",
                               stroke_width=sw, stroke_fill=255)
        return np.asarray(m, np.float32) / 255

    outline = mask(sw=stroke)
    block = np.zeros_like(outline)
    for i in range(1, depth + 1):
        block = np.maximum(block, mask(0.7 * i, i, stroke))
    fill = mask()

    # Soft drop shadow on the background.
    sh = cv2.GaussianBlur(np.maximum(outline, block), (0, 0), 7)
    M = np.float32([[1, 0, 10], [0, 1, 14]])
    sh = cv2.warpAffine(sh, M, (w, h)) * 0.38

    yy = np.arange(h, dtype=np.float32)[:, None]
    g = np.clip((yy - cap_top) / max(1, base - cap_top), 0, 1)
    top, bot = np.array(cols[0], np.float32), np.array(cols[1], np.float32)
    face = top + (bot - top) * g[..., None]
    face = np.broadcast_to(face, (h, w, 3)).copy()
    # Comic print shading: dots growing toward the bottom of the letter.
    dots = halftone(h, w, 11, lambda y: 4.6 * smoothstep(cap_top + 0.45 * (base - cap_top), base, y))
    face = face * (1 - 0.32 * dots[..., None])
    # Glossy rim on the upper-left inner edge.
    inner = mask(5, 6)
    rim = np.clip(fill - inner, 0, 1) * (1 - g * 0.7)
    face = face + (255 - face) * (0.85 * rim)[..., None]

    rgb = np.zeros((h, w, 3), np.float32)
    a = sh.copy()
    for layer_a, col in ((block, np.array(SHADOW, np.float32)),
                         (outline, np.array(INK, np.float32))):
        rgb = rgb * (1 - layer_a[..., None]) + col * layer_a[..., None]
        a = a + layer_a * (1 - a)
    rgb = rgb * (1 - fill[..., None]) + face * fill[..., None]
    a = a + fill * (1 - a)
    sprite = np.dstack([rgb / 255, a]).astype(np.float32)
    anchor = (ox + (x0 + x1) / 2, anchor_y)
    return premultiply(sprite), anchor


def premultiply(rgba):
    out = rgba.copy()
    out[..., :3] = np.clip(out[..., :3], 0, 1) * out[..., 3:4]
    return out


def starburst(rng, cx, cy, r_in, r_out, spikes, jitter):
    pts = []
    for i in range(spikes * 2):
        ang = 2 * math.pi * i / (spikes * 2) + rng.uniform(-0.06, 0.06)
        r = (r_out * rng.uniform(1 - jitter, 1 + jitter * 0.4)) if i % 2 == 0 else r_in * rng.uniform(0.92, 1.05)
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts


def kapow_sprite(word="KA-POW!", seed=0):
    """The POW starburst (980 x 680) with `word` in it, premultiplied RGBA + its centre."""
    rng = np.random.default_rng(5 + seed)
    S = 2  # supersample
    w, h = 980 * S, 680 * S
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy = w / 2, h / 2

    def squash(pts, sx, sy):
        return [(cx + (x - cx) * sx, cy + (y - cy) * sy) for x, y in pts]

    outer = squash(starburst(rng, cx, cy, 205 * S, 300 * S, 14, 0.16), 1.3, 0.95)
    d.polygon([(x + 16 * S, y + 20 * S) for x, y in outer], fill=SHADOW + (255,))
    d.polygon(outer, fill=(0, 0, 0, 255))
    d.polygon(squash(outer, 0.94, 0.92), fill=(235, 30, 45, 255))
    inner = squash(starburst(rng, cx, cy, 168 * S, 228 * S, 14, 0.12), 1.32, 0.92)
    d.polygon(inner, fill=(0, 0, 0, 255))
    d.polygon(squash(inner, 0.93, 0.9), fill=(255, 222, 40, 255))

    txt = word.strip()
    if txt:
        probe = font(POW_FONT, 100)
        valley_x = 168 * 1.32 * 0.93 * S
        size = 100 * 2.3 * valley_x / max(probe.getlength(txt), 1)
        # A short word would grow taller than the burst: cap it at 1.5x the KA-POW! letters.
        size = min(size, 1.5 * 100 * 2.3 * valley_x / probe.getlength("KA-POW!"))
        f = font(POW_FONT, int(size))
        tw = f.getlength(txt)
        cap = -f.getbbox("H", anchor="ls")[1]
        tx, ty = cx - tw / 2 - 4 * S, cy + cap / 2
        for i in range(9 * S, 0, -1):
            d.text((tx + i * 0.6, ty + i), txt, font=f, fill=(0, 0, 0, 255), anchor="ls",
                   stroke_width=8 * S, stroke_fill=(0, 0, 0, 255))
        d.text((tx, ty), txt, font=f, fill=(255, 255, 255, 255), anchor="ls",
               stroke_width=8 * S, stroke_fill=(0, 0, 0, 255))
    img = img.resize((w // S, h // S), Image.LANCZOS)
    arr = np.asarray(img, np.float32) / 255
    return premultiply(arr), (w / S / 2, h / S / 2)


# ---------------------------------------------------------------- affine helpers
def A_t(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], np.float64)


def A_r(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def A_s(sx, sy=None):
    return np.array([[sx, 0, 0], [0, sx if sy is None else sy, 0], [0, 0, 1]], np.float64)


def paste_affine(img, sprite, M):
    """Warp a premultiplied RGBA sprite with 3x3 M and composite over img (float RGB 0..255)."""
    h, w = sprite.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64) @ M.T
    x0 = max(0, int(math.floor(corners[:, 0].min())) - 2)
    x1 = min(W, int(math.ceil(corners[:, 0].max())) + 2)
    y0 = max(0, int(math.floor(corners[:, 1].min())) - 2)
    y1 = min(H, int(math.ceil(corners[:, 1].max())) + 2)
    if x1 <= x0 or y1 <= y0:
        return
    Mr = (A_t(-x0, -y0) @ M)[:2].astype(np.float32)
    scale = math.sqrt(abs(np.linalg.det(M[:2, :2])))
    src = sprite
    if scale < 0.7:  # pre-shrink to avoid aliasing on big downscales
        f = max(scale, 0.05)
        src = cv2.resize(sprite, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
        Mr = (Mr.astype(np.float64) @ np.array([[w / src.shape[1], 0, 0], [0, h / src.shape[0], 0], [0, 0, 1]]))[:2]
        Mr = Mr.astype(np.float32)
    roi = cv2.warpAffine(src, Mr, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a = roi[..., 3:4]
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + roi[..., :3] * 255


# ---------------------------------------------------------------- motion
def pop(a):
    """Spring 0 -> 1 with ~25% overshoot."""
    return 1 - math.exp(-a / 0.11) * math.cos(2 * math.pi * 3.2 * a)


def shake(t):
    a = t - T_POW
    if a < 0:
        return 0.0, 0.0, 0.0
    e = math.exp(-a / 0.13)
    return (22 * e * math.sin(2 * math.pi * 17 * a + 0.4),
            16 * e * math.cos(2 * math.pi * 21 * a),
            1.6 * e * math.sin(2 * math.pi * 13 * a))


def group_matrix(t, center, pivot_y):
    sx_, sy_, sr = shake(t)
    bob = 6 * math.sin(2 * math.pi * 0.55 * t)
    sx = sy = 1.0
    a = t - T_BOING
    if a >= 0:  # bounce on the bottom of the last line
        sq = math.exp(-a / 0.32) * math.cos(2 * math.pi * 3.6 * a)
        sy = 1 - 0.20 * sq
        sx = 1 + 0.12 * sq
    # breathe in slightly with the POW
    pa = t - T_POW
    punch = 1 + (0.06 * math.exp(-pa / 0.12) * math.cos(2 * math.pi * 4 * pa) if pa >= 0 else 0)
    return (A_t(center[0] + sx_, center[1] + sy_ + bob) @ A_r(TILT + sr) @ A_s(punch)
            @ A_t(0, pivot_y) @ A_s(sx, sy) @ A_t(0, -pivot_y))


def pow_matrix(t, at, anchor):
    a = t - T_POW
    s = 1 - math.exp(-a / 0.09) * math.cos(2 * math.pi * 3.0 * a)
    ba = t - T_BOING
    wob = 7 * math.exp(-ba / 0.35) * math.sin(2 * math.pi * 3.6 * ba) if ba >= 0 else 0
    rot = -12 + 38 * math.exp(-a / 0.1) * math.cos(2 * math.pi * 2.2 * a) + wob
    pulse = 1 + 0.025 * math.sin(2 * math.pi * 1.3 * t)
    sx_, sy_, _ = shake(t)
    M = (A_t(at[0] + sx_ * 0.6, at[1] + sy_ * 0.6) @ A_r(rot) @ A_s(at[2] * s * pulse)
         @ A_t(-anchor[0], -anchor[1]))
    return M, s


# ---------------------------------------------------------------- layout
@dataclass
class Letter:
    ch: str
    line: int
    x: float
    y: float
    rot: float
    spin: float
    t0: float
    box: tuple          # outline + block shadow around the anchor, screen px (x0, y0, x1, y1)


@dataclass
class Layout:
    lines: list
    size: float
    center: tuple
    pivot_y: float
    pow_at: tuple
    letters: list


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


def clean(text, role, uppercase=True):
    text = " ".join(str(text).split())
    return drawable(text.upper() if uppercase else text, role) if text else ""


def partitions(words, max_lines=3):
    """Every way to break words into 1..max_lines lines, keeping their order."""
    n = len(words)
    for k in range(1, min(max_lines, n) + 1):
        for cuts in itertools.combinations(range(1, n), k - 1):
            b = (0,) + cuts + (n,)
            yield [" ".join(words[i:j]) for i, j in zip(b, b[1:])]


def letter_times(lines):
    """When each visible letter starts popping: line by line, squeezed (or stretched) to land by T_LAST."""
    starts, t = [], T_L1
    for li, line in enumerate(lines):
        n = sum(1 for c in line if not c.isspace())
        st = STAG1 if li == 0 else STAG2
        starts.append([t + i * st for i in range(n)])
        t += max(n - 1, 0) * st + LINE_GAP
    last = max((s[-1] for s in starts if s), default=T_L1)
    k = min(1.8, (T_LAST - T_L1) / (last - T_L1)) if last > T_L1 else 1.0
    if abs(k - 1) > 1e-9:
        starts = [[T_L1 + (s - T_L1) * k for s in row] for row in starts]
    return starts


def place_letters(lines, size, seed=0):
    """Letters of each line laid out around the group centre at a letter size (screen px)."""
    fnt = font(FONT, int(size * SPR))
    cap = -fnt.getbbox("H", anchor="ls")[1]
    k = size / SIZE
    track, stroke, depth, pitch = TRACK * k, STROKE * SPR * k, DEPTH * SPR * k, PITCH * k
    rng = np.random.default_rng(21 + seed)
    times = letter_times(lines)
    letters = []
    for li, line in enumerate(lines):
        ly = (li - (len(lines) - 1) / 2) * pitch
        advs = [fnt.getlength(c) / SPR + track for c in line]
        x = -(sum(advs) - track) / 2
        for i, c in enumerate(line):
            if not c.isspace():
                x0, y0, x1, y1 = fnt.getbbox(c, anchor="ls")
                cy = (y0 + y1) / 2 if cap_like((x0, y0, x1, y1), cap) else -cap / 2
                cx = (x0 + x1) / 2
                box = ((x0 - cx - stroke) / SPR, (y0 - cy - stroke) / SPR,
                       (x1 - cx + stroke + 0.7 * depth) / SPR, (y1 - cy + stroke + depth) / SPR)
                j = sum(1 for L in letters if L.line == li)   # visible letters so far on this line
                letters.append(Letter(c, li, x + fnt.getlength(c) / SPR / 2, ly + rng.uniform(-8, 8) * k,
                                      rng.uniform(-4, 4), (1 if (j + li) % 2 == 0 else -1) * rng.uniform(18, 32),
                                      times[li][j], box))
            x += advs[i]
    return letters


def _corners(letters, steps=5):
    """Points along each letter's box, in letter-local coordinates (homogeneous), per letter."""
    u = np.linspace(0, 1, steps)
    out = []
    for L in letters:
        x0, y0, x1, y1 = L.box
        xs, ys = np.meshgrid(x0 + (x1 - x0) * u, y0 + (y1 - y0) * u)
        pts = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], -1)
        out.append(A_t(L.x, L.y) @ A_r(L.rot) @ pts.T)
    return np.concatenate(out, axis=1) if out else np.zeros((3, 0))


# A spread of moments after the letters have landed: the POW punch, the squash and the stretch.
CHECK_TIMES = tuple(np.round(np.arange(1.2, 4.0, 0.04), 3))


def fits(letters, center, pivot_y, pow_at):
    """True when every letter stays in title-safe through the motion and clear of the POW."""
    pts = _corners(letters)
    for t in CHECK_TIMES:
        p = group_matrix(t, center, pivot_y) @ pts
        if (p[0].min() < SAFE_X or p[0].max() > W - SAFE_X or p[1].min() < SAFE_Y or p[1].max() > H - SAFE_Y):
            return False
    if pow_at is None:
        return True
    # Keep the letters (at rest) out of an ellipse between the POW's spikes and valleys.
    rest = A_t(*center) @ A_r(TILT) @ pts
    px, py, s = pow_at
    inv = np.linalg.inv(A_t(px, py) @ A_r(-12) @ A_s(s))
    q = inv @ rest
    a, b = 245 * 1.3, 245 * 0.95
    return bool(((q[0] / a) ** 2 + (q[1] / b) ** 2).min() >= 1.0)


def ceiling(lines):
    """Biggest letter size for a line break: as tuned for two lines, a bit bigger for a lone line."""
    return SIZE * (SOLO if len(lines) == 1 else 1.0)


def fit_size(lines, center, pow_at, seed=0):
    """Largest letter size (<= ceiling) at which a line break fits; 0 if none (falls back to the smallest)."""
    def ok(size):
        letters = place_letters(lines, size, seed)
        return fits(letters, center, pivot(lines, size), pow_at)
    top = ceiling(lines)
    if ok(top):
        return top
    lo, hi = 40.0, top
    if not ok(lo):
        return 0.0
    for _ in range(10):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def pivot(lines, size):
    k = size / SIZE
    return ((len(lines) - 1) / 2) * PITCH * k + 150 * k


def layout(title, has_pow=True, uppercase=True, seed=0):
    words = clean(title, FONT, uppercase).split()
    if not words:
        raise ValueError("title is empty")
    best, best_score = None, (-1.0, 0)
    for lines in partitions(words):
        for center, pow_at, pref in PLACEMENTS if has_pow else ((CENTER, None, 1.0),):
            size = fit_size(lines, center, pow_at, seed)
            # Biggest letters win (fewer lines on a near tie), then the most even line lengths.
            score = (round(size * pref * (1 - 0.1 * (len(lines) - 1)), 1), -max(map(len, lines)))
            if score > best_score:
                best, best_score = (lines, size, center, pow_at), score
            if size >= ceiling(lines):  # later placements are less preferred and can't be bigger
                break
    lines, size, center, pow_at = best
    size = size or 40.0
    return Layout(lines, size, center, pivot(lines, size), pow_at or POW_AT, place_letters(lines, size, seed))


# ---------------------------------------------------------------- renderer
class Renderer:
    def __init__(self, title="ROBOTS REVENGE", pow="KA-POW!", uppercase=True, fps=25, seed=0):
        self.fps = fps
        word = clean(pow or "", POW_FONT, uppercase)
        self.lay = layout(title, bool(word), uppercase, seed)
        rng = np.random.default_rng(9 + seed)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.cx, self.cy = self.lay.center
        dx, dy = xx - self.cx, yy - self.cy
        self.ang = np.arctan2(dy, dx)
        rr = np.sqrt(dx * dx + dy * dy)
        rn = rr / math.hypot(W / 2, H / 2)
        # Ben-Day dots: tiny in the middle, fat toward the edges.
        S = 2
        yy2, xx2 = np.mgrid[0:H * S, 0:W * S].astype(np.float32)
        rn2 = np.sqrt(((xx2 / S - self.cx) / (W / 2)) ** 2 + ((yy2 / S - self.cy) / (H / 2)) ** 2) / math.sqrt(2)
        u = (xx2 + yy2) / math.sqrt(2)
        v = (xx2 - yy2) / math.sqrt(2)
        sp = 30 * S
        du = (u % sp) - sp / 2
        dv = (v % sp) - sp / 2
        d = np.sqrt(du * du + dv * dv)
        rad = (2.0 + 10.5 * smoothstep(0.08, 0.95, rn2)) * S
        dots = np.clip(rad - d + 0.5 * S, 0, 1)
        self.dots = cv2.resize(dots, (W, H), interpolation=cv2.INTER_AREA)[..., None]
        del yy2, xx2, rn2, u, v, du, dv, d, rad
        self.glow = np.exp(-(rn / 0.42) ** 2)[..., None]
        self.edge = smoothstep(0.55, 1.25, rn)[..., None]
        self.iris_d = np.sqrt((np.arange(W)[None, :] - self.cx) ** 2 + (np.arange(H)[:, None] - self.cy) ** 2)

        self.letters = self.build_letters()
        self.kapow, self.kapow_anchor = kapow_sprite(word, seed) if word else (None, None)
        n_lines = 26
        self.speed_ang = rng.uniform(0, 2 * math.pi, n_lines)
        self.speed_w = rng.uniform(0.006, 0.018, n_lines)
        self.speed_r0 = rng.uniform(330, 470, n_lines)

    def build_letters(self):
        lay = self.lay
        k = lay.size / SIZE
        f = font(FONT, int(lay.size * SPR))
        stroke, depth = max(2, int(STROKE * SPR * k)), max(2, int(DEPTH * SPR * k))
        cache, out = {}, []
        for L in lay.letters:
            cols = LINE_COLS[L.line % 2]
            key = (L.ch, L.line % 2)
            if key not in cache:
                cache[key] = letter_sprite(L.ch, f, cols, stroke, depth)
            spr, anc = cache[key]
            out.append({"sprite": spr, "anchor": anc, "x": L.x, "y": L.y, "rot": L.rot, "t0": L.t0, "spin": L.spin})
        return out

    # -- frame --------------------------------------------------------
    def background(self, t):
        rot = 0.18 * t
        n = 14
        s = np.sin(n * (self.ang + rot))
        ray = smoothstep(-0.12, 0.12, s)[..., None]
        light = np.array([120, 212, 255], np.float32)
        base = np.array([52, 160, 240], np.float32)
        img = base + (light - base) * ray
        img = img + (np.array([255, 255, 240], np.float32) - img) * (0.55 * self.glow)
        dot_col = np.array([22, 96, 200], np.float32)
        img = img + (dot_col - img) * (0.62 * self.dots)
        img = img * (1 - 0.35 * self.edge)
        return img

    def speed_lines(self, img, t):
        a = t - T_POW
        if a < 0 or a > 0.45:
            return
        k = 1 - a / 0.45
        layer = np.zeros((H, W), np.float32)
        r0 = self.speed_r0 + 900 * a
        for ang, wdt, rs in zip(self.speed_ang, self.speed_w, r0):
            r1 = rs + 2200
            p0 = (self.cx + rs * math.cos(ang), self.cy + rs * math.sin(ang))
            pa = (self.cx + r1 * math.cos(ang - wdt), self.cy + r1 * math.sin(ang - wdt))
            pb = (self.cx + r1 * math.cos(ang + wdt), self.cy + r1 * math.sin(ang + wdt))
            pts = np.array([p0, pa, pb], np.float32) * 4
            cv2.fillPoly(layer, [pts.astype(np.int32)], 1.0, cv2.LINE_AA, shift=2)
        img += (255 - img) * (layer * 0.9 * k)[..., None]

    def render(self, t, fi=0):
        img = self.background(t)
        bx, by, _ = shake(t)
        if abs(bx) + abs(by) > 0.5:  # background shakes too (a bit less: parallax)
            img = cv2.warpAffine(img, np.float32([[1, 0, 0.5 * bx], [0, 1, 0.5 * by]]), (W, H),
                                 borderMode=cv2.BORDER_REFLECT)
        # white impact flash on the background
        a = t - T_POW
        if 0 <= a < 0.2:
            img += (255 - img) * (0.55 * math.exp(-a / 0.05))
        self.speed_lines(img, t)

        G = group_matrix(t, self.lay.center, self.lay.pivot_y)
        for L in self.letters:
            la = t - L["t0"]
            if la <= 0:
                continue
            s = pop(la)
            if s <= 0.01:
                continue
            rot = L["rot"] + L["spin"] * math.exp(-la / 0.12) * math.cos(2 * math.pi * 2.4 * la)
            drop = -55 * math.exp(-la / 0.07)
            M = (G @ A_t(L["x"], L["y"] + drop) @ A_r(rot) @ A_s(s / SPR)
                 @ A_t(-L["anchor"][0], -L["anchor"][1]))
            paste_affine(img, L["sprite"], M)

        if a >= 0 and self.kapow is not None:
            M, s = pow_matrix(t, self.lay.pow_at, self.kapow_anchor)
            if s > 0.01:
                paste_affine(img, self.kapow, M)

        # Cartoon iris-out on the title.
        if t >= T_IRIS:
            p = float(smoothstep(T_IRIS, T_IRIS_END, t))
            rmax = math.hypot(W, H) * 0.62
            r = rmax * (1 - p) ** 1.6
            m = np.clip(r - self.iris_d + 0.5, 0, 1)
            img *= m[..., None]
        return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- audio
def norm(x):
    return x / (np.max(np.abs(x)) + 1e-12)


def pan_gains(p):
    return np.sqrt(0.5 * (1 - p)), np.sqrt(0.5 * (1 + p))


def swept_noise(rng, n, fc_fn, bw_oct):
    """Noise through a moving band-pass (STFT overlap-add)."""
    N, hop = 1024, 256
    win = np.hanning(N)
    out = np.zeros(n + N)
    freqs = np.fft.rfftfreq(N, 1 / SR)
    freqs[0] = 1
    for start in range(0, n, hop):
        spec = np.fft.rfft(rng.standard_normal(N) * win)
        fc = fc_fn((start + N / 2) / SR)
        g = np.exp(-0.5 * (np.log2(freqs / fc) / bw_oct) ** 2)
        out[start:start + N] += np.fft.irfft(spec * g) * win
    return out[:n]


def taper(x, frac=0.35):
    """Fade the tail of a one-shot smoothly to zero so it never clicks off."""
    n = int(len(x) * frac)
    x = x.copy()
    x[-n:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, n)))
    return x


def place(buf, start_s, mono, gain, pan=0.0):
    mono = taper(mono)
    i0 = int(round(start_s * SR))
    seg = mono[: max(0, buf.shape[1] - i0)]
    gl, gr = pan_gains(pan)
    buf[0, i0:i0 + len(seg)] += gain * gl * seg
    buf[1, i0:i0 + len(seg)] += gain * gr * seg


def bloop_notes(letters):
    """A rising pentatonic run per line (C D E G A C, then D E G A C D E...), squeezed for long lines."""
    notes = []
    for L in letters:
        row = [M for M in letters if M.line == L.line]
        i, n = row.index(L), len(row)
        step = i if n <= 7 else round(i * 6 / (n - 1))
        notes.append(PENTATONIC[min(L.line + step, len(PENTATONIC) - 1)])
    return notes


def synth_audio(rng, letter_times, notes):
    """The full 4.4 s soundtrack, (2, n) floats. Hits are placed in seconds, so any fps lines up."""
    n = int(round(DUR * SR))
    mix = np.zeros((2, n))
    send = np.zeros((2, n))
    t = np.arange(n) / SR

    # Whoosh while the letters fly in (panned left -> right).
    w0, w1 = 0.06, 1.0
    nw = int((w1 - w0) * SR)
    tw = np.arange(nw) / SR
    dur = w1 - w0

    def fc(x):
        if x < 0.45 * dur:
            return 350 * (2600 / 350) ** min(1, x / (0.45 * dur))
        return 2600 * (700 / 2600) ** ((x - 0.45 * dur) / (0.55 * dur))

    wh = norm(swept_noise(rng, nw, fc, 0.55)) * np.sin(np.pi * tw / dur) ** 1.5
    pan = -0.8 + 1.6 * tw / dur
    gl, gr = pan_gains(pan)
    i0 = int(w0 * SR)
    mix[0, i0:i0 + nw] += 0.55 * wh * gl
    mix[1, i0:i0 + nw] += 0.55 * wh * gr
    send[:, i0:i0 + nw] += 0.15 * wh

    # Little bloops as each letter pops (rising pentatonic).
    tb = np.arange(int(0.18 * SR)) / SR
    last = max(len(letter_times) - 1, 1)
    for i, (lt, f0) in enumerate(zip(letter_times, notes)):
        f = f0 * (1 + 0.9 * np.exp(-tb / 0.014))
        ph = 2 * np.pi * np.cumsum(f) / SR
        bl = (np.sin(ph) + 0.25 * np.sin(2 * ph)) * np.exp(-tb / 0.05) * (1 - np.exp(-tb / 0.002))
        place(mix, lt + 0.03, bl, 0.20, pan=-0.6 + 1.2 * i / last if len(letter_times) > 1 else 0.0)

    # POW: kick + snap + brassy stab, crunched, with a short crash.
    tp = np.arange(int(1.2 * SR)) / SR
    kf = 46 + 150 * np.exp(-tp / 0.03)
    kick = np.sin(2 * np.pi * np.cumsum(kf) / SR) * np.exp(-tp / 0.14)
    snap = norm(signal.sosfilt(signal.butter(2, [900, 7000], "bp", fs=SR, output="sos"),
                               rng.standard_normal(len(tp)))) * np.exp(-tp / 0.06)
    sf = 210 * (1 - 0.35 * np.minimum(tp / 0.2, 1))
    sph = 2 * np.pi * np.cumsum(sf) / SR
    stab = sum(np.sin(k * sph) / k for k in range(1, 18)) * np.exp(-tp / 0.11) * (1 - np.exp(-tp / 0.003))
    stab = signal.sosfilt(signal.butter(2, 2200, "lp", fs=SR, output="sos"), stab)
    powh = np.tanh(2.6 * (1.0 * kick + 0.75 * snap + 0.55 * norm(stab)))
    crash = norm(signal.sosfilt(signal.butter(2, 4500, "hp", fs=SR, output="sos"),
                                rng.standard_normal(len(tp)))) * np.exp(-tp / 0.4)
    powh = norm(powh + 0.22 * crash)
    place(mix, T_POW, powh, 1.0)
    place(send, T_POW, powh, 0.35)

    # BOING: twangy spring with a decaying vibrato that wobbles the pitch.
    tg = np.arange(int(1.1 * SR)) / SR
    vib = np.exp(-tg / 0.45) * np.sin(2 * np.pi * 11 * tg)
    f = 150 * (1 + 0.55 * np.minimum(tg / 0.5, 1) ** 0.6) * (1 + 0.24 * vib)
    ph = 2 * np.pi * np.cumsum(f) / SR
    twang = (np.sin(ph) + 0.5 * np.sin(2 * ph) * (0.6 + 0.4 * vib) + 0.3 * np.sin(3 * ph) + 0.12 * np.sin(5 * ph))
    boing = twang * (1 - np.exp(-tg / 0.004)) * np.exp(-tg / 0.38) * (1 + 0.35 * vib)
    pluck = norm(signal.sosfilt(signal.butter(2, [1500, 5000], "bp", fs=SR, output="sos"),
                                rng.standard_normal(len(tg)))) * np.exp(-tg / 0.012)
    boing = norm(norm(boing) + 0.35 * pluck)
    place(mix, T_BOING, boing, 0.75)
    place(send, T_BOING, boing, 0.2)

    # Slide whistle down for the iris-out.
    ts = np.arange(int((T_IRIS_END - T_IRIS + 0.06) * SR)) / SR
    sd = ts[-1]
    f = 1500 * (380 / 1500) ** (ts / sd) * (1 + 0.012 * np.sin(2 * np.pi * 7 * ts))
    whistle = np.sin(2 * np.pi * np.cumsum(f) / SR) * smoothstep(0, 0.04, ts) * (1 - smoothstep(sd - 0.06, sd, ts))
    breath = norm(signal.sosfilt(signal.butter(2, [1500, 6000], "bp", fs=SR, output="sos"),
                                 rng.standard_normal(len(ts)))) * 0.08
    place(mix, T_IRIS - 0.02, whistle + breath * smoothstep(0, 0.04, ts), 0.30)

    # Small bright room.
    nir = int(0.8 * SR)
    ti = np.arange(nir) / SR
    ir = rng.standard_normal((2, nir)) * np.exp(-6.91 * ti / 0.55)
    ir[:, : int(0.012 * SR)] = 0
    ir /= np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True))
    wet = np.stack([signal.fftconvolve(send[c], ir[c])[:n] for c in range(2)])
    mix = mix + 0.5 * wet

    mix = signal.sosfilt(signal.butter(2, 35, "hp", fs=SR, output="sos"), mix)
    mix = np.tanh(1.2 * norm(mix)) / np.tanh(1.2)
    mix *= 10 ** (-1.8 / 20) / np.max(np.abs(mix))
    mix *= 1 - smoothstep(DUR - 0.05, DUR, t)
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
def render(out, *, title="ROBOTS REVENGE", pow="KA-POW!", uppercase=True, ctx=None) -> Path:
    """Render the comic-book title card to `out` (mp4). `pow` is the starburst word ("" for none)."""
    ctx = ctx or Ctx()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = ctx.nframes(DUR)
    seconds = n / ctx.fps
    r = Renderer(title, pow, uppercase, ctx.fps, ctx.seed)
    letters = r.lay.letters
    mix = synth_audio(np.random.default_rng(77 + ctx.seed), [L.t0 for L in letters], bloop_notes(letters))
    with tempfile.TemporaryDirectory(dir=ctx.scratch("title_comic")) as td:
        wav = Path(td) / "title_comic.wav"
        media.write_wav(wav, trim_audio(mix, seconds).T)
        encode(out, wav, frames(r, n, ctx.fps, ctx.workers), ctx.fps, seconds)
    return out


def stills(out_dir, times=STILLS, *, title="ROBOTS REVENGE", pow="KA-POW!", uppercase=True, ctx=None,
           stem="comic"):
    """PNG frames at a few moments, for a quick look without encoding. Returns the paths."""
    ctx = ctx or Ctx()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r = Renderer(title, pow, uppercase, ctx.fps, ctx.seed)
    paths = []
    for tt in times:
        fi = int(round(tt * ctx.fps))
        p = out_dir / f"{stem}_t{tt:.2f}.png"
        Image.fromarray(r.render(fi / ctx.fps, fi)).save(p)
        paths.append(p)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kms.render.title_comic",
                                 description="Comic-book title card (4.4 s).")
    ap.add_argument("out", help="output .mp4")
    ap.add_argument("--title", default="ROBOTS REVENGE")
    ap.add_argument("--pow", default="KA-POW!", help='the starburst word ("" for none)')
    ap.add_argument("--keep-case", action="store_true", help="don't capitalise the words")
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
    kw = dict(title=a.title, pow=a.pow, uppercase=not a.keep_case, ctx=ctx)
    if a.stills:
        times = [float(x) for x in a.at.split(",") if x.strip()]
        for p in stills(a.stills, times, stem=Path(a.out).stem, **kw):
            print(p)
        return
    media.require_ffmpeg()
    print(render(a.out, **kw))


if __name__ == "__main__":
    main()
