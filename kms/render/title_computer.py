"""Opening title: a retro green-phosphor villain computer boots up.

A CRT terminal powers on, types its boot log, fills a progress bar, spots the
humans, glitches, then flickers the film's title onto the screen as a giant
dot-matrix sign before switching itself off. Pictures are drawn with PIL/numpy
and all sound is synthesised with numpy.

    python -m kms.render.title_computer OUT.mp4 [--title T] [--villain V] [--fps 25]
                                                [--frames N] [--seed S] [--stills DIR]
"""
import argparse
import itertools
import math
import multiprocessing as mp
import subprocess
import tempfile
import unicodedata
import wave
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import butter, fftconvolve, sosfilt

from kms.fonts import font
from kms.media import audio_args, ffmpeg, video_args
from kms.render import SR, H, W, Ctx

DUR = 6.0
MONO = "menlo-bold"

# ---------------------------------------------------------------- timeline (s)
ON_END = 0.32
L1_T, L1_CPS, L1_SLOT = 0.40, 46, 0.70       # each typed line must finish inside its slot
L2_T, L2_CPS, L2_SLOT = 1.14, 55, 0.44
BAR_T0, BAR_T1 = 1.62, 2.42
L4_T, L4_CPS, L4_SLOT = 2.48, 45, 0.40
ALERT_T0, ALERT_T1 = 2.90, 3.36
GLITCH_T0, GLITCH_T1 = 3.36, 3.52
TITLE_T0, TITLE_FILL = 3.52, 0.56
CATCH_T = 4.08  # whole title dips for one frame, like a tube catching
BEEP_T = 4.12
SUB_T, SUB_CPS, SUB_SLOT = 4.30, 42, 1.00
OFF_T0, OFF_T1 = 5.60, 6.00

# ---------------------------------------------------------------- layout (px)
X0 = 200
LINE_Y = {"l1": 245, "l2": 340, "bar": 440, "l4": 560}
TERM_PX, HDR_PX, SUB_PX = 56, 40, 52
LINE_MAX_W = {"l1": 1824 - 40 - X0, "l2": 1824 - 40 - X0, "l4": 1380 - 14 - X0}  # l4 stays clear of the robot
SUB_Y, SUB_MAX_W = 860, 1728 - 60
BAR_X1 = 1480

# the dot-matrix sign: whole cells, centred on TITLE_CY between the header and the subtitle
TITLE_CY = 493
TITLE_MAX_W, TITLE_SAFE_W, TITLE_MAX_H = 1500, 1728, 610
CELL_MAX, CELL_MIN = 60, 3
SPACE_COLS = 3

DEFAULT_TITLE = "ROBOTS REVENGE"
DEFAULT_VILLAIN = "ROBOT"
DEFAULT_SUBTITLE = "> LIVE: THE 9 O'CLOCK NEWS"
DEFAULT_MODE = "MODE: WORLD DOMINATION"

# ---------------------------------------------------------------- 5x7 dot-matrix font
GLYPHS = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11100", "10010", "10001", "10001", "10001", "10010", "11100"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    "!": ["00100", "00100", "00100", "00100", "00100", "00000", "00100"],
    "?": ["01110", "10001", "00001", "00010", "00100", "00000", "00100"],
    "'": ["01100", "00100", "01000", "00000", "00000", "00000", "00000"],
    '"': ["01010", "01010", "01010", "00000", "00000", "00000", "00000"],
    "&": ["01100", "10010", "10100", "01000", "10101", "10010", "01101"],
    "-": ["00000", "00000", "00000", "01110", "00000", "00000", "00000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ",": ["00000", "00000", "00000", "00000", "01100", "00100", "01000"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "/": ["00000", "00001", "00010", "00100", "01000", "10000", "00000"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
}
# characters with no glyph of their own, and what to show instead
FOLD = {"Ø": "O", "Æ": "AE", "Œ": "OE", "Ð": "D", "Þ": "TH", "Ł": "L", "Đ": "D", "Ħ": "H",
        "‘": "'", "’": "'", "`": "'", "´": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "..."}


def matrix_text(text):
    """The title as characters the dot-matrix sign can show.

    Upper-cases, folds accents to their base letter (É -> E), and quietly drops anything
    else (emoji, symbols), so any title renders without crashing.
    """
    out = []
    for ch in unicodedata.normalize("NFC", str(text)).upper():
        if ch.isspace():
            out.append(" ")
            continue
        for c in FOLD.get(ch, ch):
            if c in GLYPHS:
                out.append(c)
            else:
                out.extend(b for b in unicodedata.normalize("NFKD", c).upper() if b in GLYPHS)
    return " ".join("".join(out).split())


@lru_cache(maxsize=None)
def glyph(ch):
    """The 7 rows for a character; punctuation is trimmed to its lit columns."""
    rows = GLYPHS[ch]
    if ch.isalnum():
        return tuple(rows)
    lit = [c for c in range(len(rows[0])) if any(r[c] == "1" for r in rows)]
    a, b = (lit[0], lit[-1] + 1) if lit else (2, 3)
    return tuple(r[a:b] for r in rows)


def line_cols(line):
    widths = [SPACE_COLS if ch == " " else len(glyph(ch)[0]) for ch in line]
    return sum(widths) + max(0, len(widths) - 1)


def _splits(words, max_lines=3):
    n = len(words)
    for k in range(1, min(max_lines, n) + 1):
        for cuts in itertools.combinations(range(1, n), k - 1):
            bounds = (0, *cuts, n)
            yield [" ".join(words[a:b]) for a, b in zip(bounds, bounds[1:])]


def _cell_for(cols, rows):
    c = min(TITLE_MAX_W / cols, TITLE_MAX_H / rows)
    if c < 24:  # a very long line may use the whole title-safe width
        c = min(TITLE_SAFE_W / cols, TITLE_MAX_H / rows)
    return int(np.clip(math.floor(c), CELL_MIN, CELL_MAX))


def layout_title(title, max_lines=3):
    """(lines, cell px): the 1-3 line break that gives the biggest sign.

    Breaks are compared by how big they'd be inside the comfortable width, so a balanced
    stack beats one long line squeezed out to the edges of the screen.
    """
    words = matrix_text(title).split()
    if not words:
        return [], CELL_MAX
    best = None
    for lines in _splits(words, max_lines):
        cols = [line_cols(s) for s in lines]
        rows = 9 * len(lines) - 2
        comfy = min(TITLE_MAX_W / max(cols), TITLE_MAX_H / rows, CELL_MAX)
        key = (round(comfy, 3), -len(lines), -(max(cols) - min(cols)))
        if best is None or key > best[0]:
            best = (key, lines, _cell_for(max(cols), rows))
    return best[1], best[2]


# ---------------------------------------------------------------- small helpers
def type_times(text, start, cps, slot, seed):
    """When each character of a line appears (slightly uneven, like real typing).

    Lines longer than the original ones type faster so they still finish inside their slot.
    """
    n = len(text)
    fast = n > cps * slot + 1e-9
    if fast:
        cps = n / slot
    r = np.random.default_rng(seed)
    out, t = [], start
    for _ in text:
        out.append(t)
        t += (1.0 / cps) * (1 + 0.35 * (r.random() * 2 - 1))
    out = np.array(out)
    if fast and n > 1 and out[-1] > start + slot * 0.97:
        out = start + (out - start) * (slot * 0.97 / (out[-1] - start))
    return out


def shown(times, t):
    return int(np.searchsorted(times, t, side="right"))


def bar_progress(t):
    """0..1 with a couple of dramatic stalls."""
    u = np.clip((t - BAR_T0) / (BAR_T1 - BAR_T0), 0, 1)
    return float(np.interp(u, [0, 0.28, 0.42, 0.72, 0.86, 1.0], [0, 0.44, 0.47, 0.89, 0.91, 1.0]))


def flick(seed, frame):
    """Deterministic pseudo-random 0..1 per (cell, frame)."""
    v = np.sin(seed * 12.9898 + frame * 78.233) * 43758.5453
    return v - np.floor(v)


def ease(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


_NO_GLYPH = "\U000F0000"  # a private-use character no font draws: it shows the "missing glyph" box


@lru_cache(maxsize=None)
def _ink(role, ch):
    img = Image.new("L", (96, 96), 0)
    ImageDraw.Draw(img).text((16, 8), ch, font=font(role, 48), fill=255)
    return img.tobytes()


def has_glyph(role, ch):
    """True if the role's font really draws ch, rather than its missing-glyph box."""
    return ch.isspace() or _ink(role, ch) != _ink(role, _NO_GLYPH)


def _clean(text, role=MONO):
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


def _prompt(text):
    text = _clean(text)
    return text if text.startswith(">") else f"> {text}".rstrip()


def default_boot_lines(villain=DEFAULT_VILLAIN):
    v = "_".join(_clean(villain).split()) or DEFAULT_VILLAIN
    return (f"> BOOTING {v}_PROTOCOL.EXE ...", "> LOADING EVIL PLANS ...", "> HUMANS DETECTED!")


def default_os_name(villain=DEFAULT_VILLAIN):
    words = _clean(villain).split()
    word = words[-1] if words else DEFAULT_VILLAIN
    return "ROBO-OS 9000" if word == "ROBOT" else f"{word}-OS 9000"


def _mono_fit(px, texts_and_widths, min_px=20):
    """One monospaced size (<= px) at which every text fits its width."""
    cw = font(MONO, px).getlength("M") / px
    size = px
    for text, max_w in texts_and_widths:
        if text and len(text) * cw * size > max_w:
            size = min(size, int(max_w / (len(text) * cw)))
    # hinting makes small sizes a touch wider than the estimate: check the real advance
    while size > min_px and any(text and len(text) * font(MONO, size).getlength("M") > max_w
                                for text, max_w in texts_and_widths):
        size -= 1
    return max(min_px, size)


def _clip_to(text, max_w, cw):
    n = int(max_w // cw)
    return text if len(text) <= n else text[:max(1, n - 2)] + ".."


# ---------------------------------------------------------------- the scene
class Scene:
    """Everything about one render that doesn't change frame to frame."""

    def __init__(self, title=DEFAULT_TITLE, villain=DEFAULT_VILLAIN, boot_lines=None,
                 subtitle=DEFAULT_SUBTITLE, os_name=None, mode=DEFAULT_MODE, clock="21:00",
                 fps=25, seed=0):
        self.fps, self.seed = fps, seed
        self.off = 100003 * seed  # seed 0 keeps the original look exactly
        lines = list(boot_lines or ())[:3]
        defaults = default_boot_lines(villain)
        lines += defaults[len(lines):]
        self.l1, self.l2, self.l4 = (_prompt(s) for s in lines)
        self.sub = _prompt(subtitle) if _clean(subtitle) else ""

        # terminal: one size for all three lines, shrunk only if a line is too long
        px = _mono_fit(TERM_PX, [(self.l1, LINE_MAX_W["l1"]), (self.l2, LINE_MAX_W["l2"]),
                                 (self.l4, LINE_MAX_W["l4"])])
        self.font = font(MONO, px)
        self.font_px = px
        self.cw = self.font.getlength("M")
        self.l1 = _clip_to(self.l1, LINE_MAX_W["l1"], self.cw)
        self.l2 = _clip_to(self.l2, LINE_MAX_W["l2"], self.cw)
        self.l4 = _clip_to(self.l4, LINE_MAX_W["l4"], self.cw)
        spx = _mono_fit(SUB_PX, [(self.sub, SUB_MAX_W)])
        self.sub_font, self.sub_px = font(MONO, spx), spx
        self.sub_cw = self.sub_font.getlength("M")
        self.sub = _clip_to(self.sub, SUB_MAX_W, self.sub_cw)

        o = self.off
        self.T1 = type_times(self.l1, L1_T, L1_CPS, L1_SLOT, 1 + o)
        self.T2 = type_times(self.l2, L2_T, L2_CPS, L2_SLOT, 2 + o)
        self.T4 = type_times(self.l4, L4_T, L4_CPS, L4_SLOT, 4 + o)
        self.TS = type_times(self.sub, SUB_T, SUB_CPS, SUB_SLOT, 5 + o)

        self._header(os_name if os_name is not None else default_os_name(villain), mode)
        self.clock = _clean(clock)
        self.title_lines, self.cell = layout_title(title)
        self.cells = self._build_cells()

    # -- header bar
    def _header(self, os_name, mode):
        self.os_name, self.mode = _clean(os_name), _clean(mode)
        size = HDR_PX
        while size > 16:
            f = font(MONO, size)
            left = X0 + f.getlength(self.os_name)
            mid = f.getlength(self.mode) / 2
            right = W - X0 - f.getlength("00:00:00")
            if left + 30 <= W / 2 - mid and W / 2 + mid + 30 <= right:
                break
            size -= 2
        self.hdr_font = font(MONO, size)

    def clock_text(self, t):
        sec = 57 + max(0, int(np.floor(t - (TITLE_T0 - 3.0))))
        try:
            hh, mm = (int(x) for x in self.clock.split(":")[:2])
        except ValueError:
            return self.clock
        s = (hh * 3600 + mm * 60 - 3 + min(sec, 60) - 57) % 86400
        return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}"

    # -- dot-matrix title
    def _build_cells(self):
        """List of (x, y, lit, t_on, seed) for every cell of the matrix sign."""
        rng = np.random.default_rng(42 + self.off)
        cell, cells = self.cell, []
        if not self.title_lines:
            return cells
        widest = max(line_cols(s) for s in self.title_lines)
        top = TITLE_CY - (9 * len(self.title_lines) - 2) * cell / 2
        for li, line in enumerate(self.title_lines):
            ncols = line_cols(line)
            x_left = W / 2 - ncols * cell / 2
            y_top = top + li * 9 * cell
            col0 = 0
            for ch in line:
                if ch == " ":
                    col0 += SPACE_COLS + 1
                    continue
                rows = glyph(ch)
                for r, row in enumerate(rows):
                    for c, bit in enumerate(row):
                        x = x_left + (col0 + c) * cell
                        y = y_top + r * cell
                        sweep = (x - (W / 2 - widest * cell / 2)) / (widest * cell)
                        t_on = TITLE_T0 + 0.36 * sweep + 0.18 * rng.random()
                        cells.append((x, y, bit == "1", t_on, rng.random()))
                col0 += len(rows[0]) + 1
        return cells

    # -- drawing
    def draw_cursor(self, d, x, y, font_px, on):
        if on:
            d.rectangle([x + 2, y - 2, x + font_px * 0.6, y + font_px * 1.02], fill=235)

    def draw_header(self, d, t):
        d.rectangle([0, 70, W, 140], fill=205)
        d.text((X0, 105), self.os_name, font=self.hdr_font, fill=0, anchor="lm")
        d.text((W / 2, 105), self.mode, font=self.hdr_font, fill=0, anchor="mm")
        d.text((W - X0, 105), self.clock_text(t), font=self.hdr_font, fill=0, anchor="rm")

    def draw_terminal(self, d, t):
        F, CW, L1, L2, L4 = self.font, self.cw, self.l1, self.l2, self.l4
        T2 = self.T2
        blink = (t * 2.2) % 1 < 0.55
        n1, n2, n4 = shown(self.T1, t), shown(T2, t), shown(self.T4, t)
        d.text((X0, LINE_Y["l1"]), L1[:n1], font=F, fill=215)
        d.text((X0, LINE_Y["l2"]), L2[:n2], font=F, fill=215)
        cursor = None
        if t < L1_T:
            cursor = (X0, LINE_Y["l1"], blink)
        elif n1 < len(L1) or t < T2[0]:
            cursor = (X0 + n1 * CW, LINE_Y["l1"], True if n1 < len(L1) else blink)
        elif n2 < len(L2) or t < BAR_T0:
            cursor = (X0 + n2 * CW, LINE_Y["l2"], True if n2 < len(L2) else blink)

        if t >= BAR_T0 - 0.04:
            p = bar_progress(t)
            bx0, by0, bx1, by1 = X0 + 2 * CW, LINE_Y["bar"], BAR_X1, LINE_Y["bar"] + 62
            d.rectangle([bx0, by0, bx1, by1], outline=215, width=5)
            nseg = 20
            segw = (bx1 - bx0 - 20) / nseg
            for s in range(int(round(p * nseg))):
                sx = bx0 + 12 + s * segw
                d.rectangle([sx, by0 + 12, sx + segw - 7, by1 - 12], fill=225)
            pct = int(round(p * 100))
            d.text((bx1 + 36, by0 + 31), f"{pct:3d}%", font=F, fill=230 if pct == 100 else 200, anchor="lm")

        if t >= L4_T - 0.02:
            y = LINE_Y["l4"]
            inverse = ALERT_T0 <= t < ALERT_T1 and int((t - ALERT_T0) / 0.06) % 2 == 0
            txt = L4[:n4]
            if inverse:
                x1 = X0 + len(L4) * CW
                d.rectangle([X0 - 14, y - 10, x1 + 14, y + self.font_px + 14], fill=240)
                d.text((X0, y), txt, font=F, fill=0)
            else:
                d.text((X0, y), txt, font=F, fill=235)
            if n4 < len(L4):
                cursor = (X0 + n4 * CW, y, True)
            elif t < ALERT_T0:
                cursor = (X0 + n4 * CW, y, blink)
        if cursor:
            self.draw_cursor(d, cursor[0], cursor[1], self.font_px, cursor[2])
        self.draw_robot(d, t)

    def draw_robot(self, d, t):
        """Little pixel robot in the corner: eyes scan about, then go angry at the alert line."""
        if t < L2_T:
            return
        reveal = int(np.clip((t - L2_T) / 0.25, 0, 1) * len(ROBOT_HEAD))
        angry = t >= self.T4[-1] if len(self.T4) else t >= L4_T
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

    def draw_title(self, d, t, frame):
        lit_level = 1.0
        if abs(t - CATCH_T) < 0.5 / self.fps:
            lit_level = 0.35
        elif t > CATCH_T:
            lit_level = 0.93 + 0.07 * np.sin(t * 9.0)
        cell = self.cell
        off, block, radius = cell / 12, cell * 5 / 6, max(1, int(round(cell * 5 / 36)))
        for i, (x, y, lit, t_on, seed) in enumerate(self.cells):
            box = [x + off, y + off, x + off + block, y + off + block]
            val = 0.075
            if lit and t >= t_on:
                on = True
                if t < t_on + 0.14 and flick(seed * 97 + i, frame) < 0.35:
                    on = False
                if on:
                    val = lit_level
            d.rounded_rectangle(box, radius=radius, fill=int(val * 255))
        n = shown(self.TS, t)
        if self.sub and t >= SUB_T - 0.3:
            x = W / 2 - len(self.sub) * self.sub_cw / 2
            d.text((x, SUB_Y), self.sub[:n], font=self.sub_font, fill=215)
            blink = (t * 2.2) % 1 < 0.55
            self.draw_cursor(d, x + n * self.sub_cw, SUB_Y, self.sub_px, n < len(self.sub) or blink)

    def glitch(self, I, frame, k):
        """k: 0..1 glitch strength. Tears horizontal slices and sprays noise rows."""
        r = np.random.default_rng(1000 + frame + self.off)
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

    def content(self, t, frame):
        img = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(img)
        self.draw_header(d, t)
        if t < GLITCH_T0:
            self.draw_terminal(d, t)
        elif t < GLITCH_T1:
            self.draw_terminal(d, GLITCH_T0 - 0.01)
        else:
            self.draw_title(d, t, frame)
        I = np.asarray(img, np.float32) / 255.0
        if GLITCH_T0 <= t < GLITCH_T1:
            I = self.glitch(I, frame, (t - GLITCH_T0) / (GLITCH_T1 - GLITCH_T0) + 0.25)
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

    def frame(self, frame):
        t = frame / self.fps
        return crt(self.content(t, frame), t, frame, self.off)


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


# ---------------------------------------------------------------- CRT model
@lru_cache(maxsize=2)
def _crt_tables(off):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx, ny = (xx + 0.5) / W * 2 - 1, (yy + 0.5) / H * 2 - 1
    curve, zoom = 3.6, 1.075
    src_x = nx * (1 + (ny / curve) ** 2) * zoom
    src_y = ny * (1 + (nx / curve) ** 2) * zoom
    tab = {
        "map_x": ((src_x + 1) / 2 * W - 0.5).astype(np.float32),
        "map_y": ((src_y + 1) / 2 * H - 0.5).astype(np.float32),
    }
    # soft-edged screen mask with rounded corners
    edge = np.maximum(np.abs(src_x), np.abs(src_y))
    mask = np.clip((1.0 - edge) * 260, 0, 1).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.2)
    tab["mask"] = mask
    tab["rim"] = (mask * (1 - mask) * 4) ** 2
    r2 = (nx * 1.0) ** 2 + (ny * 1.1) ** 2
    tab["vignette"] = np.clip(1 - 0.42 * r2 ** 1.4, 0, 1).astype(np.float32)
    tab["sheen"] = (0.045 * np.exp(-(((xx - 0.30 * W) / 620) ** 2 + ((yy - 0.18 * H) / 260) ** 2))).astype(np.float32)
    tab["scan"] = np.tile(np.array([1.0, 1.0, 0.80, 0.56], np.float32), H // 4 + 1)[:H, None]
    tab["noise"] = [np.random.default_rng(s + off).normal(0, 0.018, (H, W)).astype(np.float32) for s in range(6)]
    return tab


GLASS = np.array([0.010, 0.030, 0.018], np.float32)
BEZEL = np.array([0.030, 0.032, 0.030], np.float32)


def crt(I, t, frame, off=0):
    tab = _crt_tables(off)
    small = cv2.resize(I, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    big = cv2.resize(cv2.GaussianBlur(small, (0, 0), 7), (W, H), interpolation=cv2.INTER_LINEAR)
    mid = cv2.GaussianBlur(I, (0, 0), 3.0)
    soft = cv2.GaussianBlur(I, (0, 0), 1.1)
    beam = 0.75 * soft + 0.45 * mid + 0.75 * big
    y0 = (t * 380) % (H + 500) - 250
    band = 1 + 0.06 * np.exp(-(((np.arange(H, dtype=np.float32) - y0) / 110) ** 2))[:, None]
    r = np.random.default_rng(frame + off)
    flicker = 1 + 0.022 * np.sin(t * 2 * np.pi * 13.7) + 0.015 * r.normal()
    noise = tab["noise"]
    beam = beam * tab["scan"] * band * flicker + noise[frame % len(noise)] * (0.6 + beam)
    beam = cv2.remap(beam, tab["map_x"], tab["map_y"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    beam = np.maximum(beam, 0) * tab["vignette"]
    hot = np.clip(beam - 0.72, 0, 1.5)
    rgb = np.empty((H, W, 3), np.float32)
    rgb[..., 0] = 0.24 * beam + 0.65 * hot
    rgb[..., 1] = 1.0 * beam
    rgb[..., 2] = 0.42 * beam + 0.45 * hot
    rgb += GLASS + tab["sheen"][..., None]
    m = tab["mask"][..., None]
    rim = tab["rim"][..., None]
    rgb = rgb * m + (BEZEL + 0.05 * rim) * (1 - m) + 0.05 * rim
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


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
    def __init__(self, seed_off=0):
        self.dry = np.zeros((2, N))
        self.wet = np.zeros((2, N))
        self.off = seed_off

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
        r = np.random.default_rng(77 + self.off)
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


REF_LIT = 216  # lit cells in ROBOTS REVENGE; busier signs get quieter blips


def build_audio(sc):
    """The whole soundtrack, (2, N) floats peaking at -1.5 dBFS."""
    r = np.random.default_rng(3 + sc.off)
    m = Mix(sc.off)
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
    for text, times, db in ((sc.l1, sc.T1, -15), (sc.l2, sc.T2, -15), (sc.l4, sc.T4, -14)):
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
    n_lit = sum(1 for c in sc.cells if c[2])
    blip_db = -22 - max(0.0, 10 * np.log10(max(n_lit, 1) / REF_LIT))
    for (x, y, lit, t_on, seed) in sc.cells:
        if lit:
            u = (t_on - TITLE_T0) / TITLE_FILL
            m.add(bleep(600 * 4 ** u * r.uniform(0.95, 1.05), 0.012, "sine", 0.005), t_on, blip_db,
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
    for ch, tc in zip(sc.sub, sc.TS):
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


def _params(title, villain, boot_lines, subtitle, os_name, mode, clock, ctx):
    return dict(title=title, villain=villain, boot_lines=tuple(boot_lines) if boot_lines else None,
                subtitle=subtitle, os_name=os_name, mode=mode, clock=clock, fps=ctx.fps, seed=ctx.seed)


def render(out, *, title=DEFAULT_TITLE, villain=DEFAULT_VILLAIN, boot_lines=None,
           subtitle=DEFAULT_SUBTITLE, os_name=None, mode=DEFAULT_MODE, clock="21:00", ctx=None) -> Path:
    """Render the 6-second boot-up title to `out` (mp4) and return its path.

    title       the film's name, shown as a giant dot-matrix sign (1-3 lines, auto-sized)
    villain     builds the default boot log ("> BOOTING ROBOT_PROTOCOL.EXE ...") and OS name
    boot_lines  up to 3 lines (boot, loading, alert) replacing the defaults, in order
    subtitle    typed under the title ("> " is added if missing)
    os_name     header left (default from villain: "ROBO-OS 9000"); mode: header centre
    clock       "HH:MM" the header clock ticks up to as the title appears
    """
    ctx = ctx or Ctx()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    params = _params(title, villain, boot_lines, subtitle, os_name, mode, clock, ctx)
    n = ctx.nframes(DUR)
    seconds = n / ctx.fps
    audio = build_audio(Scene(**params))[:, :int(round(seconds * SR))]
    with tempfile.TemporaryDirectory(dir=ctx.scratch("title_computer")) as tmp:
        wav = Path(tmp) / "title_computer.wav"
        _write_wav(wav, audio)
        _encode(out, _frames(params, n, ctx.workers), wav, ctx.fps, seconds)
    return out


STILL_TIMES = (0.16, 0.9, 2.0, 3.0, 3.44, 3.8, 4.6, 5.3, 5.75)


def stills(out_dir, times=STILL_TIMES, *, title=DEFAULT_TITLE, villain=DEFAULT_VILLAIN, boot_lines=None,
           subtitle=DEFAULT_SUBTITLE, os_name=None, mode=DEFAULT_MODE, clock="21:00", ctx=None):
    """PNG frames at a few key moments, for checking a look without encoding the video."""
    ctx = ctx or Ctx()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = Scene(**_params(title, villain, boot_lines, subtitle, os_name, mode, clock, ctx))
    paths = []
    for t in times:
        f = min(int(round(t * ctx.fps)), int(round(DUR * ctx.fps)) - 1)
        p = out_dir / f"C_{f:03d}.png"
        Image.fromarray(scene.frame(f)).save(p)
        paths.append(p)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kms.render.title_computer",
                                 description="Opening title: a villain computer boots up and shows the title.")
    ap.add_argument("out", help="output .mp4")
    ap.add_argument("--title", default=DEFAULT_TITLE)
    ap.add_argument("--villain", default=DEFAULT_VILLAIN)
    ap.add_argument("--subtitle", default=DEFAULT_SUBTITLE)
    ap.add_argument("--boot-line", action="append", dest="boot_lines", metavar="LINE",
                    help="replace a boot-log line (repeat up to 3 times)")
    ap.add_argument("--fps", type=int, default=25, choices=(25, 30))
    ap.add_argument("--frames", type=int, default=None, help="render only the first N frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stills", metavar="DIR", help="write PNG stills of key moments instead of the video")
    a = ap.parse_args(argv)
    ctx = Ctx(fps=a.fps, seed=a.seed, limit_frames=a.frames)
    kw = dict(title=a.title, villain=a.villain, subtitle=a.subtitle, boot_lines=a.boot_lines, ctx=ctx)
    if a.stills:
        for p in stills(a.stills, **kw):
            print("wrote", p)
        return
    print("wrote", render(a.out, **kw))


if __name__ == "__main__":
    main()
