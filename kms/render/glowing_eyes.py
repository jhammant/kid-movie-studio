"""Glowing eyes: a toy's eyes glow red in time with a sinister voice (the post-credits scene).

The camera finds the toy in a close-up, its two dark round eye lenses light up red, and it
answers the hero: "We'll see, Penny... We'll see...".

Two pieces, both offline and re-runnable:

  sinister_voice  text-to-speech -> pitch down -> ring mod + metallic comb + light crush/drive +
                  flanger + EQ -> dark echo + reverb tail. Checked with the local whisper CLI
                  when it's installed (does the hero's name come through?).
  render          finds the two eye lenses in the close-up (dark blob -> holes filled ->
                  distance-transform peak = centre + radius), seeded on the steadiest frame and
                  tracked forwards and backwards, then smoothed; draws a red glow (iris gradient,
                  hot pupil, two-stage bloom, red spill on the face) driven by the voice's
                  loudness. Sound: the clip's own track (2-frame fade-in, ducked under the voice)
                  + the voice + a low drone. No eyes in the shot? The voice still plays, unlit.

    python -m kms.render.glowing_eyes OUT --clip F [--voice WAV | --line "..." --hero NAME]
        [--voice-at S] [--fps 25] [--frames N] [--stills DIR]
"""
import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import scipy.signal as ss

from kms import media, tts
from kms.render import H, SR, W, Ctx
from kms.render.voices import tts_line

DEFAULT_LINE = "We'll see, Penny... We'll see..."
DEFAULT_HERO = "Penny"
REF_FPS = 25             # the per-frame glow smoothing was tuned at 25 fps; 30 is scaled to match

# Timing (seconds in the source clip)
VOICE_DELAY = 0.8        # the toy speaks this long after the camera settles
TAIL_AFTER = 1.0         # hold this long after the echo tail dies (or stop at the clip's end)
AUDIO_FADE_FRAMES = 2    # fade the clip's sound in over 2 frames, so a cut-off word can't click
DUCK_DB = -9.0           # the clip's own sound under the voice
TRACK_LEAD = 1.0         # start tracking the eyes this long before the voice

# Voice
PAUSE_MS = 1100          # silence between the two halves of the line (plus say's own ellipsis pause)
TTS_VOICE = "Daniel"
TTS_RATE = 115
SEMITONES = 4.0          # pitch down
SLOWDOWN = 1.10          # extra drawl: output duration / tts duration
SUB_MIX = 0.18           # sub-octave doubling (menace); 0 to disable
RING_HZ = 36.0
RING_MIX = 0.5
CRUSH_BITS, CRUSH_MIX = 7, 0.15
DRIVE = 1.6
ECHOES = ((0.35, 0.22), (0.70, 0.10))   # (delay s, gain) - kept low so words don't "repeat"
ECHO_LP_HZ = 1800
REVERB_RT, REVERB_MIX = 2.0, 0.28
VOICE_PEAK_DB = -1.0     # the voice wav on its own
VOICE_REL_DB = 1.0       # voice level vs the clip's own speech (95th-pct 50 ms RMS)
QUIET_DB = -40.0         # clip sound quieter than this before the voice is room tone, not talking...
STAND_IN_DB = -20.0      # ...so the voice is levelled as if someone had spoken at this level
DRONE_DB = -27.0         # RMS of the drone bed; None to disable
MIX_PEAK_DB = -1.5

# Eyes
EYE_THR = 70             # a lens pixel's brightest channel is below this (8-bit)
EYE_THRS = (EYE_THR, 50, 100)   # tried in turn when looking for the eyes with no previous fit
HOLE_MAX = 1.0           # fill holes (lens highlights) up to this x the lens area, never a whole face
SEED_HOLE_FRAC = 0.01    # ...and with no lens yet, holes up to this fraction of the frame
GLOW_SMOULDER = 0.22     # level held between the phrases
GLOW_END_LEVEL = 0.14    # faint smoulder left at the hard cut

_EYES_WORK = "glowing_eyes"


def run(cmd, **kw):
    return media.run(cmd, **kw)


def peak(x):
    return np.abs(x).max() + 1e-12


def rms_frames(x, hop, win):
    n = max(1, (len(x) - win) // hop + 1)
    return np.array([np.sqrt((x[i * hop:i * hop + win] ** 2).mean()) if len(x[i * hop:i * hop + win]) else 0.0
                     for i in range(n)])


# --------------------------------------------------------------------------- voice
def split_line(line):
    """The line's two phrases, split after the first '...': "We'll see, Penny... We'll see..."
    -> ("We'll see, Penny...", "We'll see..."). A line with no '...' break is one phrase."""
    line = line.replace("…", "...").strip()
    i = line.find("...")
    if i < 0 or not line[i + 3:].strip(" ."):
        return (line,)
    return line[:i + 3].strip(), line[i + 3:].strip()


def speak_line(line, hero, out_wav):
    """The line as plain text-to-speech (48 kHz mono wav), with a PAUSE_MS gap between its phrases."""
    parts = [tts_line(p, hero) for p in split_line(line)]
    out_wav = Path(out_wav)
    if tts.engine() == "say" or len(parts) == 1:
        text = f" [[slnc {PAUSE_MS}]] ".join(parts) if tts.engine() == "say" else parts[0]
        return tts.speak(text, out_wav, voice=TTS_VOICE, rate=TTS_RATE)
    # espeak has no silence command: speak the phrases separately and put the pause in between
    chunks = []
    for i, p in enumerate(parts):
        wav = tts.speak(p, out_wav.with_name(f"{out_wav.stem}_{i}.wav"), voice=TTS_VOICE, rate=TTS_RATE)
        chunks.append(media.read_wav(wav)[0][:, 0])
        if i < len(parts) - 1:
            chunks.append(np.zeros(int(PAUSE_MS / 1000 * SR), np.float32))
    media.write_wav(out_wav, np.concatenate(chunks))
    return out_wav


def ff_pitch(path, semis, slowdown):
    """Pitch-shift down by `semis`, output duration = input * slowdown. Mono float64 @ SR."""
    k = 2 ** (-semis / 12)
    tempo = (1 / k) / slowdown
    af = f"aresample={SR},asetrate={SR * k:.4f},aresample={SR}"
    while tempo > 2.0:          # keep each atempo stage in its sweet spot
        af += ",atempo=2.0"
        tempo /= 2.0
    if abs(tempo - 1) > 1e-4:
        af += f",atempo={tempo:.5f}"
    raw = run([media.ffmpeg(), "-v", "error", "-i", path, "-af", af, "-ac", "1", "-f", "f64le", "-"],
              capture_output=True).stdout
    return np.frombuffer(raw, np.float64).copy()


def ff_filter(x, af):
    p = run([media.ffmpeg(), "-v", "error", "-f", "f64le", "-ar", str(SR), "-ac", "1", "-i", "-",
             "-af", af, "-f", "f64le", "-"], input=x.astype(np.float64).tobytes(), capture_output=True)
    y = np.frombuffer(p.stdout, np.float64).copy()
    return y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def robotise(x):
    t = np.arange(len(x)) / SR
    x = x / peak(x)
    ring = x * np.sin(2 * np.pi * RING_HZ * t) * 1.4
    y = (1 - RING_MIX) * x + RING_MIX * ring
    d = int(0.009 * SR)                      # metallic 9 ms comb ("tin can")
    c = y.copy()
    for _ in range(3):
        c[d:] += 0.45 * c[:-d]
    y = 0.6 * y + 0.4 * c / peak(c) * peak(y)
    q = 2 ** (CRUSH_BITS - 1)                # light bit-crush
    cr = np.round(y / peak(y) * q) / q * peak(y)
    y = (1 - CRUSH_MIX) * y + CRUSH_MIX * cr
    y = np.tanh(DRIVE * y / peak(y)) / np.tanh(DRIVE)
    return ff_filter(y, "flanger=delay=2:depth=3:regen=-20:width=60:speed=0.35,"
                        "highpass=f=80,equalizer=f=2500:t=q:w=1.0:g=4,"
                        "equalizer=f=180:t=q:w=1.0:g=3,lowpass=f=7500")


def dark_tail(y, seed=0):
    pad = int((REVERB_RT * 1.6 + 1.2) * SR)
    y = np.concatenate([y, np.zeros(pad)])
    out = y.copy()
    bl, al = ss.butter(2, ECHO_LP_HZ / (SR / 2), "low")
    dark = ss.lfilter(bl, al, y)
    for dt, g in ECHOES:
        n = int(dt * SR)
        out[n:] += dark[:-n] * g
    n = int(REVERB_RT * SR)
    tt = np.arange(n) / SR
    ir = np.random.default_rng(7 + seed).standard_normal(n) * np.exp(-6.9 * tt / REVERB_RT)
    b, a = ss.butter(2, 2200 / (SR / 2), "low")
    ir = ss.lfilter(b, a, ir)
    ir /= np.sqrt((ir ** 2).sum())
    wet = ss.fftconvolve(out, ir)[:len(out)]
    wet *= np.sqrt((out ** 2).mean()) / (np.sqrt((wet ** 2).mean()) + 1e-12)
    return (1 - REVERB_MIX) * out + REVERB_MIX * wet


def build_voice(tts_wav, seed=0):
    """Returns (wet, dry): both normalised to the same scale and aligned (start = first word)."""
    x = ff_pitch(tts_wav, SEMITONES, SLOWDOWN)
    if SUB_MIX > 0:
        sub = ff_pitch(tts_wav, SEMITONES + 12, SLOWDOWN)
        n = min(len(x), len(sub))
        x = x[:n] / peak(x) + SUB_MIX * sub[:n] / peak(sub)
    dry = robotise(x)
    wet = dark_tail(dry, seed)
    dry = np.pad(dry, (0, len(wet) - len(dry)))
    env = np.abs(wet)
    i0 = max(0, int(np.argmax(env > 0.02 * env.max())) - int(0.02 * SR))
    wet, dry = wet[i0:], dry[i0:]
    win = int(0.02 * SR)
    e = rms_frames(wet, win, win)
    last = np.nonzero(e > e.max() * 10 ** (-60 / 20))[0][-1]
    end = (last + 1) * win
    wet, dry = wet[:end].copy(), dry[:end].copy()
    fade = int(0.15 * SR)
    wet[-fade:] *= np.linspace(1, 0, fade)
    g = 1 / peak(wet)
    return wet * g, dry * g


def load_voice_file(path):
    """A voice wav (any rate/channels): returns (stereo, mono) aligned so t=0 is the first sound,
    both scaled so the stereo peak is 1. The mono copy drives the eye glow."""
    raw = run([media.ffmpeg(), "-v", "error", "-i", path, "-vn", "-ac", "2", "-ar", str(SR),
               "-f", "f64le", "-"], capture_output=True).stdout
    st = np.frombuffer(raw, np.float64).reshape(-1, 2).copy()
    if len(st) == 0 or np.abs(st).max() < 1e-6:
        raise ValueError(f"the voice file {path} is silent")
    mono = st.mean(1)
    env = np.abs(mono)
    i0 = max(0, int(np.argmax(env > 0.02 * env.max())) - int(0.02 * SR))
    st, mono = st[i0:], mono[i0:]
    g = 1 / peak(st)
    return st * g, mono * g


def tail_death(wet, db=-50):
    """Seconds into `wet` where the echo tail falls below `db` re its peak RMS."""
    win = int(0.02 * SR)
    e = rms_frames(wet, win, win)
    return (np.nonzero(e > e.max() * 10 ** (db / 20))[0][-1] + 1) * win / SR


def speech_span(dry):
    win = int(0.02 * SR)
    e = rms_frames(dry, win, win)
    act = np.nonzero(e > e.max() * 0.1)[0]
    return act[0] * win / SR, (act[-1] + 1) * win / SR


def save_wav(path, mono, peak_db=VOICE_PEAK_DB):
    y = mono / peak(mono) * 10 ** (peak_db / 20)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    media.write_wav(path, np.stack([y, y], 1))


def whisper_model():
    """The whisper model to check with: the one the voice was tuned against (small.en) or whichever
    is already downloaded. None if none is, so a check never starts a big download."""
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "whisper"
    for m in ("small.en", "small", "base.en", "base", "medium.en", "medium", "turbo", "tiny.en", "tiny"):
        if (cache / f"{m}.pt").exists():
            return m
    return None


def whisper_check(wav, hero=None):
    """(transcript, heard) from the local whisper CLI, where heard says whether the hero's name is in it
    (None with no hero). None when whisper or a downloaded model isn't there. Never raises."""
    exe, model = shutil.which("whisper"), whisper_model()
    if not exe or not model:
        return None
    with tempfile.TemporaryDirectory() as d:
        cmd = [exe, str(wav), "--model", model, "--output_dir", d,
               "--fp16", "False", "--verbose", "False"]
        if not model.endswith(".en"):
            cmd += ["--language", "en"]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
        except (OSError, subprocess.SubprocessError):
            return None
        txt = next(Path(d).glob("*.txt"), None)
        if txt is None:
            return None
        text = txt.read_text().strip().replace("\n", " ")
    heard = bool(re.search(rf"\b{re.escape(hero)}", text, re.I)) if hero else None
    return text, heard


def sinister_voice(out_wav, *, line=DEFAULT_LINE, hero=DEFAULT_HERO, seed=0):
    """Write the villain's line as a sinister robot voice: a stereo 48 kHz wav peaking at -1 dBFS,
    starting on the first word and ending when the echo tail dies.

    line   split after its first '...' into two phrases with a pause between them
    hero   the name in the line, spelled out so the deep voice says it clearly
    seed   for the reverb tail, so a re-render sounds the same
    """
    out_wav = Path(out_wav)
    with tempfile.TemporaryDirectory() as tmp:
        raw = speak_line(line, hero, Path(tmp) / "tts.wav")
        wet, dry = build_voice(raw, seed)
    save_wav(out_wav, wet)
    s0, s1 = speech_span(dry)
    print(f"voice: {len(wet) / SR:.2f}s (words {s0:.2f}-{s1:.2f}s, tail dies {tail_death(wet):.2f}s) -> {out_wav}")
    if shutil.which("whisper"):
        check = whisper_check(out_wav, hero)
        if check is None:
            print("whisper: no model downloaded (or it failed), so the words weren't checked")
        else:
            text, heard = check
            note = ""
            if hero:
                note = (f"; the name {hero!r} came through" if heard else
                        f"; it didn't catch {hero!r} - have a listen, and try spelling the name differently")
            print(f"whisper hears: {text!r}{note}")
    return out_wav


# --------------------------------------------------------------------------- eye finding
def _lum(img):
    """The brightest channel: a lens is dark in all three."""
    return img if img.ndim == 2 else img.max(axis=2)


def fill_holes(mask, max_area=None):
    """Fill the holes inside a mask's blobs (a lens's highlight). With max_area, only holes up to
    that many pixels, so a light face ringed by a dark background is never swallowed whole."""
    if max_area is None:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, cnts, -1, 1, -1)
        return filled
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    filled = mask.copy()
    if hier is not None:
        small = [c for c, h in zip(cnts, hier[0]) if h[3] >= 0 and cv2.contourArea(c) <= max_area]
        if small:
            cv2.drawContours(filled, small, -1, 1, -1)
    return filled


def detect_eyes(img, prev, thr=EYE_THR):
    """Refine ((cx,cy,r),(cx,cy,r)) on `img`: dark lens blob containing each previous centre,
    holes filled, distance-transform maximum = lens centre, its value = lens radius."""
    lum = _lum(img)
    (x1, y1, r1), (x2, y2, r2) = prev
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    span = abs(x2 - x1) + r1 + r2
    x0, xa = int(max(0, cx - span * 0.9)), int(min(lum.shape[1], cx + span * 0.9))
    y0, ya = int(max(0, cy - span * 0.6)), int(min(lum.shape[0], cy + span * 0.6))
    roi = lum[y0:ya, x0:xa]
    if roi.size == 0:
        return tuple(prev)
    dark = (roi < thr).astype(np.uint8)
    k = max(3, int(max(r1, r2) * 0.12)) | 1
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    _, lab = cv2.connectedComponents(dark)
    h, w = dark.shape
    keep = np.zeros_like(dark)
    for px, py, pr in prev:
        lx = min(max(int(round(px - x0)), 0), w - 1)
        ly = min(max(int(round(py - y0)), 0), h - 1)
        r = max(2, int(pr * 0.4))
        patch = lab[max(0, ly - r):ly + r, max(0, lx - r):lx + r].ravel()
        patch = patch[patch > 0]
        if len(patch):
            keep |= (lab == np.bincount(patch).argmax()).astype(np.uint8)
    filled = fill_holes(keep, HOLE_MAX * math.pi * max(r1, r2) ** 2)
    dt = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    out = []
    for px, py, pr in prev:
        lx, ly = px - x0, py - y0
        rr = max(3, int(pr * 0.6))
        xs = slice(int(max(0, lx - rr)), int(min(w, lx + rr)))
        ys = slice(int(max(0, ly - rr)), int(min(h, ly + rr)))
        sub = dt[ys, xs]
        if sub.size == 0 or sub.max() < pr * 0.75 or sub.max() > pr * 1.25:
            out.append((px, py, pr))          # lost it this frame: hold the last good fit
            continue
        m = sub >= sub.max() * 0.92
        yy, xx = np.nonzero(m)
        wts = sub[m]
        out.append(((xx * wts).sum() / wts.sum() + xs.start + x0,
                    (yy * wts).sum() / wts.sum() + ys.start + y0, float(sub.max())))
    return tuple(out)


def _ring_light(dark, x, y, r):
    """(share of the ring 1.15r..1.6r around a disc that's light, share of that ring inside the frame)."""
    h, w = dark.shape
    R = 1.6 * r
    xa, xb = int(max(0, x - R)), int(min(w, x + R + 1))
    ya, yb = int(max(0, y - R)), int(min(h, y + R + 1))
    yy, xx = np.mgrid[ya:yb, xa:xb]
    d = np.hypot(xx - x, yy - y)
    ring = (d >= 1.15 * r) & (d <= R)
    n = int(ring.sum())
    full = math.pi * (R ** 2 - (1.15 * r) ** 2)
    if n == 0:
        return 0.0, 0.0
    return float((dark[ya:yb, xa:xb][ring] == 0).mean()), n / full


def _seed_pair(lum, thr):
    """Both eye lenses on a frame with no previous fit, or None. The same recipe as detect_eyes
    over the whole frame: dark pixels -> holes filled -> distance-transform peaks are the centres
    of dark discs (value = radius). A lens is a dark disc ringed by something lighter (the face);
    the eyes are the best pair of those: alike in size, side by side, a few radii apart."""
    h, w = lum.shape
    dark = (lum < thr).astype(np.uint8)
    k = max(3, int(min(h, w) * 0.009)) | 1
    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    filled = fill_holes(closed, SEED_HOLE_FRAC * h * w)
    dt = cv2.distanceTransform(np.pad(filled, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    rmin = max(4.0, min(h, w) * 0.012)
    nk = 2 * int(rmin) + 1
    peaks = (dt >= rmin) & (dt >= cv2.dilate(dt, np.ones((nk, nk), np.uint8)))
    ys, xs = np.nonzero(peaks)
    order = np.argsort(-dt[ys, xs])[:4000]
    discs = []                                   # biggest first; one per disc
    for j in order:
        x, y, r = float(xs[j]), float(ys[j]), float(dt[ys[j], xs[j]])
        if all((x - a) ** 2 + (y - b) ** 2 >= max(r, c) ** 2 for a, b, c in discs):
            discs.append((x, y, r))
    lenses = []
    for x, y, r in discs[:200]:
        light, inside = _ring_light(dark, x, y, r)
        if inside >= 0.6 and light >= 0.6:
            lenses.append((x, y, r, light))
    best, best_score = None, -1.0
    biggest = max((r for _, _, r, _ in lenses), default=1.0)
    for a, b in combinations(lenses, 2):
        (xa, ya, ra, la), (xb, yb, rb, lb) = sorted((a, b))
        rr = min(ra, rb) / max(ra, rb)
        dx, dy = xb - xa, yb - ya
        d, rm = math.hypot(dx, dy), (ra + rb) / 2
        if rr < 0.55 or not 2.2 <= d / rm <= 7.0 or abs(dy) > 0.6 * abs(dx):
            continue
        score = la + lb + rr + rm / biggest - abs(dy) / d
        if score > best_score:
            best, best_score = ((xa, ya, ra), (xb, yb, rb)), score
    return best


def _find(lum, thrs=EYE_THRS):
    for thr in thrs:
        seed = _seed_pair(lum, thr)
        if seed:
            return detect_eyes(lum, seed, thr), thr
    return None, None


def find_eyes(img, thr=None):
    """Find the two eye lenses on a frame (RGB or one channel) with no previous fit.

    Returns ((cx, cy, r), (cx, cy, r)), left eye first, in the frame's pixels, or None."""
    eyes, _ = _find(_lum(np.asarray(img)), (thr,) if thr else EYE_THRS)
    return tuple(tuple(float(v) for v in e) for e in eyes) if eyes else None


# --------------------------------------------------------------------------- the clip
def _cover(fps, w=W, h=H, flags="lanczos"):
    """Any clip -> fps, filling 1920x1080 (cropped, never stretched); then scaled to w x h."""
    vf = f"fps={fps},scale={W}:{H}:force_original_aspect_ratio=increase:flags={flags},crop={W}:{H}"
    return vf if (w, h) == (W, H) else vf + f",scale={w}:{h}:flags=area"


def _nice():
    return ["nice", "-n", "10"] if shutil.which("nice") else []


def motion_profile(clip, t0, fps, t1=None):
    """Camera motion per frame (mean |difference| at 160x90) from t0 to the clip's end (or t1)."""
    cmd = [media.ffmpeg(), "-v", "error", "-ss", f"{t0:.3f}", "-i", clip]
    if t1 is not None:
        cmd += ["-t", f"{max(t1 - t0, 1 / fps):.3f}"]
    cmd += ["-an", "-vf", f"fps={fps},scale=160:90:force_original_aspect_ratio=increase:flags=area,"
                          "crop=160:90,format=gray", "-f", "rawvideo", "-"]
    raw = run(cmd, capture_output=True).stdout
    f = np.frombuffer(raw, np.uint8)[:len(raw) // (160 * 90) * 160 * 90].reshape(-1, 90, 160)
    if len(f) == 0:
        raise ValueError(f"no video decoded from {clip} at {t0:.2f}s")
    d = np.abs(np.diff(f.astype(np.float32), axis=0)).mean((1, 2))
    return np.concatenate([d[:1], d]) if len(d) else np.zeros(1)


def _smooth(x, n):
    n = max(1, min(int(n), len(x)))
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), np.ones(n) / n, mode="valid")


def settle_frame(motion, fps):
    """Frame where the camera settles: the start of the last steady stretch at least 1 s long."""
    m = _smooth(motion, round(0.5 * fps))
    lo, hi = m.min(), np.percentile(m, 90)
    steady = m <= lo + max(0.25 * (hi - lo), 0.5)
    edges = np.diff(np.concatenate([[0], steady.astype(int), [0]]))
    runs = list(zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]))
    need = min(fps, max(1, len(m) // 3))
    long_runs = [r for r in runs if r[1] - r[0] >= need]
    if long_runs:
        return int(long_runs[-1][0])
    i = int(np.argmin(m))
    return int(next(a for a, b in runs if a <= i < b))


def steadiest_frame(motion, fps, i0, i1):
    """Frame in [i0, i1) where the camera moves least."""
    m = _smooth(motion, round(0.5 * fps))
    i0, i1 = max(0, min(i0, len(m) - 1)), max(1, min(i1, len(m)))
    return i0 + int(np.argmin(m[i0:max(i0 + 1, i1)]))


def _decode_cmd(clip, t0, n, fps, size, src_end):
    """ffmpeg reading n RGB frames from t0; past the clip's end its last frame is held."""
    vf = _cover(fps, *size)
    over = t0 + n / fps - src_end
    if over > 0:
        vf += f",tpad=stop_mode=clone:stop_duration={over + 1.0:.3f}"
    return [media.ffmpeg(), "-v", "error", "-ss", f"{t0:.3f}", "-i", clip, "-frames:v", str(n), "-an",
            "-vf", vf + ",format=rgb24", "-f", "rawvideo", "-"]


def decode_lum(clip, t0, t1, fps, src_end):
    """Half-res brightest-channel frames from t0 to t1 (the eyes are tracked at half res: plenty)."""
    n = max(1, int(round((t1 - t0) * fps)))
    w, h = W // 2, H // 2
    proc = subprocess.Popen([str(c) for c in _decode_cmd(clip, t0, n, fps, (w, h), src_end)],
                            stdout=subprocess.PIPE)
    frames = []
    while len(frames) < n:
        buf = proc.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        frames.append(np.frombuffer(buf, np.uint8).reshape(h, w, 3).max(axis=2))
    proc.stdout.close()
    proc.wait()
    while frames and len(frames) < n:
        frames.append(frames[-1])
    return frames


def track_eyes(frames, f0, seed_frames):
    """({frame: ((cx,cy,r),(cx,cy,r))} at 1920x1080 smoothed, seed frame), or ({}, None) with no eyes.

    frames are half-res brightest-channel frames, the first being frame f0; the eyes are looked
    for on each of seed_frames in turn and tracked forwards and backwards from the first hit."""
    seed = thr = None
    for sf in seed_frames:
        si = min(max(sf - f0, 0), len(frames) - 1)
        seed, thr = _find(frames[si])
        if seed:
            break
    if not seed:
        return {}, None
    raw = {}
    for step in (1, -1):
        prev, i = seed, si
        while 0 <= i < len(frames):
            prev = detect_eyes(frames[i], prev, thr)
            raw[i] = prev
            i += step
    idx = sorted(raw)
    arr = np.array([np.array(raw[i]).ravel() for i in idx]) * 2.0   # back to 1080p
    sm = arr.copy()
    for c in range(6):                  # centres: [1,2,1]; radii: 9-frame mean
        k = np.array([1, 2, 1]) / 4 if c % 3 != 2 else np.ones(9) / 9
        pad = len(k) // 2
        sm[:, c] = np.convolve(np.pad(arr[:, c], pad, mode="edge"), k, mode="valid")
    return {f0 + i: (tuple(sm[j, :3]), tuple(sm[j, 3:])) for j, i in enumerate(idx)}, f0 + si


# --------------------------------------------------------------------------- glow
def glow_curve(dry, voice_start, n_frames_total, fps, seed=0):
    """Per-source-frame glow intensity (dict frame -> g) from the dry voice envelope."""
    hop = SR // fps
    e = rms_frames(dry, hop, hop * 2)
    e = e / np.percentile(e[e > e.max() * 0.05], 90)
    e = np.clip(e, 0, 1.25)
    s0, s1 = speech_span(dry)
    first, last = int(s0 * fps), int(np.ceil(s1 * fps))
    attack = 1 - (1 - 0.75) ** (REF_FPS / fps)    # quick attack, slow release (per 25 fps frame)
    release = 1 - (1 - 0.22) ** (REF_FPS / fps)
    g = np.zeros(len(e) + 4 * fps)
    level = 0.0
    rng = np.random.default_rng(3 + seed)
    for i in range(len(g)):
        tgt = e[i] if i < len(e) else 0.0
        if first <= i <= last:
            tgt = max(tgt, GLOW_SMOULDER)
        elif i > last:
            fall = min(1.0, (i - last) / (0.7 * fps))
            floor = GLOW_SMOULDER + (GLOW_END_LEVEL - GLOW_SMOULDER) * fall
            tgt = max(tgt * 0.6, floor)
        coef = attack if tgt > level else release
        if i < first:
            tgt, coef = 0.0, 1.0
        level += (tgt - level) * coef
        g[i] = level * (1 + 0.035 * rng.standard_normal())
    # fade-in on the first word: never jump to full on frame one
    nfade = max(1, round(4 * fps / REF_FPS))
    for j in range(nfade):
        if first + j < len(g):
            g[first + j] *= (j + 1) / (nfade + 1)
    v0 = int(round(voice_start * fps))
    return {v0 + i: float(max(0.0, x)) for i, x in enumerate(g) if v0 + i < n_frames_total}


def _soft_disc(d, r_in=0.90, r_out=1.0):
    t = np.clip((r_out - d) / (r_out - r_in), 0, 1)
    return t * t * (3 - 2 * t)


def apply_glow(frame, eyes, g):
    """Additive red glow in (approximately) linear light, on a ROI around both eyes:
    deep-red iris inside the lens bezel, a defined bright pupil with a white-hot core,
    two-stage bloom and a little red spill on the face."""
    if g <= 0.003:
        return frame
    rmax = max(e[2] for e in eyes)
    xs, ys = [e[0] for e in eyes], [e[1] for e in eyes]
    m = rmax * 3.4
    x0, x1 = int(max(0, min(xs) - m)), int(min(W, max(xs) + m))
    y0, y1 = int(max(0, min(ys) - m)), int(min(H, max(ys) + m))
    if x1 <= x0 or y1 <= y0:
        return frame
    roi = frame[y0:y1, x0:x1].astype(np.float32) / 255.0
    lin = roi ** 2.2
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    iris = np.zeros(roi.shape[:2], np.float32)
    halo = np.zeros(roi.shape[:2], np.float32)
    pupil = np.zeros(roi.shape[:2], np.float32)
    hot = np.zeros(roi.shape[:2], np.float32)
    for cx, cy, r in eyes:
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
        iris += _soft_disc(d, 0.84, 0.96) * (0.10 + 0.90 * np.exp(-(d / 0.55) ** 2))
        halo += np.exp(-(d / 0.42) ** 2)
        pupil += _soft_disc(d, 0.21, 0.27)
        hot += np.exp(-(d / 0.10) ** 2)
    light = pupil + 0.5 * halo + 0.3 * iris
    b1 = cv2.GaussianBlur(light, (0, 0), rmax * 0.30)
    b2 = cv2.GaussianBlur(light, (0, 0), rmax * 1.30)
    deep = np.array([0.85, 0.02, 0.01], np.float32)
    red = np.array([1.0, 0.05, 0.02], np.float32)
    pup = np.array([1.0, 0.10, 0.04], np.float32)
    white = np.array([1.0, 0.58, 0.46], np.float32)
    add = (iris[..., None] * deep * (0.30 * g)
           + halo[..., None] * red * (0.45 * g)
           + pupil[..., None] * pup * (1.30 * g)
           + hot[..., None] * white * (0.85 * g ** 1.5)
           + b1[..., None] * red * (0.35 * g)
           + b2[..., None] * red * (0.45 * g))
    # red light falling on the face: lift R (additive) and pull G/B down a touch
    spill = np.clip(b2 * g * 1.0, 0, 0.40)[..., None]
    lin = lin * (1 - spill * np.array([0.0, 0.5, 0.55], np.float32))
    lin = np.clip(lin + add, 0, 1)
    frame[y0:y1, x0:x1] = (lin ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)
    return frame


# --------------------------------------------------------------------------- audio
def load_src_audio(clip, t0, t1):
    """The clip's own sound from t0 to t1 (stereo, SR); silence where it has none."""
    raw = subprocess.run([media.ffmpeg(), "-v", "error", "-ss", f"{t0:.4f}", "-t", f"{t1 - t0:.4f}",
                          "-i", str(clip), "-vn", "-ac", "2", "-ar", str(SR), "-f", "f64le", "-"],
                         capture_output=True).stdout
    a = np.frombuffer(raw[:len(raw) // 16 * 16], np.float64).reshape(-1, 2).copy()
    n = int(round((t1 - t0) * SR))
    return a[:n] if len(a) >= n else np.pad(a, ((0, n - len(a)), (0, 0)))


def drone(n, start, stop, seed=0):
    t = np.arange(n) / SR
    y = np.zeros(n)
    for f, a in ((41.2, 1.0), (41.5, 0.8), (61.7, 0.45), (82.4, 0.35), (123.5, 0.12)):
        y += a * np.sin(2 * np.pi * f * t + f)
    y *= 1 + 0.25 * np.sin(2 * np.pi * 0.23 * t)
    rumble = np.cumsum(np.random.default_rng(11 + seed).standard_normal(n))
    b, a = ss.butter(2, [25 / (SR / 2), 140 / (SR / 2)], "band")
    rumble = ss.lfilter(b, a, rumble)
    y = y / peak(y) + 0.6 * rumble / peak(rumble)
    y = np.tanh(1.5 * y)
    env = np.zeros(n)
    i0, i1 = int(start * SR), int(stop * SR)
    ramp = int(1.4 * SR)
    env[i0:i1] = 1
    env[i0:i0 + ramp] = np.linspace(0, 1, max(0, min(ramp, n - i0)))[:len(env[i0:i0 + ramp])]
    y *= env
    rms = np.sqrt((y[i0 + ramp:i1] ** 2).mean()) if i1 > i0 + ramp else peak(y)
    return y / (rms + 1e-12) * 10 ** (DRONE_DB / 20)


def speech_level_db(x):
    if len(x) == 0:
        return -240.0
    hop = int(0.05 * SR)
    return 20 * np.log10(np.percentile(rms_frames(x, hop, hop), 95) + 1e-12)


def mix_audio(src, voice, vs, fps, seed=0):
    """The clip's sound (faded in, ducked under the voice) + the voice at `vs` s + the drone,
    peaking at MIX_PEAK_DB."""
    n = len(src)
    t = np.arange(n) / SR
    fi = int(AUDIO_FADE_FRAMES / fps * SR)
    src[:fi] *= np.linspace(0, 1, fi)[:len(src[:fi]), None]
    ramp0, ramp1 = vs - 0.35, vs - 0.05
    duck = np.where(t >= ramp0, np.interp(t, [ramp0, ramp1], [1, 10 ** (DUCK_DB / 20)]), 1)
    src *= duck[:, None]
    voice = voice if voice.ndim == 2 else np.stack([voice, voice], 1)
    clip_db = speech_level_db(src[:int(max(0.5, vs - 0.4) * SR)].mean(1))
    if clip_db < QUIET_DB:
        print(f"the clip is quiet before the voice ({clip_db:.0f} dBFS): levelling the voice as ordinary speech")
        clip_db = STAND_IN_DB
    gain_db = clip_db + VOICE_REL_DB - speech_level_db(voice.mean(1))
    v = np.zeros((n, 2))
    i = int(round(vs * SR))
    seg = voice[:max(0, n - i)]
    v[i:i + len(seg)] = seg * 10 ** (gain_db / 20)
    mix = src + v
    if DRONE_DB is not None:
        dr = drone(n, max(0, vs - 1.2), n / SR, seed)
        lag = int(0.011 * SR)                                    # a little width
        mix += np.stack([dr, np.concatenate([np.zeros(lag), dr[:-lag]])], 1)
    fo = min(int(0.02 * SR), n)                                  # de-click the hard cut
    mix[n - fo:] *= np.linspace(1, 0, fo)[:, None]
    return mix / peak(mix) * 10 ** (MIX_PEAK_DB / 20)


# --------------------------------------------------------------------------- render
def render(out, *, clip, voice_wav, voice_at=None, start=0.0, end=None, eye_seed_t=None, ctx=None,
           stills=None):
    """Render the glowing-eyes scene to `out` (house format) and return its Path.

    clip        the close-up of the toy (any size or frame rate; filled to 1920x1080)
    voice_wav   the voice, e.g. from sinister_voice(); its first sound lands at voice_at
    voice_at    when the toy speaks, in seconds of the clip (None: 0.8 s after the camera settles,
                early enough for the words to fit)
    start, end  the part of the clip to use, in seconds (end None: 1 s after the echo dies, or the
                clip's end; a clip too short for the words holds its last frame)
    eye_seed_t  a moment the eyes are clearly visible (None: when the camera is steadiest while
                the voice plays, then voice_at)
    stills      a folder to save check frames into (before the voice, its loudest moment, the end...)
    """
    ctx = ctx or Ctx()
    fps = int(ctx.fps)
    out, clip = Path(out), Path(clip)
    for p in (clip, Path(voice_wav)):
        if not p.exists():
            raise FileNotFoundError(p)
    wet, dry = load_voice_file(voice_wav)
    mono = wet.mean(1)
    s0, s1 = speech_span(dry)

    # timeline, in seconds of the source clip
    t_in = round(max(0.0, start) * fps) / fps
    motion = motion_profile(clip, t_in, fps, end + 1 / fps if end is not None else None)
    src_end = t_in + len(motion) / fps
    if voice_at is None:
        settle = t_in + settle_frame(motion, fps) / fps
        latest = src_end - (s1 + TAIL_AFTER)                       # the latest start the words fit
        voice_at = max(t_in + 0.4, min(settle + VOICE_DELAY, latest))
    v_start = round(max(voice_at, t_in) * fps) / fps
    if end is None:
        t_out = v_start + tail_death(mono) + TAIL_AFTER
        if t_out > src_end:                                        # never cut the words off
            t_out = max(src_end, v_start + s1 + TAIL_AFTER)
    else:
        t_out = end
    t_out = int(t_out * fps + 1e-6) / fps
    f_in = int(round(t_in * fps))
    n_frames = max(1, int(round((t_out - t_in) * fps)))
    n_render = min(n_frames, ctx.limit_frames) if ctx.limit_frames else n_frames
    print(f"clip: src {t_in:.2f}-{t_out:.2f}s = {n_frames} frames ({n_frames / fps:.2f}s); "
          f"voice at src {v_start:.2f}s = clip {v_start - t_in:.2f}s"
          + (f" (holding the last frame from {src_end:.2f}s)" if t_out > src_end + 1e-6 else ""))

    # eye tracking + glow curve (only if any rendered frame is lit)
    glow = glow_curve(dry, v_start, int(round(t_out * fps)), fps, ctx.seed)
    tracks, seed_frame = {}, None
    if f_in + n_render > int(round(v_start * fps)):
        t0 = round(max(t_in, v_start - TRACK_LEAD) * fps) / fps
        f0 = int(round(t0 * fps))
        frames = decode_lum(clip, t0, t_out, fps, src_end)
        f_v, f_last = int(round(v_start * fps)), min(int(round(t_out * fps)), int(round(src_end * fps)))
        steady = f_in + steadiest_frame(motion, fps, f_v - f_in, f_last - f_in)   # motion[0] is frame f_in
        wanted = [int(round(eye_seed_t * fps))] if eye_seed_t is not None else []
        wanted += [steady, f_v] + list(np.linspace(f0, f0 + len(frames) - 1, 7).astype(int))
        seen = []
        for f in wanted:
            if f not in seen and f0 <= f < f0 + len(frames):
                seen.append(f)
        tracks, seed_frame = track_eyes(frames, f0, seen) if frames else ({}, None)
        del frames
        if seed_frame is None:
            print(f"warning: no eyes found in {clip.name}: the voice plays over the clip with no glow",
                  file=sys.stderr)
        else:
            (lx, ly, lr), (rx, ry, rr) = tracks[seed_frame]
            print(f"eyes: found at src {seed_frame / fps:.2f}s, ({lx:.0f},{ly:.0f}) r{lr:.0f} and "
                  f"({rx:.0f},{ry:.0f}) r{rr:.0f}; tracked {len(tracks)} frames")

    # audio
    mix = mix_audio(load_src_audio(clip, t_in, t_out), wet, v_start - t_in, fps, ctx.seed)
    if n_render < n_frames:                                        # a short test render
        mix = mix[:n_render * SR // fps].copy()
        k = min(96, len(mix))
        mix[len(mix) - k:] *= np.linspace(1, 0, k)[:, None]

    # still frames to check by eye
    still_at = {}
    if stills:
        Path(stills).mkdir(parents=True, exist_ok=True)
        lit = {f: g for f, g in glow.items() if f_in <= f < f_in + n_frames}
        marks = [("before", v_start - 0.5), ("words", v_start + 0.4), ("end", t_out - 1 / fps)]
        if lit:
            loud = max(lit, key=lit.get)
            marks.append(("loudest", loud / fps))
            fw, fl = int((v_start + s0) * fps) + 5, int((v_start + s1) * fps) - 5
            inner = {f: g for f, g in lit.items() if fw <= f <= fl}
            if inner:
                marks.append(("pause", min(inner, key=inner.get) / fps))
        for name, ts in marks:
            f = int(round(ts * fps))
            if f_in <= f < f_in + n_render:
                still_at[f] = name

    # picture
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part.mp4")
    with tempfile.TemporaryDirectory(dir=ctx.scratch(_EYES_WORK)) as tmp:
        wav = Path(tmp) / "mix.wav"
        media.write_wav(wav, mix)
        dec = subprocess.Popen(_nice() + [str(c) for c in _decode_cmd(clip, t_in, n_render, fps, (W, H), src_end)],
                               stdout=subprocess.PIPE)
        enc = subprocess.Popen(_nice() + [str(c) for c in [
            media.ffmpeg(), "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
            "-i", wav, "-map", "0:v", "-map", "1:a",
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            *media.video_args(18),
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            *media.audio_args(), "-t", f"{n_render / fps:.3f}", "-movflags", "+faststart", part]],
            stdin=subprocess.PIPE)
        fb, last = W * H * 3, None
        try:
            for k in range(n_render):
                buf = dec.stdout.read(fb)
                if len(buf) == fb:
                    last = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
                elif last is None:
                    raise RuntimeError(f"no video decoded from {clip} at {t_in:.2f}s")
                fr = last.copy()
                sf = f_in + k
                g = glow.get(sf, 0.0)
                if g > 0 and sf in tracks:
                    fr = apply_glow(fr, tracks[sf], g)
                if sf in still_at:
                    cv2.imwrite(str(Path(stills) / f"{still_at[sf]}_{sf / fps:.2f}.png"),
                                cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
                enc.stdin.write(fr.tobytes())
        finally:
            enc.stdin.close()
            dec.stdout.close()
            dec.wait()
            if enc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed writing {out}")
    os.replace(part, out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--clip", required=True, help="the close-up of the toy")
    ap.add_argument("--voice", metavar="WAV", help="a ready-made voice (default: make one from --line)")
    ap.add_argument("--line", default=DEFAULT_LINE, help="what the toy says")
    ap.add_argument("--hero", default=DEFAULT_HERO, help="the name in the line, spelled out for the voice")
    ap.add_argument("--voice-at", type=float, help="when the toy speaks, seconds into the clip")
    ap.add_argument("--start", type=float, default=0.0, help="start this far into the clip")
    ap.add_argument("--end", type=float, help="end here in the clip (default: after the echo dies)")
    ap.add_argument("--eye-seed-t", type=float, help="a moment the eyes are clearly visible")
    ap.add_argument("--fps", type=int, default=25, choices=(25, 30))
    ap.add_argument("--frames", type=int, help="render only the first N frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stills", metavar="DIR", help="save check frames here")
    a = ap.parse_args(argv)
    ctx = Ctx(fps=a.fps, seed=a.seed, limit_frames=a.frames)
    voice = a.voice
    if not voice:
        out = Path(a.out)
        voice = sinister_voice(out.with_name(out.stem + "-voice.wav"), line=a.line, hero=a.hero, seed=a.seed)
    out = render(a.out, clip=a.clip, voice_wav=voice, voice_at=a.voice_at, start=a.start, end=a.end,
                 eye_seed_t=a.eye_seed_t, ctx=ctx, stills=a.stills)
    info = media.probe(out)
    print(f"{out}: {info.duration:.2f}s" if info else out)


if __name__ == "__main__":
    main()
