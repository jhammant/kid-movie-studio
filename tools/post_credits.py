#!/usr/bin/env python3
"""ROBOTS REVENGE - post-credits scene (robot-chair clip).

Penny Sparks refuses to bow; the camera finds the little white robot on
the office chair, its eyes glow red, and it answers: "We'll see, Penny... We'll see...."

Pipeline (all offline, re-runnable):
  1. voice   macOS `say` -> pitch down -> ring mod + metallic comb + light crush/drive +
             flanger + EQ -> dark echo + reverb tail. Saved alone to extras/ for iMovie.
             Checked with the local whisper CLI (skip with --no-whisper).
  2. track   find the robot's two black eye lenses in every close-up frame
             (dark blob -> filled component -> distance-transform peak = centre + radius),
             seeded at EYE_SEED_T and tracked forwards/backwards, then smoothed.
  3. render  source from IN_POINT scaled to 1080p; red eye glow (iris gradient, hot pupil,
             two-stage bloom, red spill) driven by the voice loudness envelope; audio =
             the kid's track (2-frame fade-in, ducked under the voice) + voice + low drone.

Usage:  python3 tools/post_credits.py            (full build)
        python3 tools/post_credits.py --voice-only
        python3 tools/post_credits.py --stills     (also dump check frames to WORK)
"""
import argparse
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as ss

# --------------------------------------------------------------------------- settings
FF = "ffmpeg"
SRC = "footage/robot-chair.mp4"
KIT = "kit"
OUT = f"{KIT}/08 Post-credits - robot.mp4"
VOICE_OUT = f"{KIT}/extras/Robot voice - We'll see Penny.wav"
WORK = os.path.join(tempfile.gettempdir(), "kms-post-credits")

W, H, FPS, SR = 1920, 1080, 25, 48000

# Timing (seconds in the SOURCE clip). "and action" ends ~1.1 s; the kid's first word is ~3.4 s.
IN_POINT = 1.52          # frame-aligned (frame 38)
SETTLE_T = 22.0          # camera settles on the robot close-up
VOICE_DELAY = 0.8        # robot speaks this long after the camera settles
TAIL_AFTER = 1.0         # hold this long after the echo tail dies (or stop at source end)
AUDIO_FADE_IN = 2 / FPS  # 2-frame fade-in so the cut-off "action" can't click
DUCK_DB = -9.0           # the kid's track under the robot (it's only room tone there)

# Voice
LINE1 = "We'll see, Penny..."
LINE2 = "We'll see..."
PAUSE_MS = 1100          # silence between the phrases (plus say's own ellipsis pause)
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
VOICE_REL_DB = 1.0       # robot speech level vs the kid's speech (95th-pct 50 ms RMS)
DRONE_DB = -27.0         # RMS of the drone bed; None to disable
MIX_PEAK_DB = -1.5

# Eyes: seed positions at 1920x1080 on a settled frame (verified by eye); tracked from here.
EYE_SEED_T = 22.8
EYE_SEED = ((944.0, 306.0, 84.0), (1194.0, 290.0, 66.0))  # (cx, cy, r) left, right
TRACK_START = 21.8
GLOW_SMOULDER = 0.22     # level held between the phrases
GLOW_END_LEVEL = 0.14    # faint smoulder left at the hard cut


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


# --------------------------------------------------------------------------- voice
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
    raw = run([FF, "-v", "error", "-i", path, "-af", af, "-ac", "1", "-f", "f64le", "-"],
              capture_output=True).stdout
    return np.frombuffer(raw, np.float64).copy()


def ff_filter(x, af):
    p = run([FF, "-v", "error", "-f", "f64le", "-ar", str(SR), "-ac", "1", "-i", "-",
             "-af", af, "-f", "f64le", "-"], input=x.astype(np.float64).tobytes(),
            capture_output=True)
    y = np.frombuffer(p.stdout, np.float64).copy()
    return y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def peak(x):
    return np.abs(x).max() + 1e-12


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


def dark_tail(y):
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
    ir = np.random.default_rng(7).standard_normal(n) * np.exp(-6.9 * tt / REVERB_RT)
    b, a = ss.butter(2, 2200 / (SR / 2), "low")
    ir = ss.lfilter(b, a, ir)
    ir /= np.sqrt((ir ** 2).sum())
    wet = ss.fftconvolve(out, ir)[:len(out)]
    wet *= np.sqrt((out ** 2).mean()) / (np.sqrt((wet ** 2).mean()) + 1e-12)
    return (1 - REVERB_MIX) * out + REVERB_MIX * wet


def rms_frames(x, hop, win):
    n = max(1, (len(x) - win) // hop + 1)
    return np.array([np.sqrt((x[i * hop:i * hop + win] ** 2).mean()) for i in range(n)])


def build_voice(work):
    """Returns (wet, dry): both normalised to the same scale and aligned (start = first word)."""
    tts = os.path.join(work, "tts.aiff")
    run(["say", "-v", TTS_VOICE, "-r", str(TTS_RATE), "-o", tts,
         f"{LINE1} [[slnc {PAUSE_MS}]] {LINE2}"])
    x = ff_pitch(tts, SEMITONES, SLOWDOWN)
    if SUB_MIX > 0:
        sub = ff_pitch(tts, SEMITONES + 12, SLOWDOWN)
        n = min(len(x), len(sub))
        x = x[:n] / peak(x) + SUB_MIX * sub[:n] / peak(sub)
    dry = robotise(x)
    wet = dark_tail(dry)
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
    """A pre-made voice WAV (any rate/channels): returns (stereo, mono) aligned so t=0 is the
    first sound, both scaled so the stereo peak is 1. The mono copy drives the eye glow."""
    raw = run([FF, "-v", "error", "-i", path, "-vn", "-ac", "2", "-ar", str(SR),
               "-f", "f64le", "-"], capture_output=True).stdout
    st = np.frombuffer(raw, np.float64).reshape(-1, 2).copy()
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


def save_wav(path, mono, peak_db=-1.0):
    y = mono / peak(mono) * 10 ** (peak_db / 20)
    st = np.stack([y, y], 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wavfile.write(path, SR, (st * 32767).astype(np.int16))


def whisper_check(path, work):
    d = os.path.join(work, "whisper")
    os.makedirs(d, exist_ok=True)
    subprocess.run(["whisper", path, "--model", "small.en", "--output_dir", d,
                    "--fp16", "False", "--verbose", "False"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    txt = os.path.join(d, os.path.basename(path) + ".txt")
    if not os.path.exists(txt):
        txt = os.path.join(d, os.path.splitext(os.path.basename(path))[0] + ".txt")
    return open(txt).read().strip().replace("\n", " ") if os.path.exists(txt) else "(no transcript)"


# --------------------------------------------------------------------------- eye tracking
def detect_eyes(rgb, prev, thr=70):
    """Refine ((cx,cy,r),(cx,cy,r)) on `rgb`: dark lens blob containing each previous centre,
    holes filled, distance-transform maximum = lens centre, its value = lens radius."""
    (x1, y1, r1), (x2, y2, r2) = prev
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    span = abs(x2 - x1) + r1 + r2
    x0 = int(max(0, cx - span * 0.9)); xa = int(min(rgb.shape[1], cx + span * 0.9))
    y0 = int(max(0, cy - span * 0.6)); ya = int(min(rgb.shape[0], cy + span * 0.6))
    roi = rgb[y0:ya, x0:xa]
    dark = (roi.max(axis=2) < thr).astype(np.uint8)
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
    cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(keep)
    cv2.drawContours(filled, cnts, -1, 1, -1)
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


def decode_range(t0, t1, w, h):
    n = int(round((t1 - t0) * FPS))
    p = run([FF, "-v", "error", "-ss", f"{t0:.3f}", "-i", SRC, "-frames:v", str(n), "-an",
             "-vf", f"scale={w}:{h}:flags=area:in_color_matrix=bt709:in_range=tv,format=rgb24",
             "-f", "rawvideo", "-"], capture_output=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w, 3)


def track_eyes(t_end):
    """{source_frame_index: ((cx,cy,r),(cx,cy,r))} at 1920x1080, smoothed."""
    t0 = round(TRACK_START * FPS) / FPS
    frames = decode_range(t0, t_end, W // 2, H // 2)   # track at half res (plenty)
    f0 = int(round(t0 * FPS))
    seed_i = int(round(EYE_SEED_T * FPS)) - f0
    seed = tuple((x / 2, y / 2, r / 2) for x, y, r in EYE_SEED)
    raw = {}
    for step in (1, -1):
        prev, i = seed, seed_i
        while 0 <= i < len(frames):
            prev = detect_eyes(frames[i], prev)
            raw[i] = prev
            i += step
    idx = sorted(raw)
    arr = np.array([np.array(raw[i]).ravel() for i in idx]) * 2.0   # back to 1080p
    sm = arr.copy()
    for c in range(6):                  # centres: [1,2,1]; radii: 9-frame mean
        k = np.array([1, 2, 1]) / 4 if c % 3 != 2 else np.ones(9) / 9
        pad = len(k) // 2
        sm[:, c] = np.convolve(np.pad(arr[:, c], pad, mode="edge"), k, mode="valid")
    return {f0 + i: (tuple(sm[j, :3]), tuple(sm[j, 3:])) for j, i in enumerate(idx)}


# --------------------------------------------------------------------------- glow
def glow_curve(dry, voice_start_src, n_frames_total):
    """Per-source-frame glow intensity (dict src_frame -> g) from the dry voice envelope."""
    hop = SR // FPS
    e = rms_frames(dry, hop, hop * 2)
    e = e / np.percentile(e[e > e.max() * 0.05], 90)
    e = np.clip(e, 0, 1.25)
    s0, s1 = speech_span(dry)
    first, last = int(s0 * FPS), int(np.ceil(s1 * FPS))
    g = np.zeros(len(e) + 4 * FPS)
    level = 0.0
    rng = np.random.default_rng(3)
    for i in range(len(g)):
        tgt = e[i] if i < len(e) else 0.0
        if first <= i <= last:
            tgt = max(tgt, GLOW_SMOULDER)
        elif i > last:
            fall = min(1.0, (i - last) / (0.7 * FPS))
            floor = GLOW_SMOULDER + (GLOW_END_LEVEL - GLOW_SMOULDER) * fall
            tgt = max(tgt * 0.6, floor)
        coef = 0.75 if tgt > level else 0.22   # quick attack, slow release
        if i < first:
            tgt, coef = 0.0, 1.0
        level += (tgt - level) * coef
        g[i] = level * (1 + 0.035 * rng.standard_normal())
    # fade-in on the first word: never jump to full on frame one
    for j in range(4):
        if first + j < len(g):
            g[first + j] *= (j + 1) / 5
    v0 = int(round(voice_start_src * FPS))
    return {v0 + i: float(max(0.0, x)) for i, x in enumerate(g) if v0 + i < n_frames_total}


def _soft_disc(d, r_in=0.90, r_out=1.0):
    t = np.clip((r_out - d) / (r_out - r_in), 0, 1)
    return t * t * (3 - 2 * t)


def apply_glow(frame, eyes, g):
    """Additive red glow in (approximately) linear light, on a ROI around both eyes:
    deep-red iris inside the lens bezel, a defined bright pupil with a white-hot core,
    two-stage bloom and a little red spill on the white face."""
    if g <= 0.003:
        return frame
    rmax = max(e[2] for e in eyes)
    xs = [e[0] for e in eyes]; ys = [e[1] for e in eyes]
    m = rmax * 3.4
    x0 = int(max(0, min(xs) - m)); x1 = int(min(W, max(xs) + m))
    y0 = int(max(0, min(ys) - m)); y1 = int(min(H, max(ys) + m))
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
    # red light falling on the white face: lift R (additive) and pull G/B down a touch
    spill = np.clip(b2 * g * 1.0, 0, 0.40)[..., None]
    lin = lin * (1 - spill * np.array([0.0, 0.5, 0.55], np.float32))
    lin = np.clip(lin + add, 0, 1)
    frame[y0:y1, x0:x1] = (lin ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)
    return frame


# --------------------------------------------------------------------------- audio
def load_src_audio(t0, t1):
    raw = run([FF, "-v", "error", "-ss", f"{t0:.4f}", "-t", f"{t1 - t0:.4f}", "-i", SRC,
               "-vn", "-ac", "2", "-ar", str(SR), "-f", "f64le", "-"],
              capture_output=True).stdout
    a = np.frombuffer(raw, np.float64).reshape(-1, 2).copy()
    n = int(round((t1 - t0) * SR))
    return a[:n] if len(a) >= n else np.pad(a, ((0, n - len(a)), (0, 0)))


def drone(n, start, stop):
    t = np.arange(n) / SR
    y = np.zeros(n)
    for f, a in ((41.2, 1.0), (41.5, 0.8), (61.7, 0.45), (82.4, 0.35), (123.5, 0.12)):
        y += a * np.sin(2 * np.pi * f * t + f)
    y *= 1 + 0.25 * np.sin(2 * np.pi * 0.23 * t)
    rumble = np.cumsum(np.random.default_rng(11).standard_normal(n))
    b, a = ss.butter(2, [25 / (SR / 2), 140 / (SR / 2)], "band")
    rumble = ss.lfilter(b, a, rumble)
    y = y / peak(y) + 0.6 * rumble / peak(rumble)
    y = np.tanh(1.5 * y)
    env = np.zeros(n)
    i0, i1 = int(start * SR), int(stop * SR)
    ramp = int(1.4 * SR)
    env[i0:i1] = 1
    env[i0:i0 + ramp] = np.linspace(0, 1, min(ramp, n - i0))[:len(env[i0:i0 + ramp])]
    y *= env
    rms = np.sqrt((y[i0 + ramp:i1] ** 2).mean()) if i1 > i0 + ramp else peak(y)
    return y / (rms + 1e-12) * 10 ** (DRONE_DB / 20)


def speech_level_db(x):
    hop = int(0.05 * SR)
    return 20 * np.log10(np.percentile(rms_frames(x, hop, hop), 95) + 1e-12)


def mix_audio(clip_t0, clip_t1, voice, voice_start_src):
    src = load_src_audio(clip_t0, clip_t1)
    n = len(src)
    t = np.arange(n) / SR
    fi = int(AUDIO_FADE_IN * SR)
    src[:fi] *= np.linspace(0, 1, fi)[:, None]
    vs = voice_start_src - clip_t0
    ramp0, ramp1 = vs - 0.35, vs - 0.05
    duck = np.where(t >= ramp0, np.interp(t, [ramp0, ramp1], [1, 10 ** (DUCK_DB / 20)]), 1)
    src *= duck[:, None]
    voice = voice if voice.ndim == 2 else np.stack([voice, voice], 1)
    kid_db = speech_level_db(src[:int(max(0.5, vs - 0.4) * SR)].mean(1))
    gain_db = kid_db + VOICE_REL_DB - speech_level_db(voice.mean(1))
    v = np.zeros((n, 2))
    i = int(round(vs * SR))
    seg = voice[:max(0, n - i)]
    v[i:i + len(seg)] = seg * 10 ** (gain_db / 20)
    mix = src + v
    if DRONE_DB is not None:
        dr = drone(n, max(0, vs - 1.2), n / SR)
        lag = int(0.011 * SR)                                    # a little width
        mix += np.stack([dr, np.concatenate([np.zeros(lag), dr[:-lag]])], 1)
    fo = int(0.02 * SR)                                          # de-click the hard cut
    mix[-fo:] *= np.linspace(1, 0, fo)[:, None]
    return mix / peak(mix) * 10 ** (MIX_PEAK_DB / 20)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--voice", metavar="WAV",
                    help="use this pre-made voice file instead of generating one "
                         "(same placement, ducking and eye glow; extras WAV is left alone)")
    ap.add_argument("--voice-only", action="store_true")
    ap.add_argument("--no-whisper", action="store_true")
    ap.add_argument("--stills", action="store_true", help="dump check frames to WORK")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--voice-out", default=VOICE_OUT)
    ap.add_argument("--work", default=WORK)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)

    # 1. voice: generated (default) or a pre-made file via --voice
    if args.voice:
        wet, dry = load_voice_file(args.voice)       # wet is stereo; dry = mono for the glow
        heard_path = args.voice
        where = f"(from {args.voice})"
    else:
        wet, dry = build_voice(args.work)
        save_wav(args.voice_out, wet)
        heard_path = args.voice_out
        where = f"-> {args.voice_out}"
    mono = wet.mean(1) if wet.ndim == 2 else wet
    s0, s1 = speech_span(dry)
    print(f"voice: {len(mono) / SR:.2f}s (words {s0:.2f}-{s1:.2f}s, tail dies "
          f"{tail_death(mono):.2f}s) {where}")
    if not args.no_whisper:
        print("whisper hears:", whisper_check(heard_path, args.work))
    if args.voice_only:
        return

    # 2. timeline (source seconds)
    src_end = 708 / FPS
    try:
        nb = run(["/opt/homebrew/bin/ffprobe", "-v", "error", "-select_streams", "v:0",
                  "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0",
                  SRC], capture_output=True, text=True).stdout.strip()
        src_end = int(nb) / FPS
    except (subprocess.CalledProcessError, ValueError):
        pass
    t_in = round(IN_POINT * FPS) / FPS
    v_start = round((SETTLE_T + VOICE_DELAY) * FPS) / FPS
    t_out = min(v_start + tail_death(mono) + TAIL_AFTER, src_end)
    t_out = int(t_out * FPS) / FPS
    n_frames = int(round((t_out - t_in) * FPS))
    print(f"clip: src {t_in:.2f}-{t_out:.2f}s = {n_frames} frames ({n_frames / FPS:.2f}s); "
          f"voice at src {v_start:.2f}s = clip {v_start - t_in:.2f}s")

    # 3. tracking + glow curve
    tracks = track_eyes(t_out)
    glow = glow_curve(dry, v_start, int(round(t_out * FPS)))

    # 4. audio
    mix = mix_audio(t_in, t_out, wet, v_start)
    mix_path = os.path.join(args.work, "mix.wav")
    wavfile.write(mix_path, SR, (mix * 32767).astype(np.int16))

    # 5. render
    dec = subprocess.Popen(
        ["nice", "-n", "10", FF, "-v", "error", "-ss", f"{t_in:.3f}", "-i", SRC,
         "-frames:v", str(n_frames), "-an",
         "-vf", f"scale={W}:{H}:flags=lanczos:in_color_matrix=bt709:in_range=tv,format=rgb24",
         "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp_out = args.out + ".part.mp4"
    enc = subprocess.Popen(
        ["nice", "-n", "10", FF, "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-i", mix_path,
         "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
         "-color_range", "tv",
         "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-ac", "2",
         "-movflags", "+faststart", "-shortest", tmp_out], stdin=subprocess.PIPE)
    f_in = int(round(t_in * FPS))
    still_at = {}
    if args.stills:
        for name, ts in (("first", t_in), ("midzoom", 17.5), ("v1", v_start + 0.4),
                         ("v2", v_start + 1.2), ("pause", v_start + 2.2),
                         ("v3", v_start + 3.5), ("last", t_out - 1 / FPS)):
            still_at[int(round(ts * FPS))] = name
    fb = W * H * 3
    for k in range(n_frames):
        buf = dec.stdout.read(fb)
        if len(buf) < fb:
            print(f"decoder ended early at frame {k}", file=sys.stderr)
            break
        fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
        sf = f_in + k
        g = glow.get(sf, 0.0)
        if g > 0 and sf in tracks:
            fr = apply_glow(fr, tracks[sf], g)
        if sf in still_at:
            cv2.imwrite(os.path.join(args.work, f"still_{still_at[sf]}_{sf / FPS:.2f}.jpg"),
                        cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        enc.stdin.write(fr.tobytes())
    enc.stdin.close()
    enc.wait()
    dec.wait()
    if enc.returncode != 0:
        sys.exit("encode failed")
    os.replace(tmp_out, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
