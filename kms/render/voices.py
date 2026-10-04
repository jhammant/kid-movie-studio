"""The villain's voice: several sinister takes of one line, for the kid to choose from.

    python -m kms.render.voices OUT_DIR [--line "We'll see, Penny... We'll see..."] [--hero Penny]

Each take is a clear, deep main voice that carries the words, plus a creepy layer
underneath. Heavy effects on the main voice smear names, so the effects stay light there.
"""
import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from kms import media, tts
from kms.render import Ctx

SR = media.SR

# id: (kid-friendly name, description, recipe)
TAKES = {
    "deep": ("Deep and slow", "A deep, calm, menacing voice with a slight metal buzz.",
             dict(main=("Daniel", 110, 0.80), layers=[], ring=40, ring_mix=0.2, crush=0,
                  echo="aecho=0.8:0.5:60:0.2")),
    "classic": ("Classic robot", "Old-school computer voice with a metal ring. Names can get a bit fuzzy.",
                dict(main=("Fred", 110, 0.88), layers=[], ring=32, ring_mix=0.35, crush=0,
                     echo="aecho=0.8:0.4:40:0.2")),
    "whisper": ("Creepy whisper", "A deep voice with a breathy ghost whisper underneath. Names can get fuzzy.",
                dict(main=("Daniel", 105, 0.76), layers=[("Daniel", 105, 1.0, 0.5, "whisper")], ring=0,
                     ring_mix=0, crush=0, echo="aecho=0.8:0.5:80:0.2")),
    "glitchy": ("Glitchy evil", "A crunchy, glitchy, evil robot. Names can get a bit fuzzy.",
                dict(main=("Ralph", 110, 0.86), layers=[], ring=55, ring_mix=0.25, crush=7,
                     echo="aecho=0.8:0.4:50:0.2")),
    "chorus": ("Monster chorus", "Three deep voices at once, like a whole army.",
               dict(main=("Daniel", 110, 0.82), layers=[("Daniel", 110, 0.72, 0.6), ("Daniel", 110, 0.92, 0.45)],
                    ring=0, ring_mix=0, crush=0, echo="aecho=0.8:0.5:70:0.2")),
    "sinister": ("Sinister", "Slow and sinister, with a dark echo that hangs in the air.", dict(chain="sinister")),
}


def tts_line(line, hero=None, plain_name=True):
    """The line as the speech engine should read it, with a long pause at the first '...' break.

    Names are left as written: checked with whisper, plain spellings came through the effects
    and spelled-out ones ("Pen-nee") didn't. If the computer mangles a name, set `hero_say` in
    movie.yaml to a sounds-like spelling and the studio uses that instead."""
    text = line
    if tts.engine() == "say":
        text = re.sub(r"\.\.\.\s+", "... [[slnc 900]] ", text, count=1)
    return text


def _speak(text, voice, rate, pitch, tmp, tag):
    raw = tts.speak(text, Path(tmp) / f"{tag}_raw.wav", voice=voice, rate=rate)
    low = Path(tmp) / f"{tag}.wav"
    # pitch down; atempo keeps every layer the same length so they stay in sync
    media.run([media.ffmpeg(), "-v", "error", "-y", "-i", raw, "-af",
               f"aresample={SR},asetrate={SR}*{pitch},aresample={SR},atempo={pitch:.4f}*0.95", "-ac", "1", low])
    return media.read_wav(low)[0][:, 0]


def render_voice(out_wav, *, take="deep", line="We'll see, Penny... We'll see...", hero="Penny", seed=7):
    """Write one take of the villain's line as a stereo 48 kHz wav, levelled to -16 LUFS."""
    if take not in TAKES:
        raise KeyError(f"unknown voice {take!r}; choose from {', '.join(TAKES)}")
    t = TAKES[take][2]
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if t.get("chain") == "sinister":  # the post-credits voice: its own, heavier effects chain
        from kms.render.glowing_eyes import sinister_voice
        return sinister_voice(out_wav, line=line, hero=hero, seed=seed)
    text = tts_line(line, hero)
    with tempfile.TemporaryDirectory() as tmp:
        x = _speak(text, *t["main"], tmp, "main")
        x = x / (np.abs(x).max() + 1e-6)
        n = np.arange(len(x)) / SR
        if t["ring"]:
            x = (1 - t["ring_mix"]) * x + t["ring_mix"] * x * np.sin(2 * np.pi * t["ring"] * n) * 1.6
        if t["crush"]:
            steps = 2 ** t["crush"]
            crushed = np.repeat(np.round(x[::5] * steps) / steps, 5)[: len(x)]
            x = 0.6 * x + 0.4 * crushed
        for i, (voice, rate, pitch, gain, *style) in enumerate(t["layers"]):
            y = _speak(text, voice, rate, pitch, tmp, f"layer{i}")
            if style == ["whisper"]:
                # breathy ghost: swap the tone for noise that follows the voice's envelope
                env = np.convolve(np.abs(y), np.ones(400) / 400, mode="same")
                noise = np.random.default_rng(seed).standard_normal(len(y))
                hp = noise - np.convolve(noise, np.ones(8) / 8, mode="same")
                y = hp * env * 6 + 0.3 * y
            y = y / (np.abs(y).max() + 1e-6)
            m = min(len(x), len(y))
            x = np.concatenate([x[:m] + gain * y[:m], x[m:]])
        fx = Path(tmp) / "fx.wav"
        media.write_wav(fx, x / (np.abs(x).max() + 1e-6) * 0.9)
        media.run([media.ffmpeg(), "-v", "error", "-y", "-i", fx, "-af",
                   f"lowshelf=f=150:g=4,{t['echo']},apad=pad_dur=1.0,loudnorm=I=-16:TP=-1.5:LRA=11,aresample={SR}",
                   "-ac", "2", out_wav])
    return out_wav


def render(out, *, take="deep", line="We'll see, Penny... We'll see...", hero="Penny", picture=None,
           picture_at=0.0, ctx=None):
    """A preview video: the take over a still or a clip (`picture`), else over dark static."""
    ctx = ctx or Ctx()
    out = Path(out)
    wav = render_voice(out.with_suffix(".wav"), take=take, line=line, hero=hero, seed=ctx.seed + 7)
    dur = (media.duration(wav) or 4.0) + 0.8
    if ctx.limit_frames:
        dur = ctx.limit_frames / ctx.fps
    if picture and Path(picture).exists():
        src = ["-ss", f"{picture_at:.2f}", "-t", f"{dur:.2f}", "-i", str(picture)]
        if Path(picture).suffix.lower() in (".png", ".jpg", ".jpeg"):
            src = ["-loop", "1", "-t", f"{dur:.2f}", "-i", str(picture)]
    else:
        src = ["-f", "lavfi", "-t", f"{dur:.2f}", "-i", f"color=c=0x0b0e14:s=1920x1080:r={ctx.fps},noise=alls=12:allf=t"]
    media.run([media.ffmpeg(), "-v", "error", "-y", *src, "-i", wav, "-filter_complex",
               f"[0:v]scale=1920:1080:flags=area,fps={ctx.fps},format=yuv420p[v];[1:a]adelay=600|600,apad[a]",
               "-map", "[v]", "-map", "[a]", "-t", f"{dur:.2f}", *media.video_args(20), *media.audio_args(),
               "-movflags", "+faststart", out])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--line", default="We'll see, Penny... We'll see...")
    ap.add_argument("--hero", default="Penny")
    ap.add_argument("--picture", help="a clip or still to play the voice over")
    ap.add_argument("--picture-at", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=25)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for take, (name, _, _) in TAKES.items():
        render(out / f"{take}.mp4", take=take, line=a.line, hero=a.hero, picture=a.picture,
               picture_at=a.picture_at, ctx=Ctx(fps=a.fps))
        print(take, name)


if __name__ == "__main__":
    main()
