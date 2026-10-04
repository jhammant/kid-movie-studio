"""Make several sinister robot-voice takes of the post-credits line to choose from.

    python3 tools/robot_voices.py

Writes voices/Voice X - <name>.wav plus a preview mp4 of each over the robot close-up.
"""
import subprocess
import wave
from pathlib import Path

import numpy as np

LINE = "We'll see, Pen-nee... [[slnc 900]] We'll see...."  # spelled so the deep voices say it clearly
OUT = Path("kit/previews/voices")
ROBOT = "footage/robot-chair.mp4"
ROBOT_AT = 21.6  # steady robot close-up
SR = 48000

# Each take: a clear, deep main voice that carries the words, plus a creepy layer
# underneath (heavy effects on the main voice smear the name "Penny").
TAKES = {
    "A": dict(name="Deep and slow", main=("Daniel", 110, 0.80), layers=[], ring=40, ring_mix=0.2,
              crush=0, echo="aecho=0.8:0.5:60:0.2"),
    "B": dict(name="Classic robot", main=("Fred", 110, 0.88), layers=[], ring=32, ring_mix=0.35, name_as="Penny",
              crush=0, echo="aecho=0.8:0.4:40:0.2"),
    "C": dict(name="Creepy whisper", main=("Daniel", 105, 0.76), layers=[("Daniel", 105, 1.0, 0.5, "whisper")],
              ring=0, ring_mix=0, crush=0, echo="aecho=0.8:0.5:80:0.2"),
    "D": dict(name="Glitchy evil", main=("Ralph", 110, 0.86), layers=[], ring=55, ring_mix=0.25, name_as="Penny",
              crush=7, echo="aecho=0.8:0.4:50:0.2"),
    "E": dict(name="Monster chorus", main=("Daniel", 110, 0.82),
              layers=[("Daniel", 110, 0.72, 0.6), ("Daniel", 110, 0.92, 0.45)],
              ring=0, ring_mix=0, crush=0, echo="aecho=0.8:0.5:70:0.2"),
}


def run(cmd):
    subprocess.run(cmd, check=True)


def read_wav(path):
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        return x.reshape(-1, w.getnchannels()).mean(1), w.getframerate()


def write_wav(path, x, sr):
    y = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y.tobytes())


def speak(voice, rate, pitch, tmp, tag, name_as="Pen-nee"):
    raw = tmp / f"{tag}.aiff"
    run(["say", "-v", voice, "-r", str(rate), "-o", str(raw), LINE.replace("Pen-nee", name_as)])
    low = tmp / f"{tag}.wav"
    # pitch down; atempo keeps each layer the same length so they stay in sync
    run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-af",
         f"aresample={SR},asetrate={SR}*{pitch},aresample={SR},atempo={pitch:.4f}*0.95",
         "-ac", "1", str(low)])
    return read_wav(low)[0]


def make(key, t, tmp):
    name_as = t.get("name_as", "Pen-nee")
    x = speak(*t["main"], tmp, f"{key}_main", name_as)
    x = x / (np.abs(x).max() + 1e-6)
    n = np.arange(len(x)) / SR
    if t["ring"]:
        x = (1 - t["ring_mix"]) * x + t["ring_mix"] * x * np.sin(2 * np.pi * t["ring"] * n) * 1.6
    if t["crush"]:
        steps = 2 ** t["crush"]
        crushed = np.repeat(np.round(x[::5] * steps) / steps, 5)[: len(x)]
        x = 0.6 * x + 0.4 * crushed
    for i, (voice, rate, pitch, gain, *style) in enumerate(t["layers"]):
        y = speak(voice, rate, pitch, tmp, f"{key}_layer{i}", name_as)
        if style == ["whisper"]:
            # breathy ghost: replace the tone with noise that follows the voice's envelope
            env = np.convolve(np.abs(y), np.ones(400) / 400, mode="same")
            noise = np.random.default_rng(7).standard_normal(len(y))
            hp = noise - np.convolve(noise, np.ones(8) / 8, mode="same")
            y = hp * env * 6 + 0.3 * y
        y = y / (np.abs(y).max() + 1e-6)
        m = min(len(x), len(y))
        x = np.concatenate([x[:m] + gain * y[:m], x[m:]])
    write_wav(tmp / f"{key}_fx.wav", x / (np.abs(x).max() + 1e-6) * 0.9, SR)
    final = OUT / f"Voice {key} - {t['name']}.wav"
    run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp / f"{key}_fx.wav"), "-af",
         f"lowshelf=f=150:g=4,{t['echo']},apad=pad_dur=1.0,loudnorm=I=-16:TP=-1.5:LRA=11,aresample={SR}",
         "-ac", "2", str(final)])
    return final


def preview(key, t, voice):
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(voice)], capture_output=True, text=True).stdout) + 0.8
    run(["ffmpeg", "-v", "error", "-y", "-ss", str(ROBOT_AT), "-t", f"{dur:.2f}", "-i", ROBOT, "-i", str(voice),
         "-filter_complex", "[0:v]scale=1920:1080:flags=area,fps=25,format=yuv420p[v];"
         "[1:a]adelay=600|600,apad[a]", "-map", "[v]", "-map", "[a]", "-shortest",
         "-c:v", "libx264", "-crf", "20", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
         str(OUT / f"Voice {key} - {t['name']}.mp4")])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / ".work"
    tmp.mkdir(exist_ok=True)
    for key, t in TAKES.items():
        v = make(key, t, tmp)
        preview(key, t, v)
        print(key, v.name)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()


if __name__ == "__main__":
    main()
