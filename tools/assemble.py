"""Assemble the whole Robots Revenge film from the picks, as a watchable preview.

    python3 tools/assemble.py --title A --channel A --signal A [--out path]

Every piece is conformed to 1920x1080 / 25 fps / AAC 48 kHz stereo, then joined
with hard cuts (it's the news). Name captions are overlaid when their PNGs exist.
Missing pieces are skipped with a warning, so it works while agents are still rendering.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

FOOTAGE = Path("footage")
SRC = FOOTAGE
KIT = Path("kit")
PREV = KIT / "previews"
LT = KIT / "lower-thirds"

TITLES = {"A": "Title A - Blockbuster.mp4", "B": "Title B - Comic Book.mp4",
          "C": "Title C - Robot Computer.mp4", "D": "Title D - Warning.mp4"}
CHANNELS = {"A": "News A - SBC News.mp4", "B": "News B - Robot Watch 9.mp4",
            "C": "News C - Nine OClock News.mp4", "D": "News D - Penny Sparks Live.mp4"}
SIGNALS = {"A": "Signal A - Glitch Static Beep.mp4", "B": "Signal B - Robots Hacked.mp4",
           "C": "Signal C - Just Static.mp4"}

# The grown-up's desk shots are levelled here; the kid's clips (04/05) are pre-levelled a touch
# louder by tools/loudness.py so the kid sits slightly forward.
DIALOGUE = "loudnorm=I=-18:TP=-2:LRA=11"
# generated titles/stings measured 5-7 dB hotter than the dialogue; sit them just above it
GRAPHICS = "loudnorm=I=-17:TP=-2:LRA=11"


def plan(args):
    """(label, path, start, duration, overlays, audio_filter) in running order."""
    return [
        ("Title card", PREV / TITLES[args.title], 0, None, [], GRAPHICS),
        ("9 o'clock news intro", PREV / CHANNELS[args.channel], 0, None, [], GRAPHICS),
        ("Rex's intro", SRC / "desk-intro.mp4", 0.8, 32.8,
         [(LT / "Lower third - Rex Newsome.png", 1.5, 7.0)], DIALOGUE),
        ("Penny live", KIT / "04 Penny LIVE - water background.mp4", 0, None,
         [(LT / "Lower third - Penny Sparks LIVE.png", 1.0, 7.0)], None),
        ("Penny: help!", KIT / "05 Penny HELP - robot background.mp4", 0, None, [], None),
        ("Signal lost", PREV / SIGNALS[args.signal], 0, None, [], GRAPHICS),
        ("Rex: Penny?!", FOOTAGE / "desk-worried.mp4", 0, None, [], DIALOGUE),
        ("End credits", KIT / "07 End Credits.mp4", 0, None, [], None),
        ("Post-credits robot", KIT / "08 Post-credits - robot.mp4", 0, None, [], None),
        ("To Be Continued", KIT / "09 To Be Continued.mp4", 0, None, [], GRAPHICS),
    ]


def readable(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                        str(path)], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip()


def conform(path, start, dur, overlays, afilter, out):
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(start)]
    if dur:
        cmd += ["-t", str(dur)]
    cmd += ["-i", str(path)]
    live = [(p, a, b) for p, a, b in overlays if p.exists()]
    for p, _, _ in live:
        cmd += ["-loop", "1", "-i", str(p)]
    cmd += ["-f", "lavfi", "-t", "0.1", "-i", "anullsrc=r=48000:cl=stereo"]
    silent_idx = 1 + len(live)
    v = ("[0:v]scale=1920:1080:flags=lanczos,fps=25,format=yuv420p,setsar=1[v0]")
    last = "v0"
    for i, (_, a, b) in enumerate(live, start=1):
        fade = f"[{i}:v]format=rgba,fade=t=in:st={a}:d=0.4:alpha=1,fade=t=out:st={b - 0.4}:d=0.4:alpha=1[o{i}]"
        v += f";{fade};[{last}][o{i}]overlay=0:0:enable='between(t,{a},{b})':shortest=1[v{i}]"
        last = f"v{i}"
    has_audio = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                                "stream=index", "-of", "csv=p=0", str(path)],
                               capture_output=True, text=True).stdout.strip()
    if has_audio:
        a = f"[0:a]aresample=48000,aformat=channel_layouts=stereo{(',' + afilter) if afilter else ''}[a]"
    else:  # pad silence for the whole clip
        a = f"[{silent_idx}:a]apad[a]"
    cmd += ["-filter_complex", f"{v};{a}", "-map", f"[{last}]", "-map", "[a]",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="A", choices=TITLES)
    ap.add_argument("--channel", default="A", choices=CHANNELS)
    ap.add_argument("--signal", default="A", choices=SIGNALS)
    ap.add_argument("--out", default=str(KIT / "Robots Revenge - full preview.mp4"))
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for n, (label, path, start, dur, overlays, afilter) in enumerate(plan(args)):
            if not path.exists() or not readable(path):
                print(f"  skip  {label}: {path.name} not ready", file=sys.stderr)
                continue
            out = Path(tmp) / f"{n:02d}.mp4"
            conform(path, start, dur, overlays, afilter, out)
            parts.append(out)
            print(f"  ok    {label}")
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in parts))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", "-movflags", "+faststart", args.out], check=True)
    print(args.out)


if __name__ == "__main__":
    main()
