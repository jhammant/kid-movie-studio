"""Look at the footage before deciding anything.

For every clip in footage/: its size, frame rate and length; a contact sheet (a strip of
frames to look at); a speech-activity strip that shows dead air and screams; and, if the
`whisper` command is installed, a transcript. Everything is written to kit/analysis/, with
a summary in kit/analysis/README.md that a person (or Claude) can read to plan the film.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from kms import media

VIDEO = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mts"}


def contact_sheet(clip, out, frames=6, width=320):
    d = media.duration(clip) or 1
    media.run([media.ffmpeg(), "-v", "error", "-y", "-i", clip, "-vf",
               f"fps={frames}/{d:.3f},scale={width}:-2,tile={frames}x1", "-frames:v", "1", out])
    return out


def speech_strip(clip, step=0.25):
    """One character per `step` seconds: █ loud, ▄ quiet, · silent."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        r = subprocess.run([media.ffmpeg(), "-v", "error", "-y", "-i", str(clip), "-vn", "-ac", "1", "-ar", "16000",
                            str(wav)], capture_output=True)
        if r.returncode != 0 or not wav.exists():
            return ""
        x, sr = media.read_wav(wav)
    x = x[:, 0]
    hop = int(sr * step)
    rms = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2)) for i in range(0, len(x) - hop + 1, hop)])
    if not len(rms):
        return ""
    db = 20 * np.log10(rms + 1e-9)
    floor = np.percentile(db, 10)
    return "".join("█" if v > floor + 24 else "▄" if v > floor + 10 else "·" for v in db)


def transcribe(clip, out_dir):
    if not shutil.which("whisper"):
        return None
    r = subprocess.run(["whisper", str(clip), "--model", "small.en", "--output_dir", str(out_dir),
                        "--fp16", "False"], capture_output=True, text=True)
    txt = Path(out_dir) / (Path(clip).stem + ".txt")
    return txt.read_text().strip() if r.returncode == 0 and txt.exists() else None


def analyse(p, transcripts=True, log=print):
    out = p.kit_dir("analysis")
    clips = sorted(f for f in p.footage.glob("*") if f.suffix.lower() in VIDEO) if p.footage.exists() else []
    if not clips:
        log("  There's nothing in footage/ yet. Copy your clips in first.")
        return None
    lines = ["# What's in the footage", "",
             "| clip | size | fps | length | sound |", "|---|---|---|---|---|"]
    detail = []
    for c in clips:
        info = media.probe(c)
        if info is None:
            lines.append(f"| {c.name} | unreadable | | | |")
            continue
        lines.append(f"| {c.name} | {info.width}x{info.height}{' (portrait!)' if info.portrait else ''} | "
                     f"{info.fps:.0f} | {info.duration:.1f} s | {'yes' if info.has_audio else 'no'} |")
        sheet = contact_sheet(c, out / f"{c.stem}.jpg")
        strip = speech_strip(c) if info.has_audio else ""
        text = transcribe(c, out) if transcripts and info.has_audio else None
        block = [f"## {c.name}", "", f"![{c.stem}]({sheet.name})", ""]
        if strip:
            block += ["Sound, one mark per ¼ s (█ loud, ▄ quiet, · silent):", "", "```text", strip, "```", ""]
        if text:
            block += ["What's said:", "", f"> {text}", ""]
        detail += block
        log(f"  looked at {c.name}")
    report = out / "README.md"
    report.write_text("\n".join(lines + [""] + detail))
    return report
