"""Level a clip's dialogue in place (video stream copied).

    python -m kms.loudness <clip.mp4> [target LUFS] [--audio-from SRC --start S --dur D]

A speech leveller, not a plain gain: kids' takes mix quiet talking with very loud
screams, so plain normalising turns the talking *down* to fit the screams. This
compresses first (lifts talking, holds screams), then rides the level to the target
and limits peaks. `kms assemble` uses the same chain for every scene's `level:`
(-17 LUFS by default; give the kid -15 to sit them slightly forward).
--audio-from rebuilds the track from the camera original.
"""
import argparse
import os
import subprocess
import tempfile

from kms.media import ffmpeg

LEVELLER = ("highpass=f=70,"
            "acompressor=threshold=0.06:ratio=4:attack=8:release=250:makeup=2.5:knee=4,"
            "loudnorm=I={target}:TP=-1.5:LRA=7,"
            "alimiter=limit=0.84:level=false,aresample=48000")


def level(path, target=-15.0, audio_from=None, start=0.0, dur=None):
    fd, tmp = tempfile.mkstemp(suffix=".mp4", dir=os.path.dirname(path))
    os.close(fd)
    cmd = [ffmpeg(), "-v", "error", "-y", "-i", path]
    if audio_from:
        cmd += ["-ss", str(start)] + (["-t", str(dur)] if dur else []) + ["-i", audio_from]
    cmd += ["-map", "0:v", "-map", f"{1 if audio_from else 0}:a", "-c:v", "copy",
            "-af", LEVELLER.format(target=target), "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart", tmp]
    subprocess.run(cmd, check=True)
    os.replace(tmp, path)


def integrated(path):
    out = subprocess.run([ffmpeg(), "-hide_banner", "-i", path, "-vn", "-af", "ebur128=framelog=quiet",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    tail = out[out.rindex("Integrated loudness"):]
    return float(tail.split("I:")[1].split("LUFS")[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("target", nargs="?", type=float, default=-15.0)
    ap.add_argument("--audio-from")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float)
    a = ap.parse_args()
    before = integrated(a.clip)
    level(a.clip, a.target, a.audio_from, a.start, a.dur)
    print(f"{os.path.basename(a.clip)}: {before:.1f} -> {integrated(a.clip):.1f} LUFS")
