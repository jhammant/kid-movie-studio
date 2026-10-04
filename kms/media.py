"""ffmpeg helpers shared by every tool.

The house format: 1920x1080, 25 fps (or 30), H.264 yuv420p, AAC 48 kHz stereo.
ffmpeg builds without drawtext are fine: text is drawn with PIL and raw frames are piped in.
"""
import json
import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

W, H, FPS, SR = 1920, 1080, 25, 48000


def ffmpeg():
    return os.environ.get("KMS_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"


def ffprobe():
    return os.environ.get("KMS_FFPROBE") or shutil.which("ffprobe") or "ffprobe"


def require_ffmpeg():
    if not shutil.which(ffmpeg()):
        raise SystemExit("ffmpeg isn't installed. On a Mac: `brew install ffmpeg`. "
                         "On Linux: `sudo apt install ffmpeg`.")


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


@dataclass
class Probe:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    rotation: int = 0

    @property
    def portrait(self):
        w, h = (self.height, self.width) if self.rotation in (90, 270) else (self.width, self.height)
        return h > w


def probe(path):
    """Size, frame rate, length and audio of a clip, or None if ffprobe can't read it yet."""
    r = subprocess.run([ffprobe(), "-v", "error", "-show_entries",
                        "format=duration:stream=codec_type,width,height,avg_frame_rate:stream_side_data=rotation",
                        "-of", "json", str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        info = json.loads(r.stdout)
        duration = float(info["format"]["duration"])
    except (ValueError, KeyError):
        return None
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    if video is None:
        return Probe(duration, 0, 0, 0.0, audio)
    num, _, den = video.get("avg_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den) if den and float(den) else 0.0
    rot = 0
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = abs(int(sd["rotation"])) % 360
    return Probe(duration, int(video.get("width", 0)), int(video.get("height", 0)), fps, audio, rot)


def readable(path):
    """True once a file exists and ffprobe can read its length (an agent may still be writing it)."""
    p = Path(path)
    if not p.exists():
        return False
    info = probe(p)
    return info is not None and info.duration > 0


def duration(path):
    info = probe(path)
    return info.duration if info else None


def write_wav(path, samples, sr=SR):
    """Write float samples (n,) or (n, channels) in -1..1 as 16-bit PCM."""
    x = np.asarray(samples, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def read_wav(path):
    """(float samples (n, channels), sample rate) from a 16-bit PCM wav."""
    with wave.open(str(path), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
        return x.reshape(-1, w.getnchannels()), w.getframerate()


def video_args(crf=18):
    return ["-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p"]


# RGB frames -> HD video colour: convert with the BT.709 matrix and tag it, so players
# don't guess (an untagged file can look washed out or too saturated)
BT709_VF = "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"
BT709_TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]


def audio_args():
    return ["-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-ac", "2"]


class FrameWriter:
    """Pipe raw RGB frames into an H.264 file, optionally muxing a wav at the end.

        with FrameWriter(out, fps=25, audio=wav_path) as fw:
            for frame in frames:
                fw.write(frame)          # uint8 (H, W, 3) RGB
    """

    def __init__(self, out, fps=FPS, size=(W, H), audio=None, crf=18, pix_fmt="rgb24"):
        self.out, self.audio = Path(out), audio
        w, h = size
        cmd = [ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix_fmt,
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
        if audio:
            cmd += ["-i", str(audio), "-map", "0:v", "-map", "1:a"] + audio_args() + ["-shortest"]
        cmd += ["-vf", BT709_VF] + video_args(crf) + BT709_TAGS + ["-movflags", "+faststart", str(self.out)]
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.out}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type:
            self.proc.kill()
            self.proc.wait()
            return False
        self.close()
        return False


def web_copy(src, dst, width=960):
    """A small, fast-starting copy for the pick page."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    run([ffmpeg(), "-v", "error", "-y", "-i", src, "-vf", f"scale={width}:-2", "-c:v", "libx264", "-crf", "25",
         "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", dst])


def extract_frame(src, at, dst, width=None):
    vf = ["-vf", f"scale={width}:-2"] if width else []
    run([ffmpeg(), "-v", "error", "-y", "-ss", f"{at:.3f}", "-i", src, "-frames:v", "1", *vf, dst])


def loudness(path, start=None, length=None):
    """Integrated loudness (LUFS) and true peak (dBFS) of a file or a section of it."""
    cmd = [ffmpeg(), "-hide_banner", "-nostats"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if length is not None:
        cmd += ["-t", str(length)]
    cmd += ["-i", str(path), "-vn", "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-"]
    err = subprocess.run(cmd, capture_output=True, text=True).stderr
    if "Integrated loudness" not in err:
        return None, None
    tail = err[err.rindex("Integrated loudness"):]
    lufs = float(tail.split("I:")[1].split("LUFS")[0])
    peak = float(tail.split("Peak:")[1].split("dBFS")[0]) if "Peak:" in tail else None
    return lufs, peak
