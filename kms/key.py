"""Key every `key:` scene in movie.yaml: plain-wall green screen over its background.

Long takes are keyed in parallel time slices. Each slice keys 1 s early (preroll) so the
matte has settled before it starts writing, so there's no seam at the joins. Backgrounds
shorter than the take are looped with a crossfade so the loop point doesn't jump.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from kms import media


def plate_window(scene, src_duration):
    """(start, length) of the empty wall: `plate: {from, for}`, else the last 4 s of the take."""
    plate = scene.get("plate") or {}
    if "from" in plate:
        return float(plate["from"]), float(plate.get("for", 3.0))
    length = min(3.0, max(0.5, src_duration * 0.25))
    return max(0.0, src_duration - length - 0.5), length


def loop_background(src, out, need, fps, xfade=0.6):
    """Loop a short background clip into one at least `need` seconds long."""
    dur = media.duration(src)
    copies = 1
    while copies * (dur - xfade) + xfade < need:
        copies += 1
    if copies == 1:
        return Path(src)
    inputs, chain, prev = [], "", "[v0]"
    for i in range(copies):
        inputs += ["-i", str(src)]
        chain += f"[{i}:v]scale=1920:1080:flags=area,fps={fps},format=yuv420p,settb=AVTB[v{i}];"
    for i in range(1, copies):
        chain += f"{prev}[v{i}]xfade=transition=fade:duration={xfade}:offset={i * (dur - xfade):.3f}[x{i}];"
        prev = f"[x{i}]"
    media.run([media.ffmpeg(), "-v", "error", "-y", *inputs, "-filter_complex", chain.rstrip(";"),
               "-map", prev, "-an", "-c:v", "libx264", "-crf", "14", "-preset", "fast", out])
    return Path(out)


def key_scene(p, scene, slices=3, green=True, limit_frames=None):
    """Key one scene into kit/keyed/<id>.mp4 (and <id>-green.mp4 for iMovie). Returns the path."""
    src = p.footage_file(scene.target)
    info = media.probe(src)
    if info is None:
        raise FileNotFoundError(f"{src.name} isn't in footage/ yet (scene {scene.id!r}).")
    plate = scene.get("plate") or {}
    plate_src = p.footage_file(plate.get("clip", scene.target))
    plate_start, plate_len = plate_window(scene, info.duration)
    start = float(scene.get("from") or 0)
    end = float(scene.get("to") or (plate_start if plate_src == src else info.duration))
    dur = max(0.2, end - start)
    out_dir = p.kit_dir("keyed")
    out = out_dir / f"{scene.id}.mp4"
    gout = out_dir / f"{scene.id}-green.mp4" if green else None
    bg = None
    if scene.get("background"):
        bg_src = p.footage_file(scene.get("background"))
        if not media.readable(bg_src):
            raise FileNotFoundError(f"The background {bg_src.name} isn't in footage/ yet (scene {scene.id!r}).")
        bg = loop_background(bg_src, out_dir / f"{scene.id}-background.mp4", dur, p.fps)
    common = ["--plate", str(plate_src), "--plate-start", str(plate_start), "--plate-dur", str(plate_len),
              "--fps", str(p.fps)]
    if scene.get("wall"):
        common += ["--wall", ",".join(str(int(x)) for x in scene.get("wall"))]
    if scene.get("min_area"):
        common += ["--min-area", str(int(scene.get("min_area")))]
    if limit_frames:
        dur = min(dur, limit_frames / p.fps)
        slices = 1
    slices = max(1, min(slices, int(dur // 2) or 1))
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        tmp = Path(tmp)
        part = round(dur / slices, 3)
        procs = []
        for k in range(slices):
            s = start + k * part
            d = dur - (slices - 1) * part if k == slices - 1 else part
            cmd = [sys.executable, "-m", "kms.keyer", str(src), str(tmp / f"c{k}.mp4"), *common,
                   "--start", f"{s:.3f}", "--dur", f"{d:.3f}", "--preroll", "1.0"]
            if bg:
                cmd += ["--bg", str(bg), "--bg-start", f"{k * part:.3f}"]
            if gout:
                cmd += ["--green-out", str(tmp / f"g{k}.mp4")]
            log = open(tmp / f"log{k}.txt", "w")
            procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        failed = []
        for k, (proc, log) in enumerate(procs):
            if proc.wait() != 0:
                failed.append(k)
            log.close()
        if failed:
            raise RuntimeError(f"keying {scene.id} failed: " + (tmp / f"log{failed[0]}.txt").read_text()[-800:])
        for prefix, target in (("c", out), ("g", gout)):
            if target is None:
                continue
            lst = tmp / f"{prefix}.txt"
            lst.write_text("".join(f"file '{tmp / f'{prefix}{k}.mp4'}'\n" for k in range(slices)))
            media.run([media.ffmpeg(), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                       "-c", "copy", "-movflags", "+faststart", target])
    return out


def key_all(p, only=None, slices=3, green=True, log=print):
    done = []
    for s in p.scenes:
        if s.kind != "key" or (only and s.id not in only):
            continue
        log(f"  keying {s.id} ({s.target}) …")
        done.append(key_scene(p, s, slices=slices, green=green))
        log(f"  ok     {done[-1].relative_to(p.root)}")
    return done
