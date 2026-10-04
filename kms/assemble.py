"""Put the film together from the running order and the director's picks.

Every piece is conformed to 1920x1080 / the project's fps / AAC 48 kHz stereo, levelled,
captioned, and joined with hard cuts (it's the news). Pieces that aren't ready are skipped
with a note, so you can watch "the film so far" while the rest renders.
"""
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from kms import media
from kms.loudness import LEVELLER
from kms.packs import get_pack
from kms.previews import preview_file

# generated titles and stings measure hotter than dialogue; sit them level with it. They're
# measured and given an exact gain: one-pass loudnorm is unreliable on a 5-second sting.
GRAPHICS = "graphics"
GRAPHICS_LUFS = -17.0
LEVELS = {"dialogue": -18.0, "speech": -17.0, "loud": -15.0, "kid": -15.0, "quiet": -20.0}


@dataclass
class Piece:
    label: str
    path: Path
    start: float = 0.0
    dur: float | None = None
    overlays: list = field(default_factory=list)   # [(png, from_s, to_s)]
    afilter: str | None = None
    note: str = ""
    ready: bool = True


def audio_filter(level, default):
    """An ffmpeg audio filter for a scene's `level:` (a name, a LUFS number, or none)."""
    level = default if level is None else level
    if level in ("none", False):
        return None
    if level == "graphics":
        return GRAPHICS
    target = LEVELS.get(level) if isinstance(level, str) else float(level)
    if target is None:
        raise ValueError(f"level {level!r}: use a LUFS number like -17, or one of {', '.join(LEVELS)}, none")
    return LEVELLER.format(target=target)


def plan(p, make_missing=True, ctx=None, log=print):
    """The pieces in running order, making `make:` pieces as needed."""
    from kms.render import Ctx
    ctx = ctx or Ctx(fps=p.fps)
    pack = get_pack(p.pack_name)
    caps = pack.captions(p, p.kit_dir("captions"))
    out = []
    for s in p.ordered_scenes():
        overlays = []
        if s.id in caps:
            a, b = (s.get("caption_at") or [1.0, 7.0])[:2]
            overlays.append((caps[s.id], float(a), float(b)))
        if s.kind == "pick":
            oid, picked = pack.choice(p, s.target)
            path = preview_file(p, s.target, oid)
            note = f"{s.target} = {oid}" + ("" if picked else "  (not picked yet, so the default)")
            out.append(Piece(s.label, path, afilter=GRAPHICS, note=note, ready=media.readable(path)))
        elif s.kind == "make":
            path = p.kit_dir("pieces") / f"{s.id}.mp4"
            ready = True
            if make_missing:
                try:
                    pack.make(p, s, path, ctx)
                except FileNotFoundError as e:
                    ready, note = False, str(e)
            ready = ready and media.readable(path)
            out.append(Piece(s.label, path, afilter=audio_filter(s.get("level"), "graphics" if s.target != "credits" else "none"),
                             note="" if ready else f"needs {s.get('clip') or 'its footage'}", ready=ready))
        elif s.kind == "clip":
            path = p.footage_file(s.target)
            start = float(s.get("from") or 0)
            dur = float(s.get("to")) - start if s.get("to") is not None else None
            out.append(Piece(s.label, path, start, dur, overlays, audio_filter(s.get("level"), "speech"),
                             note=s.target, ready=media.readable(path)))
        elif s.kind == "key":
            path = p.kit / "keyed" / f"{s.id}.mp4"
            out.append(Piece(s.label, path, 0.0, None, overlays, audio_filter(s.get("level"), "speech"),
                             note="run `kms key`" if not path.exists() else s.target, ready=media.readable(path)))
    return out


def conform(piece, out, fps):
    cmd = [media.ffmpeg(), "-v", "error", "-y", "-ss", f"{piece.start:.3f}"]
    if piece.dur:
        cmd += ["-t", f"{piece.dur:.3f}"]
    cmd += ["-i", str(piece.path)]
    live = [(png, a, b) for png, a, b in piece.overlays if Path(png).exists()]
    for png, _, _ in live:
        cmd += ["-loop", "1", "-i", str(png)]
    cmd += ["-f", "lavfi", "-t", "0.1", "-i", "anullsrc=r=48000:cl=stereo"]
    silent = 1 + len(live)
    v = f"[0:v]scale=1920:1080:flags=lanczos:force_original_aspect_ratio=decrease,pad=1920:1080:-1:-1," \
        f"fps={fps},format=yuv420p,setsar=1[v0]"
    last = "v0"
    for i, (_, a, b) in enumerate(live, start=1):
        v += (f";[{i}:v]format=rgba,fade=t=in:st={a}:d=0.4:alpha=1,fade=t=out:st={b - 0.4}:d=0.4:alpha=1[o{i}]"
              f";[{last}][o{i}]overlay=0:0:enable='between(t,{a},{b})':shortest=1[v{i}]")
        last = f"v{i}"
    info = media.probe(piece.path)
    afilter = piece.afilter
    if afilter == GRAPHICS:
        lufs, _ = media.loudness(piece.path, piece.start or None, piece.dur)
        gain = GRAPHICS_LUFS - lufs if lufs is not None and lufs > -60 else 0.0
        afilter = f"volume={gain:.2f}dB,alimiter=limit=0.84:level=false"
    if info and info.has_audio:
        a = f"[0:a]aresample=48000,aformat=channel_layouts=stereo{(',' + afilter) if afilter else ''}[a]"
    else:
        a = f"[{silent}:a]apad[a]"
    cmd += ["-filter_complex", f"{v};{a}", "-map", f"[{last}]", "-map", "[a]", *media.video_args(18),
            *media.audio_args(), "-shortest", "-movflags", "+faststart", str(out)]
    media.run(cmd)


def assemble(p, out=None, log=print, ctx=None):
    """Make the film. Returns (path, pieces) where pieces says what was used or skipped."""
    out = Path(out) if out else p.film_path
    pieces = plan(p, ctx=ctx, log=log)
    with tempfile.TemporaryDirectory(dir=p.kit_dir()) as tmp:
        parts = []
        for n, piece in enumerate(pieces):
            if not piece.ready:
                log(f"  skip  {piece.label}: {piece.note or 'not ready'}")
                continue
            part = Path(tmp) / f"{n:02d}.mp4"
            conform(piece, part, p.fps)
            parts.append(part)
            log(f"  ok    {piece.label}" + (f"  ({piece.note})" if piece.note else ""))
        if not parts:
            raise RuntimeError("Nothing is ready to assemble yet. Run `kms next` to see what to do first.")
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{x}'\n" for x in parts))
        media.run([media.ffmpeg(), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                   "-movflags", "+faststart", out])
    return out, pieces
