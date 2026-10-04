"""Get the finished film out: a small copy for a phone, the pieces for iMovie, or a folder
for a home media server (Plex, Jellyfin) with a poster. Sharing is a grown-up decision."""
import shutil
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

from kms import media
from kms.packs import get_pack


def _film(p):
    if not media.readable(p.film_path):
        raise FileNotFoundError("There's no finished film yet. Run `kms assemble` first.")
    return p.film_path


def phone(p):
    out = p.film_path.with_name(p.film_path.stem + " (phone).mp4")
    media.run([media.ffmpeg(), "-v", "error", "-y", "-i", _film(p), "-vf", "scale=1280:-2", "-c:v", "libx264",
               "-crf", "24", "-maxrate", "3M", "-bufsize", "6M", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
               "-movflags", "+faststart", out])
    return out


def imovie(p):
    """Numbered pieces in running order, the captions, and the green-screen takes."""
    from kms.assemble import plan
    out = p.kit_dir("iMovie")
    for n, piece in enumerate(plan(p, make_missing=False), start=1):
        if piece.ready:
            shutil.copy2(piece.path, out / f"{n:02d} {piece.label}{piece.path.suffix}")
    for f in sorted((p.kit / "captions").glob("*.png")):
        shutil.copy2(f, out / f"Caption - {f.stem}.png")
    for f in sorted((p.kit / "keyed").glob("*-green.mp4")):
        shutil.copy2(f, out / f"Green screen - {f.name}")
    return out


def media_server(p, dest=None, year=None):
    year = year or date.today().year
    name = f"{p.title.title()} ({year})"
    folder = Path(dest or p.kit) / name
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_film(p), folder / f"{name}.mp4")
    pack = get_pack(p.pack_name)
    cover, _ = pack.choice(p, "cover")
    frame = p.kit / "previews" / "cover" / f"{cover}.jpg" if cover else None
    if frame and frame.exists():
        from kms.render import cover_art
        cover_art.render(frame, folder, **pack.cover_words(p))
    nfo = folder / "movie.nfo"  # Jellyfin and Kodi read this; Plex uses the folder name
    nfo.write_text(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<movie>\n"
        f"  <title>{escape(p.title.title())}</title>\n  <year>{year}</year>\n"
        f"  <plot>{escape(p.tagline)}</plot>\n  <mpaa>U</mpaa>\n  <genre>Family</genre>\n"
        f"  <director>{escape(p.director)}</director>\n  <studio>kid-movie-studio</studio>\n</movie>\n")
    return folder
