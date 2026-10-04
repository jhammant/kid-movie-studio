"""Render every option of every choice group as a real preview, in parallel.

Previews ARE the finished pieces (full quality 1080p), so what the kid picks is exactly
what ends up in the film. Each one also gets a small web copy for the pick page. A manifest
remembers what words each preview was made with, so editing movie.yaml re-renders only
the groups it affects.
"""
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kms import media
from kms.packs import get_pack

COVER_CANDIDATES = 6


def inputs(p, group):
    """The parts of movie.yaml a group's previews depend on (a change means re-render)."""
    s = p.story
    base = {"fps": p.fps, "pack": p.pack_name}
    by_group = {
        "title": {"title": p.title, "villain": s.get("villain"), "programme": s.get("programme")},
        "channel": {"programme": s.get("programme"), "channels": p.data.get("channels"), "crawl": s.get("crawl"),
                    "headline": s.get("headline"), "director": p.director},
        "signal": {"title": p.title, "villain": s.get("villain"), "lead_in": _lead_in_stamp(p)},
        "voice": {"line": s.get("villain_line"), "hero": s.get("hero"), "say": s.get("hero_say"),
                  "picture": _villain_stamp(p)},
        "eyes": {"line": s.get("villain_line"), "hero": s.get("hero"), "say": s.get("hero_say"),
                 "voice": p.picks.get("voice"), "picture": _villain_stamp(p)},
        "ending": {"ending": s.get("ending")},
        "cover": {"title": p.title, "tagline": p.tagline, "cast": p.cast, "channel": p.picks.get("channel"),
                  "frames": _cover_stamp(p)},
    }
    return {**base, **by_group.get(group, {}), "seed": seed_for(p, group)}


def _stamp(path):
    try:
        st = Path(path).stat()
        return [Path(path).name, st.st_size, int(st.st_mtime)]
    except OSError:
        return None


def _lead_in_stamp(p):
    src, start = get_pack(p.pack_name).lead_in(p)
    return [_stamp(src), start] if src else None


def _villain_stamp(p):
    s = next((s for s in p.scenes if s.kind == "make" and s.target == "villain-line"), None)
    return _stamp(p.footage_file(s.get("clip"))) if s and s.get("clip") else None


def _cover_stamp(p):
    return [_stamp(f) for f in sorted((p.kit / "previews" / "cover").glob("frame-*.jpg"))
            if not f.stem.endswith("-poster")]


def _seeds_path(p):
    return p.kit / "previews" / "seeds.json"


def _seeds(p):
    try:
        return json.loads(_seeds_path(p).read_text())
    except (OSError, ValueError):
        return {}


def seed_for(p, group):
    return int(_seeds(p).get(group, 0))


def bump_seeds(p, groups):
    """New variations: a new random seed for each group, so its previews re-render differently."""
    s = _seeds(p)
    for g in groups:
        s[g] = int(s.get(g, 0)) + 1
    _seeds_path(p).parent.mkdir(parents=True, exist_ok=True)
    _seeds_path(p).write_text(json.dumps(s))
    return list(groups)


def digest(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


class Manifest:
    def __init__(self, p):
        self.path = p.kit / "previews" / "manifest.json"
        try:
            self.data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.data = {}

    def fresh(self, key, want, path):
        return self.data.get(key) == want and media.readable(path) if Path(path).suffix == ".mp4" \
            else self.data.get(key) == want and Path(path).exists()

    def set(self, key, value):
        self.data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True))


def preview_file(p, group, option):
    if group == "cover":
        return p.kit / "previews" / "cover" / f"{option}-poster.jpg"
    return p.preview_path(group, option)


def web_file(p, group, option):
    ext = ".jpg" if group == "cover" else ".mp4"
    return p.kit / "web" / f"{group}-{option}{ext}"


def cover_candidates(p, n=COVER_CANDIDATES):
    """Pull the sharpest frames from the footage as poster candidates (frame-1.jpg …)."""
    import cv2
    out = p.kit_dir("previews", "cover")
    clips = [p.footage_file(f) for _, f, kind in p.footage_needed() if kind == "shot"]
    clips = [c for c in dict.fromkeys(clips) if media.readable(c)]
    keyed = sorted((p.kit / "keyed").glob("*.mp4")) if (p.kit / "keyed").exists() else []
    clips = [k for k in keyed if not k.stem.endswith(("-green", "-background"))] + clips
    if not clips:
        return []
    picks = []
    per = max(1, -(-n // len(clips)))
    for clip in clips:
        d = media.duration(clip) or 0
        scored = []
        for k in range(per * 3):
            at = d * (k + 0.5) / (per * 3)
            tmp = out / f".probe-{k}.jpg"
            media.extract_frame(clip, at, tmp, width=480)
            img = cv2.imread(str(tmp), 0)
            tmp.unlink(missing_ok=True)
            if img is not None:
                scored.append((cv2.Laplacian(img, cv2.CV_64F).var() * (img.std() + 1), at))
        for _, at in sorted(scored, reverse=True)[:per]:
            picks.append((clip, at))
    files = []
    for i, (clip, at) in enumerate(picks[:n], start=1):
        f = out / f"frame-{i}.jpg"
        media.extract_frame(clip, at, f)
        files.append(f)
    return files


def jobs(p, groups=None):
    """[(group, option)] for everything that needs rendering, given the manifest."""
    pack = get_pack(p.pack_name)
    man = Manifest(p)
    todo = []
    for g in pack.resolved_groups(p):
        if g.kind == "text" or (groups and g.key not in groups):
            continue
        want = digest(inputs(p, g.key))
        for o in g.options:
            if g.locked and o.id != g.locked:
                continue
            if not man.fresh(f"{g.key}/{o.id}", want, preview_file(p, g.key, o.id)):
                todo.append((g.key, o.id))
    return todo


def render_one(p, group, option, ctx):
    """Render one option and its web copy (used in a worker process)."""
    pack = get_pack(p.pack_name)
    out = preview_file(p, group, option)
    out.parent.mkdir(parents=True, exist_ok=True)
    pack.render_option(p, group, option, out, ctx)
    web = web_file(p, group, option)
    web.parent.mkdir(parents=True, exist_ok=True)
    if group == "cover":
        from PIL import Image
        Image.open(out).convert("RGB").resize((480, 720)).save(web, quality=85)
    else:
        media.web_copy(out, web)
        thumb(web, THUMB_AT.get(group, 0.66))
    return out


# where in each preview its still is taken: once the title, logo or card has landed
THUMB_AT = {"title": 0.8, "channel": 0.88, "signal": 0.85, "voice": 0.5, "eyes": 0.45, "ending": 0.6}


def thumb(web, at=0.66):
    """A still for the pick page to show before the video plays (iPads show nothing otherwise)."""
    media.extract_frame(web, (media.duration(web) or 1) * at, web.with_suffix(".jpg"))
    return web.with_suffix(".jpg")


def render_all(p, groups=None, jobs_n=3, log=print):
    """Render every stale option, `jobs_n` at a time, each in its own process."""
    if (groups is None or "cover" in groups) and not _cover_stamp(p):
        cover_candidates(p)
        p.reload()
    todo = jobs(p, groups)
    if not todo:
        log("  every preview is up to date")
        return []
    log(f"  rendering {len(todo)} previews, {jobs_n} at a time. The pick page fills in as they finish.")
    done, failed = [], []
    man = Manifest(p)
    wants = {g: digest(inputs(p, g)) for g in {g for g, _ in todo}}

    def work(job):
        g, o = job
        r = subprocess.run([sys.executable, "-m", "kms", "_render", "--project", str(p.path), g, o],
                           capture_output=True, text=True)
        return job, r

    with ThreadPoolExecutor(max_workers=jobs_n) as ex:
        for fut in as_completed([ex.submit(work, j) for j in todo]):
            (g, o), r = fut.result()
            if r.returncode == 0:
                done.append((g, o))
                man.set(f"{g}/{o}", wants[g])  # only this process writes the manifest
                log(f"  ok    {g}/{o}")
            else:
                failed.append((g, o))
                log(f"  FAIL  {g}/{o}: {(r.stderr or r.stdout).strip().splitlines()[-1:] or '?'}")
    if failed:
        raise RuntimeError(f"{len(failed)} previews failed: " + ", ".join(f"{g}/{o}" for g, o in failed))
    return done


def status(p):
    """{group: {option: ready?}} for the pick page and the guide."""
    pack = get_pack(p.pack_name)
    out = {}
    for g in pack.resolved_groups(p):
        out[g.key] = {o.id: (g.kind == "text" or web_file(p, g.key, o.id).exists()) for o in g.options}
    return out
