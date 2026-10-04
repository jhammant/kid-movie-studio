"""A film project: movie.yaml, its folders, and the director's picks.

    my-film/
      movie.yaml        what the film is about, who's in it, the running order, the picks
      footage/          your camera clips (never leaves your computer)
      kit/              everything the studio makes: previews, pieces, the finished film
"""
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

PICKS_HEADER = ("# ---- The director's picks. The pick page writes these; `kms assemble` reads only these.\n"
                "# Leave this block last in the file.\n")
SCENE_KINDS = ("pick", "make", "clip", "key")


class ProjectError(Exception):
    """Something in movie.yaml needs fixing; the message says what, in plain words."""


@dataclass
class Scene:
    id: str
    kind: str      # pick | make | clip | key
    target: str    # the group, the thing to make, or the footage file
    raw: dict = field(default_factory=dict)

    def get(self, key, default=None):
        return self.raw.get(key, default)

    @property
    def label(self):
        return self.raw.get("label") or self.id.replace("-", " ").capitalize()


class Project:
    def __init__(self, path):
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise ProjectError(f"No movie.yaml at {self.path}. Start one with `kms init`.")
        self.root = self.path.parent
        self.reload()

    def reload(self):
        try:
            self.data = yaml.safe_load(self.path.read_text()) or {}
        except yaml.YAMLError as e:
            raise ProjectError(f"movie.yaml has a typo YAML can't read: {e}") from e
        if not isinstance(self.data, dict):
            raise ProjectError("movie.yaml should be a list of `key: value` settings.")
        self.scenes = self._scenes()
        return self

    @classmethod
    def find(cls, start=None):
        """The movie.yaml in this folder or the nearest parent."""
        here = Path(start or Path.cwd()).resolve()
        if here.is_file():
            return cls(here)
        for d in [here, *here.parents]:
            if (d / "movie.yaml").exists():
                return cls(d / "movie.yaml")
        raise ProjectError("There's no movie.yaml here. Make one with `kms init my-film`, "
                           "or run this inside your film's folder.")

    # ---- what the film is
    @property
    def title(self):
        return str(self.data.get("title") or "MY MOVIE").strip()

    @property
    def tagline(self):
        return str(self.data.get("tagline") or "").strip()

    @property
    def fps(self):
        fps = int(self.data.get("fps") or 25)
        if fps not in (24, 25, 30):
            raise ProjectError(f"fps is {fps}; use 25 (UK/EU) or 30 (US).")
        return fps

    @property
    def pack_name(self):
        return str(self.data.get("pack") or "news")

    @property
    def crew(self):
        return self.data.get("crew") or {}

    @property
    def director(self):
        return str(self.crew.get("director") or "The Director")

    @property
    def grownup(self):
        return str(self.crew.get("grownup") or "")

    @property
    def cast(self):
        return [c for c in (self.data.get("cast") or []) if isinstance(c, dict) and c.get("character")]

    @property
    def story(self):
        return self.data.get("story") or {}

    def story_text(self, key, default=""):
        value = self.story.get(key, default)
        return value if isinstance(value, list) else str(value if value is not None else default)

    @property
    def choices(self):
        return self.data.get("choices") or {}

    @property
    def picks(self):
        p = self.data.get("picks") or {}
        return p if isinstance(p, dict) else {}

    def fill(self, text):
        """Fill {director}, {initial}, {grownup}, {title} in a piece of text from movie.yaml."""
        d = self.director
        return (str(text).replace("{director}", d).replace("{initial}", d[:1].upper())
                .replace("{grownup}", self.grownup).replace("{title}", self.title))

    # ---- folders
    @property
    def footage(self):
        return self.root / "footage"

    @property
    def kit(self):
        return self.root / "kit"

    def kit_dir(self, *parts):
        d = self.kit.joinpath(*parts)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def footage_file(self, name):
        p = Path(name)
        return p if p.is_absolute() else self.footage / p

    def preview_path(self, group, option, ext=".mp4"):
        return self.kit / "previews" / group / f"{option}{ext}"

    @property
    def film_path(self):
        safe = re.sub(r"[^\w\s-]", "", self.title).strip() or "film"
        return self.kit / f"{safe.title()}.mp4"

    # ---- the running order
    def _scenes(self):
        out, seen = [], set()
        for i, raw in enumerate(self.data.get("scenes") or []):
            if not isinstance(raw, dict):
                raise ProjectError(f"Scene {i + 1} should be a set of `key: value` lines.")
            kinds = [k for k in SCENE_KINDS if k in raw]
            if "make" in kinds:  # a made piece may use a clip, e.g. the villain's line over a close-up
                kinds = [k for k in kinds if k != "clip"]
            if len(kinds) != 1:
                raise ProjectError(f"Scene {i + 1} needs exactly one of {', '.join(SCENE_KINDS)} "
                                   f"(it has {', '.join(kinds) or 'none'}).")
            kind = kinds[0]
            target = str(raw[kind])
            sid = str(raw.get("id") or f"{kind}-{Path(target).stem}")
            if sid in seen:
                raise ProjectError(f"Two scenes are called {sid!r}; give one a different id.")
            seen.add(sid)
            out.append(Scene(sid, kind, target, raw))
        return out

    def scene(self, sid):
        return next((s for s in self.scenes if s.id == sid), None)

    def ordered_scenes(self):
        """Scenes in the director's running order (picks.order), else as written."""
        order = self.picks.get("order")
        if not order:
            return list(self.scenes)
        by_id = {s.id: s for s in self.scenes}
        out = [by_id[i] for i in order if i in by_id]
        return out + [s for s in self.scenes if s.id not in order]  # scenes added since

    def footage_needed(self):
        """[(scene, file, what it's for)] for every footage file the running order uses."""
        need = []
        for s in self.scenes:
            if s.kind in ("clip", "key"):
                need.append((s, s.target, "shot"))
            if s.kind == "key" and s.get("background"):
                need.append((s, s.get("background"), "background"))
            if s.kind == "make" and s.get("clip"):
                need.append((s, s.get("clip"), "shot"))
        return need

    # ---- picks
    def set_picks(self, updates, replace=False):
        """Save picks into the picks: block at the end of movie.yaml, keeping everything above."""
        picks = {} if replace else dict(self.picks)
        for k, v in updates.items():
            if v is None or v == "":
                picks.pop(k, None)
            else:
                picks[k] = v
        picks["updated"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        text = self.path.read_text()
        m = re.search(r"^picks:.*$", text, flags=re.M)
        head = text[:m.start()] if m else text.rstrip("\n") + "\n\n" + PICKS_HEADER
        block = yaml.safe_dump({"picks": picks}, sort_keys=False, allow_unicode=True, width=100)
        tmp = self.path.with_suffix(".yaml.tmp")
        tmp.write_text(head + block)
        tmp.replace(self.path)
        self.reload()
        return self.picks

    # ---- progress through the guide
    @property
    def progress_path(self):
        return self.kit / "progress.json"

    def progress(self):
        try:
            return json.loads(self.progress_path.read_text())
        except (OSError, ValueError):
            return {}

    def mark(self, step, done=True):
        p = self.progress()
        if done:
            p[step] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        else:
            p.pop(step, None)
        self.kit.mkdir(parents=True, exist_ok=True)
        self.progress_path.write_text(json.dumps(p, indent=2))
        return p
