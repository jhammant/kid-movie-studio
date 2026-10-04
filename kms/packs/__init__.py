"""Packs: the looks and sounds a film can have, as choices for the kid.

A pack declares choice groups (the opening title, the channel name, the cut-out...). Each
group has options with a kid-friendly name and a one-line description, and the pack knows
how to render a real preview of each option with the project's own words. The kid picks
on the pick page; `kms assemble` uses only the picks.
"""
from dataclasses import dataclass, field


@dataclass
class Option:
    id: str
    name: str                 # what the kid sees: "Robot computer"
    desc: str = ""            # one line: "Green computer screen typing itself out, with beeps."
    params: dict = field(default_factory=dict)


@dataclass
class Group:
    key: str                  # "title"
    heading: str              # "Opening title"
    blurb: str                # "The very first thing on screen."
    options: list
    kind: str = "video"       # video | image | text: what the pick page shows for each option
    default: str | None = None
    locked: str | None = None  # a grown-up locked this choice; it isn't offered

    def option(self, oid):
        return next((o for o in self.options if o.id == oid), None)

    @property
    def default_id(self):
        if self.locked:
            return self.locked
        if self.default and self.option(self.default):
            return self.default
        return self.options[0].id if self.options else None

    def to_dict(self):
        return {"key": self.key, "heading": self.heading, "blurb": self.blurb, "kind": self.kind,
                "default": self.default_id, "locked": self.locked,
                "options": [{"id": o.id, "name": o.name, "desc": o.desc} for o in self.options]}


class Pack:
    """Base class: subclasses fill in groups(), render_option() and make()."""
    name = "base"

    def groups(self, project):
        raise NotImplementedError

    def render_option(self, project, group, option, out, ctx):
        raise NotImplementedError

    def make(self, project, scene, out, ctx):
        raise NotImplementedError

    # shared plumbing
    def resolved_groups(self, project):
        """The pack's groups after the grown-up's hide/lock settings in movie.yaml."""
        out = []
        for g in self.groups(project):
            rule = project.choices.get(g.key) or {}
            hide = set(rule.get("hide") or [])
            g.options = [o for o in g.options if o.id not in hide] or g.options
            lock = rule.get("lock")
            if lock and g.option(str(lock)):
                g.locked = str(lock)
            out.append(g)
        return out

    def group(self, project, key):
        return next((g for g in self.resolved_groups(project) if g.key == key), None)

    def choice(self, project, key):
        """(option id, came_from_pick) for a group: the lock, the kid's pick, else the default."""
        g = self.group(project, key)
        if g is None:
            return None, False
        if g.locked:
            return g.locked, False
        picked = project.picks.get(key)
        if picked is not None and g.option(str(picked)):
            return str(picked), True
        return g.default_id, False


def get_pack(name):
    if name == "news":
        from kms.packs.news import NewsPack
        return NewsPack()
    raise KeyError(f"There's no {name!r} pack yet. Packs available: news.")
