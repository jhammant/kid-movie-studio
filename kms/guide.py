"""The parent's guide: what to do next, who does it, and what to say.

A film is a team effort. Some steps are the grown-up's (the computer bits), some you do
together (planning, filming, the premiere), and every creative decision is the director's
turn: the kid picks from real previews. `kms next` works out where you are from the
project's files and tells you the next step; `kms guide` prints the whole plan for the day.
"""
from dataclasses import dataclass, field

from kms import media

GROWNUP, TOGETHER, DIRECTOR = "grown-up", "together", "director"
WHO = {GROWNUP: "👤 GROWN-UP", TOGETHER: "🤝 TOGETHER", DIRECTOR: "⭐ DIRECTOR'S TURN"}


@dataclass
class Step:
    id: str
    who: str
    title: str
    minutes: str
    done: bool = False
    skip: bool = False
    body: list = field(default_factory=list)      # what to do, in plain words
    run: list = field(default_factory=list)       # commands for the grown-up
    say: str = ""                                 # something to say to your director
    tips: list = field(default_factory=list)
    problems: list = field(default_factory=list)  # things to fix before moving on

    def to_dict(self):
        return {k: getattr(self, k) for k in ("id", "who", "title", "minutes", "done", "skip", "body", "run", "say",
                                               "tips", "problems")}


def _mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return 0


def steps(p):
    """Every step for this project, each marked done or not."""
    from kms.packs import get_pack
    from kms.previews import jobs

    pack = get_pack(p.pack_name)
    prog = p.progress()
    director = p.director
    out = []

    # 1. plan --------------------------------------------------------------------------
    out.append(Step(
        "plan", TOGETHER, "Plan the film together", "10–15 min", done="plan" in prog,
        body=[f"Sit down with {director} and open movie.yaml. Read the questions in it out loud and type "
              "what they say: what the film is called, who's in it, who the villain is and what the villain says.",
              "Don't decide how it looks yet. That's what the pick page is for."],
        run=[f"open {p.path.name}   # or any text editor", "kms done plan      # when you're happy with it"],
        say=f"You're the director, {director}. That means you get to decide what everything looks like. "
            "What's our film called?",
        tips=["Short and loud beats long: two or three minutes of film is plenty.",
              "Made-up character names are half the fun. Let them pick yours too."]))

    # 2. shoot -------------------------------------------------------------------------
    needed = p.footage_needed()
    shots, problems, missing = [], [], 0
    for scene, name, kind in needed:
        f = p.footage_file(name)
        info = media.probe(f) if f.exists() else None
        ok = info is not None and info.duration > 0
        missing += 0 if ok else 1
        if kind == "background":
            what = scene.get("background_shot") or f"Something to show behind {scene.label.lower()}. No people needed."
        else:
            what = scene.get("shot") or scene.label
        what = " ".join(str(what).split())
        line = f"{'✓' if ok else '□'} footage/{name}: {what}"
        if scene.kind == "key" and kind == "shot":
            line += "  ⚠ Then film 5 seconds of the EMPTY wall: same light, camera still."
        shots.append(line)
        if info is not None:
            if info.portrait:
                problems.append(f"{name} is portrait (tall). Turn the phone sideways and film it again.")
            if info.fps and abs(info.fps - p.fps) > 1:
                problems.append(f"{name} is {info.fps:.0f} fps but movie.yaml says {p.fps}. "
                                "It'll still work; set fps in movie.yaml to match your camera for smoother motion.")
            if not info.has_audio and kind == "shot":
                problems.append(f"{name} has no sound.")
            if scene.kind == "key" and kind == "shot":
                plate = scene.get("plate") or {}
                end = float(plate.get("from", 0)) + float(plate.get("for", 0))
                if plate and end > info.duration + 0.1:
                    problems.append(f"{name}: the empty wall (plate: from {plate.get('from')} for "
                                    f"{plate.get('for')}) runs past the end of the clip ({info.duration:.1f} s).")
    out.append(Step(
        "shoot", TOGETHER, "Filming day", "30–60 min", done=missing == 0 and not problems, skip=not needed,
        body=["Here's your shot list. Film each one, copy the clips to your computer and name them like this, "
              "in the footage/ folder:", *shots],
        say="You say 'Action!' and 'Cut!'. That's the director's job. Want to do another take?",
        tips=["Film sideways (landscape), with the camera still: a tripod, or a pile of books.",
              "For the plain-wall shots, wear colours that are different from the wall. A green wall "
              "and a green jumper means an invisible jumper!",
              "Light the wall evenly and stand a step or two in front of it, so there are no hard shadows.",
              "Record a second or two before anyone talks, and after they finish.",
              "Takes don't need to be perfect. The funny mistakes are often the best bits."],
        problems=problems))

    # 3. key ---------------------------------------------------------------------------
    keyed = [s for s in p.scenes if s.kind == "key"]
    key_todo = []
    for s in keyed:
        out_f = p.kit / "keyed" / f"{s.id}.mp4"
        srcs = [p.footage_file(s.target)] + ([p.footage_file(s.get("background"))] if s.get("background") else [])
        if not out_f.exists() or any(_mtime(src) > _mtime(out_f) for src in srcs):
            key_todo.append(s.id)
    out.append(Step(
        "key", GROWNUP, "Green-screen the wall shots", "5–20 min, computer time", done=not key_todo,
        skip=not keyed,
        body=[f"The computer cuts {director} out of the plain wall and puts them in the scene "
              f"({', '.join(key_todo) or 'all done'})."],
        run=["kms key"],
        say="The computer is doing the magic bit now. Want to draw the poster while we wait?",
        tips=["It takes a minute or two for every 10 seconds of footage.",
              "If an arm or a hat goes see-through, the clothes were too close to the wall colour. "
              "Film that bit again, or just keep it: it's a robot attack, after all."]))

    # 4. previews ----------------------------------------------------------------------
    todo = jobs(p)
    out.append(Step(
        "previews", GROWNUP, "Make the choices", "5–15 min, computer time", done=not todo,
        body=[f"The studio makes a real preview of every option {director} can choose from: "
              "the title styles, the channel names, the cut-outs, the voices, the endings."
              + (f" {len(todo)} to make." if todo else "")],
        run=["kms previews", "kms picks        # start the pick page now; it fills in as previews finish"],
        say="I'm making you some choices. You're going to pick your favourites!",
        tips=["The previews are the real thing, not a sketch: what they pick is exactly what goes in the film."]))

    # 5. pick --------------------------------------------------------------------------
    groups = [g for g in pack.resolved_groups(p) if not g.locked and g.options]
    unpicked = [g.heading for g in groups if p.picks.get(g.key) is None]
    out.append(Step(
        "pick", DIRECTOR, f"{director} picks how the film looks", "10–20 min", done=not unpicked,
        body=[f"Hand the tablet (or the laptop) to {director}. They watch each preview and tap Pick. "
              "They can change their mind as often as they like.",
              ("Still to pick: " + ", ".join(unpicked)) if unpicked else "Everything is picked."],
        run=["kms picks        # prints a link (and a QR code) to open on the tablet"],
        say="Which one makes you laugh? Which one would you want to see at the cinema?",
        tips=["Don't vote! If they pick the one you like least, that's the one. You can suggest; they decide.",
              "The 'Anything to change?' box at the bottom is for their ideas. Type them in if they can't yet."]))

    # 6. assemble ----------------------------------------------------------------------
    film = p.film_path
    newest = max([_mtime(p.path)] + [_mtime(f) for f in (p.kit / "keyed").glob("*.mp4")]
                 if (p.kit / "keyed").exists() else [_mtime(p.path)])
    out.append(Step(
        "assemble", GROWNUP, "Put the film together", "2–5 min, computer time",
        done=film.exists() and _mtime(film) >= newest,
        body=["The studio joins everything in the running order, with the picks, captions and levelled sound."],
        run=["kms assemble"],
        tips=["Want to watch the film so far? Assemble any time. It skips the bits that aren't ready."]))

    # 7. watch -------------------------------------------------------------------------
    watched = prog.get("watch")
    out.append(Step(
        "watch", TOGETHER, "The premiere", "10 min", done=bool(watched) and film.exists(),
        body=["Watch it together on the biggest screen you have. Dim the lights. Popcorn is optional but recommended.",
              f"To change anything: {director} re-picks (kms picks), then kms assemble again.",
              "Want brand new options for something? kms previews --fresh title (or channel, signal, voice…)."],
        run=[f"open \"{film}\"", "kms done watch"],
        say="What was the best bit? Is there anything you'd change?",
        tips=["Ask them to say thank you to the cast in the credits. Directors do that."]))

    # 8. share -------------------------------------------------------------------------
    out.append(Step(
        "share", GROWNUP, "Share it (grown-ups only)", "5 min", done="share" in prog,
        body=["Sharing is a grown-up decision. The film has your family's faces and voices in it, so keep it in "
              "the family unless you've thought it through.",
              "kms export --phone makes a small copy to send; --imovie copies the pieces for more editing; "
              "--media-server makes a folder with a poster for Plex or Jellyfin."],
        run=["kms export --phone", "kms done share"],
        tips=["Everything so far stayed on this computer. Nothing was uploaded."]))
    return out


def current(all_steps):
    return next((s for s in all_steps if not s.done and not s.skip), None)


def render_step(s, number, total, detail=True):
    lines = [f"Step {number} of {total} · {WHO[s.who]} · {s.title}  ({s.minutes})", ""]
    if not detail:
        return lines[0]
    lines += [f"  {b}" for b in s.body]
    if s.problems:
        lines += ["", "  Fix these first:"] + [f"  ⚠ {x}" for x in s.problems]
    if s.run:
        lines += ["", "  Run:"] + [f"    {r}" for r in s.run]
    if s.say:
        lines += ["", f"  Say: \"{s.say}\""]
    if s.tips:
        lines += [""] + [f"  Tip: {t}" for t in s.tips]
    return "\n".join(lines)


def overview(all_steps):
    shown = [s for s in all_steps if not s.skip]
    cur = current(all_steps)
    marks = []
    for i, s in enumerate(shown, start=1):
        mark = "✓" if s.done else ("→" if s is cur else "·")
        marks.append(f"  {mark} {i}. {s.title:<38} {WHO[s.who]}")
    return "\n".join(marks)


def next_text(p):
    all_steps = steps(p)
    shown = [s for s in all_steps if not s.skip]
    cur = current(all_steps)
    head = f"🎬 {p.title}  ·  directed by {p.director}\n\n{overview(all_steps)}\n"
    if cur is None:
        return head + ("\nThat's a wrap! 🎉 The film is made, watched and shared.\n"
                       "Fancy a sequel? `kms init sequel` and start again: the director already knows the ropes.")
    return head + "\n" + render_step(cur, shown.index(cur) + 1, len(shown))


def guide_text(p):
    all_steps = [s for s in steps(p) if not s.skip]
    parts = [f"🎬 The plan for {p.title}, directed by {p.director}\n",
             "👤 GROWN-UP steps are computer jobs. 🤝 TOGETHER steps are the fun ones. "
             "⭐ DIRECTOR'S TURN means the kid decides.\n"]
    for i, s in enumerate(all_steps, start=1):
        parts.append(("✓ " if s.done else "") + render_step(s, i, len(all_steps)) + "\n")
    return "\n".join(parts)
