"""kms: kid movie studio. Make a real-looking movie with your kid.

Start here:   kms init my-film   then   cd my-film && kms next
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from kms import guide, media
from kms.project import Project, ProjectError

EXAMPLE = Path(__file__).parent / "examples" / "robots" / "movie.yaml"


def say(msg=""):
    print(msg, flush=True)


def cmd_init(a):
    root = Path(a.dir)
    if (root / "movie.yaml").exists():
        raise ProjectError(f"{root / 'movie.yaml'} already exists. `cd {root}` and run `kms next`.")
    root.mkdir(parents=True, exist_ok=True)
    (root / "footage").mkdir(exist_ok=True)
    (root / "kit").mkdir(exist_ok=True)
    text = EXAMPLE.read_text()
    if a.director:
        text = text.replace("  director: Sam ", f"  director: {a.director} ", 1)
    if a.title:
        text = text.replace("title: ROBOTS REVENGE", f"title: {a.title.upper()}", 1)
    (root / "movie.yaml").write_text(text)
    say(f"🎬 A new film, in {root}/\n")
    say("  movie.yaml   what it's about, who's in it, the running order (it starts as an example: Robots Revenge)")
    say("  footage/     put your camera clips here")
    say("  kit/         everything the studio makes\n")
    say(f"Next: cd {root} && kms next")


def cmd_next(a):
    p = Project.find()
    if a.json:
        steps = guide.steps(p)
        cur = guide.current(steps)
        say(json.dumps({"title": p.title, "director": p.director, "current": cur.id if cur else None,
                        "steps": [s.to_dict() for s in steps]}, indent=1))
    else:
        say(guide.next_text(p))


def cmd_guide(a):
    say(guide.guide_text(Project.find()))


def cmd_done(a):
    p = Project.find()
    known = [s.id for s in guide.steps(p)]
    if a.step not in known:
        raise ProjectError(f"There's no step called {a.step!r}. Steps: {', '.join(known)}")
    p.mark(a.step, done=not a.undo)
    say(guide.next_text(p))


def cmd_check(a):
    p = Project.find()
    shoot = next(s for s in guide.steps(p) if s.id == "shoot")
    say("\n".join(shoot.body[1:]) or "No footage is needed for this running order.")
    if shoot.problems:
        say("\nFix these:\n" + "\n".join(f"  ⚠ {x}" for x in shoot.problems))
        sys.exit(1)


def cmd_analyse(a):
    from kms.analyse import analyse
    p = Project.find()
    report = analyse(p, transcripts=not a.no_transcripts)
    if report:
        say(f"\nWritten to {report.relative_to(p.root)}")


def cmd_key(a):
    from kms.key import key_all
    media.require_ffmpeg()
    p = Project.find()
    done = key_all(p, only=a.scenes or None, slices=a.slices, green=not a.no_green)
    say("Nothing to key: no `key:` scenes in movie.yaml." if not done else "\nNext: kms next")


def cmd_previews(a):
    from kms import previews
    media.require_ffmpeg()
    p = Project.find()
    if a.fresh:
        seeds = previews.bump_seeds(p, a.fresh)
        say(f"  fresh options for: {', '.join(seeds)}")
    previews.render_all(p, groups=a.group or a.fresh or None, jobs_n=a.jobs)
    say("\nNext: kms picks   (the director's turn!)")


def cmd_render_one(a):
    from kms import previews
    from kms.render import Ctx
    p = Project(a.project)
    ctx = Ctx(fps=p.fps, seed=previews.seed_for(p, a.group), workers=a.workers)
    previews.render_one(p, a.group, a.option, ctx)


def cmd_picks(a):
    from kms.picks import server
    p = Project.find()
    if a.set:
        updates = dict(kv.split("=", 1) for kv in a.set)
        p.set_picks(server.validate(p, updates))
    if a.import_file:
        data = json.loads(Path(a.import_file).read_text() if a.import_file != "-" else sys.stdin.read())
        server.import_picks(p, data)
    if a.set or a.import_file or a.show:
        from kms.packs import get_pack
        pack = get_pack(p.pack_name)
        for g in pack.resolved_groups(p):
            oid, picked = pack.choice(p, g.key)
            o = g.option(oid)
            mark = "★" if picked else ("🔒" if g.locked else "·")
            say(f"  {mark} {g.heading:<26} {o.name if o else '–'}" + ("" if picked or g.locked else "  (not picked yet)"))
        if p.picks.get("order"):
            say(f"    Running order: {' → '.join(p.picks['order'])}")
        if p.picks.get("note"):
            say(f"    Note from the director: {p.picks['note']}")
        return
    if a.artifact:
        files = server.build_artifact(p, a.artifact)
        say(json.dumps(files, indent=1))
        return
    server.serve(p, port=a.port, host="127.0.0.1" if a.local_only else "0.0.0.0")


def cmd_assemble(a):
    from kms.assemble import assemble
    from kms.render import Ctx
    media.require_ffmpeg()
    p = Project.find()
    out, pieces = assemble(p, a.out, ctx=Ctx(fps=p.fps))
    skipped = [x for x in pieces if not x.ready]
    lufs, peak = media.loudness(out)
    say(f"\n🎞  {out}  ({media.duration(out):.0f} s, {lufs:.1f} LUFS, peak {peak:.1f} dBFS)")
    if skipped:
        say(f"   {len(skipped)} scene(s) skipped because they aren't ready; `kms next` says what to do.")
    say("\nNext: kms next")


def cmd_export(a):
    from kms import export
    p = Project.find()
    if not (a.phone or a.imovie or a.media_server):
        a.phone = True
    if a.phone:
        say(f"  phone copy: {export.phone(p)}")
    if a.imovie:
        say(f"  iMovie pieces: {export.imovie(p)}   (File › Import Media in iMovie)")
    if a.media_server:
        say(f"  media-server folder: {export.media_server(p, None if a.media_server is True else a.media_server)}")


def cmd_demo(a):
    from kms.demo import make_demo
    media.require_ffmpeg()
    root = Path(a.dir)
    if (root / "movie.yaml").exists() and not a.force:
        raise ProjectError(f"{root} already has a movie.yaml (use --force to overwrite the demo).")
    say("🤖 Making a demo film with toy robots (no real people in it)…")
    movie = make_demo(root, fps=a.fps, short=a.short, log=say)
    if a.film:
        import random
        from kms import previews
        from kms.assemble import assemble
        from kms.key import key_all
        from kms.packs import get_pack
        p = Project(movie)
        key_all(p, log=say)
        previews.render_all(p, jobs_n=a.jobs, log=say)
        pack = get_pack(p.pack_name)
        rnd = random.Random(a.seed)
        p.set_picks({g.key: rnd.choice(g.options).id for g in pack.resolved_groups(p) if g.options})
        out, _ = assemble(p, log=say)
        p.mark("plan")
        say(f"\n🎞  {out}")
    say(f"\nNext: cd {root} && kms next")


def cmd_skill(a):
    src = Path(__file__).parent / "skill" / "SKILL.md"
    if not a.install:
        say(f"The Claude Code skill is at {src}\nInstall it with: kms skill --install")
        return
    dest = Path.home() / ".claude" / "skills" / "kid-movie-director"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest / "SKILL.md")
    say(f"Installed: {dest / 'SKILL.md'}\nIn Claude Code, say \"let's make a movie with my kid\" or /kid-movie-director.")


def build_parser():
    ap = argparse.ArgumentParser(prog="kms", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", metavar="COMMAND")

    s = sub.add_parser("init", help="start a new film folder")
    s.add_argument("dir", nargs="?", default="my-film")
    s.add_argument("--title")
    s.add_argument("--director", help="the kid's first name or nickname")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("next", help="what to do next, who does it, and what to say")
    s.add_argument("--json", action="store_true", help="machine-readable (for the Claude Code director skill)")
    s.set_defaults(fn=cmd_next)

    s = sub.add_parser("guide", help="the whole plan for the day, step by step")
    s.set_defaults(fn=cmd_guide)

    s = sub.add_parser("done", help="tick off a step you do away from the computer (plan, watch, share)")
    s.add_argument("step")
    s.add_argument("--undo", action="store_true")
    s.set_defaults(fn=cmd_done)

    s = sub.add_parser("check", help="check the footage against the shot list")
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("analyse", aliases=["analyze"], help="contact sheets, sound strips and transcripts of the footage")
    s.add_argument("--no-transcripts", action="store_true")
    s.set_defaults(fn=cmd_analyse)

    s = sub.add_parser("key", help="green-screen the plain-wall scenes")
    s.add_argument("scenes", nargs="*", help="scene ids (default: all `key:` scenes)")
    s.add_argument("--slices", type=int, default=3, help="parallel slices per take")
    s.add_argument("--no-green", action="store_true", help="skip the pure-green copy for iMovie")
    s.set_defaults(fn=cmd_key)

    s = sub.add_parser("previews", help="render every option for the director to choose from")
    s.add_argument("--group", action="append", help="only this choice group (repeatable)")
    s.add_argument("--fresh", action="append", help="new variations for this group (repeatable)")
    s.add_argument("--jobs", type=int, default=3, help="previews to render at once")
    s.set_defaults(fn=cmd_previews)

    s = sub.add_parser("picks", help="the director's pick page (serve it, or show/set/import picks)")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--local-only", action="store_true", help="only this computer can open it")
    s.add_argument("--artifact", metavar="DIR", help="build a static copy to publish as a claude.ai artifact")
    s.add_argument("--import", dest="import_file", metavar="JSON", help="load picks from a JSON file ('-' for stdin)")
    s.add_argument("--set", nargs="+", metavar="GROUP=OPTION", help="set picks by hand")
    s.add_argument("--show", action="store_true", help="show the current picks")
    s.set_defaults(fn=cmd_picks)

    s = sub.add_parser("assemble", help="put the film together from the picks")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_assemble)

    s = sub.add_parser("export", help="a phone copy, iMovie pieces, or a media-server folder")
    s.add_argument("--phone", action="store_true")
    s.add_argument("--imovie", action="store_true")
    s.add_argument("--media-server", nargs="?", const=True, metavar="DIR")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("skill", help="the Claude Code assistant-director skill")
    s.add_argument("--install", action="store_true", help="copy it into ~/.claude/skills/")
    s.set_defaults(fn=cmd_skill)

    s = sub.add_parser("demo", help="a demo film with toy robots, no real people")
    s.add_argument("dir", nargs="?", default="robot-demo")
    s.add_argument("--film", action="store_true", help="also render, pick at random and assemble")
    s.add_argument("--short", action="store_true", help="shorter clips (for tests)")
    s.add_argument("--fps", type=int, default=25)
    s.add_argument("--jobs", type=int, default=3)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_demo)
    return ap


def render_parser():
    """Internal: `kms _render` renders one preview in its own process (kms previews runs several)."""
    ap = argparse.ArgumentParser(prog="kms _render")
    ap.add_argument("--project", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("group")
    ap.add_argument("option")
    ap.set_defaults(fn=cmd_render_one)
    return ap


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["_render"]:
        ap, argv = render_parser(), argv[1:]
    else:
        ap = build_parser()
    a = ap.parse_args(argv)
    if not getattr(a, "fn", None):
        ap.print_help()
        say("\nNew here? Try:  kms demo   (a toy-robot film)   or   kms init my-film")
        return 0
    try:
        a.fn(a)
    except ProjectError as e:
        say(f"⚠ {e}")
        return 2
    except FileNotFoundError as e:
        say(f"⚠ {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
