"""Rolling end credits: the film's title, then who did what, rolling up a black screen.

    python -m kms.render.credits out.mp4 [--title "ROBOTS REVENGE"] [--entries credits.json]

`entries` is a list of credit lines, each one of:

    ("title", text)          the film's title, big and red
    ("gap", px)              empty space
    ("role", role, name)     two columns: the job on the left, the person on the right
    ("heading", text)        a yellow section heading, e.g. CAST
    ("small", text)          a small grey line, e.g. where it was filmed

The roll moves at the example's speed (its credits took 16 s), so it lasts as long as the
content needs, then holds on the last line for 3 s and fades out. The audio is silence.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from kms import fonts, media
from kms.render import Ctx, H, SR, W, split_title

SAFE_X = 96            # title-safe margin
GUTTER = 30            # half the gap between the role and name columns
COL_W = W // 2 - GUTTER - SAFE_X
ROLL_SPEED = 117.5     # px per second: the example's 1880 px roll took 16 s
HOLD = 3.0             # seconds on the last line before the end
FADE = 1.0             # fade to black over the last second
END_Y = H // 2 - 40    # where the bottom of the last line comes to rest

TITLE = "ROBOTS REVENGE"
ENTRIES = (
    ("gap", 120),
    ("role", "Director", "Sam"),
    ("role", "Second-in-Command Director", "Alex"),
    ("role", "Tech Editor", "Alex"),
    ("role", "Producers", "Alex & Sam"),
    ("gap", 110),
    ("heading", "CAST"),
    ("gap", 30),
    ("role", "Rex Newsome", "Alex"),
    ("role", "Penny Sparks", "Sam"),
    ("gap", 160),
    ("small", "Filmed on location at Robot HQ"),
)

# kind: (font role, size, colour, row advance, extra line spacing when wrapped)
STYLE = {
    "title": ("avenir-bold", 110, (235, 40, 40), 150, 128),
    "heading": ("avenir-bold", 56, (255, 210, 60), 90, 66),
    "role": ("avenir-medium", 46, (170, 170, 180), 95, 56),
    "name": ("avenir-bold", 54, (255, 255, 255), 95, 64),
    "small": ("avenir-medium", 38, (150, 150, 160), 70, 48),
}


def default_entries(title=TITLE):
    """The example credits, with the film's title on top."""
    return [("title", title)] + [tuple(e) for e in ENTRIES]


def prepare_entries(title, entries):
    """The credit lines to roll: the example for None, else `entries` with the title on top
    unless they already have a ("title", ...) line."""
    if entries is None:
        return default_entries(title)
    entries = [tuple(e) for e in entries]
    if title and not any(e[0] == "title" for e in entries):
        entries = [("title", title), ("gap", 120)] + entries
    return entries


def _width(fnt, text):
    return fnt.getlength(text)


def _wrap(text, fnt, max_w, max_lines):
    """Lines of text that each fit max_w, splitting between words (balanced for two lines),
    or None if it can't be done in max_lines."""
    if _width(fnt, text) <= max_w:
        return [text]
    words = text.split()
    if len(words) < 2 or max_lines < 2:
        return None
    best = None
    for k in range(1, len(words)):  # the most even two-line split
        lines = [" ".join(words[:k]), " ".join(words[k:])]
        widest = max(_width(fnt, line) for line in lines)
        if widest <= max_w and (best is None or widest < best[0]):
            best = (widest, lines)
    if best:
        return best[1]
    lines, cur = [], []
    for w in words:  # greedy, for more lines
        if cur and _width(fnt, " ".join(cur + [w])) > max_w:
            lines.append(" ".join(cur))
            cur = []
        cur.append(w)
    lines.append(" ".join(cur))
    if len(lines) <= max_lines and all(_width(fnt, line) <= max_w for line in lines):
        return lines
    return None


def fit_lines(text, kind, max_w, max_lines=2, min_scale=0.7, nudge=0.9):
    """(font, lines, line spacing) for text in a style: one line at full size if it fits (or a
    touch smaller, down to `nudge`), else wrapped, shrinking to min_scale, else whatever it
    takes to stay inside max_w."""
    role, size, _, _, spacing = STYLE[kind]
    s = size
    while s >= size * nudge:  # a little smaller on one line beats wrapping
        fnt = fonts.font(role, s)
        if _width(fnt, text) <= max_w:
            return fnt, [text], spacing * s / size
        s -= 1
    s = size
    while s >= size * min_scale:
        fnt = fonts.font(role, s)
        lines = _wrap(text, fnt, max_w, max_lines)
        if lines:
            return fnt, lines, spacing * s / size
        s -= 2
    fnt = fonts.font(role, size * min_scale)
    lines = _wrap(text, fnt, max_w, 4) or [text]
    fnt = min((fonts.fit(role, line, max_w, start=int(size * min_scale)) for line in lines), key=lambda f: f.size)
    return fnt, lines, spacing * fnt.size / size


def fit_title(text, max_w):
    """(font, lines, spacing) for the film's title: one line, or up to three when one line
    would have to shrink below three quarters of full size."""
    role, size, _, _, spacing = STYLE["title"]
    fnt = fonts.fit(role, text, max_w, start=size)
    lines = [text]
    if fnt.size < size * 0.75 and len(text.split()) > 1:
        lines = split_title(text, max_lines=3)
        fnt = min((fonts.fit(role, line, max_w, start=size) for line in lines), key=lambda f: f.size)
    return fnt, lines, spacing * fnt.size / size


def layout(entries):
    """Draw operations [(x, y, text, font, colour, anchor)] on the strip, and where the content ends.
    The strip starts a screen below the frame so the first line rolls in."""
    ops, y = [], H
    for item in entries:
        kind = item[0]
        if kind == "gap":
            y += int(item[1])
        elif kind == "title":
            fnt, lines, spacing = fit_title(str(item[1]), W - 2 * SAFE_X)
            for k, line in enumerate(lines):
                ops.append((W / 2, y + k * spacing, line, fnt, STYLE["title"][2], "mt"))
            y += STYLE["title"][3] + (len(lines) - 1) * spacing
        elif kind in ("heading", "small"):
            fnt, lines, spacing = fit_lines(str(item[1]), kind, W - 2 * SAFE_X, max_lines=3)
            for k, line in enumerate(lines):
                ops.append((W / 2, y + k * spacing, line, fnt, STYLE[kind][2], "mt"))
            y += STYLE[kind][3] + (len(lines) - 1) * spacing
        elif kind == "role":
            # classic two-column credit: role right-aligned, name left-aligned
            rf, rlines, rsp = fit_lines(str(item[1]), "role", COL_W)
            nf, nlines, nsp = fit_lines(str(item[2]), "name", COL_W)
            for k, line in enumerate(rlines):
                ops.append((W / 2 - GUTTER, y + k * rsp, line, rf, STYLE["role"][2], "rt"))
            for k, line in enumerate(nlines):
                ops.append((W / 2 + GUTTER, y - 4 + k * nsp, line, nf, STYLE["name"][2], "lt"))
            y += STYLE["role"][3] + max((len(rlines) - 1) * rsp, (len(nlines) - 1) * nsp)
        else:
            raise ValueError(f"unknown credit kind {kind!r}: use title, gap, role, heading or small")
    return ops, int(round(y))


def build_strip(entries):
    """(the credits as one tall black image, where the content ends on it)."""
    ops, content_end = layout(entries)
    strip = Image.new("RGB", (W, content_end + H // 2 + 60), "black")
    d = ImageDraw.Draw(strip)
    for x, y, text, fnt, colour, anchor in ops:
        d.text((x, y), text, font=fnt, fill=colour, anchor=anchor)
    return strip, content_end


def timing(content_end):
    """(roll seconds, travel in px): scroll until the last line sits just above mid-screen."""
    travel = content_end - END_Y
    return travel / ROLL_SPEED, travel


def render(out, *, title=TITLE, entries=None, ctx=None):
    """Rolling end credits. entries=None gives the example; entries without a ("title", ...)
    line get `title` on top. Lasts roll + 3 s hold (19 s for the example)."""
    ctx = ctx or Ctx()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    strip, content_end = build_strip(prepare_entries(title, entries))
    secs, travel = timing(content_end)
    total = secs + HOLD
    nf = ctx.nframes(total)
    dur = nf / ctx.fps
    fd, path = tempfile.mkstemp(suffix=".png", dir=ctx.scratch("credits"))
    os.close(fd)
    try:
        strip.save(path)
        expr = f"-min(t/{secs:.6f},1)*{travel}"
        media.run([media.ffmpeg(), "-v", "error", "-y",
                   "-f", "lavfi", "-i", f"color=black:s={W}x{H}:r={ctx.fps}:d={total:.6f}",
                   "-framerate", str(ctx.fps), "-loop", "1", "-i", path,
                   "-f", "lavfi", "-i", f"anullsrc=r={SR}:cl=stereo",
                   "-filter_complex",
                   f"[0:v][1:v]overlay=x=0:y='{expr}':shortest=1,"
                   f"fade=t=out:st={total - FADE:.6f}:d={FADE},format=yuv420p[v]",
                   "-map", "[v]", "-map", "2:a", "-t", f"{dur:.6f}", "-frames:v", str(nf),
                   *media.video_args(crf=16), *media.audio_args(), "-movflags", "+faststart", out])
    finally:
        os.unlink(path)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output .mp4")
    ap.add_argument("--title", default=TITLE)
    ap.add_argument("--entries", metavar="JSON", help='a JSON list like [["role", "Director", "Sam"], ["gap", 100]]')
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--frames", type=int, help="render only the first N frames")
    args = ap.parse_args(argv)
    entries = json.loads(Path(args.entries).read_text()) if args.entries else None
    media.require_ffmpeg()
    print(render(args.out, title=args.title, entries=entries,
                 ctx=Ctx(fps=args.fps, limit_frames=args.frames)))


if __name__ == "__main__":
    main()
