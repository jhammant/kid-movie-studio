"""Fill the picks page with whichever preview videos exist, as small web copies.

Prints the `files` mapping (published path -> local web copy) as JSON for the Artifact publish.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

PREVIEWS = Path("kit/previews")
OPTIONS = {
    ("title", "A"): "Title A - Blockbuster.mp4",
    ("title", "B"): "Title B - Comic Book.mp4",
    ("title", "C"): "Title C - Robot Computer.mp4",
    ("title", "D"): "Title D - Warning.mp4",
    ("channel", "A"): "News A - SBC News.mp4",
    ("channel", "B"): "News B - Robot Watch 9.mp4",
    ("channel", "C"): "News C - Nine OClock News.mp4",
    ("channel", "D"): "News D - Penny Sparks Live.mp4",
    ("signal", "A"): "Signal A - Glitch Static Beep.mp4",
    ("signal", "B"): "Signal B - Robots Hacked.mp4",
    ("signal", "C"): "Signal C - Just Static.mp4",
    ("voice", "A"): "voices/Voice A - Deep and slow.mp4",
    ("voice", "B"): "voices/Voice B - Classic robot.mp4",
    ("voice", "C"): "voices/Voice C - Creepy whisper.mp4",
    ("voice", "D"): "voices/Voice D - Glitchy evil.mp4",
    ("voice", "E"): "voices/Voice E - Monster chorus.mp4",
    ("voice", "F"): "voices/Voice F - Sinister original.mp4",
}


def main(page_path):
    page = Path(page_path)
    vdir = page.parent / "v"
    vdir.mkdir(exist_ok=True)
    html = page.read_text()
    files = {}
    for (key, oid), name in OPTIONS.items():
        src = PREVIEWS / name
        if not src.exists():
            continue
        # an agent may still be writing it: skip files ffprobe can't read yet
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", str(src)], capture_output=True, text=True)
        if probe.returncode != 0 or not probe.stdout.strip():
            continue
        web = vdir / f"{key}-{oid.lower()}.mp4"
        if not web.exists() or web.stat().st_mtime < src.stat().st_mtime:
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", "scale=960:-2", "-c:v", "libx264",
                 "-crf", "25", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                 "-movflags", "+faststart", str(web)], check=True)
        rel = f"v/{web.name}"
        files[rel] = str(web)
        # set this option's src in the SECTIONS table
        pat = re.compile(r'(\{ id: "%s", [^\n]*?src: )(null|"[^"]*")' % oid)
        block_start = html.index(f'key: "{key}"')
        block_end = html.index("]}", block_start)
        block = pat.sub(lambda m: f'{m.group(1)}"{rel}"', html[block_start:block_end], count=1)
        html = html[:block_start] + block + html[block_end:]
    page.write_text(html)
    print(json.dumps(files))


if __name__ == "__main__":
    main(sys.argv[1])
