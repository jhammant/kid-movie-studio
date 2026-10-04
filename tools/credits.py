"""Rolling end credits for Robots Revenge. """
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 25
FONT = "/System/Library/Fonts/Avenir Next.ttc"  # has ë

CREDITS = [
    ("title", "ROBOTS REVENGE"),
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
]


def font(index, size):
    return ImageFont.truetype(FONT, size, index=index)


def build_strip():
    title, heading = font(0, 110), font(0, 56)  # Avenir Next Bold
    role_f, name_f, small = font(5, 46), font(0, 54), font(5, 38)
    strip = Image.new("RGB", (W, 6000), "black")
    d = ImageDraw.Draw(strip)
    y = H  # start below the frame so it rolls in
    for item in CREDITS:
        kind = item[0]
        if kind == "gap":
            y += item[1]
        elif kind == "title":
            d.text((W / 2, y), item[1], font=title, fill=(235, 40, 40), anchor="mt")
            y += 150
        elif kind == "heading":
            d.text((W / 2, y), item[1], font=heading, fill=(255, 210, 60), anchor="mt")
            y += 90
        elif kind == "role":
            # classic two-column credit: role right-aligned, name left-aligned
            d.text((W / 2 - 30, y), item[1], font=role_f, fill=(170, 170, 180), anchor="rt")
            d.text((W / 2 + 30, y - 4), item[2], font=name_f, fill="white", anchor="lt")
            y += 95
        elif kind == "small":
            d.text((W / 2, y), item[1], font=small, fill=(150, 150, 160), anchor="mt")
            y += 70
    return strip.crop((0, 0, W, y + H // 2 + 60)), y


def main(out):
    strip, content_end = build_strip()
    path = out.rsplit(".", 1)[0] + "_strip.png"
    strip.save(path)
    # scroll until the last line sits mid-screen, then hold
    travel = content_end - H // 2 + 40
    secs, hold = 16.0, 3.0
    expr = f"-min(t/{secs},1)*{travel}"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=black:s={W}x{H}:r={FPS}:d={secs + hold}",
         "-loop", "1", "-i", path, "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
         "-filter_complex", f"[0:v][1:v]overlay=x=0:y='{expr}':shortest=1,fade=t=out:st={secs + hold - 1}:d=1,format=yuv420p[v]",
         "-map", "[v]", "-map", "2:a", "-t", str(secs + hold), "-c:v", "libx264", "-crf", "16",
         "-c:a", "aac", out],
        check=True)


if __name__ == "__main__":
    main(sys.argv[1])
