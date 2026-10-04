"""A demo film with no real people in it: drawn toy robots in front of a painted wall.

    kms demo my-demo             # footage + movie.yaml, ready for `kms next`
    kms demo my-demo --film      # ...and render every option, pick at random, assemble

The wall take comes with the subject's true outline, so tests can score the keyer.
"""
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import yaml

from kms import media, tts

W, H = 1920, 1080
WALL_BGR = np.array([136, 172, 150], np.float32)   # a sage painted wall, not chroma green
EXAMPLE = Path(__file__).parent / "examples" / "robots" / "movie.yaml"

LINES = {
    "desk-intro": "Good evening. This is the nine o'clock news. The robot invasion continues. "
                  "Let's go live to Penny Sparks, on the battlefield.",
    "wall-take": "This is Penny Sparks, live from the battlefield. The robots are everywhere. Help!",
    "desk-worried": "Penny? Penny?! We seem to have lost Penny.",
}


# ---------------------------------------------------------------- drawing
def robot(img, mask, cx, cy, s, t, mouth=0.0, wave=0.0, body=(150, 120, 96), head=(196, 196, 204),
          limbs=(120, 100, 84), joints=(60, 60, 70)):
    """Draw a toy robot (BGR) centred at cx, with its feet at cy, scale s. Also fills mask."""
    def rect(x0, y0, x1, y1, col, r=0):
        p0, p1 = (int(cx + x0 * s), int(cy + y0 * s)), (int(cx + x1 * s), int(cy + y1 * s))
        for target, c in ((img, col), (mask, 255)):
            if r:
                rr = int(r * s)
                cv2.rectangle(target, (p0[0] + rr, p0[1]), (p1[0] - rr, p1[1]), c, -1, cv2.LINE_AA)
                cv2.rectangle(target, (p0[0], p0[1] + rr), (p1[0], p1[1] - rr), c, -1, cv2.LINE_AA)
                for q in ((p0[0] + rr, p0[1] + rr), (p1[0] - rr, p0[1] + rr), (p0[0] + rr, p1[1] - rr),
                          (p1[0] - rr, p1[1] - rr)):
                    cv2.circle(target, q, rr, c, -1, cv2.LINE_AA)
            else:
                cv2.rectangle(target, p0, p1, c, -1, cv2.LINE_AA)

    rect(-70, -40, -20, 0, joints)                           # legs
    rect(20, -40, 70, 0, joints)
    rect(-110, -330, 110, -40, body, r=18)                    # body
    rect(-60, -260, 60, -170, (40, 40, 200), r=10)           # chest panel
    for i in range(3):
        on = (math.sin(t * 6 + i * 2) > 0)
        cv2.circle(img, (int(cx + (-30 + 30 * i) * s), int(cy - 215 * s)), int(10 * s),
                   (60, 230, 255) if on else (40, 90, 120), -1, cv2.LINE_AA)
    a = math.radians(-20 - 70 * wave)                          # waving arm with a microphone
    sx, sy = cx + 110 * s, cy - 300 * s
    ex, ey = sx + math.cos(a) * 150 * s, sy + math.sin(a) * 150 * s
    for target, c in ((img, limbs), (mask, 255)):
        cv2.line(target, (int(sx), int(sy)), (int(ex), int(ey)), c, int(30 * s), cv2.LINE_AA)
        cv2.line(target, (int(cx - 110 * s), int(cy - 300 * s)), (int(cx - 150 * s), int(cy - 150 * s)), c,
                 int(30 * s), cv2.LINE_AA)
    for target, c in ((img, (30, 30, 30)), (mask, 255)):
        cv2.circle(target, (int(ex), int(ey)), int(26 * s), c, -1, cv2.LINE_AA)
    rect(-12, -380, 12, -330, joints)                        # neck
    rect(-120, -560, 120, -370, head, r=40)                   # head
    rect(-6, -620, 6, -560, joints)                          # antenna
    for target, c in ((img, (40, 40, 230)), (mask, 255)):
        cv2.circle(target, (int(cx), int(cy - 625 * s)), int(16 * s), c, -1, cv2.LINE_AA)
    for ex_ in (-55, 55):                                      # eyes: dark lenses
        cv2.circle(img, (int(cx + ex_ * s), int(cy - 480 * s)), int(32 * s), (20, 20, 24), -1, cv2.LINE_AA)
        cv2.circle(img, (int(cx + ex_ * s - 9 * s), int(cy - 490 * s)), int(8 * s), (230, 230, 240), -1,
                   cv2.LINE_AA)
    mh = int((6 + 34 * mouth) * s)                             # mouth opens with the voice
    cv2.rectangle(img, (int(cx - 50 * s), int(cy - 420 * s - mh / 2)), (int(cx + 50 * s), int(cy - 420 * s + mh / 2)),
                  (30, 30, 36), -1, cv2.LINE_AA)


def envelope(wav, fps, n):
    """Per-frame loudness 0..1 of a wav, for moving mouths."""
    if wav is None or not Path(wav).exists():
        return np.zeros(n)
    x, sr = media.read_wav(wav)
    x = np.abs(x.mean(1))
    hop = sr // fps
    e = np.array([x[i * hop:(i + 1) * hop].mean() if i * hop < len(x) else 0 for i in range(n)])
    return np.clip(e / (e.max() + 1e-6) * 1.4, 0, 1)


def speak_or_babble(text, out_wav, seconds, voice="Daniel"):
    """The line with text-to-speech if there is one, else robot babble, padded to `seconds`."""
    try:
        tts.speak(text, out_wav, voice=voice, rate=165)
        x, sr = media.read_wav(out_wav)
    except Exception:  # no TTS on this machine: beeps that rise and fall like speech
        sr = media.SR
        t = np.arange(int(seconds * 0.8 * sr)) / sr
        x = (0.3 * np.sin(2 * np.pi * (300 + 120 * np.sin(2 * np.pi * 3 * t)) * t) *
             (np.sin(2 * np.pi * 4 * t) > 0))[:, None]
    n = int(seconds * sr)
    y = np.zeros((n, 1), np.float32)
    lead = int(0.4 * sr)
    m = min(len(x), n - lead)
    y[lead:lead + m] = x[:m, :1]
    y += np.random.default_rng(1).normal(0, 0.003, y.shape)  # room tone
    media.write_wav(out_wav, np.repeat(y, 2, axis=1), sr)
    return out_wav


def _clip(path, seconds, fps, draw, audio=None):
    n = int(round(seconds * fps))
    with media.FrameWriter(path, fps=fps, audio=audio, pix_fmt="bgr24") as fw:
        for i in range(n):
            fw.write(draw(i, i / fps))
    return path


# ---------------------------------------------------------------- the clips
def wall_take(out, fps=25, talk=6.0, empty=3.0, truth_dir=None, seed=0):
    """The robot reporter in front of the plain wall, then it walks off and the wall is empty."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    shade = (0.92 + 0.10 * (1 - yy / H) - 0.06 * ((xx - W / 2) / W) ** 2)[..., None]
    total = talk + 1.5 + empty
    wav = speak_or_babble(LINES["wall-take"], Path(out).with_suffix(".wav"), total)
    mouth = envelope(wav, fps, int(total * fps) + 1)
    if truth_dir:
        Path(truth_dir).mkdir(parents=True, exist_ok=True)
    noise = [rng.normal(0, 2.0, (H, W, 3)).astype(np.float32) for _ in range(6)]  # sensor grain

    def draw(i, t):
        drift = 1.0 + 0.04 * math.sin(t * 0.7)               # the light drifts a little
        img = WALL_BGR * shade * drift
        img = img + noise[i % len(noise)]
        frame = np.clip(img, 0, 255).astype(np.uint8)
        mask = np.zeros((H, W), np.uint8)
        if t < talk + 1.5:
            walk = max(0.0, t - talk) / 1.5                  # walks off to the right at the end
            cx = W * 0.5 + 60 * math.sin(t * 1.3) + walk * W * 0.9
            cy = H * 0.98 + 8 * math.sin(t * 5)
            q = 4  # the soft shadow is drawn small and scaled up: it's blurry anyway
            shadow = np.zeros((H // q, W // q), np.uint8)
            robot(np.zeros((H // q, W // q, 3), np.uint8), shadow, (cx + 40) / q, cy / q, 1.25 / q, t)
            shadow = cv2.GaussianBlur(shadow, (0, 0), 25 / q).astype(np.float32) / 255 * 0.22
            shadow = cv2.resize(shadow, (W, H), interpolation=cv2.INTER_LINEAR)
            frame = np.clip(frame * (1 - shadow[..., None]), 0, 255).astype(np.uint8)
            # a bright toy: like real green screen, wear colours that aren't the wall's
            robot(frame, mask, cx, cy, 1.25, t, mouth=mouth[min(i, len(mouth) - 1)],
                  wave=0.5 + 0.5 * math.sin(t * 2.2), body=(50, 60, 205), head=(40, 175, 250),
                  limbs=(185, 95, 40), joints=(40, 40, 50))
        if truth_dir and i % 5 == 0:
            cv2.imwrite(str(Path(truth_dir) / f"{i:05d}.png"), mask)
        return frame

    _clip(out, total, fps, draw, audio=wav)
    Path(wav).unlink(missing_ok=True)
    return {"to": talk, "plate_from": talk + 1.5 + 0.3, "plate_for": empty - 0.6}


def battlefield(out, fps=25, seconds=4.0, seed=0):
    """A background: robots marching across a field under a stormy sky, with water spray."""
    rng = np.random.default_rng(seed)
    yy = np.linspace(0, 1, H)[:, None, None]
    sky = (np.array([90, 70, 60]) * (1 - yy) + np.array([170, 150, 140]) * yy).astype(np.float32)
    drops = rng.uniform(0, 1, (220, 4))

    def draw(i, t):
        img = np.broadcast_to(sky, (H, W, 3)).copy()
        cv2.rectangle(img, (0, int(H * 0.72)), (W, H), (60, 110, 80), -1)
        mask = np.zeros((H, W), np.uint8)
        for k in range(5):
            x = (k * 430 + t * 120) % (W + 300) - 150
            robot(img, mask, x, H * 0.80 + 10 * math.sin(t * 6 + k), 0.38, t + k,
                  body=(80 + 20 * k, 90, 140), head=(170, 170, 180))
        for d in drops:                                         # water spray arcing up from the left
            u = (t * 0.9 + d[0]) % 1.0
            x = 120 + d[1] * 300 + u * 900
            y = H * 0.75 - math.sin(u * math.pi) * (300 + 250 * d[2])
            cv2.circle(img, (int(x), int(y)), int(4 + 4 * d[3]), (240, 220, 200), -1, cv2.LINE_AA)
        return np.clip(img, 0, 255).astype(np.uint8)

    wav = Path(out).with_suffix(".wav")
    n = int(seconds * media.SR)
    noise = np.random.default_rng(2).normal(0, 0.05, (n, 2))
    media.write_wav(wav, noise * np.linspace(0.6, 1, n)[:, None])
    _clip(out, seconds, fps, draw, audio=wav)
    wav.unlink(missing_ok=True)


def desk(out, line, fps=25, seconds=6.0, worried=False):
    """The newsreader robot at the studio desk."""
    wav = speak_or_babble(line, Path(out).with_suffix(".wav"), seconds, voice="Daniel")
    mouth = envelope(wav, fps, int(seconds * fps) + 1)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    back = np.dstack([90 + 60 * (1 - yy / H), 40 + 30 * (1 - yy / H), 20 + 10 * (1 - yy / H)])
    rings = ((np.hypot(xx - W * 0.72, yy - H * 0.42) // 60) % 2 == 0) & (np.hypot(xx - W * 0.72, yy - H * 0.42) < 380)
    back[rings] += (30, 20, 10)

    def draw(i, t):
        img = back.copy()
        mask = np.zeros((H, W), np.uint8)
        jitter = 6 * math.sin(t * 9) if worried else 0
        robot(img, mask, W * 0.36 + jitter, H * 0.97, 1.2, t, mouth=mouth[min(i, len(mouth) - 1)])
        cv2.rectangle(img, (0, int(H * 0.74)), (W, H), (40, 60, 110), -1)
        cv2.rectangle(img, (0, int(H * 0.74)), (W, int(H * 0.76)), (90, 120, 200), -1)
        return np.clip(img, 0, 255).astype(np.uint8)

    _clip(out, seconds, fps, draw, audio=wav)
    wav.unlink(missing_ok=True)


def robot_walk(out, fps=25, seconds=3.0):
    """A giant robot foot coming down on the camera until it fills the screen."""
    def draw(i, t):
        u = t / seconds
        img = np.zeros((H, W, 3), np.float32)
        img[:] = (150, 130, 110)
        cv2.rectangle(img, (0, int(H * 0.6)), (W, H), (60, 100, 70), -1)
        s = 0.2 + 3.2 * u ** 2.2
        cx, cy = W / 2, H * 0.55 + u * H * 0.6
        cv2.rectangle(img, (int(cx - 260 * s), int(cy - 160 * s)), (int(cx + 260 * s), int(cy)), (70, 70, 80), -1)
        cv2.rectangle(img, (int(cx - 90 * s), int(cy - 600 * s)), (int(cx + 90 * s), int(cy - 150 * s)),
                      (110, 110, 120), -1)
        return np.clip(img, 0, 255).astype(np.uint8)

    wav = Path(out).with_suffix(".wav")
    n = int(seconds * media.SR)
    tt = np.arange(n) / media.SR
    stomps = sum(np.exp(-np.clip(tt - k, 0, None) * 9) * (tt >= k) * np.sin(2 * np.pi * 55 * (tt - k))
                 for k in np.arange(0.3, seconds, 0.9))
    media.write_wav(wav, np.repeat((0.5 * stomps)[:, None], 2, axis=1))
    _clip(out, seconds, fps, draw, audio=wav)
    wav.unlink(missing_ok=True)


def robot_chair(out, fps=25, seconds=6.0):
    """A close-up of a little robot sitting very still: the post-credits shot."""
    def draw(i, t):
        img = np.zeros((H, W, 3), np.float32)
        img[:] = (40, 46, 60)
        mask = np.zeros((H, W), np.uint8)
        robot(img, mask, W / 2 + 3 * math.sin(t * 1.7), H * 1.9, 2.6, 0.0, head=(225, 225, 230))
        return np.clip(img, 0, 255).astype(np.uint8)

    wav = Path(out).with_suffix(".wav")
    media.write_wav(wav, np.random.default_rng(3).normal(0, 0.004, (int(seconds * media.SR), 2)))
    _clip(out, seconds, fps, draw, audio=wav)
    wav.unlink(missing_ok=True)


# ---------------------------------------------------------------- the demo project
def make_demo(root, fps=25, short=False, log=print):
    """Write a demo project (movie.yaml + synthetic footage) into `root`. Returns movie.yaml."""
    root = Path(root)
    footage = root / "footage"
    footage.mkdir(parents=True, exist_ok=True)
    k = 0.4 if short else 1.0
    log("  drawing the robot newsreader …")
    desk(footage / "desk-intro.mp4", LINES["desk-intro"], fps, 9.0 * k if short else 9.0)
    log("  drawing the reporter in front of the painted wall …")
    w = wall_take(footage / "wall-take.mp4", fps, talk=7.0 * k, empty=3.0, truth_dir=root / "kit" / "truth")
    log("  drawing the battlefield …")
    battlefield(footage / "water-robots.mp4", fps, 4.0 * k)
    log("  drawing the giant robot foot …")
    robot_walk(footage / "robot-walk.mp4", fps, 3.0 * k if not short else 1.2)
    desk(footage / "desk-worried.mp4", LINES["desk-worried"], fps, 4.0 * k if not short else 2.0, worried=True)
    robot_chair(footage / "robot-chair.mp4", fps, 7.0 * k if not short else 3.0)

    data = yaml.safe_load(EXAMPLE.read_text())
    data["fps"] = fps
    data["crew"] = {"director": "Robo", "grownup": "Bolt"}
    data["cast"] = [{"character": "Rex Newsome", "role": "Newsreader", "actor": "Bolt"},
                    {"character": "Penny Sparks", "role": "Roving Reporter · The Battlefield", "actor": "Robo"}]
    for s in data["scenes"]:
        if s.get("id") == "desk-intro":
            s.pop("from", None), s.pop("to", None)
            s["caption_at"] = [0.6, min(5.0, 9.0 * k - 0.5)]
        if s.get("id") == "penny-live":
            s["to"] = round(w["to"], 2)
            s["plate"] = {"from": round(w["plate_from"], 2), "for": round(w["plate_for"], 2)}
            s["caption_at"] = [0.5, round(min(5.0, w["to"] - 0.5), 2)]
        if s.get("id") == "villain-line":
            s["voice_at"] = 0.5
    text = EXAMPLE.read_text()
    head = text[:text.index("title:")]
    body = yaml.safe_dump({k_: v for k_, v in data.items() if k_ != "picks"}, sort_keys=False,
                          allow_unicode=True, width=100)
    movie = root / "movie.yaml"
    movie.write_text(head.replace("# Fill this in TOGETHER.", "# A DEMO: toy robots only, drawn by the computer.\n#\n# Fill this in TOGETHER.")
                     + body + "\n# ---- The director's picks. The pick page writes these; `kms assemble` reads only these.\n"
                     "# Leave this block last in the file.\npicks: {}\n")
    return movie


if __name__ == "__main__":
    import sys
    print(make_demo(sys.argv[1] if len(sys.argv) > 1 else "demo-film"))
