"""Clean-plate keyer for the sage-green wall shots .

The wall isn't chroma green and the light drifted between takes, so a plain
chroma key leaks. Instead, per frame:
  1. fit a smooth per-channel gain field that maps the empty-wall clean plate
     onto this frame (estimated only from pixels currently believed to be wall),
  2. score each pixel by how far it is from the corrected plate, mostly in
     chromaticity (a shadow on the wall keeps the wall's hue; hair/skin don't),
  3. clean the matte (largest blobs, hole fill, feathered edge, temporal smooth),
  4. despill green from the edge band.

Writes either a straight composite over a background clip, or the subject over pure
chroma green so iMovie's own Green/Blue Screen tool can key it perfectly.
"""
import argparse
import subprocess

import cv2
import numpy as np

W, H = 1920, 1080
WALL_X0, WALL_X1 = 108, 1712  # garbage matte: the painted wall's left/right edges at 1080p
CHROMA_GREEN = np.array([64, 177, 0], np.float32)  # BGR, broadcast chroma green


def read_frames(path, start=0.0, dur=None):
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-i", path]
    if dur:
        cmd += ["-t", str(dur)]
    cmd += ["-vf", f"scale={W}:{H}:flags=area", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = W * H * 3
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    p.wait()


def clean_plate(path, start, dur):
    frames = list(read_frames(path, start, dur))[::5]
    return np.median(np.stack(frames), axis=0).astype(np.float32)


class Keyer:
    def __init__(self, plate, lo=0.045, hi=0.11):
        self.plate = plate + 1.0
        self.lo, self.hi = lo, hi
        self.prev_alpha = None
        self.small = (W // 8, H // 8)
        self.plate_s = cv2.resize(self.plate, self.small, interpolation=cv2.INTER_AREA)
        gm = np.zeros((H, W), np.float32)
        gm[:, WALL_X0:WALL_X1] = 1
        self.garbage = cv2.GaussianBlur(gm, (0, 0), 2)

    def _gain(self, frame, bg_mask):
        """Relight the plate: one global colour balance x a local *brightness* field.

        Letting the local field change hue too means a subject that suddenly
        appears somewhere new (the kid popping up at the lens) gets learned as wall
        and keyed out, so local correction is brightness-only.
        """
        f_s = cv2.resize(frame, self.small, interpolation=cv2.INTER_AREA)
        m_s = cv2.resize(bg_mask, self.small, interpolation=cv2.INTER_AREA)
        glob = (f_s * m_s[..., None]).sum((0, 1)) / ((self.plate_s * m_s[..., None]).sum((0, 1)) + 1e-3)
        balance = glob / glob.mean()
        sig = 10
        num = cv2.GaussianBlur(f_s.sum(2) * m_s, (0, 0), sig)
        den = cv2.GaussianBlur(self.plate_s.sum(2) * m_s, (0, 0), sig)
        lum = (num + 1e-3) / (den + 1e-3)
        # where no wall was visible nearby, fall back to the global brightness
        cover = cv2.GaussianBlur(m_s, (0, 0), sig)
        lum = np.where(cover > 0.05, lum, glob.mean())
        g = lum[..., None] * balance
        return cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)

    def _score(self, frame, plate_c):
        f = frame + 1.0
        fs, ps = f.sum(2, keepdims=True), plate_c.sum(2, keepdims=True)
        chroma = np.linalg.norm(f / fs - plate_c / ps, axis=2)
        ratio = (fs / ps)[..., 0]
        # much darker than the wall even after relighting = dark hair, not a shadow
        dark = np.clip((0.45 - ratio) / 0.2, 0, 1)
        bright = np.clip((ratio - 1.35) / 0.25, 0, 1)
        # chroma is noisy in near-black pixels; damp it there
        trust = np.clip(fs[..., 0] / 120.0, 0.25, 1)
        chroma = cv2.GaussianBlur(chroma * trust, (0, 0), 1.2)
        a = np.clip((chroma - self.lo) / (self.hi - self.lo), 0, 1)
        return np.maximum(a, np.maximum(dark, bright))

    def alpha(self, frame_u8):
        frame = frame_u8.astype(np.float32)
        if self.prev_alpha is None:
            bg = np.ones((H, W), np.float32)
            bg[:, W // 4: 3 * W // 4] = 0.0  # first guess: subject is roughly central
        else:
            bg = 1 - cv2.dilate(self.prev_alpha, np.ones((41, 41), np.uint8))
        for _ in range(2):
            plate_c = self.plate * self._gain(frame + 1.0, np.clip(bg, 0, 1))
            self.last_plate = plate_c - 1.0
            a = self._score(frame, plate_c) * self.garbage
            bg = 1 - cv2.dilate((a > 0.3).astype(np.float32), np.ones((41, 41), np.uint8))
        a = self._cleanup(a)
        if self.prev_alpha is not None:
            # smooth flicker on still edges only; fast movement would leave a ghost trail
            stable = np.abs(a - self.prev_alpha) < 0.25
            a = np.where(stable, 0.7 * a + 0.3 * self.prev_alpha, a)
        self.prev_alpha = a
        return a

    def _cleanup(self, a):
        hard = (a > 0.5).astype(np.uint8)
        hard = cv2.morphologyEx(hard, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(hard, 8)
        keep = np.zeros_like(hard)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] > 4000:
                keep[lab == i] = 1
        # fill holes (white shirt patches that happen to match the wall), but
        # leave real see-through gaps open (e.g. between a raised arm and head)
        inv = (1 - keep).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
        solid = keep.copy()
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            touches_edge = x == 0 or y == 0 or x + w >= W or y + h >= H
            if touches_edge:
                continue
            confident_wall = area > 400 and a[lab == i].mean() < 0.12
            if not confident_wall:
                solid[lab == i] = 1
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        # soft edge: keep the measured alpha in a thin band, solid inside
        inner = cv2.erode(solid, np.ones((5, 5), np.uint8)).astype(np.float32)
        band = cv2.dilate(solid, np.ones((7, 7), np.uint8)).astype(np.float32)
        out = np.maximum(inner, a * band)
        return cv2.GaussianBlur(out, (0, 0), 1.0) * self.garbage


def decontaminate(frame_u8, a, plate):
    """Un-mix the wall out of soft edges: frame = a*fg + (1-a)*wall, solve for fg.
    Removes the light halo around motion-blurred hair/hat."""
    f = frame_u8.astype(np.float32)
    a3 = a[..., None]
    est = (f - (1 - a3) * plate) / np.maximum(a3, 1e-3)
    edge = (a3 > 0.15) & (a3 < 0.97)
    return np.clip(np.where(edge, est, f), 0, 255)


def despill(f, a):
    edge = np.clip(1 - np.abs(a - 0.5) * 2, 0, 1)
    edge = cv2.GaussianBlur(cv2.dilate(edge, np.ones((9, 9), np.uint8)), (0, 0), 3)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    limit = np.maximum(r, b)
    g2 = np.where(g > limit, limit + (g - limit) * (1 - edge), g)
    return np.dstack([b, g2, r])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--plate", required=True)
    ap.add_argument("--plate-start", type=float, default=9.0)
    ap.add_argument("--plate-dur", type=float, default=6.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float)
    ap.add_argument("--bg", help="background clip; omit for pure chroma green")
    ap.add_argument("--bg-start", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=1.0, help="scale the subject about bottom-centre")
    ap.add_argument("--shift-x", type=int, default=0)
    ap.add_argument("--alpha-out", help="also write the matte as a greyscale video")
    ap.add_argument("--preroll", type=float, default=0.0,
                    help="key this many seconds before --start without writing them, so the matte has settled")
    ap.add_argument("--fg-until", type=float,
                    help="source time (s) after which the subject's layer fades out; its audio keeps playing")
    ap.add_argument("--green-out", help="also write the subject over pure chroma green (for iMovie's keyer)")
    args = ap.parse_args()

    plate = clean_plate(args.plate, args.plate_start, args.plate_dur)
    keyer = Keyer(plate)
    bg_iter = read_frames(args.bg, args.bg_start) if args.bg else None
    green = np.broadcast_to(CHROMA_GREEN, (H, W, 3)).copy()

    def encoder(out):
        return subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
             "-r", "25", "-i", "-", "-ss", str(args.start)] + (["-t", str(args.dur)] if args.dur else []) +
            ["-i", args.src, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "16",
             "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", out],
            stdin=subprocess.PIPE)

    enc = encoder(args.out)
    genc = encoder(args.green_out) if args.green_out else None
    aenc = None
    if args.alpha_out:
        aenc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}",
             "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", args.alpha_out],
            stdin=subprocess.PIPE)

    M = cv2.getRotationMatrix2D((W / 2, H), 0, args.scale)
    M[0, 2] += args.shift_x
    pre = int(round(min(args.preroll, args.start) * 25))
    read_start = args.start - pre / 25.0
    read_dur = args.dur + pre / 25.0 if args.dur else None
    for j, frame in enumerate(read_frames(args.src, read_start, read_dur)):
        a = keyer.alpha(frame)
        if j < pre:
            continue  # warming up the matte; not part of the output
        i = j - pre
        if args.fg_until is not None:
            t = args.start + i / 25.0
            a = a * float(np.clip((args.fg_until - t) / 0.08 + 1, 0, 1))
        fg = despill(decontaminate(frame, a, keyer.last_plate), a)
        if args.scale != 1.0 or args.shift_x:
            fg = cv2.warpAffine(fg, M, (W, H), flags=cv2.INTER_LINEAR)
            a = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR)
        if bg_iter is not None:
            bgf = next(bg_iter, None)
            back = bgf.astype(np.float32) if bgf is not None else green
        else:
            back = green
        out = fg * a[..., None] + back * (1 - a[..., None])
        enc.stdin.write(np.clip(out, 0, 255).astype(np.uint8).tobytes())
        if genc:
            gout = fg * a[..., None] + green * (1 - a[..., None])
            genc.stdin.write(np.clip(gout, 0, 255).astype(np.uint8).tobytes())
        if aenc:
            aenc.stdin.write((a * 255).astype(np.uint8).tobytes())
    for e in (enc, genc):
        if e:
            e.stdin.close()
            e.wait()
    if aenc:
        aenc.stdin.close()
        aenc.wait()


if __name__ == "__main__":
    main()
