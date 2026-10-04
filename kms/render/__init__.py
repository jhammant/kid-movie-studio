"""Generators: every graphic, sound and card the studio can make.

Each module here exposes

    render(out, *, <its own keyword params>, ctx=Ctx()) -> Path

and a small standalone CLI (`python -m kms.render.<name> OUT ...`). All text comes in as
parameters, with fun defaults from the "Robots Revenge" example. Nothing reads movie.yaml
directly: the packs (kms/packs) turn a project and an option into these calls.
"""
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

W, H, SR = 1920, 1080, 48000


@dataclass
class Ctx:
    """How to render, as opposed to what.

    fps           25 (UK/EU) or 30 (US). Timelines are written in seconds.
    seed          for anything random, so a re-render looks the same.
    limit_frames  tests only: render just the first N frames (audio trimmed to match).
    workdir       scratch space; a fresh temp dir by default.
    workers       processes for frame rendering.
    """
    fps: int = 25
    seed: int = 0
    limit_frames: int | None = None
    workdir: Path | None = None
    workers: int = field(default_factory=lambda: max(1, (os.cpu_count() or 2) // 2))

    def nframes(self, seconds):
        """Frames for a piece that lasts `seconds`, honouring limit_frames."""
        n = int(round(seconds * self.fps))
        return min(n, self.limit_frames) if self.limit_frames else n

    def scratch(self, name):
        base = Path(self.workdir) if self.workdir else Path(tempfile.gettempdir()) / "kms"
        d = base / name
        d.mkdir(parents=True, exist_ok=True)
        return d


def split_title(title, max_lines=3):
    """Break a title into balanced lines: 'ROBOTS REVENGE' -> ['ROBOTS', 'REVENGE']."""
    words = title.split()
    if len(words) <= 1:
        return words or [""]
    best, best_cost = [title], float("inf")
    for n in range(1, min(max_lines, len(words)) + 1):
        lines = _balanced(words, n)
        widest = max(len(line) for line in lines)
        cost = widest * (1 + 0.18 * (n - 1))  # prefer fewer lines unless one gets too long
        if cost < best_cost:
            best, best_cost = lines, cost
    return best


def _balanced(words, n):
    total = sum(len(w) for w in words) + len(words) - 1
    target, lines, cur = total / n, [], []
    for i, w in enumerate(words):
        remaining_words = len(words) - i
        remaining_lines = n - len(lines)
        if cur and (len(" ".join(cur + [w])) > target * 1.15 or remaining_words < remaining_lines):
            lines.append(" ".join(cur))
            cur = []
        cur.append(w)
    lines.append(" ".join(cur))
    return lines
