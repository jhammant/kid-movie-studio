"""The plain-wall keyer on a synthetic take: a toy robot in front of a painted (not green)
wall, with drifting light, sensor grain and a soft shadow, then the empty wall."""
import cv2
import numpy as np
import pytest

from kms import demo, media
from kms.keyer import key_clip


@pytest.fixture(scope="module")
def wall_take(tmp_path_factory):
    root = tmp_path_factory.mktemp("wall")
    clip = root / "wall-take.mp4"
    info = demo.wall_take(clip, fps=25, talk=2.0, empty=2.0, truth_dir=root / "truth")
    return clip, info, root


def test_keyer_matte_matches_the_true_outline(wall_take):
    # Arrange
    clip, info, root = wall_take
    alpha = root / "alpha.mp4"

    # Act
    key_clip(clip, root / "keyed.mp4", plate=clip, plate_start=info["plate_from"], plate_dur=info["plate_for"],
             dur=info["to"], alpha_out=alpha, fps=25)

    # Assert
    cap = cv2.VideoCapture(str(alpha))
    ious, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        truth = cv2.imread(str(root / "truth" / f"{i:05d}.png"), 0) if i % 5 == 0 else None
        if truth is not None and truth.any():
            a, t = frame[..., 0] > 127, truth > 127
            ious.append((a & t).sum() / (a | t).sum())
        i += 1
    assert len(ious) >= 5
    assert min(ious) > 0.9, f"worst IoU {min(ious):.3f}"


def test_keyed_output_is_house_format(wall_take):
    # Arrange
    clip, info, root = wall_take
    bg = root / "bg.mp4"
    demo.battlefield(bg, fps=25, seconds=1.0)

    # Act
    out = key_clip(clip, root / "over-bg.mp4", plate=clip, plate_start=info["plate_from"],
                   plate_dur=info["plate_for"], dur=1.0, bg=bg, fps=25, green_out=root / "green.mp4")

    # Assert
    for f in (out, root / "green.mp4"):
        p = media.probe(f)
        assert (p.width, p.height) == (1920, 1080)
        assert p.has_audio
        assert abs(p.duration - 1.0) < 0.15


def test_despill_skips_a_wall_with_no_strong_colour():
    # Arrange
    from kms.keyer import Keyer
    grey_wall = np.full((1080, 1920, 3), 128, np.float32)

    # Act
    k = Keyer(grey_wall)

    # Assert
    assert k.spill is None
