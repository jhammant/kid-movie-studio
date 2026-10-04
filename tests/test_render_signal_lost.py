import subprocess

import pytest

from kms import media
from kms.render import Ctx
from kms.render import signal_lost


def assert_house_format(path, fps, frames):
    info = media.probe(path)
    assert info is not None
    assert (info.width, info.height) == (1920, 1080)
    assert info.fps == pytest.approx(fps, abs=0.01)
    assert info.has_audio
    assert info.duration == pytest.approx(frames / fps, abs=0.15)


@pytest.fixture(scope="module")
def tiny_source(tmp_path_factory):
    """A 1 s noisy test-pattern clip with a tone, standing in for the previous scene."""
    out = tmp_path_factory.mktemp("src") / "source.mp4"
    subprocess.run([media.ffmpeg(), "-v", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=1",
                    "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=48000:duration=1",
                    "-vf", "noise=alls=30:allf=t", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(out)], check=True)
    return out


@pytest.mark.parametrize("style", ["glitch-beep", "hacked", "static"])
def test_each_style_renders_without_a_source(tmp_path, style):
    # Arrange
    out = tmp_path / f"{style}.mp4"
    ctx = Ctx(limit_frames=6, workdir=tmp_path)

    # Act
    result = signal_lost.render(out, style=style, source=None, villain="DINOSAUR", ctx=ctx)

    # Assert
    assert result == out
    assert_house_format(out, 25, 6)


def test_renders_from_a_source_clip(tmp_path, tiny_source):
    # Arrange
    out = tmp_path / "from_source.mp4"
    ctx = Ctx(limit_frames=8, workdir=tmp_path)

    # Act
    signal_lost.render(out, style="hacked", source=tiny_source, ctx=ctx)

    # Assert
    assert_house_format(out, 25, 8)


def test_renders_at_30_fps(tmp_path, tiny_source):
    # Arrange
    out = tmp_path / "thirty.mp4"
    ctx = Ctx(fps=30, limit_frames=9, workdir=tmp_path)

    # Act
    signal_lost.render(out, style="B", source=tiny_source, source_start=0.2, ctx=ctx)

    # Assert
    assert_house_format(out, 30, 9)


def test_old_letter_styles_are_aliases():
    # Arrange
    letters = ["A", "b", "C"]

    # Act
    names = [signal_lost.style_name(s) for s in letters]

    # Assert
    assert names == ["glitch-beep", "hacked", "static"]


def test_unknown_style_is_refused():
    # Arrange
    style = "fireworks"

    # Act / Assert
    with pytest.raises(ValueError):
        signal_lost.style_name(style)


@pytest.mark.parametrize("villain, plural", [
    ("ROBOT", "ROBOTS"), ("Dinosaur", "DINOSAURS"), ("ALIEN SPACE HAMSTER", "ALIEN SPACE HAMSTERS"),
    ("SNOWMAN", "SNOWMEN"), ("WITCH", "WITCHES"), ("SPY", "SPIES"), ("OCTOPUS", "OCTOPUSES"),
    ("ZOMBIES", "ZOMBIES"), ("HUMAN", "HUMANS"), ("WOLF", "WOLVES"), ("MOUSE", "MICE"),
])
def test_villains_get_proper_plurals(villain, plural):
    # Arrange / Act
    result = signal_lost.plural(villain)

    # Assert
    assert result == plural


def test_hacked_screen_words_follow_the_villain():
    # Arrange
    villain, ident = "dinosaur", "DINO DAILY NEWS"

    # Act
    words = signal_lost.hacked_words(villain, ident)

    # Assert
    assert words["control"] == "DINOSAUR CONTROL 100%"
    assert words["message"] == "BROADCAST TAKEN OVER BY DINOSAURS"
    assert words["status"].startswith("DDN://")
    assert "BEEP BOOP" not in words["ticker"]


def test_default_hacked_words_match_the_original():
    # Arrange / Act
    words = signal_lost.hacked_words("ROBOT", "ROBOTS REVENGE NEWS")

    # Assert
    assert words["status"] == "RRN://LIVE-FEED  >>  STATUS:"
    assert words["control"] == "ROBOT CONTROL 100%"
    assert words["ticker"] == ["BEEP BOOP", "THE ROBOTS ARE IN CHARGE NOW", "HUMANS PLEASE STAND BY",
                               "ALL NEWS IS NOW ROBOT NEWS"]


@pytest.mark.parametrize("fps", [25, 30])
def test_phase_lengths_keep_their_timing_at_any_frame_rate(fps):
    # Arrange
    tl = signal_lost.TIMELINE["glitch-beep"]

    # Act
    seconds = sum(int(round(tl[p] * fps)) for p in tl) / fps

    # Assert
    assert seconds == pytest.approx(5.0, abs=0.05)


@pytest.mark.slow
def test_full_length_hacked_takeover(tmp_path):
    # Arrange
    out = tmp_path / "hacked.mp4"

    # Act
    signal_lost.render(out, style="hacked", ctx=Ctx(workdir=tmp_path))

    # Assert
    assert_house_format(out, 25, 129)  # 0.24 s lead-in + 0.72 + 1.00 + 3.20 s
