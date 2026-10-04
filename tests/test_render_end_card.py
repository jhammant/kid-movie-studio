import pytest
from PIL import Image

from kms import media
from kms.render import Ctx
from kms.render import end_card


def assert_house_format(path, fps, frames):
    info = media.probe(path)
    assert info is not None
    assert (info.width, info.height) == (1920, 1080)
    assert info.fps == pytest.approx(fps, abs=0.01)
    assert info.has_audio
    assert info.duration == pytest.approx(frames / fps, abs=0.15)


@pytest.mark.parametrize("text, dots", [("To Be Continued", True), ("The End", False)])
def test_both_endings_render(tmp_path, text, dots):
    # Arrange
    out = tmp_path / "end.mp4"
    ctx = Ctx(limit_frames=6, workdir=tmp_path)

    # Act
    result = end_card.render(out, text=text, dots=dots, ctx=ctx)

    # Assert
    assert result == out
    assert_house_format(out, 25, 6)


def test_renders_at_30_fps(tmp_path):
    # Arrange
    out = tmp_path / "end30.mp4"
    ctx = Ctx(fps=30, limit_frames=9, workdir=tmp_path)

    # Act
    end_card.render(out, text="To Be Continued... Maybe!", ctx=ctx)

    # Assert
    assert_house_format(out, 30, 9)


@pytest.mark.parametrize("text, dots, expected", [
    ("To Be Continued", True, ("To Be Continued", 5, "")),
    ("To Be Continued...", True, ("To Be Continued", 3, "")),
    ("To Be Continued… Maybe!", True, ("To Be Continued", 3, "Maybe!")),
    ("The End", False, ("The End", 0, "")),
])
def test_dots_in_the_text_become_landings(text, dots, expected):
    # Arrange / Act
    parsed = end_card.parse(text, dots)

    # Assert
    assert parsed == expected


def test_default_timeline_matches_the_original_frames():
    # Arrange
    tl = end_card.Timeline(25, 5)

    # Act
    frames = (tl.eyes_on, tl.blink, tl.text_on, tl.lands, tl.pulse, tl.cut, tl.total, tl.still)

    # Assert
    assert frames == (12, (37, 42, 45, 51), 54, [74, 83, 92, 101, 110], (114, 128), 128, 144, 113)


def test_long_text_shrinks_or_breaks_to_fit():
    # Arrange
    text = "To Be Continued Next Week On The Same Channel"

    # Act
    _, size, rows, _ = end_card.layout(text, True)

    # Assert
    assert max(width for width, _ in rows) <= end_card.MAX_TEXT_W
    assert size >= 120


def test_still_is_a_full_frame(tmp_path):
    # Arrange
    out = tmp_path / "still.png"

    # Act
    end_card.still(out, text="The End", dots=False)

    # Assert
    assert Image.open(out).size == (1920, 1080)


@pytest.mark.slow
def test_full_length_to_be_continued(tmp_path):
    # Arrange
    out = tmp_path / "tbc.mp4"

    # Act
    end_card.render(out, ctx=Ctx(workdir=tmp_path))

    # Assert
    assert_house_format(out, 25, 144)  # 5.76 s
