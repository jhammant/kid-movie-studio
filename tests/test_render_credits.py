"""Rolling end credits."""
import numpy as np
import pytest

from kms import media
from kms.render import W, Ctx
from kms.render import credits as cr


def test_credits_limited_render_is_house_format(tmp_path):
    # Arrange
    ctx = Ctx(fps=25, limit_frames=5, workdir=tmp_path / "work")
    out = tmp_path / "credits.mp4"

    # Act
    cr.render(out, ctx=ctx)

    # Assert
    info = media.probe(out)
    assert (info.width, info.height) == (1920, 1080)
    assert info.fps == pytest.approx(25)
    assert info.has_audio
    assert info.duration == pytest.approx(5 / 25, abs=0.15)


def test_credits_with_custom_entries_at_30fps(tmp_path):
    # Arrange
    ctx = Ctx(fps=30, limit_frames=6, workdir=tmp_path / "work")
    entries = [("role", "Director", "Mia"), ("gap", 80), ("heading", "CAST"),
               ("role", "Dr Dino", "Mia"), ("small", "Filmed in the garden")]

    # Act
    cr.render(tmp_path / "dino.mp4", title="DINO DISASTER", entries=entries, ctx=ctx)

    # Assert
    info = media.probe(tmp_path / "dino.mp4")
    assert info.fps == pytest.approx(30) and info.has_audio
    assert info.duration == pytest.approx(6 / 30, abs=0.15)


def test_example_rolls_for_sixteen_seconds():
    # Arrange
    entries = cr.prepare_entries(cr.TITLE, None)

    # Act
    _, content_end = cr.build_strip(entries)
    secs, travel = cr.timing(content_end)

    # Assert
    assert secs == pytest.approx(16.0)
    assert secs + cr.HOLD == pytest.approx(19.0)


def test_roll_length_follows_the_content():
    # Arrange
    short = [("title", "HI")]
    long = cr.default_entries() + [("role", f"Helper {i}", "Someone") for i in range(20)]

    # Act
    short_secs = cr.timing(cr.build_strip(short)[1])[0]
    long_secs = cr.timing(cr.build_strip(long)[1])[0]

    # Assert
    assert short_secs < 16 < long_secs


def test_title_goes_on_top_unless_entries_have_one():
    # Arrange
    plain = [("role", "Director", "Mia")]
    titled = [("title", "MY FILM"), ("role", "Director", "Mia")]

    # Act
    got_plain, got_titled = cr.prepare_entries("DINO", plain), cr.prepare_entries("DINO", titled)

    # Assert
    assert got_plain[0] == ("title", "DINO")
    assert got_titled == titled


def test_long_roles_never_cross_into_the_names():
    # Arrange
    entries = [("role", "Chief Dinosaur Wrangler and Assistant Special Effects Supervisor",
                "Professor Bartholomew Fizzlewick"),
               ("role", "Supercalifragilisticexpialidociousness Coordinator", "Chloë"),
               ("role", "Snacks", "Grandma, Grandpa, Auntie Jo and the Dog Next Door Who Keeps Barking")]

    # Act
    strip, _ = cr.build_strip(entries)

    # Assert
    px = np.asarray(strip).max(axis=2)
    middle = px[:, W // 2 - cr.GUTTER + 4:W // 2 + cr.GUTTER - 4]
    assert middle.max() < 40, "role and name columns touch"
    assert px[:, :cr.SAFE_X].max() < 40 and px[:, -cr.SAFE_X:].max() < 40


def test_unknown_entry_kind_is_refused():
    # Arrange
    entries = [("banana", "x")]

    # Act / Assert
    with pytest.raises(ValueError):
        cr.layout(entries)


@pytest.mark.slow
def test_full_credits_are_nineteen_seconds(tmp_path):
    # Arrange
    ctx = Ctx(fps=25, workdir=tmp_path / "work")
    out = tmp_path / "credits.mp4"

    # Act
    cr.render(out, ctx=ctx)

    # Assert
    info = media.probe(out)
    assert info.duration == pytest.approx(19.0, abs=0.05)
    assert info.has_audio
