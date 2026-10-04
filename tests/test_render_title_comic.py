import numpy as np
import pytest

from kms.media import probe
from kms.render import Ctx
from kms.render import title_comic as tc

LONG = "ATTACK OF THE GIANT SPACE HAMSTERS"


def assert_house_format(path, fps, frames):
    info = probe(path)
    assert info is not None
    assert (info.width, info.height) == (1920, 1080)
    assert info.fps == pytest.approx(fps, abs=0.01)
    assert info.has_audio
    assert info.duration == pytest.approx(frames / fps, abs=0.15)


def test_default_title_renders_in_house_format(tmp_path):
    # Arrange
    ctx = Ctx(limit_frames=6, workdir=tmp_path / "work")
    out = tmp_path / "comic.mp4"

    # Act
    result = tc.render(out, ctx=ctx)

    # Assert
    assert result == out
    assert_house_format(out, 25, 6)


def test_long_title_renders(tmp_path):
    # Arrange
    ctx = Ctx(limit_frames=6, workdir=tmp_path / "work")
    out = tmp_path / "long.mp4"

    # Act
    tc.render(out, title=LONG, pow="BOOM!", ctx=ctx)

    # Assert
    assert_house_format(out, 25, 6)


def test_renders_at_30_fps(tmp_path):
    # Arrange
    ctx = Ctx(fps=30, limit_frames=6, workdir=tmp_path / "work")
    out = tmp_path / "comic30.mp4"

    # Act
    tc.render(out, title="THE DINOSAUR DISCO", ctx=ctx)

    # Assert
    assert_house_format(out, 30, 6)


def test_unusual_characters_and_no_pow_render(tmp_path):
    # Arrange
    ctx = Ctx(limit_frames=2, workdir=tmp_path / "work")
    out = tmp_path / "odd.mp4"

    # Act
    tc.render(out, title="Chloë's Piñata & R2-D2 3000!", pow="", ctx=ctx)

    # Assert
    assert_house_format(out, 25, 2)


@pytest.mark.parametrize("title", ["ROBOTS REVENGE", "ZAP!", "THE DINOSAUR DISCO", LONG,
                                   "R2-D2 & ME", "LES ÉLÈVES FOUS", "SUPERCALIFRAGILISTICEXPIALI"])
def test_layout_stays_inside_title_safe_and_clear_of_the_pow(title):
    # Arrange / Act
    lay = tc.layout(title)

    # Assert
    assert 1 <= len(lay.lines) <= 3
    assert tc.fits(lay.letters, lay.center, lay.pivot_y, lay.pow_at)


def test_default_title_keeps_the_original_layout():
    # Arrange / Act
    lay = tc.layout("ROBOTS REVENGE")

    # Assert
    assert lay.lines == ["ROBOTS", "REVENGE"]
    assert lay.size == tc.SIZE
    assert lay.pow_at == tc.POW_AT
    assert [round(L.t0, 3) for L in lay.letters][::6] == [0.16, 0.58, 0.94]


def test_letters_land_before_the_pow_and_every_letter_gets_a_note():
    # Arrange
    lay = tc.layout(LONG)

    # Act
    notes = tc.bloop_notes(lay.letters)

    # Assert
    assert max(L.t0 for L in lay.letters) <= tc.T_LAST + 1e-9
    assert len(notes) == len(lay.letters)


def test_pow_hit_lands_at_t_pow_in_seconds():
    # Arrange
    lay = tc.layout("ROBOTS REVENGE")
    times = [L.t0 for L in lay.letters]

    # Act
    mix = tc.synth_audio(np.random.default_rng(77), times, tc.bloop_notes(lay.letters))

    # Assert: the POW is the loudest moment, right at T_POW (seconds, not frames)
    peak = np.abs(mix).max(axis=0).argmax() / tc.SR
    assert tc.T_POW <= peak < tc.T_POW + 0.1


@pytest.mark.slow
def test_full_length_render(tmp_path):
    # Arrange
    ctx = Ctx(workdir=tmp_path / "work")
    out = tmp_path / "full.mp4"

    # Act
    tc.render(out, ctx=ctx)

    # Assert
    assert_house_format(out, 25, round(tc.DUR * 25))
