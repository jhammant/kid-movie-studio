import pytest

from kms.fonts import font
from kms.media import probe
from kms.render import Ctx
from kms.render import title_blockbuster as tb

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
    out = tmp_path / "blockbuster.mp4"

    # Act
    result = tb.render(out, ctx=ctx)

    # Assert
    assert result == out
    assert_house_format(out, 25, 6)


def test_long_title_renders(tmp_path):
    # Arrange
    ctx = Ctx(limit_frames=6, workdir=tmp_path / "work")
    out = tmp_path / "long.mp4"

    # Act
    tb.render(out, title=LONG, ctx=ctx)

    # Assert
    assert_house_format(out, 25, 6)


def test_renders_at_30_fps(tmp_path):
    # Arrange
    ctx = Ctx(fps=30, limit_frames=6, workdir=tmp_path / "work")
    out = tmp_path / "blockbuster30.mp4"

    # Act
    tb.render(out, title="THE DINOSAUR DISCO", ctx=ctx)

    # Assert
    assert_house_format(out, 30, 6)


def test_unusual_characters_render(tmp_path):
    # Arrange
    ctx = Ctx(limit_frames=2, workdir=tmp_path / "work")
    out = tmp_path / "odd.mp4"

    # Act
    tb.render(out, title="Chloë's Piñata & R2-D2 3000!", ctx=ctx)

    # Assert
    assert_house_format(out, 25, 2)


@pytest.mark.parametrize("title", ["ROBOTS REVENGE", "ZAP!", "THE DINOSAUR DISCO", LONG,
                                   "R2-D2 & ME", "LES ÉLÈVES FOUS", "SUPERCALIFRAGILISTICEXPIALI"])
def test_layout_stays_inside_title_safe(title):
    # Arrange
    scale = 1.05 / tb.SS  # canvas -> screen at the end of the push-in (the biggest it gets)

    def to_screen(v, canvas_mid, screen_mid):
        return screen_mid + (v - canvas_mid) * scale

    # Act
    lines = tb.layout(title)

    # Assert
    assert 1 <= len(lines) <= 3
    top = lines[0].base - lines[0].cap - lines[0].over
    bottom = lines[-1].base + lines[-1].desc + 9 * tb.SS  # the extrusion hangs below
    assert to_screen(top, tb.CH / 2, tb.H / 2) >= 54
    assert to_screen(bottom, tb.CH / 2, tb.H / 2) <= tb.H - 54
    for ln in lines:
        f = font(tb.FONT, ln.size)
        width = sum(f.getlength(c) for c in ln.text) + ln.track * (len(ln.text) - 1)
        assert to_screen(tb.CW / 2 - width / 2, tb.CW / 2, tb.W / 2) >= 96


def test_default_title_breaks_like_the_original():
    # Arrange / Act
    lines = tb.layout("ROBOTS REVENGE")

    # Assert
    assert [(ln.text, ln.finish) for ln in lines] == [("ROBOTS", "steel"), ("REVENGE", "red")]


def test_short_function_word_becomes_a_kicker():
    # Arrange / Act
    lines = tb.layout("THE DINOSAUR DISCO")

    # Assert
    assert [ln.text for ln in lines] == ["THE", "DINOSAUR", "DISCO"]
    assert lines[0].kicker and lines[0].cap < 0.6 * lines[1].cap


def test_empty_title_is_rejected():
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        tb.layout("   ")


@pytest.mark.slow
def test_full_length_render(tmp_path):
    # Arrange
    ctx = Ctx(workdir=tmp_path / "work")
    out = tmp_path / "full.mp4"

    # Act
    tb.render(out, ctx=ctx)

    # Assert
    assert_house_format(out, 25, round(tb.DUR * 25))
