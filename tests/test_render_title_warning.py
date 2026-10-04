import numpy as np
import pytest

from kms.media import probe
from kms.render import H, W, Ctx
from kms.render import title_warning as tw

ODD_TITLES = [
    "Café Crème & 2 Robots' 🤖 #1?",
    "ÆØÅ ÉCLAIR — L'ÎLE!",
    "🤖🤖🤖",
    "",
    "SUPERCALIFRAGILISTICEXPIALI",
    "a b c d e f g h i j",
]


def assert_house_format(path, fps, frames):
    info = probe(path)
    assert info is not None
    assert (info.width, info.height) == (1920, 1080)
    assert info.fps == pytest.approx(fps, abs=0.01)
    assert info.has_audio
    assert info.duration == pytest.approx(frames / fps, abs=0.15)


def test_default_title_renders_in_house_format(tmp_path):
    # Arrange
    ctx = Ctx(limit_frames=6, workdir=tmp_path)
    out = tmp_path / "warning.mp4"

    # Act
    result = tw.render(out, ctx=ctx)

    # Assert
    assert result == out
    assert_house_format(out, 25, 6)


def test_long_title_renders_in_house_format(tmp_path):
    # Arrange
    ctx = Ctx(limit_frames=6, workdir=tmp_path)
    out = tmp_path / "hamsters.mp4"

    # Act
    tw.render(out, title="ATTACK OF THE GIANT SPACE HAMSTERS", villain="GIANT SPACE HAMSTER", ctx=ctx)

    # Assert
    assert_house_format(out, 25, 6)


def test_renders_at_30_fps(tmp_path):
    # Arrange
    ctx = Ctx(fps=30, limit_frames=6, workdir=tmp_path)
    out = tmp_path / "warning30.mp4"

    # Act
    tw.render(out, title="THE DINOSAUR DISCO", villain="DINOSAUR", ctx=ctx)

    # Assert
    assert_house_format(out, 30, 6)


@pytest.mark.parametrize("title", ODD_TITLES)
def test_odd_titles_render_the_slam_without_crashing(title):
    # Arrange
    scene = tw.Scene(title=title, villain=title or "ROBOT", alert=title, warning=title, subtitle=title)

    # Act
    frame = scene.frame(int((tw.SLAM_T + 0.3) * 25))

    # Assert
    assert frame.shape == (H, W, 3) and frame.dtype == np.uint8
    assert len(scene.title_lines) <= 3


@pytest.mark.parametrize("title", ["ROBOTS REVENGE", "THE DINOSAUR DISCO", "ZAP!",
                                   "ATTACK OF THE GIANT SPACE HAMSTERS", "SUPERCALIFRAGILISTICEXPIALI"])
def test_chrome_title_stays_inside_title_safe_area(title):
    # Arrange
    scene = tw.Scene(title=title)

    # Act
    x0, y0, x1, y1 = scene.T_BOX

    # Assert
    assert 1 <= len(scene.title_lines) <= 3
    assert x0 >= 96 and x1 <= W - 96
    assert y0 >= 246 and y1 <= 853  # below the WARNING tag, above the subtitle


def test_default_layout_keeps_the_original_title():
    # Arrange
    from kms import fonts
    system_font = not fonts.font_path("impact")[0].startswith(str(fonts.BUNDLED))

    # Act
    lines, px = tw.layout_title("ROBOTS REVENGE")

    # Assert: the original's two lines; at its exact size with the font it was designed in
    assert lines == ["ROBOTS", "REVENGE"]
    if system_font:
        assert px == 300
    else:
        assert 240 <= px <= 300


def test_alert_comes_from_the_villain_unless_overridden():
    # Arrange / Act
    default = tw.Scene(villain="Dinosaur").alert
    custom = tw.Scene(villain="Dinosaur", alert="Dino disco detected").alert

    # Assert
    assert default == "DINOSAUR INVASION DETECTED"
    assert custom == "DINO DISCO DETECTED"


def test_characters_the_font_cannot_draw_are_dropped():
    # Arrange
    title = "Robots 🤖 Café ☃"

    # Act
    lines, _ = tw.layout_title(title)

    # Assert
    assert " ".join(lines) == "ROBOTS CAFÉ"


@pytest.mark.parametrize("fps", [25, 30])
def test_slam_flash_lands_at_the_same_second_at_any_fps(fps):
    # Arrange
    scene = tw.Scene(fps=fps)
    slam = int(round(tw.SLAM_T * fps))

    # Act
    before, at = scene.frame(slam - 1).mean(), scene.frame(slam).mean()

    # Assert
    assert at > before + 60  # the white impact flash


@pytest.mark.slow
def test_full_length_render(tmp_path):
    # Arrange
    ctx = Ctx(workdir=tmp_path)
    out = tmp_path / "warning_full.mp4"

    # Act
    tw.render(out, ctx=ctx)

    # Assert
    assert_house_format(out, 25, int(tw.DUR * 25))
