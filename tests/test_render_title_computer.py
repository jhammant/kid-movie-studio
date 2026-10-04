import string

import numpy as np
import pytest

from kms.media import probe
from kms.render import H, W, Ctx
from kms.render import title_computer as tc

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
    out = tmp_path / "computer.mp4"

    # Act
    result = tc.render(out, ctx=ctx)

    # Assert
    assert result == out
    assert_house_format(out, 25, 6)


def test_long_title_renders_in_house_format(tmp_path):
    # Arrange
    ctx = Ctx(limit_frames=6, workdir=tmp_path)
    out = tmp_path / "hamsters.mp4"

    # Act
    tc.render(out, title="ATTACK OF THE GIANT SPACE HAMSTERS", villain="GIANT SPACE HAMSTER", ctx=ctx)

    # Assert
    assert_house_format(out, 25, 6)


def test_renders_at_30_fps(tmp_path):
    # Arrange
    ctx = Ctx(fps=30, limit_frames=6, workdir=tmp_path)
    out = tmp_path / "computer30.mp4"

    # Act
    tc.render(out, title="THE DINOSAUR DISCO", villain="DINOSAUR", ctx=ctx)

    # Assert
    assert_house_format(out, 30, 6)


def test_every_letter_and_digit_has_a_5x7_glyph():
    # Arrange
    wanted = string.ascii_uppercase + string.digits + "!?'&-.,:"

    # Act
    missing = [ch for ch in wanted if ch not in tc.GLYPHS]
    shapes = {ch: (len(rows), {len(r) for r in rows}) for ch, rows in tc.GLYPHS.items()}

    # Assert
    assert missing == []
    assert all(shape == (7, {5}) for shape in shapes.values()), shapes
    assert all(set("".join(rows)) <= {"0", "1"} for rows in tc.GLYPHS.values())


def test_matrix_text_folds_accents_and_drops_the_rest():
    # Arrange
    title = "Café crème! 🤖 #1"

    # Act
    text = tc.matrix_text(title)

    # Assert
    assert text == "CAFE CREME! 1"


@pytest.mark.parametrize("title", ODD_TITLES)
def test_odd_titles_render_a_title_frame_without_crashing(title):
    # Arrange
    scene = tc.Scene(title=title, villain=title or "ROBOT", subtitle=title)

    # Act
    frame = scene.frame(int(4.6 * 25))

    # Assert
    assert frame.shape == (H, W, 3) and frame.dtype == np.uint8
    assert len(scene.title_lines) <= 3


@pytest.mark.parametrize("title", ["ROBOTS REVENGE", "THE DINOSAUR DISCO", "ZAP!",
                                   "ATTACK OF THE GIANT SPACE HAMSTERS", "SUPERCALIFRAGILISTICEXPIALI"])
def test_dot_matrix_sign_stays_inside_title_safe_area(title):
    # Arrange
    scene = tc.Scene(title=title)

    # Act
    xs = [x for x, *_ in scene.cells]
    ys = [y for _, y, *_ in scene.cells]

    # Assert
    assert 1 <= len(scene.title_lines) <= 3
    assert min(xs) >= 96 and max(xs) + scene.cell <= W - 96
    assert min(ys) >= 140 and max(ys) + scene.cell <= 860  # between the header bar and the subtitle


def test_default_layout_keeps_the_original_sign():
    # Arrange / Act
    lines, cell = tc.layout_title("ROBOTS REVENGE")

    # Assert
    assert (lines, cell) == (["ROBOTS", "REVENGE"], 36)


def test_villain_builds_the_boot_log_and_header():
    # Arrange
    scene = tc.Scene(villain="Dinosaur")

    # Act
    lines = (scene.l1, scene.l2, scene.l4)

    # Assert
    assert lines == ("> BOOTING DINOSAUR_PROTOCOL.EXE ...", "> LOADING EVIL PLANS ...", "> HUMANS DETECTED!")
    assert scene.os_name == "DINOSAUR-OS 9000"
    assert tc.Scene().os_name == "ROBO-OS 9000"


def test_boot_lines_override_keeps_missing_lines_from_the_defaults():
    # Arrange
    scene = tc.Scene(boot_lines=["WAKING UP THE T-REX"])

    # Act
    lines = (scene.l1, scene.l2, scene.l4)

    # Assert
    assert lines == ("> WAKING UP THE T-REX", "> LOADING EVIL PLANS ...", "> HUMANS DETECTED!")


def test_typing_and_beeps_land_on_the_same_seconds_at_25_and_30_fps():
    # Arrange
    s25, s30 = tc.Scene(fps=25), tc.Scene(fps=30)

    # Act
    a25, a30 = tc.build_audio(s25), tc.build_audio(s30)

    # Assert
    assert np.array_equal(s25.T1, s30.T1) and np.array_equal(s25.TS, s30.TS)
    assert np.array_equal(a25, a30)


@pytest.mark.slow
def test_full_length_render(tmp_path):
    # Arrange
    ctx = Ctx(workdir=tmp_path)
    out = tmp_path / "computer_full.mp4"

    # Act
    tc.render(out, ctx=ctx)

    # Assert
    assert_house_format(out, 25, int(tc.DUR * 25))
