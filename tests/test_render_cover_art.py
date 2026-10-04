import numpy as np
import pytest
from PIL import Image

from kms.render import cover_art


@pytest.fixture
def frame(tmp_path):
    """A synthetic 16:9 still: colour bars with a gradient."""
    x = np.linspace(0, 1, 1920)[None, :, None]
    y = np.linspace(0, 1, 1080)[:, None, None]
    img = (np.concatenate([x * 255 + 0 * y, y * 255 + 0 * x, (1 - x) * 200 + 0 * y], axis=2)).astype(np.uint8)
    path = tmp_path / "frame.png"
    Image.fromarray(img).save(path)
    return path


def test_render_writes_a_2_by_3_poster_and_16_by_9_fanart(tmp_path, frame):
    # Arrange
    out_dir = tmp_path / "art"

    # Act
    poster, fanart = cover_art.render(frame, out_dir)

    # Assert
    p, f = Image.open(poster), Image.open(fanart)
    assert (p.format, f.format) == ("JPEG", "JPEG")
    assert p.width * 3 == p.height * 2
    assert f.width * 9 == f.height * 16


@pytest.mark.parametrize("title", ["ZAP!", "ROBOTS REVENGE", "ATTACK OF THE GIANT SPACE HAMSTERS"])
def test_titles_of_one_to_three_lines_fit_the_poster(title):
    # Arrange
    max_w = cover_art.POSTER_W - 60

    # Act
    lines, size = cover_art.title_lines(title, 3, max_w, sizes=(250, 210, 170))

    # Assert
    assert 1 <= len(lines) <= 3
    assert all(cover_art.font("impact", size).getlength(line) <= max_w for line in lines)


def test_poster_crop_follows_the_subject():
    # Arrange
    size = (1920, 1080)

    # Act
    centre = cover_art.crop_box(size, 1000 / 1125)
    right = cover_art.crop_box(size, 1000 / 1125, crop_x=0.9)

    # Assert
    assert centre == (480, 0, 1440, 1080)
    assert right == (960, 0, 1920, 1080)  # clamped to the frame


def test_any_size_frame_works(tmp_path):
    # Arrange
    small = Image.new("RGB", (1280, 720), (30, 60, 90))

    # Act
    poster = cover_art.poster(small, tmp_path / "p.jpg", title="ATTACK OF THE GIANT SPACE HAMSTERS")
    fanart = cover_art.fanart(small, tmp_path / "f.jpg", title="ATTACK OF THE GIANT SPACE HAMSTERS")

    # Assert
    assert Image.open(poster).size == (1000, 1500)
    assert Image.open(fanart).size == (1920, 1080)
