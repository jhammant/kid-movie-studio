"""The news open, its pips, and the lower-third / BREAKING NEWS overlays."""
import numpy as np
import pytest
from PIL import Image
from scipy.signal import butter, sosfiltfilt

from kms import media
from kms.render import SR, Ctx
from kms.render import news_intro as ni

SAFE_X = 96


def onsets(samples, sr=SR, band=(900, 1100)):
    """(onset times, durations) of 1 kHz bursts, from a 5 ms RMS envelope."""
    y = sosfiltfilt(butter(4, band, "band", fs=sr, output="sos"), samples)
    win = int(0.005 * sr)
    rms = np.sqrt(np.convolve(y ** 2, np.ones(win) / win, "same"))
    above = rms >= 0.5 * rms.max()
    on = np.where(above[1:] & ~above[:-1])[0] / sr
    off = np.where(~above[1:] & above[:-1])[0] / sr
    return on, off - on


def alpha(path):
    im = Image.open(path)
    return im, np.asarray(im)[..., 3]


@pytest.mark.parametrize("fps", [25, 30])
def test_intro_limited_render_is_house_format(tmp_path, fps):
    # Arrange
    ctx = Ctx(fps=fps, limit_frames=6, workers=2, workdir=tmp_path / "work")
    out = tmp_path / "intro.mp4"

    # Act
    ni.render(out, ctx=ctx)

    # Assert
    info = media.probe(out)
    assert (info.width, info.height) == (1920, 1080)
    assert info.fps == pytest.approx(fps)
    assert info.has_audio
    assert info.duration == pytest.approx(6 / fps, abs=0.15)


def test_intro_with_another_hour_channel_and_no_strapline_renders(tmp_path):
    # Arrange
    ctx = Ctx(fps=25, limit_frames=3, workers=1, workdir=tmp_path / "work")
    out = tmp_path / "six.mp4"

    # Act
    ni.render(out, programme="THE SIX O'CLOCK NEWS", channel="", hour=18, headline="DINOSAURS ESCAPE", ctx=ctx)

    # Assert
    info = media.probe(out)
    assert info.has_audio and info.width == 1920


def test_pip_track_is_on_the_seconds():
    # Arrange
    n = int(ni.DUR * SR)

    # Act
    on, length = onsets(ni.pip_track(n))

    # Assert
    np.testing.assert_allclose(on, [1, 2, 3, 4, 5, 6], atol=0.004)
    np.testing.assert_allclose(length, [0.1] * 5 + [0.5], atol=0.01)


def test_mixed_soundtrack_keeps_the_pips_on_the_seconds():
    # Arrange
    mix = ni.synth_audio(seed=0)

    # Act
    on, length = onsets(mix.mean(1))

    # Assert
    assert len(mix) == int(ni.DUR * SR)
    np.testing.assert_allclose(on, [1, 2, 3, 4, 5, 6], atol=0.004)
    assert length[-1] == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize("hour", [0, 9, 18, 21])
def test_clock_reads_the_hour_on_the_long_pip(hour):
    # Arrange
    secs = hour * 3600

    # Act
    first_pip, long_pip = ni.clock_time(ni.T_PIP, secs), ni.clock_time(ni.T_HOUR, secs)

    # Assert
    assert long_pip == secs
    assert first_pip == secs - 5


def test_crawl_follows_headline_and_hour():
    # Arrange / Act
    default = ni.crawl_lines(ni.HEADLINE, None, 21)
    six = ni.crawl_lines("Dinosaurs escape", None, 18)
    given = ni.crawl_lines("TEDDY LOST", ["TEDDY LOST", "SEARCH CONTINUES"], 12)

    # Assert
    assert default == ["ROBOT INVASION CONTINUES", "PEOPLE TOLD TO STAY IN THEIR HOMES",
                       "RESIDENTS FIGHT BACK WITH WATER", "MORE ON THIS STORY AT NINE"]
    assert six == ["DINOSAURS ESCAPE", "MORE ON THIS STORY AT SIX"]
    assert given == ["TEDDY LOST", "SEARCH CONTINUES"]


def test_long_programme_title_wraps_inside_the_panel():
    # Arrange
    long_title = "THE SUPER AMAZING SATURDAY MORNING DINOSAUR NEWS SHOW"

    # Act
    short_lines, _, _ = ni.build_title("THE NINE O’CLOCK NEWS")
    long_lines, cap, _ = ni.build_title(long_title)

    # Assert
    assert len(short_lines) == 1
    assert len(long_lines) == 2
    assert all(t.width <= ni.TITLE_W for t in long_lines)
    assert 2.45 * cap <= ni.PANEL_BOT - ni.PANEL_TOP


def test_bad_hour_is_refused(tmp_path):
    # Arrange
    ctx = Ctx(limit_frames=1, workdir=tmp_path)

    # Act / Assert
    with pytest.raises(ValueError):
        ni.render(tmp_path / "x.mp4", hour=25, ctx=ctx)


def test_lower_third_is_a_transparent_overlay(tmp_path):
    # Arrange
    out = tmp_path / "rex.png"

    # Act
    path = ni.lower_third(out, name="Rex Newsome", role="Newsreader")

    # Assert
    im, a = alpha(path)
    assert im.mode == "RGBA" and im.size == (1920, 1080)
    assert a[:700].max() == 0
    assert (a[800:] == 255).sum() > 10_000


def test_live_lower_third_adds_the_live_bug(tmp_path):
    # Arrange
    out = tmp_path / "penny.png"

    # Act
    ni.lower_third(out, name="Penny Sparks", role="Roving Reporter · The Battlefield", live=True)

    # Assert
    _, a = alpha(out)
    assert (a[:200, :500] == 255).sum() > 1000
    assert a[200:700].max() == 0


@pytest.mark.parametrize("name, role", [
    ("Professor Bartholomew Fizzlewick", "Chief Scientist · Volcano Island"),
    ("Her Royal Highness Princess Anastasia Wilhelmina Featherstonehaugh-Smythe of Volcano Island",
     "Supreme Commander of the Intergalactic Dinosaur Rescue Squadron · Volcano Island North Beach"),
])
def test_long_names_stay_title_safe(tmp_path, name, role):
    # Arrange
    out = tmp_path / "long.png"

    # Act
    ni.lower_third(out, name=name, role=role, live=True)

    # Assert
    _, a = alpha(out)
    solid = a > 128
    assert not solid[:, :SAFE_X].any()
    assert not solid[:, -SAFE_X:].any()


def test_breaking_banner_fits_a_long_headline(tmp_path):
    # Arrange
    headline = ("Dinosaurs escape from the natural history museum and head for the ice cream shop "
                "where they eat absolutely everything including the freezer")

    # Act
    path = ni.breaking_banner(tmp_path / "banner.png", headline=headline)

    # Assert
    im, a = alpha(path)
    assert im.mode == "RGBA" and im.size == (1920, 1080)
    assert a[:700].max() == 0
    assert (a[800:] == 255).sum() > 50_000
    solid = a > 128
    assert not solid[:, :SAFE_X].any() and not solid[:, -SAFE_X + 1:].any()


def test_lower_thirds_cli_writes_the_example_set(tmp_path):
    # Arrange
    outdir = tmp_path / "captions"

    # Act
    ni.main(["--lower-thirds", str(outdir)])

    # Assert
    names = sorted(p.name for p in outdir.glob("*.png"))
    assert names == ["Breaking banner - Robot Invasion.png", "Lower third - Penny Sparks LIVE.png",
                     "Lower third - Rex Newsome.png"]


@pytest.mark.slow
def test_full_intro_is_ten_seconds(tmp_path):
    # Arrange
    ctx = Ctx(fps=25, workdir=tmp_path / "work")
    out = tmp_path / "intro.mp4"

    # Act
    ni.render(out, ctx=ctx)

    # Assert
    info = media.probe(out)
    assert info.duration == pytest.approx(10.0, abs=0.05)
    assert info.fps == pytest.approx(25) and info.has_audio
    _, peak = media.loudness(out)
    assert peak < 0
