import math
import wave

import cv2
import numpy as np
import pytest

from kms import demo, media, tts
from kms.render import SR, Ctx
from kms.render import glowing_eyes

W, H = 1920, 1080
BACKGROUND = (40, 46, 60)        # robot_chair's dark wall (BGR)
HEAD = (225, 225, 230)


def draw_robot(cx, cy, s):
    """An RGB frame of the demo's toy robot, drawn as robot_chair draws it, at any place and size."""
    img = np.zeros((H, W, 3), np.float32)
    img[:] = BACKGROUND
    demo.robot(img, np.zeros((H, W), np.uint8), cx, cy, s, 0.0, head=HEAD)
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)


def drawn_eyes(cx, cy, s):
    """Where kms.demo.robot draws the two eye lenses: ((cx, cy, r), (cx, cy, r))."""
    return (cx - 55 * s, cy - 480 * s, 32 * s), (cx + 55 * s, cy - 480 * s, 32 * s)


def wandering(t):
    """An off-centre robot, smaller, whole body in frame, drifting sideways and bobbing."""
    return W * 0.3 + 150 * math.sin(0.8 * t), H * 0.97 + 12 * math.sin(1.3 * t), 1.3


def stand_in_voice(path, seconds=2.4):
    """A 'voice' with no text-to-speech: two buzzy phrases with a pause, as a stereo wav."""
    t = np.arange(int(seconds * SR)) / SR
    buzz = 0.3 * np.sign(np.sin(2 * np.pi * 110 * t)) + 0.2 * np.sin(2 * np.pi * 220 * t)
    on = ((t > 0.1) & (t < 1.0)) | ((t > 1.5) & (t < 2.2))
    env = np.convolve(on.astype(float), np.ones(480) / 480, mode="same")
    x = buzz * env
    media.write_wav(path, np.stack([x, x], 1))
    return path


def assert_house_format(path, fps, frames):
    info = media.probe(path)
    assert info is not None
    assert (info.width, info.height) == (1920, 1080)
    assert info.fps == pytest.approx(fps, abs=0.01)
    assert info.has_audio
    assert info.duration == pytest.approx(frames / fps, abs=0.15)


def frame_at(path, index, tmp_path):
    """Frame number `index` of a rendered file, as RGB."""
    png = tmp_path / f"frame_{index}.png"
    media.run([media.ffmpeg(), "-v", "error", "-y", "-i", path, "-vf", f"select=eq(n\\,{index})",
               "-frames:v", "1", png])
    return cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="module")
def chair_clip(tmp_path_factory):
    out = tmp_path_factory.mktemp("chair") / "robot-chair.mp4"
    demo.robot_chair(out, fps=25, seconds=2.0)
    return out


@pytest.fixture(scope="module")
def wandering_clip(tmp_path_factory):
    out = tmp_path_factory.mktemp("wander") / "wandering.mp4"
    with media.FrameWriter(out, fps=25) as fw:
        for i in range(40):
            fw.write(draw_robot(*wandering(i / 25)))
    return out


@pytest.fixture(scope="module")
def battlefield_clip(tmp_path_factory):
    out = tmp_path_factory.mktemp("battle") / "battlefield.mp4"
    demo.battlefield(out, fps=25, seconds=1.5)
    return out


@pytest.fixture
def voice(tmp_path):
    return stand_in_voice(tmp_path / "voice.wav")


@pytest.mark.parametrize("cx, cy, s", [
    (W / 2, H * 1.9, 2.6),       # robot_chair's close-up
    (W * 0.28, H * 0.93, 1.1),   # off-centre, the whole robot ringed by the dark wall
    (W * 0.75, H * 1.3, 1.6),    # off to the right
])
def test_finds_both_lenses_on_a_demo_frame(cx, cy, s):
    # Arrange
    frame = draw_robot(cx, cy, s)
    expected = drawn_eyes(cx, cy, s)

    # Act
    eyes = glowing_eyes.find_eyes(frame)

    # Assert
    assert eyes is not None
    for (x, y, r), (ex, ey, er) in zip(eyes, expected):
        assert x == pytest.approx(ex, abs=3)
        assert y == pytest.approx(ey, abs=3)
        assert r == pytest.approx(er, abs=max(3, 0.05 * er))


def test_finds_no_eyes_on_a_plain_frame():
    # Arrange
    frames = [np.full((H, W, 3), 200, np.uint8), np.zeros((H, W, 3), np.uint8)]

    # Act
    found = [glowing_eyes.find_eyes(f) for f in frames]

    # Assert
    assert found == [None, None]


def test_tracks_a_robot_moving_off_centre(wandering_clip):
    # Arrange
    frames = glowing_eyes.decode_lum(wandering_clip, 0.0, 40 / 25, 25, 40 / 25)

    # Act
    tracks, seed = glowing_eyes.track_eyes(frames, 0, [20, 0])

    # Assert
    assert seed == 20
    assert sorted(tracks) == list(range(40))
    for f, (left, right) in tracks.items():
        (lx, ly, lr), (rx, ry, _) = drawn_eyes(*wandering(f / 25))
        assert left[0] == pytest.approx(lx, abs=3) and left[1] == pytest.approx(ly, abs=3)
        assert right[0] == pytest.approx(rx, abs=3) and right[1] == pytest.approx(ry, abs=3)
        assert left[2] == pytest.approx(lr, abs=4)


@pytest.mark.parametrize("line, parts", [
    ("We'll see, Penny... We'll see...", ("We'll see, Penny...", "We'll see...")),
    ("You haven't seen the last of me, Max… ha ha", ("You haven't seen the last of me, Max...", "ha ha")),
    ("I'll be back...", ("I'll be back...",)),
    ("Goodbye.", ("Goodbye.",)),
])
def test_line_splits_after_its_first_ellipsis(line, parts):
    # Arrange / Act
    result = glowing_eyes.split_line(line)

    # Assert
    assert result == parts


def test_glow_is_dark_before_the_voice_and_lit_while_it_speaks(voice):
    # Arrange
    _, dry = glowing_eyes.load_voice_file(voice)

    # Act
    glow = glowing_eyes.glow_curve(dry, 1.0, 200, 25)

    # Assert
    assert min(glow) == 25
    assert glow[25] < 0.3                       # eases in on the first word
    assert max(glow.values()) > 0.8
    assert glow[int(25 * 1.5)] >= glowing_eyes.GLOW_SMOULDER * 0.8   # smoulders between the phrases


def test_short_render_is_house_format(tmp_path, chair_clip, voice):
    # Arrange
    out = tmp_path / "eyes.mp4"
    ctx = Ctx(limit_frames=8, workdir=tmp_path)

    # Act
    result = glowing_eyes.render(out, clip=chair_clip, voice_wav=voice, ctx=ctx)

    # Assert
    assert result == out
    assert_house_format(out, 25, 8)


def test_short_render_at_30_fps_with_the_eyes_lit(tmp_path, chair_clip, voice):
    # Arrange
    out = tmp_path / "eyes30.mp4"
    ctx = Ctx(fps=30, limit_frames=12, workdir=tmp_path)

    # Act
    glowing_eyes.render(out, clip=chair_clip, voice_wav=voice, voice_at=0.0, ctx=ctx)

    # Assert
    assert_house_format(out, 30, 12)
    lit = frame_at(out, 11, tmp_path)
    (lx, ly, lr), _ = drawn_eyes(W / 2, H * 1.9, 2.6)
    r, g, _ = lit[int(ly + 0.3 * lr), int(lx + 0.3 * lr)].astype(int)
    assert r > g + 60                            # the lens glows red


def test_clip_with_no_eyes_still_renders(tmp_path, battlefield_clip, voice, capsys):
    # Arrange
    out = tmp_path / "no_eyes.mp4"
    ctx = Ctx(limit_frames=8, workdir=tmp_path)

    # Act
    glowing_eyes.render(out, clip=battlefield_clip, voice_wav=voice, voice_at=0.1, ctx=ctx)

    # Assert
    assert_house_format(out, 25, 8)
    assert "no eyes found" in capsys.readouterr().err


def test_sinister_voice_writes_a_stereo_48k_wav(tmp_path, monkeypatch):
    # Arrange
    if tts.engine() is None:
        pytest.skip("no text-to-speech on this machine")
    monkeypatch.setattr(glowing_eyes, "whisper_check", lambda wav, hero=None: None)  # keep it quick
    out = tmp_path / "voice.wav"

    # Act
    result = glowing_eyes.sinister_voice(out, line="We'll see, Max... We'll see...", hero="Max")

    # Assert
    assert result == out
    with wave.open(str(out)) as w:
        assert (w.getnchannels(), w.getframerate()) == (2, 48000)
        assert w.getnframes() / w.getframerate() > 2.0
    x, _ = media.read_wav(out)
    assert 20 * np.log10(np.abs(x).max()) == pytest.approx(glowing_eyes.VOICE_PEAK_DB, abs=0.2)


@pytest.mark.slow
def test_full_render_lights_the_eyes_with_the_voice(tmp_path):
    # Arrange
    clip = tmp_path / "robot-chair.mp4"
    demo.robot_chair(clip, fps=25, seconds=7.0)
    voice = tmp_path / "voice.wav"
    if tts.engine():
        glowing_eyes.sinister_voice(voice)
    else:
        stand_in_voice(voice)
    _, dry = glowing_eyes.load_voice_file(voice)
    glow = glowing_eyes.glow_curve(dry, 0.8, 10 ** 6, 25)
    loudest = max(glow, key=glow.get)
    out = tmp_path / "eyes.mp4"

    # Act
    glowing_eyes.render(out, clip=clip, voice_wav=voice, ctx=Ctx(workdir=tmp_path))

    # Assert
    info = media.probe(out)
    assert (info.width, info.height, info.has_audio) == (1920, 1080, True)
    assert 3.0 < info.duration <= 7.05
    (lx, ly, lr), _ = drawn_eyes(W / 2, H * 1.9, 2.6)
    spot = (int(ly + 0.3 * lr), int(lx + 0.3 * lr))
    before = frame_at(out, 5, tmp_path)[spot].astype(int)
    lit = frame_at(out, loudest, tmp_path)[spot].astype(int)
    assert before.max() < 60                     # a dark lens before the voice
    assert lit[0] > 200 and lit[0] > lit[1] + 80  # glowing red at the loudest word
    lufs, peak = media.loudness(out)
    assert peak == pytest.approx(-1.5, abs=0.6)
