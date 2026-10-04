from kms import media
from kms.assemble import assemble, audio_filter, plan
from kms.key import key_all
from kms.project import Project


def test_level_names_and_numbers_become_filters():
    # Act / Assert
    assert "loudnorm=I=-15.0" in audio_filter("kid", "speech")
    assert "loudnorm=I=-16.0" in audio_filter(-16, "speech")
    assert audio_filter("none", "speech") is None
    assert audio_filter(None, "graphics") == "graphics"  # measured and gained exactly in conform()


def test_the_film_so_far_skips_what_isnt_ready(demo_project, tmp_path):
    # Arrange: footage only, nothing rendered or keyed yet
    p = Project(demo_project)

    # Act
    pieces = plan(p, make_missing=False)

    # Assert
    ready = {x.label for x in pieces if x.ready}
    assert "Desk intro" in ready and "Robot walk" in ready
    assert not any(x.ready for x in pieces if x.label == "Penny live")  # needs `kms key`


def test_assemble_keys_captions_and_joins_in_the_directors_order(demo_project, tmp_path):
    # Arrange
    p = Project(demo_project)
    key_all(p, log=lambda *_: None)
    p.set_picks({"order": ["robot-walk", "desk-intro", "penny-live", "desk-worried"]})
    expected = sum(x.dur or media.duration(x.path) for x in plan(p, make_missing=False) if x.ready
                   and x.path.suffix == ".mp4" and x.label in ("Robot walk", "Desk intro", "Penny live", "Desk worried"))

    # Act
    out, pieces = assemble(p, tmp_path / "film.mp4", log=lambda *_: None)

    # Assert
    info = media.probe(out)
    assert (info.width, info.height, round(info.fps)) == (1920, 1080, 25)
    assert info.has_audio
    assert [x.label for x in pieces][:4] == ["Robot walk", "Desk intro", "Penny live", "Desk worried"]
    assert info.duration >= expected - 0.5
    assert (p.kit / "captions" / "penny-live.png").exists()
