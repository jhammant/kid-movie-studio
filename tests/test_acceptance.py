"""The acceptance check from the design: every visible creative choice comes from a pick the
kid made, and changing one pick and assembling again changes the film accordingly."""
import hashlib

import pytest

from kms import media, previews
from kms.assemble import assemble
from kms.packs import get_pack
from kms.project import Project


def first_frame_hash(path, tmp_path, name):
    png = tmp_path / f"{name}.png"
    media.extract_frame(path, 2.5, png, width=320)
    return hashlib.sha1(png.read_bytes()).hexdigest()


@pytest.mark.slow
def test_changing_one_pick_changes_the_film(demo_project, tmp_path):
    # Arrange: two title styles rendered; the director picks one
    p = Project(demo_project)
    text = p.path.read_text().replace("# choices:\n#   title: {hide: [warning]}",
                                      "choices:\n  title: {hide: [blockbuster, warning]}")
    p.path.write_text(text)
    p.reload()
    previews.render_all(p, groups=["title"], jobs_n=2, log=lambda *_: None)
    p.set_picks({"title": "comic", "order": ["title", "desk-intro"]})
    first, _ = assemble(p, tmp_path / "a.mp4", log=lambda *_: None)

    # Act: they change their mind
    p.set_picks({"title": "computer"})
    second, pieces = assemble(p, tmp_path / "b.mp4", log=lambda *_: None)

    # Assert
    assert pieces[0].note == "title = computer"
    assert first_frame_hash(first, tmp_path, "a") != first_frame_hash(second, tmp_path, "b")


def test_unpicked_choices_are_flagged_as_placeholders(demo_project):
    # Arrange
    p = Project(demo_project)
    pack = get_pack(p.pack_name)

    # Act
    _, picked = pack.choice(p, "signal")

    # Assert: nothing picked yet, so it's a default, and the guide says it's the director's turn
    assert picked is False
