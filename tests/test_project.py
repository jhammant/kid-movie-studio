import pytest
import yaml

from kms.project import Project, ProjectError


def test_example_loads_with_every_scene_kind(project):
    # Arrange / Act
    kinds = {s.kind for s in project.scenes}

    # Assert
    assert project.title == "ROBOTS REVENGE"
    assert project.director == "Sam"
    assert kinds == {"pick", "make", "clip", "key"}
    assert project.scene("villain-line").get("clip") == "robot-chair.mp4"


def test_set_picks_keeps_the_file_above_and_the_block_last(project):
    # Arrange
    before = project.path.read_text()
    head = before[:before.index("picks:")]

    # Act
    project.set_picks({"title": "comic", "note": "make it sparkly"})
    project.set_picks({"channel": "ch2"})
    after = project.path.read_text()

    # Assert
    assert after.startswith(head)  # every comment and setting above is untouched
    assert after.rstrip().splitlines()[-1].startswith("  ")  # picks: is still the last block
    picks = yaml.safe_load(after)["picks"]
    assert picks["title"] == "comic" and picks["channel"] == "ch2" and picks["note"] == "make it sparkly"


def test_clearing_a_pick_removes_it(project):
    # Arrange
    project.set_picks({"title": "comic"})

    # Act
    project.set_picks({"title": None})

    # Assert
    assert "title" not in project.picks


def test_running_order_follows_the_directors_pick(project):
    # Arrange
    ids = [s.id for s in project.scenes]
    order = [ids[-1]] + ids[:-1]

    # Act
    project.set_picks({"order": order})

    # Assert
    assert [s.id for s in project.ordered_scenes()] == order


def test_fill_puts_the_directors_name_in(project):
    # Act
    text = project.fill("{initial}BC NEWS / {director} Broadcasting Corporation")

    # Assert
    assert text == "SBC NEWS / Sam Broadcasting Corporation"


def test_a_scene_with_two_kinds_is_explained(tmp_path):
    # Arrange
    (tmp_path / "movie.yaml").write_text("title: X\nscenes:\n  - pick: title\n    clip: a.mp4\n")

    # Act / Assert
    with pytest.raises(ProjectError, match="exactly one"):
        Project(tmp_path / "movie.yaml")


def test_find_walks_up_from_a_subfolder(project):
    # Act
    found = Project.find(project.root / "footage")

    # Assert
    assert found.path == project.path
