from kms import guide
from kms.project import Project


def test_a_new_film_starts_with_planning_together(project):
    # Act
    cur = guide.current(guide.steps(project))

    # Assert
    assert cur.id == "plan"
    assert cur.who == guide.TOGETHER
    assert "Sam" in cur.say


def test_after_planning_the_guide_gives_a_shot_list(project):
    # Arrange
    project.mark("plan")

    # Act
    cur = guide.current(guide.steps(project))

    # Assert
    assert cur.id == "shoot"
    shots = "\n".join(cur.body)
    assert "footage/wall-take.mp4" in shots
    assert "EMPTY wall" in shots
    assert "footage/water-robots.mp4: Toy robots on the lawn" in shots


def test_every_step_says_who_does_it_and_the_kid_gets_a_turn(project):
    # Act
    steps = guide.steps(project)

    # Assert
    assert {s.who for s in steps} == {guide.GROWNUP, guide.TOGETHER, guide.DIRECTOR}
    pick = next(s for s in steps if s.id == "pick")
    assert pick.who == guide.DIRECTOR and "Sam" in pick.title


def test_with_the_footage_filmed_the_next_step_is_keying(demo_project):
    # Arrange
    p = Project(demo_project)
    p.mark("plan")

    # Act
    steps = {s.id: s for s in guide.steps(p)}

    # Assert
    assert steps["shoot"].done, steps["shoot"].problems
    assert guide.current(list(steps.values())).id == "key"


def test_next_text_shows_the_whole_journey(project):
    # Act
    text = guide.next_text(project)

    # Assert
    assert "→ 1. Plan the film together" in text
    assert "DIRECTOR'S TURN" in text
