import shutil
from pathlib import Path

import pytest

from kms.project import Project

EXAMPLE = Path(__file__).parent.parent / "kms" / "examples" / "robots" / "movie.yaml"


@pytest.fixture
def project(tmp_path):
    """A fresh copy of the example film, with empty footage/ and kit/ folders."""
    shutil.copy(EXAMPLE, tmp_path / "movie.yaml")
    (tmp_path / "footage").mkdir()
    (tmp_path / "kit").mkdir()
    return Project(tmp_path / "movie.yaml")


@pytest.fixture(scope="session")
def _demo_master(tmp_path_factory):
    from kms.demo import make_demo
    root = tmp_path_factory.mktemp("demo-master")
    make_demo(root, short=True, log=lambda *_: None)
    return root


@pytest.fixture
def demo_project(_demo_master, tmp_path):
    """A fresh copy of the short toy-robot demo: synthetic footage that matches its movie.yaml."""
    root = tmp_path / "demo"
    shutil.copytree(_demo_master, root)
    return root / "movie.yaml"
