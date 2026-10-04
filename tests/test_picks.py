import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from kms.picks import server


@pytest.fixture
def live(project):
    key = server.token(project)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(project, key))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", key, project
    httpd.shutdown()
    httpd.server_close()


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req)


def test_validate_rejects_options_that_dont_exist(project):
    # Act / Assert
    with pytest.raises(ValueError):
        server.validate(project, {"title": "sparkly-unicorns"})


def test_validate_keeps_known_picks_and_trims_the_note(project):
    # Act
    clean = server.validate(project, {"title": "comic", "note": "x" * 5000, "hack": "no"})

    # Assert
    assert clean["title"] == "comic"
    assert len(clean["note"]) == server.MAX_NOTE
    assert "hack" not in clean


def test_a_locked_choice_cant_be_changed_from_the_page(project):
    # Arrange
    text = project.path.read_text().replace("# choices:\n#   title: {hide: [warning]}\n#   signal: {lock: static}",
                                            "choices:\n  signal: {lock: static}")
    project.path.write_text(text)
    project.reload()

    # Act
    clean = server.validate(project, {"signal": "hacked"})

    # Assert
    assert "signal" not in clean


def test_the_page_needs_its_key(live):
    # Arrange
    base, key, _ = live

    # Act / Assert
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(f"{base}/")
    assert e.value.code == 403
    page = urllib.request.urlopen(f"{base}/?k={key}").read().decode()
    assert "ROBOTS REVENGE" in page and "/*KMS_DATA*/" not in page


def test_a_tap_on_the_page_saves_into_movie_yaml(live):
    # Arrange
    base, key, project = live

    # Act
    state = json.loads(post(f"{base}/api/picks?k={key}", {"title": "warning", "note": "more robots"}).read())

    # Assert
    assert state["picks"]["title"] == "warning"
    project.reload()
    assert project.picks["title"] == "warning" and project.picks["note"] == "more robots"


def test_media_cant_escape_the_web_folder(live):
    # Arrange
    base, key, project = live
    (project.root / "secret.txt").write_text("nope")

    # Act / Assert
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(f"{base}/media/..%2F..%2Fsecret.txt?k={key}")
    assert e.value.code == 404


def test_artifact_build_embeds_state_and_lists_files(project, tmp_path):
    # Act
    files = server.build_artifact(project, tmp_path / "art")

    # Assert
    html = (tmp_path / "art" / "index.html").read_text()
    assert '"backend": "artifact"' in html
    assert "fonts/Bangers-Regular.ttf" in files and "qrcode.js" in files
