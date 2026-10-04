"""The pick page, served from this computer so a tablet on the same Wi-Fi can open it.

    kms picks                 # serve it; prints the link (the page shows a QR code for it)
    kms picks --artifact DIR  # or build a static copy to publish as a claude.ai artifact

Picks are saved straight into the picks: block of movie.yaml. Every request needs the
page's key (in the link), so other people on the network can't wander in.
"""
import json
import mimetypes
import secrets
import shutil
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from kms.packs import get_pack
from kms.previews import web_file

HERE = Path(__file__).parent
FONT = Path(__file__).parent.parent / "fonts" / "Bangers-Regular.ttf"
MAX_NOTE = 2000


def state(p, media_prefix="media/"):
    """Everything the page needs: the groups, which previews are ready, the picks, the scenes."""
    p.reload()
    pack = get_pack(p.pack_name)
    groups = [g.to_dict() for g in pack.resolved_groups(p)]
    ready, thumbs = {}, {}
    for g in groups:
        ready[g["key"]], thumbs[g["key"]] = {}, {}
        for o in g["options"]:
            f = web_file(p, g["key"], o["id"])
            ok = g["kind"] != "text" and f.exists()
            ready[g["key"]][o["id"]] = f"{media_prefix}{f.name}" if ok else None
            t = f.with_suffix(".jpg")
            thumbs[g["key"]][o["id"]] = f"{media_prefix}{t.name}" if ok and g["kind"] == "video" and t.exists() else None
    scenes = [{"id": s.id, "label": s.label, "desc": _scene_desc(s, pack, p)} for s in p.scenes]
    picks = {k: v for k, v in p.picks.items() if k != "updated"}
    return {"title": p.title, "director": p.director, "groups": groups, "ready": ready, "thumbs": thumbs,
            "picks": picks,
            "scenes": scenes, "orderLocked": bool((p.choices.get("order") or {}).get("lock"))}


def _scene_desc(s, pack, p):
    if s.kind == "pick":
        g = pack.group(p, s.target)
        return f"The {g.heading.lower()} you pick" if g else ""
    if s.kind == "make":
        return {"credits": "The credits", "villain-line": "The villain's last word"}.get(s.target, "")
    return " ".join(str(s.get("shot") or s.target).split())[:90]


def validate(p, body):
    """Only known groups/options, a list of known scene ids, and a short note get through."""
    if not isinstance(body, dict):
        raise ValueError("expected an object of picks")
    pack = get_pack(p.pack_name)
    groups = {g.key: g for g in pack.resolved_groups(p)}
    clean = {}
    for k, v in body.items():
        if k in groups:
            if v is not None and (not isinstance(v, str) or groups[k].option(v) is None):
                raise ValueError(f"{v!r} isn't an option for {k}")
            if groups[k].locked:
                continue
            clean[k] = v
        elif k == "order":
            ids = {s.id for s in p.scenes}
            if v is not None and (not isinstance(v, list) or not all(isinstance(i, str) and i in ids for i in v)):
                raise ValueError("order must list scene ids")
            clean[k] = v or None
        elif k == "note":
            clean[k] = str(v or "")[:MAX_NOTE]
    return clean


def page_html(p, backend, data_state):
    data = json.dumps({"backend": backend, "state": data_state}).replace("</", "<\\/")
    return (HERE / "page.html").read_text().replace("/*KMS_DATA*/", data)


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))  # TEST-NET: nothing is sent, this only picks the route
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def token(p):
    f = p.kit_dir() / ".pick-key"
    if not f.exists():
        f.write_text(secrets.token_urlsafe(6))
        f.chmod(0o600)
    return f.read_text().strip()


def make_handler(p, key):
    web = p.kit / "web"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _authorised(self, url):
            return secrets.compare_digest((parse_qs(url.query).get("k") or [""])[0], key)

        def _file(self, path):
            if not path.is_file():
                return self._send(404, "not ready yet", "text/plain")
            size = path.stat().st_size
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            if rng and rng.startswith("bytes="):  # Safari on iPad won't play video without ranges
                a, _, b = rng[6:].partition("-")
                start = int(a) if a else max(0, size - int(b))
                end = int(b) if a and b else size - 1
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            with open(path, "rb") as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = fh.read(min(1 << 16, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/fonts/Bangers-Regular.ttf":
                return self._file(FONT)
            if url.path == "/qrcode.js":
                return self._file(HERE / "qrcode.js")
            if not self._authorised(url):
                return self._send(403, "This link needs its key. Ask your grown-up for the link from `kms picks`.",
                                  "text/plain; charset=utf-8")
            if url.path in ("/", "/index.html"):
                return self._send(200, page_html(p, "local", state(p)), "text/html; charset=utf-8")
            if url.path == "/api/state":
                return self._send(200, json.dumps(state(p)))
            if url.path.startswith("/media/"):
                name = Path(url.path).name  # no directories: only files in kit/web
                return self._file(web / name)
            return self._send(404, "not found", "text/plain")

        def do_POST(self):
            url = urlparse(self.path)
            if not self._authorised(url):
                return self._send(403, "forbidden", "text/plain")
            if url.path != "/api/picks":
                return self._send(404, "not found", "text/plain")
            n = int(self.headers.get("Content-Length") or 0)
            if n > 64 * 1024:
                return self._send(413, "too big", "text/plain")
            try:
                updates = validate(p, json.loads(self.rfile.read(n) or b"{}"))
            except (ValueError, json.JSONDecodeError) as e:
                return self._send(400, str(e), "text/plain")
            p.set_picks(updates)
            return self._send(200, json.dumps(state(p)))

    return Handler


def serve(p, port=8765, host="0.0.0.0", log=print):
    key = token(p)
    httpd = ThreadingHTTPServer((host, port), make_handler(p, key))
    ip = "127.0.0.1" if host in ("127.0.0.1", "localhost") else lan_ip()
    url = f"http://{ip}:{port}/?k={key}"
    log(f"\n  The pick page is live:\n\n    {url}\n")
    log("  Open it on this computer, then scan the QR code at the bottom with the tablet's camera")
    log("  (the tablet needs to be on the same Wi-Fi). Picks save into movie.yaml as they're tapped.")
    log("  Press Ctrl-C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log("\n  Stopped. Run `kms next` to see what's next.")
    finally:
        httpd.server_close()
    return url


def build_artifact(p, out_dir):
    """A static copy for a claude.ai artifact with the db capability. Returns {published: local}."""
    out = Path(out_dir)
    (out / "v").mkdir(parents=True, exist_ok=True)
    (out / "fonts").mkdir(exist_ok=True)
    st = state(p, media_prefix="v/")
    files = {}
    for g, opts in st["ready"].items():
        for o, rel in opts.items():
            for r in (rel, st["thumbs"].get(g, {}).get(o)):
                if r:
                    shutil.copy2(p.kit / "web" / Path(r).name, out / r)
                    files[r] = str(out / r)
    shutil.copy2(FONT, out / "fonts" / FONT.name)
    shutil.copy2(HERE / "qrcode.js", out / "qrcode.js")
    files["fonts/" + FONT.name] = str(out / "fonts" / FONT.name)
    files["qrcode.js"] = str(out / "qrcode.js")
    (out / "index.html").write_text(page_html(p, "artifact", st))
    (out / "files.json").write_text(json.dumps(files, indent=1))
    return files


def import_picks(p, data):
    """Load picks read back from the artifact's db (or any JSON) into movie.yaml."""
    data = {k: v for k, v in data.items() if k not in ("updatedAt", "updated")}
    return p.set_picks(validate(p, data))
