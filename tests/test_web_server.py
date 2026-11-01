"""网页版 HTTP：单机 mock 开局、轮询 / 选择、分享模式鉴权与人满。"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from tests.helpers import ROOT

sys.path.insert(0, str(ROOT / "web"))
import server  # noqa: E402


def _args(tmp_path, **kw):
    a = argparse.Namespace(
        save="web", new=True, mock=True, incognito=False, share=False,
        code="", max_games=2, budget=0, host="127.0.0.1", port=0,
    )
    vars(a).update(kw)
    return a


def _http(method: str, url: str, body=None, cookies=None):
    data = None if body is None else json.dumps(body).encode()
    h = {"Content-Type": "application/json"} if body is not None else {}
    if cookies:
        h["Cookie"] = "; ".join(f"{k}={v}" for k, v in cookies.items())
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            raw = r.read()
            ctype = r.headers.get("Content-Type", "")
            payload = json.loads(raw) if "json" in ctype else raw
            return r.status, dict(r.headers), payload
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            payload = json.loads(raw)
        except Exception:  # noqa: BLE001
            payload = raw
        return e.code, dict(e.headers), payload


def _serve(monkeypatch, game=None, share=None):
    monkeypatch.setattr(server, "GAME", game)
    monkeypatch.setattr(server, "SHARE", share)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    return base, httpd, thread


def _shutdown(httpd, thread):
    httpd.shutdown()
    thread.join(timeout=5)


class StubGame:
    """只够 Handler 用的假局：不跑引擎，避免和 HTTP 断言抢 answer。"""

    def __init__(self):
        self.io = server.WebIO()
        self.engine = None
        self.args = argparse.Namespace(incognito=False)

    def side_panel(self):
        return {"status": "stub", "location": "庭院"}

    def stop(self):
        with self.io.cond:
            self.io.stop = True
            self.io.cond.notify_all()

    def restart(self, new_game: bool):
        with self.io.cond:
            self.io.lines.append({"id": len(self.io.lines), "text": f"restart:{new_game}", "kind": "meta"})


@pytest.fixture
def stub_server(monkeypatch):
    game = StubGame()
    with game.io.cond:
        game.io.waiting = True
        game.io.allow_free = True
        game.io.options = [{"i": 0, "text": "告辞", "kind": "leave", "intent": "", "line": False, "act": ""}]
        game.io.lines = [{"id": 0, "text": "开场", "kind": "out"}]
    base, httpd, thread = _serve(monkeypatch, game=game)
    try:
        yield base, game
    finally:
        _shutdown(httpd, thread)


@pytest.fixture
def solo_server(tmp_path, monkeypatch):
    """真实 mock 引擎 + 网页服：端到端冒烟。"""
    monkeypatch.setenv("GAME_DATA_DIR", str(tmp_path))
    game = server.Game(_args(tmp_path))
    game.start(new_game=True)
    deadline = time.time() + 30
    while time.time() < deadline:
        with game.io.cond:
            if game.io.waiting or game.io.finished or game.io.error:
                break
        time.sleep(0.05)
    assert game.io.waiting, f"引擎未就绪: finished={game.io.finished} error={game.io.error!r}"
    base, httpd, thread = _serve(monkeypatch, game=game)
    try:
        yield base, game
    finally:
        game.stop()
        _shutdown(httpd, thread)


@pytest.fixture
def share_server(tmp_path, monkeypatch):
    monkeypatch.setenv("GAME_DATA_DIR", str(tmp_path))
    sh = server.Share(_args(tmp_path, share=True, code="雪见,初雪", max_games=1, budget=1000))
    base, httpd, thread = _serve(monkeypatch, share=sh)
    try:
        yield base, sh
    finally:
        sh.stop_all()
        _shutdown(httpd, thread)


def test_stub_serves_page_state_and_choose(stub_server):
    base, game = stub_server
    status, _, html = _http("GET", base + "/")
    assert status == 200 and b"<" in html[:64]

    status, _, state = _http("GET", base + "/api/state")
    assert status == 200
    assert state["waiting"] is True and state["lines"][0]["text"] == "开场"
    assert state["panel"]["location"] == "庭院"
    assert len(state["options"]) == 1

    status, _, body = _http("POST", base + "/api/choose", {"index": 99})
    assert status == 400 and body["ok"] is False

    with game.io.cond:
        game.io.waiting = False
    status, _, body = _http("POST", base + "/api/choose", {"index": 0})
    assert status == 409

    with game.io.cond:
        game.io.waiting = True
        game.io.answer = None
    status, _, body = _http("POST", base + "/api/choose", {"index": 0})
    assert status == 200 and body["ok"] is True
    with game.io.cond:
        assert game.io.answer == {"index": 0}

    with game.io.cond:
        game.io.waiting = True
        game.io.allow_free = True
        game.io.answer = None
    status, _, body = _http("POST", base + "/api/choose", {"text": "  环顾四周  "})
    assert status == 200 and body["ok"] is True
    with game.io.cond:
        assert game.io.answer == {"text": "环顾四周"}

    status, _, body = _http("POST", base + "/api/restart", {"new": True})
    assert status == 200 and body["ok"] is True
    assert any(l["text"] == "restart:True" for l in game.io.lines)


def test_stub_rejects_oversized_and_unknown(stub_server):
    base, _ = stub_server
    req = urllib.request.Request(
        base + "/api/choose",
        data=b"x" * 5000,
        headers={"Content-Type": "application/json", "Content-Length": "5000"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(req, timeout=5)
    assert ei.value.code == 413
    assert _http("GET", base + "/api/nope")[0] == 404
    assert _http("POST", base + "/api/nope", {})[0] == 404


def test_solo_mock_engine_exposes_options_over_http(solo_server):
    base, _ = solo_server
    status, _, state = _http("GET", base + "/api/state")
    assert status == 200
    assert state["waiting"] is True
    assert len(state["options"]) >= 1
    assert state["panel"].get("location") or state["panel"].get("status")


def test_share_requires_invite_code(share_server):
    base, sh = share_server
    status, _, body = _http("GET", base + "/api/state")
    assert status == 401 and body["error"]

    status, _, html = _http("GET", base + "/")
    assert status == 200 and "邀请码".encode() in html
    assert _http("GET", base + "/?code=wrong")[0] == 401

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    try:
        opener.open(urllib.request.Request(base + "/?code=%E5%88%9D%E9%9B%AA"), timeout=5)  # 初雪
        assert False, "expected 303"
    except urllib.error.HTTPError as e:
        assert e.code == 303
        cookies = e.headers.get_all("Set-Cookie") or []
        blob = "\n".join(cookies)
        assert "ayaka_auth=" in blob and "ayaka_sid=" in blob
        auth = next(c.split("=", 1)[1].split(";", 1)[0] for c in cookies if c.startswith("ayaka_auth="))
        sid = next(c.split("=", 1)[1].split(";", 1)[0] for c in cookies if c.startswith("ayaka_sid="))
        assert auth == sh.token() and len(sid) == 32

        status, _, state = _http("GET", base + "/api/state", cookies={"ayaka_auth": auth, "ayaka_sid": sid})
        assert status == 200 and "lines" in state
        # Wait briefly for Share.game() to start the mock engine thread
        time.sleep(0.2)
        status, _, body = _http(
            "GET", base + "/api/state",
            cookies={"ayaka_auth": auth, "ayaka_sid": "b" * 32},
        )
        assert status == 503 and "人满" in body["error"]
