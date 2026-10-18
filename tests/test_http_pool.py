"""连接复用的 HTTP 客户端：一回合只握手一次；卡住的握手快速重连；错误和 urllib 一致。"""
import http.client
import json
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llm import http_pool


class Server:
    """HTTP/1.1 keep-alive 假服务，记下一共接了几条连接。"""

    def __init__(self):
        outer = self
        self.connections = 0

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self):
                outer.connections += 1
                super().setup()

            def log_message(self, *a):
                pass

            def _send(self, code, body, extra=None):
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                data = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if self.path == "/close":
                    return self._send(200, data, {"Connection": "close"})
                if self.path == "/drop":                      # 回完话就把连接关掉，却没告诉客户端（模拟空闲太久被对方断开）
                    self._send(200, data)
                    self.close_connection = True
                    self.connection.close()
                    return
                if self.path == "/slow":
                    time.sleep(1.0)
                    return self._send(200, data)
                if self.path == "/bad":
                    return self._send(400, b'{"error": "nope"}')
                self._send(200, json.dumps({"echo": json.loads(data or b"{}"), "ua": self.headers.get("User-Agent")}).encode())

        class S(ThreadingHTTPServer):
            daemon_threads = True
            block_on_close = False

        self.httpd = S(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def srv():
    http_pool.close_all()
    s = Server()
    yield s
    http_pool.close_all()
    s.stop()


def post(srv, path="/", body=None, **kw):
    return http_pool.request("POST", srv.url + path, data=json.dumps(body or {"a": 1}).encode(),
                             headers={"Content-Type": "application/json"}, **kw)


def test_requests_reuse_one_connection(srv):
    before = http_pool.stats()
    for i in range(5):
        assert json.loads(post(srv, body={"i": i}))["echo"] == {"i": i}
    assert srv.connections == 1                                           # 五个请求，只建了一条连接
    after = http_pool.stats()
    assert after["connects"] - before["connects"] == 1 and after["reused"] - before["reused"] == 4
    assert json.loads(post(srv))["ua"].startswith("Python-urllib/")        # 和 urllib 发的 User-Agent 一样


def test_a_connection_the_server_closes_is_not_reused(srv):
    post(srv, "/close")
    post(srv)
    assert srv.connections == 2


def test_a_stale_pooled_connection_is_replaced_silently(srv):
    post(srv, "/drop")                                                     # 服务端悄悄把它关了
    time.sleep(0.1)
    assert json.loads(post(srv, body={"after": "drop"}))["echo"] == {"after": "drop"}      # 调用方一点察觉不到
    assert srv.connections == 2


def test_http_errors_look_like_urllib_errors(srv):
    with pytest.raises(urllib.error.HTTPError) as e:
        post(srv, "/bad")
    assert e.value.code == 400 and b"nope" in e.value.read()
    assert json.loads(post(srv))["echo"] == {"a": 1}                       # 出过错的连接照样能继续用（响应已读完）


def test_a_slow_answer_times_out_as_timeouterror(srv):
    with pytest.raises(TimeoutError):
        post(srv, "/slow", timeout=0.2)
    assert json.loads(post(srv))["echo"] == {"a": 1}


def test_a_stalled_handshake_is_retried_immediately(srv, monkeypatch):
    real = http.client.HTTPConnection.connect
    calls = {"n": 0}

    def flaky(self):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise TimeoutError("握手超时")
        return real(self)
    monkeypatch.setattr(http.client.HTTPConnection, "connect", flaky)
    before = http_pool.stats()["connect_retries"]
    t = time.time()
    assert json.loads(post(srv))["echo"] == {"a": 1}
    assert calls["n"] == 3 and http_pool.stats()["connect_retries"] - before == 2 and time.time() - t < 2     # 重连不退避、不等


def test_connection_failures_raise_urlerror_quickly():
    http_pool.close_all()
    t = time.time()
    with pytest.raises(urllib.error.URLError):
        http_pool.request("POST", "http://127.0.0.1:9/x", data=b"{}", headers={}, timeout=5)
    assert time.time() - t < 3


def test_a_configured_proxy_falls_back_to_urllib(monkeypatch):
    monkeypatch.setattr("urllib.request.getproxies", lambda: {"https": "http://proxy:8080"})
    monkeypatch.setattr("urllib.request.proxy_bypass", lambda host: host in ("localhost", "127.0.0.1"))
    assert http_pool._proxied("https", "openrouter.ai") is True            # 设了代理：走 urllib，保持原来的行为
    assert http_pool._proxied("https", "127.0.0.1") is False               # 本机地址照例绕过代理
    used = {}
    monkeypatch.setattr(http_pool, "_legacy", lambda *a: used.setdefault("legacy", b"via-urllib"))
    assert http_pool.request("POST", "https://openrouter.ai/api/v1/x", data=b"{}", headers={}) == b"via-urllib"
    monkeypatch.setattr("urllib.request.getproxies", lambda: {})
    assert http_pool._proxied("https", "openrouter.ai") is False


def test_each_thread_has_its_own_connection(srv):
    post(srv)
    out = []
    t = threading.Thread(target=lambda: out.append(post(srv)))
    t.start()
    t.join()
    assert out and srv.connections == 2                                    # 连接不跨线程共用，避免两个线程抢同一条
