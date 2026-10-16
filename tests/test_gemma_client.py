"""用一个本地假 vLLM 服务测 GemmaClient，不需要真服务。"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llm.base import ContextLengthError, LLMError, LLMRequest, RetryableError
from llm.gemma_client import GemmaClient, to_wire


class FakeVLLM:
    def __init__(self):
        self.requests = []
        self.behaviors = []          # 依次弹出：callable(payload) -> (status, body)
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status, body):
                data = json.dumps(body, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.headers.get("Authorization") != "Bearer k":
                    return self._send(401, {"error": "unauthorized"})
                self._send(200, {"data": [{"id": "heretic", "max_model_len": 32768}]})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(payload)
                beh = outer.behaviors.pop(0) if outer.behaviors else None
                status, body = beh(payload) if beh else (200, reply("ok"))
                self._send(status, body)

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def reply(text, finish="stop", reasoning=None):
    msg = {"role": "assistant", "content": text}
    if reasoning:
        msg["reasoning_content"] = reasoning
    return {"choices": [{"message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


@pytest.fixture
def srv():
    s = FakeVLLM()
    yield s
    s.close()


def client(srv, **kw):
    return GemmaClient(srv.url, "heretic", api_key="k", sleep=lambda s: None, **kw)


def req(**kw):
    return LLMRequest(task="t", messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], **kw)


def test_to_wire_alternation_and_merge_system():
    msgs = [{"role": "system", "content": "S"}, {"role": "assistant", "content": "A"},
            {"role": "user", "content": "U1"}, {"role": "user", "content": "U2"}]
    w = to_wire(msgs)
    assert [m["role"] for m in w] == ["system", "user", "assistant", "user"]
    assert w[-1]["content"] == "U1\n\nU2"
    w2 = to_wire(msgs, merge_system=True)
    assert w2[0]["role"] == "user" and w2[0]["content"].startswith("S\n\n")


def test_list_models_and_auth(srv):
    assert client(srv).list_models()[0]["id"] == "heretic"
    with pytest.raises(LLMError, match="401"):
        GemmaClient(srv.url, "heretic", api_key="wrong").list_models()


def test_json_schema_payload_and_thinking_strip(srv):
    srv.behaviors.append(lambda p: (200, reply('<think>嗯</think>{"a":1}', reasoning="内部思考")))
    r = client(srv).complete(req(json_schema={"type": "object"}, schema_name="x"))
    p = srv.requests[-1]
    assert p["response_format"]["type"] == "json_schema"
    assert p["response_format"]["json_schema"]["name"] == "x"
    assert r.text == '{"a":1}' and r.reasoning == "内部思考" and r.usage["prompt_tokens"] == 10


def test_structured_downgrade(srv):
    srv.behaviors += [lambda p: (400, {"error": "response_format json_schema not supported"}),
                      lambda p: (400, {"error": "guided decoding backend error"}),
                      lambda p: (200, reply("{}"))]
    c = client(srv)
    c.complete(req(json_schema={"type": "object"}))
    assert c.structured == "off"
    assert "response_format" not in srv.requests[-1]
    c.complete(req(json_schema={"type": "object"}))       # 降级对之后的请求持续生效
    assert "response_format" not in srv.requests[-1]


def test_retry_on_5xx(srv):
    srv.behaviors += [lambda p: (503, {"error": "busy"}), lambda p: (200, reply("好"))]
    assert client(srv).complete(req()).text == "好"


def test_retry_exhausted(srv):
    srv.behaviors += [lambda p: (500, {"error": "x"})] * 5
    with pytest.raises(RetryableError):
        client(srv, max_retries=2).complete(req())


def test_context_length(srv):
    srv.behaviors.append(lambda p: (400, {"error": "This model's maximum context length is 32768 tokens"}))
    with pytest.raises(ContextLengthError):
        client(srv).complete(req(json_schema={"type": "object"}))


def test_chat_template_kwargs_passthrough(srv):
    client(srv, chat_template_kwargs={"enable_thinking": False}).complete(req())
    assert srv.requests[-1]["chat_template_kwargs"] == {"enable_thinking": False}


def test_connection_refused():
    c = GemmaClient("http://127.0.0.1:9/v1", "heretic", max_retries=0, timeout=2)
    with pytest.raises(RetryableError):
        c.list_models()
