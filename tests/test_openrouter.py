"""OpenRouter 后端：配置按后端隔离、key 不串、key 规整与 401 诊断、请求头、200+error 的重试分类、
推理档位、自动选最便宜模型、--check、花费统计。"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import main as main_mod
from config.settings import clean_key, load_settings
from core.player_agent import ScriptedPlayer
from engine.game_engine import GameEngine
from llm import model_picker
from llm.base import LLMError, LLMRequest, RetryableError
from llm.diagnostics import run_check
from llm.gemma_client import describe_key
from llm.mock_backend import MockBackend
from main import build_llm

KEY = "sk-or-v1-" + "0123456789abcdef" * 4
STEP = ('{"thought": "回应", "tool": "respond", "args": {"action": "垂眸", "speech": "多谢您前来。", "emotion": "平静", '
        '"intent": "回应", "affinity_delta": 1, "trust_delta": 1, "tension_delta": 0, "task_signals": [], '
        '"wants_to_end": false, "new_facts": [], "lies": []}}')


def model(mid, pin, pout, ctx=131072, params=("response_format", "structured_outputs"), mandatory=False,
          modalities=("text",)):
    m = {"id": mid, "context_length": ctx, "supported_parameters": list(params),
         "pricing": {"prompt": str(pin), "completion": str(pout)},
         "architecture": {"input_modalities": list(modalities), "output_modalities": ["text"]}}
    if mandatory:
        m["reasoning"] = {"mandatory": True}
    return m


# 价格是每 token 美元；GAME_MIX = 36.6 万 prompt + 2 万 completion
CATALOG = [
    model("inference-net/schematron-v2-turbo", 3e-8, 1.5e-7),         # 抽取专用：标价最低，但不是通用对话厂商
    model("~deepseek/deepseek-flash-latest", 2e-8, 6e-7),             # 浮动别名
    model("openai/gpt-6-luna:batch", 5e-8, 2.5e-7),                   # 批处理
    model("qwen/qwen-tiny:free", 0, 0),                               # 免费档
    model("deepseek/deepseek-flash", 2e-8, 6e-7, params=("response_format", "reasoning")),   # 每局 $0.019
    model("z-ai/glm-cheap-out", 1e-7, 1e-8),                          # 输出最便宜，但每局 $0.037
    model("xiaomi/mimo-flash", 1.4e-7, 2.8e-7),                       # 每局 $0.057
    model("qwen/qwen-think", 1e-8, 1e-8, mandatory=True),             # 强制推理
    model("google/gemma-small", 1e-8, 1e-8, ctx=8192),                # 上下文不够
    model("mistralai/no-json", 1e-8, 1e-8, params=("temperature",)),  # 不支持结构化输出
    model("openai/gpt-4o", 2.5e-6, 1e-5),
    model("openrouter/auto", -1, -1),
]


def reply(text, cost=None, finish="stop", completion_tokens=5):
    usage = {"prompt_tokens": 10, "completion_tokens": completion_tokens}
    if cost is not None:
        usage["cost"] = cost
    return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}], "usage": usage}


class FakeOpenRouter:
    def __init__(self, models=None):
        self.models = models if models is not None else [model("openai/gpt-4o", 2.5e-6, 1e-5)]
        self.requests, self.headers, self.behaviors, self.gets = [], [], [], []
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

            def _authed(self):
                return self.headers.get("Authorization") == f"Bearer {KEY}"

            def do_GET(self):
                outer.gets.append(self.path)
                if self.path.endswith("/key"):
                    if not self._authed():
                        return self._send(401, {"error": {"code": 401, "message": "User not found."}})
                    return self._send(200, {"data": {"usage": 0.0123, "limit_remaining": None, "is_free_tier": False,
                                                     "free_model_daily_requests": {"used": 0, "limit": 1000,
                                                                                   "remaining": 1000}}})
                self._send(200, {"data": outer.models})          # 和 OpenRouter 一样：/models 不需要 key

            def do_POST(self):
                outer.headers.append(dict(self.headers))
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(payload)
                if not self._authed():
                    return self._send(401, {"error": {"code": 401, "message": "User not found."}})
                beh = outer.behaviors.pop(0) if outer.behaviors else None
                self._send(*(beh(payload) if beh else (200, reply("好"))))

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/api/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


_MP = None


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    global _MP
    _MP = monkeypatch                                     # 所有 setenv 都经 monkeypatch，测完自动还原
    for k in list(os.environ):
        if k.startswith(("OPENROUTER_", "GEMMA_", "LLM_", "MODEL_")):
            monkeypatch.delenv(k)
    monkeypatch.setenv("HOME", str(tmp_path))            # ~/.openrouter_key、~/.agentrt/api_key 都落在临时目录
    main_mod._RESOLVED.clear()
    return tmp_path


def settings(tmp, **env):
    for k, v in env.items():
        _MP.setenv(k, v)
    return load_settings(env_file=tmp / "none.env")


@pytest.fixture
def srv():
    s = FakeOpenRouter()
    yield s
    s.close()


def client(clean_env, srv, **env):
    env = {"OPENROUTER_MODEL": "openai/gpt-4o", **env}
    s = settings(clean_env, OPENROUTER_BASE_URL=srv.url, OPENROUTER_API_KEY=KEY, OPENROUTER_MAX_RETRIES="2", **env)
    c = build_llm(s.llm)
    c._sleep = lambda x: None
    return c


def req(**kw):
    return LLMRequest(task="t", messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], **kw)


# ---------------------------------------------------------------- 配置

def test_default_is_openrouter_cheapest_and_never_borrows_the_vllm_key(clean_env):
    (clean_env / ".agentrt").mkdir()
    (clean_env / ".agentrt/api_key").write_text("vllm-secret")
    llm = settings(clean_env, GEMMA_API_KEY="vllm-secret-2", GEMMA_BASE_URL="http://127.0.0.1:8000/v1").llm
    assert llm.backend == "openrouter" and llm.base_url == "https://openrouter.ai/api/v1"
    assert llm.model == "cheapest" and llm.reasoning_effort == "none" and llm.headers == {"X-Title": "ayaka_game"}
    assert llm.api_key == "" and "OPENROUTER_API_KEY" in llm.key_source       # 自建服务的 key 不会发给第三方
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        build_llm(llm)


def test_key_file_is_read_and_source_never_contains_the_key(clean_env):
    (clean_env / ".openrouter_key").write_text(f"  {KEY}\n")
    llm = settings(clean_env).llm
    assert llm.api_key == KEY
    assert "~/.openrouter_key" in llm.key_source and KEY not in llm.key_source
    llm = settings(clean_env, OPENROUTER_API_KEY="env-key", OPENROUTER_MODEL="google/gemma-3-27b-it").llm
    assert llm.api_key == "env-key" and llm.key_source == "OPENROUTER_API_KEY"
    assert llm.model == "google/gemma-3-27b-it"


@pytest.mark.parametrize("raw,note", [
    (f'"{KEY}"\n', ""),
    (f"Bearer {KEY}", "Bearer"),
    (f'curl ... -H "Authorization: Bearer {KEY}" \\\n -d ...', "提取"),
    (f"{KEY[:30]}\n{KEY[30:]}", "换行"),                          # 终端里折行粘贴
])
def test_pasted_key_is_cleaned(raw, note, clean_env):
    (clean_env / ".openrouter_key").write_text(raw)
    llm = settings(clean_env).llm
    assert llm.api_key == KEY and note in llm.key_source and KEY not in llm.key_source


def test_key_shape_never_leaks_the_key():
    d = describe_key(KEY)
    assert d == "长度 73，sk-or-v1- 开头" and KEY[9:] not in d
    assert "中间有空白" in describe_key("Bearer abc") and describe_key("") == "空"
    assert clean_key("vllm-key-with Bearer", "GEMMA") == ("vllm-key-with Bearer", "")   # 只规整，不乱抠别家的 key


def test_gemma_backend_keeps_old_variables(clean_env):
    llm = settings(clean_env, LLM_BACKEND="gemma", GEMMA_BASE_URL="http://10.0.0.5:8000/v1", GEMMA_API_KEY="k",
                   OPENROUTER_MODEL="openai/gpt-4o-mini", OPENROUTER_REASONING_EFFORT="low").llm
    assert (llm.backend, llm.base_url, llm.model, llm.api_key) == ("gemma", "http://10.0.0.5:8000/v1", "heretic", "k")
    assert llm.headers == {} and llm.reasoning_effort == ""


def test_unknown_backend_is_an_error(clean_env):
    with pytest.raises(ValueError, match="LLM_BACKEND"):
        settings(clean_env, LLM_BACKEND="gemini")


# ---------------------------------------------------------------- 选最便宜的模型

def test_rank_applies_hard_filters_then_prices_a_whole_game():
    ranked = model_picker.rank(CATALOG)
    assert [c.id for c in ranked] == ["deepseek/deepseek-flash", "z-ai/glm-cheap-out", "xiaomi/mimo-flash", "openai/gpt-4o"]
    assert abs(ranked[0].game_cost() - (366_000 * 2e-8 + 20_000 * 6e-7)) < 1e-12
    assert ranked[0].reasoning and not ranked[1].reasoning
    why = {m["id"]: model_picker.why_not(m) for m in CATALOG}
    assert why["inference-net/schematron-v2-turbo"] == "不是通用对话厂商"
    assert why["~deepseek/deepseek-flash-latest"] == "浮动别名" and why["openai/gpt-6-luna:batch"] == "批处理"
    assert why["qwen/qwen-tiny:free"] == "免费档" and why["qwen/qwen-think"] == "强制推理"
    assert why["google/gemma-small"] == "上下文不够" and why["mistralai/no-json"] == "不支持结构化输出"
    assert model_picker.why_not(CATALOG[-1], vendors=("*",)) == "价格异常"        # 路由器标价 -1


def test_free_and_vendor_overrides():
    assert model_picker.resolve("cheapest-free", CATALOG).id == "qwen/qwen-tiny:free"
    assert model_picker.rank(CATALOG, vendors=("*",))[0].id == "inference-net/schematron-v2-turbo"
    with pytest.raises(LookupError):
        model_picker.resolve("cheapest", [model("openai/x", 1e-7, 1e-7, params=())])


def test_build_llm_resolves_cheapest_once(clean_env):
    srv = FakeOpenRouter(CATALOG)
    try:
        s = settings(clean_env, OPENROUTER_BASE_URL=srv.url, OPENROUTER_API_KEY=KEY)
        c = build_llm(s.llm)
        assert c.model == s.llm.model == "deepseek/deepseek-flash"
        assert "cheapest → deepseek/deepseek-flash" in s.llm.model_note and "一整局估约 $0.019" in s.llm.model_note
        n = len(srv.gets)
        assert build_llm(s.llm, "cheapest").model == "deepseek/deepseek-flash" and len(srv.gets) == n   # 缓存
        c.complete(req())
        assert srv.requests[-1]["model"] == "deepseek/deepseek-flash"
        s2 = settings(clean_env, OPENROUTER_CHEAPEST_VENDORS="xiaomi")
        main_mod._RESOLVED.clear()
        assert build_llm(s2.llm).model == "xiaomi/mimo-flash"
    finally:
        srv.close()


# ---------------------------------------------------------------- 线上行为（本地假服务）

def test_auth_header_title_name_and_reasoning_off(clean_env, srv):
    c = client(clean_env, srv)
    assert c.complete(req()).text == "好" and c.name == "openrouter"
    h, p = srv.headers[-1], srv.requests[-1]
    assert h["Authorization"] == f"Bearer {KEY}" and h["X-Title"] == "ayaka_game"
    assert p["reasoning"] == {"effort": "none"}


def test_reasoning_param_dropped_when_model_rejects_it(clean_env, srv):
    c = client(clean_env, srv)
    srv.behaviors.append(lambda p: (400, {"error": {"code": 400, "message": "Reasoning is mandatory for this endpoint "
                                                                            "and cannot be disabled."}}))
    assert c.complete(req()).text == "好"
    assert "reasoning" not in srv.requests[-1]
    c.complete(req())
    assert "reasoning" not in srv.requests[-1]                   # 之后都不再发


def test_upstream_error_in_200_body(clean_env, srv):
    c = client(clean_env, srv)
    srv.behaviors += [lambda p: (200, {"error": {"code": 502, "message": "upstream overloaded"}}),
                      lambda p: (200, {"error": {"code": 429, "message": "rate limited"}}),
                      lambda p: (200, reply("好了"))]
    assert c.complete(req()).text == "好了"                    # 502 / 429 → 重试
    srv.behaviors.append(lambda p: (200, {"error": {"code": 400, "message": "bad request"}}))
    with pytest.raises(LLMError) as e:
        c.complete(req())
    assert not isinstance(e.value, RetryableError) and "400" in str(e.value)


def test_401_says_where_the_key_came_from_and_its_shape(clean_env, srv):
    (clean_env / ".openrouter_key").write_text("sk-or-v1-revoked")
    bad = build_llm(settings(clean_env, OPENROUTER_BASE_URL=srv.url, OPENROUTER_MODEL="openai/gpt-4o").llm)
    with pytest.raises(LLMError) as e:
        bad.complete(req())
    msg = str(e.value)
    assert "~/.openrouter_key" in msg and "长度 16，sk-or-v1- 开头" in msg and "User not found" in msg
    assert "revoked" not in msg
    c = client(clean_env, srv)
    srv.behaviors.append(lambda p: (402, {"error": {"code": 402, "message": "Insufficient credits"}}))
    with pytest.raises(LLMError, match="余额"):
        c.complete(req())


def test_check_validates_the_key_before_spending(clean_env):
    srv = FakeOpenRouter([model(f"vendor/m{i}", 1e-6, 1e-6) for i in range(40)]
                         + [model("openai/gpt-4o", 2.5e-6, 1e-5), model("openai/gpt-4o-mini", 1.5e-7, 6e-7)])
    try:
        out = []
        c = client(clean_env, srv)
        srv.behaviors += [lambda p: (200, reply("你好"))] + [lambda p: (200, reply(STEP))] * 3
        assert run_check(c, out.append)
        assert any("结构化通过 3/3" in x for x in out)
        assert all(r.get("max_tokens") in (64, 700) for r in srv.requests)          # 结构化探测用游戏里 NPC 的参数
        assert any("共 42 个" in x for x in out) and not any("vendor/m3" in x for x in out)   # 不把几百个 id 全打出来
        assert any("key 有效" in x and "已用 $0.0123" in x for x in out)
        out.clear()
        _MP.delenv("OPENROUTER_API_KEY")
        (clean_env / ".openrouter_key").write_text("sk-or-v1-revoked")
        bad = build_llm(settings(clean_env, OPENROUTER_BASE_URL=srv.url, OPENROUTER_MODEL="openai/gpt-4o").llm)
        n = len(srv.requests)
        assert not run_check(bad, out.append)
        assert any("key 校验失败" in x and "长度 16" in x for x in out) and len(srv.requests) == n   # 没发对话请求
        out.clear()
        c2 = client(clean_env, srv, OPENROUTER_MODEL="openai/gpt-4o-2099")
        assert not run_check(c2, out.append)
        assert any("OPENROUTER_MODEL" in x and "openai/gpt-4o" in x for x in out)
    finally:
        srv.close()


# ---------------------------------------------------------------- 花费

class CostlyMock(MockBackend):
    def complete(self, req):
        r = super().complete(req)
        r.usage["cost"] = 0.001
        return r


def test_cost_is_summed_from_usage(tmp_path):
    s = load_settings(env_file=tmp_path / "none.env")
    s.game.data_dir = tmp_path / "data"
    s.game.seed = 0
    eng = GameEngine(s, CostlyMock(seed=0), ScriptedPlayer([{"kind": "talk", "npc": "ayaka"}, ("打扰了", "礼貌问候"),
                                                            {"kind": "leave"}, "q"]), out=lambda x: None)
    eng.run()
    st = eng.tracer.summary()
    assert st["llm_calls"] > 0 and abs(st["cost_usd"] - round(0.001 * st["llm_calls"], 4)) < 1e-9
    assert "花费 $" in eng.stats_line()


# ---------------------------------------------------------------- tools/pick_model.py

def test_pick_model_probes_and_pins_first_passing(clean_env, monkeypatch, capsys):
    from tools import pick_model
    monkeypatch.setattr(os, "environ", dict(os.environ))     # load_dotenv 会往环境变量里写，测完还原
    srv = FakeOpenRouter(CATALOG)
    try:
        env = clean_env / ".env"
        env.write_text("LLM_BACKEND=openrouter\nOPENROUTER_MODEL=cheapest   # 注释\nOPENROUTER_API_KEY_FILE=~/.openrouter_key\n")
        settings(clean_env, OPENROUTER_BASE_URL=srv.url, OPENROUTER_API_KEY=KEY)
        srv.behaviors += [lambda p: (200, reply("这不是 JSON")),           # 第一名：3 次里坏 1 次 → 不算通过
                          lambda p: (200, reply(STEP)), lambda p: (200, reply(STEP))] \
            + [lambda p: (200, reply(STEP))] * 3                           # 第二名：3/3
        code = pick_model.main(["--probe", "3", "--write", "--env", str(env)])
        out = capsys.readouterr().out
        assert code == 0, out
        assert "✗ deepseek/deepseek-flash" in out and "✓ z-ai/glm-cheap-out" in out
        assert "2/3" in out and "3/3" in out
        lines = env.read_text().splitlines()
        assert lines[0] == "LLM_BACKEND=openrouter" and lines[2] == "OPENROUTER_API_KEY_FILE=~/.openrouter_key"
        assert lines[1].startswith("OPENROUTER_MODEL=z-ai/glm-cheap-out ")
        assert [r["model"] for r in srv.requests] == ["deepseek/deepseek-flash"] * 3 + ["z-ai/glm-cheap-out"] * 3 \
            + ["xiaomi/mimo-flash"] * 3                                    # 前 N 个都测，选最便宜的全过者
        assert "选中：z-ai/glm-cheap-out" in out
    finally:
        srv.close()


def test_probe_failure_explains_truncation(clean_env, srv):
    from llm.diagnostics import structured_probe
    c = client(clean_env, srv)
    srv.behaviors.append(lambda p: (200, reply('{"thought": "' + "\n" * 300, finish="length", completion_tokens=700)))
    r = structured_probe(c)
    assert not r.ok and "finish_reason=length" in r.detail and "700/700" in r.detail and "空白死循环" in r.detail
    srv.behaviors.append(lambda p: (200, reply('{"thought": "' + "很长的思考" * 200, finish="length", completion_tokens=700)))
    assert "写太长被截断" in structured_probe(c).detail
