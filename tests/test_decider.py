"""便宜的分支判定（Jev）：请求格式、回包兼容、出错退回关键词、各层判定怎么用。"""
import io
import json
import urllib.error

import llm.decider as decider_mod
from core.branch import BranchAgent
from llm.decider import JevDecider, ScriptedDecider, parse_response
from tests.helpers import make_god
from tests.test_living_world import living
from core.player_agent import ScriptedPlayer
from engine.game_engine import GameEngine
from llm.mock_backend import MockBackend
from tests.test_physical import settings


def engine(tmp_path, state, script, decider):
    save = tmp_path / "data/saves/p.json"
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps({"world": state}))
    out = []
    eng = GameEngine(settings(tmp_path), MockBackend(seed=0), ScriptedPlayer(script), out=out.append, save_name="p",
                     decider=decider)
    res = eng.run()
    return eng, res, out

Q = {"is_bug": {"type": "noul", "criteria": {"true": "a", "false": "b"}, "instructions": "?"},
     "team": {"type": "choice", "criteria": {"account": "a", "payments": "p"}, "instructions": "?"},
     "urgency": {"type": "score", "criteria": ["low", "mid", "high"], "instructions": "?"}}


def test_parse_response_tolerates_shapes():
    a = parse_response(Q, {"decisions": {"is_bug": {"answer": True, "probabilities": {"true": 0.9, "false": 0.1}},
                                         "team": {"answer": "payments", "probabilities": {"payments": 0.8}},
                                         "urgency": {"answer": 2}}})
    assert a["is_bug"].answer is True and a["is_bug"].p == 0.9
    assert a["team"].answer == "payments" and a["urgency"].answer == 2
    b = parse_response(Q, {"is_bug": "false", "team": {"probabilities": {"account": 0.7, "payments": 0.3}},
                           "urgency": {"value": "mid"}})
    assert b["is_bug"].answer is False and b["team"].answer == "account" and b["urgency"].answer == 1
    assert parse_response(Q, {"team": {"answer": "frontend"}}) is None      # 不在选项里的答案不收


def test_parse_response_real_jev_shapes():
    """线上 jev-1.13 的真实回包：noul 只给「是」的概率，choice/score 的答案放在 choice/score 键里。"""
    data = {"answers": {"is_bug": {"type": "noul", "noul": 0.98},
                        "team": {"type": "choice", "choice": "payments", "probabilities": {"account": 0.1, "payments": 0.9}, "confidence": 0.9},
                        "urgency": {"type": "score", "score": 1.89, "probabilities": {"0": 0, "1": 0.11, "2": 0.89}, "confidence": 0.83}}}
    a = parse_response(Q, data)
    assert a["is_bug"].answer is True and a["is_bug"].yes(0.9)
    assert a["team"].answer == "payments" and a["team"].p == 0.9
    assert a["urgency"].answer == 2
    low = parse_response({"is_bug": Q["is_bug"]}, {"answers": {"is_bug": {"type": "noul", "noul": 0.05}}})["is_bug"]
    assert low.answer is False and low.p == 0.95 and not low.yes()          # 0.05 = 九成五不是


class FakeHTTP:
    def __init__(self, body=None, fail=False):
        self.body, self.fail, self.sent = body, fail, []

    def __call__(self, method, url, data=None, headers=None, timeout=0):          # 替换 http_pool.request
        self.sent.append(json.loads(data.decode()))
        if self.fail:
            raise urllib.error.URLError("down")
        return json.dumps(self.body).encode()


def test_jev_request_format_cost_and_cache(monkeypatch):
    http = FakeHTTP({"decisions": {"is_bug": {"answer": True, "probabilities": {"true": 0.95}}}, "usage": {"cost": 1e-6}})
    monkeypatch.setattr(decider_mod.http_pool, "request", http)
    logs = []
    d = JevDecider("sk-or-v1-x", tracer=type("T", (), {"log": lambda self, k, **kw: logs.append((k, kw))})())
    out = d.decide("t", {"ticket": "x"}, {"is_bug": Q["is_bug"]})
    assert out["is_bug"].yes(0.9)
    assert http.sent[0] == {"model": "typesafe/jev-1.13", "questions": {"is_bug": Q["is_bug"]}, "state": {"ticket": "x"}}
    assert logs[0][0] == "decision" and logs[0][1]["cost"] == 1e-6
    d.decide("t", {"ticket": "x"}, {"is_bug": Q["is_bug"]})
    assert len(http.sent) == 1                                      # 同样的问题不重复花钱


def test_jev_failure_falls_back_and_disables(monkeypatch):
    http = FakeHTTP(fail=True)
    monkeypatch.setattr(decider_mod.http_pool, "request", http)
    d = JevDecider("k", max_failures=2)
    assert d.decide("t", {"a": 1}, {"is_bug": Q["is_bug"]}) is None
    assert d.decide("t", {"a": 2}, {"is_bug": Q["is_bug"]}) is None
    assert d.decide("t", {"a": 3}, {"is_bug": Q["is_bug"]}) is None and len(http.sent) == 2   # 连着失败就停用


def test_free_action_without_keywords_goes_to_jev():
    g, phys, ex, act = living({"act_id": "act2", "phase_idx": 1, "clues": ["c_case"], "location": "ritou"})
    assert act.parse("我远远地缀在宗次后面，看他收工后去哪") is None            # 关键词认不出
    g.decider = ScriptedDecider({"action": {"verb": ("follow", 0.9), "target": ("soji", 0.9)}})
    assert act.parse("我远远地缀在宗次后面，看他收工后去哪") == ("follow", "soji")
    g.decider = ScriptedDecider({"action": {"verb": ("follow", 0.5)}})
    assert act.parse("我远远地缀在宗次后面") is None                           # 没把握就不当行动


def test_branch_uses_jev_before_llm():
    g = make_god({"act_id": "act3", "flags": []})
    g.decider = ScriptedDecider({"proposal": {"proposal": ("proposed_mercy", 0.85)}})
    calls = []
    llm = type("A", (), {"classify": lambda self, t, p: calls.append(t)})()
    prop = BranchAgent(g, llm, mode="llm").classify("ayaka", "这孩子也不容易，就当没这回事吧")
    assert prop and prop["flag"] == "proposed_mercy" and not calls          # Jev 有把握，大模型没被调用
    g.decider = ScriptedDecider({"proposal": {"proposal": ("none", 0.9)}})
    assert BranchAgent(g, llm, mode="llm").classify("ayaka", "您觉得呢？") is None and not calls


def test_judged_hypothesis_dispatches_thoma(tmp_path):
    say = "我总觉得，那天傍晚来送东西的那个孩子跟茶碗的事脱不了干系。"
    d = ScriptedDecider({"turn": {"h_double": (True, 0.9), "threat": (False, 0.9), "specific": (1, 1.0)}})
    eng, _, out = engine(tmp_path, {"act_id": "act2", "phase_idx": 0, "clues": ["c_case", "c_shops"]},
                         [{"kind": "talk", "npc": "ayaka"}, (say, "自由发言"), "q"], decider=d)
    assert "dispatched:h_double" in eng.god.flags                         # 关键词认不出，Jev 认出来了
    assert any(tag == "turn" for tag, *_ in d.calls)


def test_threat_alerts_the_npc(tmp_path):
    d = ScriptedDecider({"turn": {"threat": (True, 0.9), "specific": (0, 1.0)}})
    eng, _, _ = engine(tmp_path, {"act_id": "act2", "phase_idx": 1, "location": "ritou", "clues": ["c_case"]},
                       [{"kind": "talk", "npc": "soji"}, ("不说实话我就把你送去天领奉行！", "自由发言"), "q"], decider=d)
    assert "alerted:soji" in eng.god.flags


class FakeLLM:
    name = "fake"

    def __init__(self, text):
        self.text, self.reqs = text, []

    def complete(self, req):
        from llm.base import LLMResponse
        self.reqs.append(req)
        return LLMResponse(self.text, usage={})


def test_llm_decider_answers_the_same_questions():
    from llm.decider import LLMDecider
    llm = FakeLLM('{"is_bug": {"answer": true, "p": 0.8}, "team": {"answer": "payments", "p": 0.9}, '
                  '"urgency": {"answer": 2, "p": 0.7}}')
    d = LLMDecider(llm, type("P", (), {"max_tokens": 300})())
    out = d.decide("t", {"ticket": "Pay 之后白屏"}, Q)
    assert out["is_bug"].yes(0.75) and out["team"].answer == "payments" and out["urgency"].answer == 2
    assert "Pay 之后白屏" in llm.reqs[0].messages[1]["content"]
    bad = LLMDecider(FakeLLM("不是 JSON"), type("P", (), {"max_tokens": 300})(), max_failures=1)
    assert bad.decide("t", {"a": 1}, Q) is None and bad.decide("t", {"a": 2}, Q) is None   # 失败退回关键词，之后停用


def test_auto_mode_prefers_jev_then_llm():
    from config.settings import load_settings
    from llm.decider import Decider, LLMDecider, make_decider
    s = load_settings()
    s.harness.decider_mode = "auto"
    s.llm.backend, s.llm.api_key = "gemma", ""
    assert isinstance(make_decider(s, "gemma", None, (FakeLLM("{}"), None)), LLMDecider)
    assert type(make_decider(s, "mock", None, (FakeLLM("{}"), None))) is Decider          # mock 不判
    s.llm.backend, s.llm.api_key = "openrouter", "sk-or-v1-abc"
    assert isinstance(make_decider(s, "openrouter", None, (FakeLLM("{}"), None)), JevDecider)


def test_typed_farewell_ends_the_talk_without_npc_consent(tmp_path):
    d = ScriptedDecider({"turn": {"leaving": (True, 0.95), "threat": (False, 0.9), "specific": (0, 1.0)}})
    eng, res, _ = engine(tmp_path, {"act_id": "act2", "phase_idx": 0, "clues": ["c_case"]},
                         [{"kind": "talk", "npc": "ayaka"}, ("时候不早了，我该走了，改天再聊。", "自由发言"), "q"], decider=d)
    assert [r.end_reason.name for r in res.sessions] == ["PLAYER_LEFT"]      # 一句话就走，不必等对方同意


def test_not_leaving_keeps_talking(tmp_path):
    d = ScriptedDecider({"turn": {"leaving": (False, 0.95), "threat": (False, 0.9), "specific": (0, 1.0)}})
    eng, res, _ = engine(tmp_path, {"act_id": "act2", "phase_idx": 0, "clues": ["c_case"]},
                         [{"kind": "talk", "npc": "ayaka"}, ("改天再聊聊茶道吧，今天先说案子。", "自由发言"), "q"], decider=d)
    assert res.sessions[0].end_reason.name != "PLAYER_LEFT"


def test_mock_backend_never_builds_a_network_decider_even_if_jev_is_configured():
    """.env 里写了 DECIDER_MODE=jev、环境里又有 key：离线 mock 跑（含测试）也不能真的联网调 jev。"""
    from types import SimpleNamespace
    from llm.decider import make_decider
    s = SimpleNamespace(harness=SimpleNamespace(decider_mode="jev", decider_model="m", decider_url="http://x", decider_timeout_s=1.0),
                        llm=SimpleNamespace(api_key="sk-or-v1-real", backend="openrouter"))
    assert make_decider(s, "mock").name == "off"
    assert make_decider(s, "openrouter").name == "jev"
