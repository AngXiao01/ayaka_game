"""反思的焦点问题步骤，以及记忆被取用后刷新“最近被用到”。"""
import json

import pytest

from config.settings import TaskParams
from core.player_agent import ScriptedPlayer
from engine.game_engine import GameEngine
from llm.mock_backend import MockBackend
from llm.reflection_llm_adapter import ReflectionLLMAdapter, validate_focus
from llm.schemas import ValidationError
from models.memory import MemoryItem, MemoryStore
from tests.test_context import settings

OLD = "平藏上回提醒过：茶室高窗的插销早就有些松了"


def test_validate_focus():
    assert validate_focus({"questions": ["他为什么愿意帮我", "他为什么愿意帮我", "哈", "对不上的那句话是真的吗"]}) == [
        "他为什么愿意帮我", "对不上的那句话是真的吗"]                                      # 去重，太短的丢掉
    with pytest.raises(ValidationError):
        validate_focus({"questions": ["", "短"]})


class Spy:
    """记下发来的请求的后端。"""
    name = "spy"

    def __init__(self, texts):
        self.texts, self.reqs = list(texts), []

    def complete(self, req):
        from llm.base import LLMResponse
        self.reqs.append(req)
        return LLMResponse(text=self.texts.pop(0))


def test_adapter_shows_each_question_with_its_own_memories():
    ok = json.dumps({"impression": "对方可信", "doubts": [], "next": "", "evidence": [1]}, ensure_ascii=False)
    spy = Spy([json.dumps({"questions": ["他为什么愿意帮我"]}, ensure_ascii=False), ok])
    a = ReflectionLLMAdapter(spy, TaskParams(0.2, 400), None, 0)
    assert a.focal_questions("绫华", "旅行者", ["记忆甲", "记忆乙"], "谈话") == ["他为什么愿意帮我"]
    out = a.reflect("绫华", "旅行者", "谈话", [], ["记忆甲", "记忆乙", "记忆丙"], [("他为什么愿意帮我", [2, 1])])
    user = spy.reqs[1].messages[1]["content"]
    assert "问题1：他为什么愿意帮我" in user and "[2] 记忆丙" in user and "[1] 记忆乙" in user and "[0]" not in user
    assert "整体看法" in spy.reqs[1].messages[0]["content"] and out["evidence"] == [1]


class FocusBackend(MockBackend):
    def __init__(self, questions):
        super().__init__(seed=0)
        self.questions = questions

    def _reflect_focus(self, meta):
        return {"questions": self.questions}


def _run(tmp_path, monkeypatch, backend, name="c"):
    import core.npc_agent as na
    monkeypatch.setattr(na, "REFLECT_AT", 1)
    save = tmp_path / f"data/saves/{name}.json"
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps({"world": {"act_id": "act2", "phase_idx": 1, "clues": ["c_case", "c_shops"]}}))
    eng = GameEngine(settings(tmp_path), backend, ScriptedPlayer(
        [{"kind": "talk", "npc": "ayaka"}, ("你好，聊聊茶会吧。", "闲聊"), ("再聊聊。", "闲聊"), {"kind": "leave"}, "q"]),
        out=lambda s: None, save_name=name)
    old = MemoryItem(text=OLD, kind="fact", importance=3, world_time=0)
    eng.npcs["ayaka"].memory.add(old)
    eng.run()
    ev = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    return eng, ev, old


def test_focal_question_pulls_in_an_old_memory_the_chat_never_touched(tmp_path, monkeypatch):
    eng, ev, old = _run(tmp_path, monkeypatch, FocusBackend(["茶室高窗的插销是不是早就松了"]))
    focus = [e for e in ev if e["event"] == "reflection_focus"]
    assert focus and old.id in focus[0]["retrieved"][0]["memories"]                     # 按问题翻出了早先的那条
    ask = next(e for e in ev if e["event"] == "llm" and e["task"] == "reflect")
    body = ask["messages"][1]["content"]
    assert "问题1：茶室高窗的插销是不是早就松了" in body and OLD in body              # 反思时它挂在问题下面
    refl = next(m for m in eng.npcs["ayaka"].memory.items if m.kind == "reflection")
    assert refl.extra["focus"] == ["茶室高窗的插销是不是早就松了"] and set(refl.extra["evidence"]) <= {m.id for m in eng.npcs["ayaka"].memory.items}


def test_when_no_question_can_be_picked_it_falls_back_to_recent_memories(tmp_path, monkeypatch):
    eng, ev, _ = _run(tmp_path, monkeypatch, FocusBackend([]))
    assert any(e["event"] == "fallback" and e.get("where") == "reflect_focus" for e in ev)          # 记下是在哪一步退回的
    assert not any(e["event"] == "reflection_focus" for e in ev)
    ask = next(e for e in ev if e["event"] == "llm" and e["task"] == "reflect")
    assert "最近的记忆（编号）" in ask["messages"][1]["content"]                          # 退回原来的做法
    assert any(m.kind == "reflection" for m in eng.npcs["ayaka"].memory.items)


def test_retrieval_refreshes_last_access_and_persists_on_flush(tmp_path):
    p = tmp_path / "m.jsonl"
    s = MemoryStore(p)
    s.add(MemoryItem(text="茶室的窗插销有些松", kind="fact", world_time=1))
    s.add(MemoryItem(text="今天的点心是三彩团子", kind="fact", world_time=2))
    got = s.retrieve("窗插销", k=1, now=30)
    assert got[0].accessed == 30 and s.items[1].accessed == 0                           # 被取到的刷新，没取到的不动
    assert MemoryStore(p).items[0].accessed == 0                                         # 还没落盘（不是每次检索都重写文件）
    s.flush()
    assert MemoryStore(p).items[0].accessed == 30
    s.retrieve("窗插销", k=1, now=40, touch=False)
    assert s.items[0].accessed == 30                                                     # 只看不动
    assert [m.kind for m in s.retrieve("窗插销", k=3, now=40, exclude_kinds=("fact",))] == []


def test_a_recently_used_memory_is_not_treated_as_old(tmp_path):
    s = MemoryStore(tmp_path / "m.jsonl")
    s.items = [MemoryItem(text="同一句话", kind="dialogue_summary", world_time=0),
               MemoryItem(text="同一句话", kind="fact", world_time=0, accessed=95)]      # 前者排在前面，平局时本会选它
    assert s.retrieve("同一句话", k=1, now=100, touch=False)[0].kind == "fact"           # 最近被想起过的更“新”


def test_replay_still_matches_with_focal_reflection_and_access_times(tmp_path, monkeypatch):
    from tools import replay_game
    eng, ev, _ = _run(tmp_path, monkeypatch, FocusBackend(["茶室高窗的插销是不是早就松了"]), name="r")
    assert any(e["event"] == "reflection_focus" for e in ev)
    out = replay_game.replay(eng.tracer.path)
    assert out["ok"], out
