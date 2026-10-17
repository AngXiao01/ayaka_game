"""反思的去留：同一个问题的新结论替换旧结论；过了保鲜期的不再带进上下文；最多留几条。"""
import json

import core.npc_agent as na
from core.player_agent import ScriptedPlayer
from engine.game_engine import GameEngine
from models.memory import MemoryItem
from models.retrieval import bigram_jaccard
from tests.test_physical import settings
from tests.test_reflection_focus import FocusBackend

Q1 = "对方为什么愿意帮一个刚认识的人"
Q1_AGAIN = "对方为什么愿意帮刚认识的人"
Q2 = "托马夜里到底看见了什么"


def agent(tmp_path):
    f = tmp_path / "data/saves/c.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"world": {"act_id": "act2", "phase_idx": 1, "clues": ["c_case", "c_shops"]}}))
    eng = GameEngine(settings(tmp_path), FocusBackend([Q1]), ScriptedPlayer(["q"]), out=lambda s: None, save_name="c")
    return eng, eng.npcs["ayaka"]


def reflection(agent_, text, focus, t):
    item, _ = agent_.memory.add(MemoryItem(text=text, kind="reflection", importance=4, world_time=t, extra={"focus": focus}))
    return item


def kinds(agent_):
    return [m.text for m in agent_.memory.items if m.kind == "reflection"]


def test_same_question_replaces_the_old_answer_but_other_questions_stay(tmp_path):
    assert bigram_jaccard(Q1, Q1_AGAIN) >= na.SAME_QUESTION > bigram_jaccard(Q1, Q2)
    _, a = agent(tmp_path)
    old = reflection(a, "我觉得他是个好人，所以愿意帮", [Q1], 5)
    other = reflection(a, "托马的话还有疑点，要再问问", [Q2], 6)
    new = reflection(a, "他其实另有目的，我不太放心了", [Q1_AGAIN], 20)
    retired = a._retire_reflections(new, [Q1_AGAIN], 20)
    assert retired == [{"id": old.id, "why": "same_question"}]
    assert kinds(a) == [other.text, new.text]


def test_expired_reflections_are_dropped_and_hidden(tmp_path):
    _, a = agent(tmp_path)
    old = reflection(a, "很久以前的看法", [Q2], 0)
    assert a.latest_reflection() == old.text and a.latest_reflection(now=na.REFLECTION_TTL) == old.text     # 刚好到期日还在
    assert a.latest_reflection(now=na.REFLECTION_TTL + 1) == ""                                            # 过了就不再带进上下文
    new = reflection(a, "新的看法", [Q1], na.REFLECTION_TTL + 1)
    assert a._retire_reflections(new, [Q1], na.REFLECTION_TTL + 1) == [{"id": old.id, "why": "expired"}]
    assert a.latest_reflection(now=na.REFLECTION_TTL + 1) == new.text and [m.id for m in a._live_reflections(na.REFLECTION_TTL + 1)] == [new.id]


def test_at_most_a_few_reflections_are_kept_newest_first(tmp_path):
    _, a = agent(tmp_path)
    for i, q in enumerate(["问题甲甲甲甲", "问题乙乙乙乙", "问题丙丙丙丙", "问题丁丁丁丁", "问题戊戊戊戊"]):
        item = reflection(a, f"看法{i}", [q], i)
        a._retire_reflections(item, [q], i)
    assert kinds(a) == ["看法2", "看法3", "看法4"] and len(kinds(a)) == na.MAX_REFLECTIONS


def test_retired_reflections_are_gone_from_the_file_and_the_index(tmp_path):
    eng, a = agent(tmp_path)
    old = reflection(a, "旧看法旧看法旧看法", [Q1], 1)
    a.memory.retrieve("看法", k=2)                                                # 建好索引
    new = reflection(a, "新看法新看法新看法", [Q1_AGAIN], 2)
    a._retire_reflections(new, [Q1_AGAIN], 2)
    assert [m.id for m in a.memory.retrieve("看法", k=5)] == [new.id]             # 索引跟着删除重建，没有错位
    on_disk = [json.loads(l)["id"] for l in a.memory.path.read_text().splitlines()]
    assert old.id not in on_disk and new.id in on_disk


def test_a_real_session_replaces_the_stale_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(na, "REFLECT_AT", 1)
    eng, a = agent(tmp_path)
    eng.player = ScriptedPlayer([{"kind": "talk", "npc": "ayaka"}, ("你好，聊聊茶会吧。", "闲聊"), ("再聊聊。", "闲聊"),
                                 {"kind": "leave"}, "q"])
    old = reflection(a, "上一回的看法", [Q1], 0)
    eng.run()
    now = [m for m in a.memory.items if m.kind == "reflection"]
    assert old.id not in [m.id for m in now] and len(now) == 1 and now[0].extra["focus"] == [Q1]
    ev = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    assert next(e for e in ev if e["event"] == "reflection")["retired"] == [{"id": old.id, "why": "same_question"}]
