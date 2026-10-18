"""新事实 vs 相近旧事实：新增 / 更新 / 作废 / 不用存（Mem0 的 ADD / UPDATE / DELETE / NONE）。"""
import json

from config.settings import TaskParams
from core.player_agent import ScriptedPlayer
from engine.game_engine import GameEngine
from llm.base import LLMResponse
from llm.memory_update_llm_adapter import MemoryUpdateLLMAdapter, validate_decisions
from llm.mock_backend import MockBackend
from models.memory import MemoryItem
from tests.test_physical import settings

OLD = "旅行者喜欢喝茶"
NEW = "旅行者不喜欢喝茶"
S = {"importance": 3, "topics": ["闲聊"]}


class Fake:
    """按预设回答的更新器，同时数调用次数。"""
    def __init__(self, answer=None):
        self.answer, self.calls = answer or {}, 0

    def decide(self, npc, facts, neighbors):
        self.calls += 1
        return {k: dict(v) for k, v in self.answer.items()}


def agent(tmp_path, updater=None):
    f = tmp_path / "data/saves/c.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"world": {"act_id": "act2", "phase_idx": 1, "clues": ["c_case", "c_shops"]}}))
    eng = GameEngine(settings(tmp_path), MockBackend(seed=0), ScriptedPlayer(["q"]), out=lambda s: None, save_name="c")
    a = eng.npcs["ayaka"]
    a.updater = updater
    return a


def facts(a):
    return [m.text for m in a.memory.items if m.kind == "fact"]


def seed(a):
    old, _ = a.memory.add(MemoryItem(text=OLD, kind="fact", importance=3, world_time=1))
    return old


def test_without_the_step_a_contradiction_is_swallowed_as_a_duplicate(tmp_path):
    """基线：字面相似度 0.62 ≥ 0.6，“不喜欢”被当成“喜欢”的重复合并——矛盾没了，旧的那句还在。"""
    a = agent(tmp_path)
    old = seed(a)
    a._store_facts([NEW], S, 5, {})
    assert facts(a) == [OLD] and old.hits == 2


def test_a_contradiction_replaces_the_old_fact(tmp_path):
    a = agent(tmp_path, None)
    old = seed(a)
    a.updater = Fake({0: {"event": "DELETE", "id": old.id, "text": ""}})
    log = a._store_facts([NEW], S, 5, {})
    assert facts(a) == [NEW] and log[0]["event"] == "DELETE" and log[0]["old"] == OLD
    assert [json.loads(l)["text"] for l in a.memory.path.read_text().splitlines() if '"fact"' in l] == [NEW]      # 文件里也只剩新的


def test_an_update_rewrites_in_place_and_keeps_the_id(tmp_path):
    a = agent(tmp_path)
    old = seed(a)
    a.updater = Fake({0: {"event": "UPDATE", "id": old.id, "text": "旅行者喜欢喝热茶"}})
    log = a._store_facts(["旅行者喜欢喝热的茶"], S, 5, {})
    assert facts(a) == ["旅行者喜欢喝热茶"] and a.memory.items[0].id == old.id and a.memory.items[0].hits == 2 and log[0]["event"] == "UPDATE"


def test_an_update_that_invents_things_is_not_applied(tmp_path):
    a = agent(tmp_path)
    old = seed(a)
    a.updater = Fake({0: {"event": "UPDATE", "id": old.id, "text": "旅行者是天领奉行安插的探子，来自须弥"}})
    log = a._store_facts(["旅行者喜欢喝热的茶"], S, 5, {})
    assert OLD in facts(a) and "旅行者喜欢喝热的茶" in facts(a) and log[0]["event"] == "ADD"          # 没有依据：两条都留着


def test_none_stores_nothing_and_unrelated_facts_do_not_ask(tmp_path):
    a = agent(tmp_path)
    old = seed(a)
    fake = Fake({0: {"event": "NONE", "id": "", "text": ""}})
    a.updater = fake
    log = a._store_facts(["旅行者爱喝茶"], S, 5, {})
    assert facts(a) == [OLD] and log == [{"fact": "旅行者爱喝茶", "event": "NONE"}] and fake.calls == 1
    a._store_facts(["托马会弹三味线，但弹得不太好"], S, 6, {})
    assert fake.calls == 1 and "托马会弹三味线，但弹得不太好" in facts(a)                              # 没有相近旧事实：不问模型


def test_a_failed_call_falls_back_to_the_old_behaviour(tmp_path):
    a = agent(tmp_path)
    seed(a)
    a.updater = Fake({})                                                                    # 调用失败返回 {}
    a._store_facts([NEW], S, 5, {})
    assert facts(a) == [OLD]                                                                # 退回字面去重（旧行为）


def test_decisions_are_validated():
    ids = [{"a1", "a2"}, {"b1"}]
    out = validate_decisions({"decisions": [{"i": 0, "event": "update", "id": "a2", "text": "合并句"},
                                            {"i": 1, "event": "DELETE", "id": "zzz"},              # 编号不在候选里
                                            {"i": 7, "event": "ADD"}, {"i": 0, "event": "NONE"}]}, 2, ids)
    assert out[0] == {"event": "UPDATE", "id": "a2", "text": "合并句"} and out[1]["event"] == "ADD" and 7 not in out
    assert validate_decisions({"decisions": [{"i": 0, "event": "UPDATE", "id": "a1"}]}, 1, ids[:1])[0]["event"] == "ADD"   # 没写合并句
    assert validate_decisions({"decisions": [{"i": 0, "event": "怪事"}]}, 1, ids[:1])[0]["event"] == "ADD"


class Spy:
    name = "spy"

    def __init__(self, text):
        self.text, self.reqs = text, []

    def complete(self, req):
        self.reqs.append(req)
        return LLMResponse(text=self.text)


def test_the_adapter_shows_each_fact_with_its_neighbours():
    spy = Spy(json.dumps({"decisions": [{"i": 0, "event": "DELETE", "id": "m1", "text": ""}]}))
    out = MemoryUpdateLLMAdapter(spy, TaskParams(0.0, 400), None, 0).decide("绫华", [NEW], [[("m1", OLD)]])
    user = spy.reqs[0].messages[1]["content"]
    assert f"新事实 0：{NEW}" in user and f"[m1] {OLD}" in user and out == {0: {"event": "DELETE", "id": "m1", "text": ""}}
