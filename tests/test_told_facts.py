"""“你已经告诉过对方的事实”记的是她当时实际说的话，不是当初给她的指令。"""
import json

from core.player_agent import ScriptedPlayer
from engine.game_engine import GameEngine
from llm.mock_backend import MockBackend
from tests.test_physical import settings
from tools import preview_prompt as pp


def test_the_established_fact_is_what_she_actually_said_not_the_instruction(tmp_path):
    f = tmp_path / "data/saves/c.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"world": {"act_id": "act1", "phase_idx": 0, "location": "kamisato_estate", "clues": []}}))
    eng = GameEngine(settings(tmp_path), MockBackend(seed=0), ScriptedPlayer(
        [{"kind": "talk", "npc": "ayaka"}, {"kind": "ask", "signal": "asked_purpose"}, {"kind": "leave"}, "q"]), out=lambda s: None, save_name="c")
    eng.run()
    told = eng.god.told.get("asked_purpose")
    assert told and "把事情说清楚" not in told                                           # 存的是 NPC 的台词
    assert json.loads(json.dumps(eng.god.to_dict()))["told"]["asked_purpose"] == told      # 进存档
    d = eng.god.npc_task_detail(eng.personas["ayaka"], 30)
    fact = next(x["fact"] for x in d["achieved_signals"] if x["id"] == "asked_purpose")
    assert fact == f"你当时说的是：「{told}」"
    assert "把事情说清楚" not in fact and "郑重地问" not in fact                          # 没有祈使句可被当成指令再执行一遍


def test_without_a_record_it_never_falls_back_to_the_instruction(tmp_path):
    """旧存档里已经达成、但没记过台词的信号：退回 desc（一句描述），绝不退回 reveal（指令）。"""
    s = settings(tmp_path)
    (tmp_path / "data/saves").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/saves/c.json").write_text(json.dumps({"world": {"act_id": "act2", "phase_idx": 0, "location": "kamisato_estate",
                                                                      "clues": ["c_case", "c_shops"], "achieved": ["asked_purpose"]}}))
    pv = pp.capture("c", "ayaka", say="你好", intent="闲聊", settings=s)
    # 给 NPC 的请求（评估 / 说话 / 记录）里不能有指令式的原文；核对员的请求里有作者原文，是拿来比对事实的，它不会照着执行
    block = "\n".join(m["content"] for r in pv.requests if r.task != "npc_audit" for m in r.messages)
    assert "把事情说清楚" not in block and "最后郑重地问对方愿不愿意帮忙" not in block
    audit = "\n".join(m["content"] for r in pv.requests if r.task == "npc_audit" for m in r.messages)
    assert "三天前的夜里" in audit                                                      # 核对员拿得到准确的时间线，能发现“茶会前夜”这类口误


def test_what_ayaka_knows_about_the_link_between_the_two_thefts_is_bounded(tmp_path):
    """真实 trace：被问“为什么谈到花见坂的失窃”，她答“时间完全重合……这绝非巧合”——设定里她只知道同一晚，真相恰恰是两件事撞在了一起。"""
    from tests.test_physical import PERSONAS  # noqa: F401  (persona files are loaded by the engine)
    for act in ("act1", "act2"):
        f = tmp_path / f"data/saves/{act}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"world": {"act_id": act, "phase_idx": 0, "location": "kamisato_estate", "clues": ["c_case", "c_shops"] if act == "act2" else []}}))
        pv = pp.capture(act, "ayaka", say="为什么你谈论到花见坂的失窃", intent="自由发言", settings=settings(tmp_path))
        step = next(r for r in pv.requests if r.task == "npc_step").messages[-1]["content"]
        assert "不要断言「绝非巧合」" in step, act
