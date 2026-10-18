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
    assert fact == f"你当时是这么说的：「{told}」"
    assert "把事情说清楚" not in fact and "郑重地问" not in fact                          # 没有祈使句可被当成指令再执行一遍


def test_without_a_record_it_never_falls_back_to_the_instruction(tmp_path):
    """旧存档里已经达成、但没记过台词的信号：退回 desc（一句描述），绝不退回 reveal（指令）。"""
    s = settings(tmp_path)
    (tmp_path / "data/saves").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/saves/c.json").write_text(json.dumps({"world": {"act_id": "act2", "phase_idx": 0, "location": "kamisato_estate",
                                                                      "clues": ["c_case", "c_shops"], "achieved": ["asked_purpose"]}}))
    pv = pp.capture("c", "ayaka", say="你好", intent="闲聊", settings=s)
    block = "\n".join(m["content"] for r in pv.requests for m in r.messages)
    assert "把事情说清楚" not in block and "最后郑重地问对方愿不愿意帮忙" not in block
