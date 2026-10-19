"""即兴设定：可以编，但编了就要记、要一致；不许编案情；没说出口的不记。"""
import json

from config.settings import load_settings
from core.npc_agent import NPCReply
from core.player_agent import ScriptedPlayer
from engine.game_engine import GameEngine
from llm.mock_backend import MockBackend

AYAKA = {"kind": "talk", "npc": "ayaka"}
THOMA = {"kind": "talk", "npc": "thoma"}
LEAVE = {"kind": "leave"}


def settings(tmp_path):
    s = load_settings(env_file=tmp_path / "none.env")
    s.game.data_dir = tmp_path / "data"
    s.harness.game_token_budget = 0
    return s


def npc_prompts(eng, npc):
    ev = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    return [e["messages"][-1]["content"] for e in ev
            if e.get("event") == "llm" and e["task"] == "npc_step" and e["span"].get("npc") == npc]


def test_improvised_fact_is_kept_reused_and_case_fact_blocked(tmp_path):
    script = [AYAKA, ("你小时候养过什么小动物吗？", "闲聊"), ("那它后来怎么样了？", "闲聊"), LEAVE, "q"]
    eng = GameEngine(settings(tmp_path), MockBackend(seed=0), ScriptedPlayer(script), out=lambda s: None, save_name="i")
    eng.run()
    ev = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    w = next(e for e in ev if e["event"] == "canon_write")
    assert [k["fact"] for k in w["kept"]] == ["小时候养过一只叫小雪的白猫，爱趴在茶室纸拉门边晒太阳"]
    assert w["dropped"][0]["fact"] == "茶盏其实是托马藏起来的" and "案情" in w["dropped"][0]["reason"]
    later = npc_prompts(eng, "ayaka")[-1]                      # 同一次交谈的下一轮就能看到
    assert "你之前说过的设定" in later and "小雪" in later
    assert all("茶盏其实是托马藏起来的" not in m.text for m in eng.npcs["ayaka"].memory.items)


def test_world_fact_is_shared_across_characters_and_persists(tmp_path):
    s = settings(tmp_path)
    script = [AYAKA, ("托马会什么乐器吗？", "闲聊"), LEAVE, THOMA, ("听说你会乐器？", "闲聊"), "q"]
    eng = GameEngine(s, MockBackend(seed=0), ScriptedPlayer(script), out=lambda s: None, save_name="w")
    eng.run()
    thoma_prompt = npc_prompts(eng, "thoma")[-1]
    assert "托马会弹三味线" in thoma_prompt and "神里绫华说过" in thoma_prompt
    eng2 = GameEngine(s, MockBackend(seed=0), ScriptedPlayer(["q"]), out=lambda s: None, save_name="w")
    assert any("三味线" in m.text for m in eng2.world_canon.items)


def test_unsaid_fact_not_recorded(tmp_path):
    eng = GameEngine(settings(tmp_path), MockBackend(seed=0), ScriptedPlayer(["q"]), out=lambda s: None, save_name="u")
    r = NPCReply(speech="今天天气不错。", new_facts=[{"fact": "我其实会说至冬语", "about": "self"}])
    res = eng.npcs["ayaka"].remember_improvisations(r, 0, {})
    assert res["kept"] == [] and "没有说出" in res["dropped"][0]["reason"]


def test_duplicate_improvisation_merged(tmp_path):
    eng = GameEngine(settings(tmp_path), MockBackend(seed=0), ScriptedPlayer(["q"]), out=lambda s: None, save_name="d")
    a = eng.npcs["ayaka"]
    for _ in range(2):
        a.remember_improvisations(NPCReply(speech="我最喜欢冬天的第一场雪。",
                                           new_facts=[{"fact": "最喜欢冬天的第一场雪", "about": "self"}]), 0, {})
    canon = [m for m in a.memory.items if m.kind == "canon"]
    assert len(canon) == 1 and canon[0].hits == 2
