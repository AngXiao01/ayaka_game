"""说谎与对质：谎要记、要圆；对质结果按规则判定；拆穿能撬开嘴。"""
import json

from config.settings import load_settings
from core.npc_agent import NPCReply
from core.player_agent import ScriptedPlayer
from engine.game_engine import GameEngine
from llm.mock_backend import MockBackend
from tests.helpers import make_god


def test_confront_outcomes_are_rule_based():
    g = make_god()
    lie = g.add_testimony("soji", "那晚我早睡了，谁也没见", "lie", truth="见过小吉", lie_id="soji_denial",
                          contradicted_by=["c_messenger"])
    assert g.confront(lie["id"], "c_letter", 20)["outcome"] == "unproven"
    assert g.confront(lie["id"], None, 90)["outcome"] == "unproven"        # 有据可查的谎，光凭信任撬不开
    assert g.confront(lie["id"], "c_messenger", 20)["outcome"] == "caught"
    assert lie["caught"] and "caught:soji_denial" in g.flags and g.testimonies_of("soji") == []
    white = g.add_testimony("yohei", "门窗都锁得好好的", "lie", truth="后窗忘了关")
    assert g.confront(white["id"], None, 20)["outcome"] == "unproven"
    assert g.confront(white["id"], None, 45)["outcome"] == "caught"       # 临时的谎：够信任就心虚认了
    true = g.add_testimony("aen", "那晚看见小吉抱着布包往离岛去了", "truth")
    assert g.confront(true["id"], "c_letter", 50)["outcome"] == "truth"


def settings(tmp_path):
    s = load_settings(env_file=tmp_path / "none.env")
    s.game.data_dir = tmp_path / "data"
    s.harness.game_token_budget = 0
    return s


def start_at_ritou(tmp_path, name):
    save = tmp_path / "data/saves" / f"{name}.json"
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps({"world": {"day": 2, "phase_idx": 1, "location": "ritou", "act_id": "act2",
                                          "clues": ["c_case", "c_shops", "c_messenger", "c_letter"]}}))


def test_lie_then_caught_unlocks_confession(tmp_path):
    start_at_ritou(tmp_path, "lie")
    script = [{"kind": "talk", "npc": "soji"}, ("丢东西那晚，小吉来找过你吧？", "推进任务"), ("真的谁也没见？", "试探"),
              {"kind": "confront_menu"}, {"kind": "testimony"}, {"kind": "evidence", "clue": "c_messenger"},
              ("那他找你帮什么忙？", "推进任务"), "q"]
    out = []
    eng = GameEngine(settings(tmp_path), MockBackend(seed=0), ScriptedPlayer(script), out=out.append, save_name="lie")
    eng.run()
    g = eng.god
    assert g.testimonies[0]["said"] == "小吉？那晚我早睡了，谁也没见" and g.testimonies[0]["caught"]
    assert "c_soji" in g.clues                                   # 拆穿之后撬开了嘴
    assert any("拆穿了宗次的谎" in j for j in g.journal) and any("拆穿" in o for o in out)
    ev = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    prompts = [e["messages"][-1]["content"] for e in ev if e.get("task") == "npc_step"]
    assert any("你对对方说过的谎" in p for p in prompts)        # 说过的谎进了情境，要圆住
    assert any("瞒不住了" in p for p in prompts)                # 对质结果告诉了 NPC
    assert any(m.kind == "lie" for m in eng.npcs["soji"].memory.items)
    assert next(e for e in ev if e["event"] == "confront")["outcome"] == "caught"


def test_false_accusation_costs_trust(tmp_path):
    start_at_ritou(tmp_path, "acc")
    eng = GameEngine(settings(tmp_path), MockBackend(seed=0), ScriptedPlayer(["q"]), out=lambda s: None, save_name="acc")
    eng.god.add_testimony("aen", "那晚看见小吉抱着布包往离岛去了", "truth")
    eng.god.location = "komore_teahouse"
    eng.god.minute = eng.god.phase_start(2, 1) + 30
    trust0 = eng.npcs["aen"].relationship.trust
    eng.player = ScriptedPlayer([{"kind": "talk", "npc": "aen"}, {"kind": "confront_menu"}, {"kind": "testimony"},
                                 {"kind": "evidence", "clue": "c_letter"}, "q"])
    eng.run()
    assert eng.npcs["aen"].relationship.trust <= trust0 - 2


def test_unsaid_lie_not_recorded(tmp_path):
    eng = GameEngine(settings(tmp_path), MockBackend(seed=0), ScriptedPlayer(["q"]), out=lambda s: None, save_name="u")
    r = NPCReply(speech="今天天气不错。", lies=[{"said": "我昨天去过离岛", "truth": "没去过", "lie_id": "improvised"}])
    assert eng.npcs["ayaka"].record_lies(r, eng.god, 0)["kept"] == [] and eng.god.testimonies == []
