"""用 MockBackend 跑整局：开放调查、存读档、冒犯、后端挂掉。"""
import json

from config.settings import load_settings
from core.player_agent import AutoPlayer, ScriptedPlayer
from engine.game_engine import GameEngine
from llm.base import LLMError
from llm.mock_backend import MockBackend

TALK_AYAKA = {"kind": "talk", "npc": "ayaka"}
LEAVE = {"kind": "leave"}


def make_settings(tmp_path):
    s = load_settings(env_file=tmp_path / "none.env")
    s.game.data_dir = tmp_path / "data"
    s.game.seed = 1
    s.harness.game_token_budget = 0
    return s


def test_cooperative_run_reaches_good_ending(tmp_path):
    s = make_settings(tmp_path)
    out = []
    eng = GameEngine(s, MockBackend(seed=1), AutoPlayer(seed=1, out=out.append), out=out.append, save_name="t")
    res = eng.run()
    assert res.quest_complete, res.ending
    assert len(eng.god.clues) >= 5 and len({r.npc for r in res.sessions}) >= 3     # 跟多个角色打过交道
    assert any("调查开始" in line for line in out)
    save = json.loads((tmp_path / "data/saves/t.json").read_text())
    assert save["world"]["ending"] == res.ending and "aen" in save["relationships"]
    assert (tmp_path / "data/memories/t/ayaka.jsonl").is_file()
    tasks = {json.loads(l).get("task") for l in eng.tracer.path.read_text().splitlines()}
    assert {"npc_step", "player_options", "summary"} <= tasks
    # 这一局的进展主要来自动作直达（关键问题、出示、汇报、提议）；离线模型偶尔会顺口上报一个，
    # 上报的信号照样要过审核，不能靠嘴推进。审核员本身由 test_reviewer.py 覆盖
    st = eng.tracer.stats
    assert st["signals_by_action"] >= 8 and st["signals_claimed"] <= 2


def test_resume_from_save_keeps_world_and_memory(tmp_path):
    s = make_settings(tmp_path)
    eng = GameEngine(s, MockBackend(seed=0), ScriptedPlayer([TALK_AYAKA, 1, 1, 1, LEAVE, "q"]),
                     out=lambda x: None, save_name="r")
    eng.run()
    eng2 = GameEngine(s, MockBackend(seed=0), ScriptedPlayer(["q"]), out=lambda x: None, save_name="r")
    assert eng2.resumed and eng2.god.met["ayaka"] == 1 and eng2.god.journal
    assert eng2.npcs["ayaka"].last_meeting()
    eng3 = GameEngine(s, MockBackend(seed=0), ScriptedPlayer(["q"]), out=lambda x: None, save_name="r",
                      new_game=True)
    assert not eng3.resumed and not eng3.npcs["ayaka"].memory.items


def test_offensive_player_lowers_affinity(tmp_path):
    s = make_settings(tmp_path)
    lines = [TALK_AYAKA] + [("你这白鹭公主也不过如此", "冒犯")] * 3 + ["q"]
    eng = GameEngine(s, MockBackend(seed=0), ScriptedPlayer(lines), out=lambda x: None, save_name="o")
    eng.run()
    assert eng.npcs["ayaka"].relationship.affinity < 30


class DeadBackend:
    name = "dead"

    def complete(self, req):
        raise LLMError("服务挂了")


def test_backend_down_falls_back_without_crash(tmp_path):
    s = make_settings(tmp_path)
    player = ScriptedPlayer([TALK_AYAKA, 1, ("你好", "自由发言"), "q"])
    eng = GameEngine(s, DeadBackend(), player, out=lambda x: None, save_name="d")
    res = eng.run()
    assert res.stopped_by_player and eng.tracer.stats["fallbacks"] > 0
    assert any(o.text == "近来可好？看您气色不错。" for o in player.seen_options[1])   # 人物卡里的兜底选项
