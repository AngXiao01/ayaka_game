"""心智状态：一方以为对方知道什么 / 对什么感兴趣（core/mind.py）；两个小 agent 各管一边，互相看不到。"""
import json
from pathlib import Path

from config.settings import load_settings
from core import mind
from core.player_agent import ScriptedPlayer
from engine.game_engine import GameEngine
from llm.mock_backend import MockBackend

ROOT = Path(__file__).resolve().parent.parent


def test_clean_drops_junk_and_falls_back_to_cautious_values():
    out = mind.clean({"items": [{"topic": "画家", "belief": "胡写", "interest": 9, "share": "乱写"},
                                {"topic": "", "belief": "knows"}, {"topic": "x" * 30}, "不是对象"]})
    assert out == [{"topic": "画家", "belief": "maybe", "interest": 2, "share": "undecided", "why": ""}]
    assert mind.clean("坏的") == [] and mind.clean({"items": "坏的"}) == []


def test_merge_overwrites_by_topic_and_caps_the_list():
    a = mind.merge([], [{"topic": "茶会", "belief": "knows", "interest": 1, "share": "tell", "why": ""}], 1)
    b = mind.merge(a, [{"topic": "茶会", "belief": "unknown", "interest": 2, "share": "withhold", "why": "还没到火候"}], 2)
    assert len(b) == 1 and b[0]["belief"] == "unknown" and b[0]["turn"] == 2
    many = mind.merge([], [{"topic": f"话题{i}", "belief": "maybe", "interest": i % 3, "share": "tell", "why": ""} for i in range(4)], 1)
    for k in range(4):
        many = mind.merge(many, [{"topic": f"另{k}{j}", "belief": "maybe", "interest": 1, "share": "tell", "why": ""} for j in range(4)], 2 + k)
    assert len(many) == mind.MAX_ITEMS


def test_the_player_side_always_carries_the_do_not_assume_rule():
    text = mind.render_player([], "托马")
    assert "托马知道什么，玩家本来是不清楚的" in text and "询问或试探" in text
    assert "多半不知道" in mind.render_player([{"topic": "画家", "belief": "unknown", "why": "外地人"}], "托马")
    assert mind.render_npc([]) == ""
    npc = mind.render_npc([{"topic": "画家", "belief": "unknown", "interest": 2, "share": "withhold", "why": "还没到火候"}])
    assert "先不说" in npc and "很感兴趣" in npc


def test_a_turn_updates_both_sides_and_feeds_the_next_prompts(tmp_path):
    s = load_settings(env_file=tmp_path / "none.env")
    s.game.data_dir, s.game.seed, s.harness.game_token_budget = tmp_path / "data", 0, 0
    s.game.content_dir, s.game.quest_id, s.game.npc_id = ROOT / "content", "duval_main", "ayaka"
    seen = {"npc_step": [], "player_options": []}

    class Recording(MockBackend):
        def complete(self, req):
            if req.task in seen:
                seen[req.task].append("\n".join(m.get("content") or "" for m in req.messages))
            return super().complete(req)

    statuses = []
    eng = GameEngine(s, Recording(seed=0), ScriptedPlayer(
        [{"kind": "move", "dest": "kamisato_estate"}, {"kind": "talk", "npc": "ayaka"}, ("你好，聊聊茶会的事吧。", "闲聊"), ("茶会明天开始吗", "闲聊"),
         {"kind": "leave"}, "q"]), out=lambda x: None, save_name="m", new_game=True)
    eng.status = statuses.append
    eng.run()
    state = eng.god.minds["ayaka"]
    assert state["npc"] and state["player"]                                    # 两边各有各的判断
    assert all("share" in x for x in state["npc"])
    events = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    sides = {e["side"] for e in events if e.get("event") == "mind"}
    assert sides == {"npc", "player"}
    assert eng.god.to_dict()["minds"]["ayaka"] == state                         # 存档里带着，重开游戏还在
    assert any("## 你对对方的判断" in x for x in seen["npc_step"][1:])           # 第一轮之后，NPC 的提示词里带着她自己的判断
    assert not any("## 你对对方的判断" in x for x in seen["npc_step"][:1])
    assert all("以为" in x and "询问或试探" in x for x in seen["player_options"])
    assert all("是谁（站在面前就看得出来的）" in x and "别问显而易见的问题" in x for x in seen["player_options"])   # 选项 agent 知道对方是谁、不问明摆着的事
    assert statuses[:2] == ["options", ""] and set(statuses) == {"options", ""}                          # 准备台词时界面显示的不是“对方正在想”    # 选项 agent 每次都带着“别替对方预设知道”的规矩
