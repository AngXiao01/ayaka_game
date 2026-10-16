"""渐进式披露（人物卡索引 + 按需注入）与上下文重置后的结构化交接。"""
import json
from pathlib import Path

from config.settings import load_settings
from core.player_agent import ScriptedPlayer
from engine.game_engine import GameEngine
from llm.messages import estimate_tokens
from llm.mock_backend import MockBackend
from models.personality import Personality

ROOT = Path(__file__).resolve().parent.parent
AYAKA = Personality.load_all(ROOT / "content/npcs")["ayaka"]
TALK_AYAKA = {"kind": "talk", "npc": "ayaka"}


def engine(tmp_path, lines, **harness):
    s = load_settings(env_file=tmp_path / "none.env")
    s.game.data_dir = tmp_path / "data"
    s.game.seed = 0
    s.harness.game_token_budget = 0
    for k, v in harness.items():
        setattr(s.harness, k, v)
    return GameEngine(s, MockBackend(seed=0), ScriptedPlayer(lines), out=lambda x: None)


def events(eng):
    return [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]


def test_index_card_is_shorter_and_keeps_the_voice():
    full, idx = AYAKA.to_prompt("旅行者", "full"), AYAKA.to_prompt("旅行者", "index")
    assert estimate_tokens(idx) < estimate_tokens(full)
    assert AYAKA.anecdotes[0].split("，")[0] in full and AYAKA.anecdotes[0].split("，")[0] not in idx
    for kept in (AYAKA.speech_style[0], AYAKA.taboos[0]):
        assert kept.format(player_name="旅行者") in idx
    assert "人物细节索引" in idx and "托马" in idx


def test_details_are_routed_by_topic():
    names = lambda text, intent=None: [t for t, _ in AYAKA.relevant_details(text, intent, 0)]   # noqa: E731
    assert "人际" in names("托马最近在忙什么？", "打听情报")
    assert "人际" in names("绫人大人还好吗？", "打听情报")               # 只说了名字的后两个字
    assert any("看法" in t for t in names("你怎么看眼狩令？", "推进任务"))
    assert names("今天的事就说到这里", "推进任务") == []                  # 不相关就什么都不附


def test_index_mode_injects_details_and_offers_load_persona(tmp_path):
    eng = engine(tmp_path, [TALK_AYAKA, ("托马最近在忙什么？", "自由发言"), "q"], persona_mode="index")
    eng.run()
    assert "load_persona" in eng.npcs["ayaka"].registry.names()
    assert "托马" in eng.npcs["ayaka"].registry.execute("load_persona", {"section": "人际"})[0]
    steps = [e for e in events(eng) if e["event"] == "llm" and e["task"] == "npc_step"]
    assert "人物细节索引" in steps[-1]["messages"][0]["content"]
    assert "本轮相关的人物细节：人际" in steps[-1]["messages"][-1]["content"]
    eng_full = engine(tmp_path / "full", [TALK_AYAKA, ("托马最近在忙什么？", "自由发言"), "q"], persona_mode="full")
    assert "load_persona" not in eng_full.npcs["ayaka"].registry.names()


def test_handoff_after_reset_is_structured_and_grounded(tmp_path):
    line = "上次说的那个募捐的人你查到了吗"
    eng = engine(tmp_path, [TALK_AYAKA, (line, "打听情报"), {"kind": "leave"},
                            TALK_AYAKA, ("我又来了", "礼貌问候"), "q"])
    eng.run()
    ev = events(eng)
    w = next(e for e in ev if e["event"] == "memory_write")
    assert any(line in t for t in w["handoff"]["open_threads"])
    assert "约好三天后在雪见茶会上碰面" in w["handoff_dropped"]["promises"]      # 对话里没说过，不算约定
    second = [e for e in ev if e["event"] == "llm" and e["task"] == "npc_step" and e["span"].get("session") == 2]
    last = second[0]["messages"][-1]["content"]
    assert "## 上次见面" in last and line in last
