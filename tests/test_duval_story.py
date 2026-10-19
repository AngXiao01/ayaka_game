"""雪见茶会与多出来的客人（content/）：茶会前的怪事、茶会上的两位杜瓦尔、找破绽、三种收场。"""
import json
from pathlib import Path

import pytest

from config.settings import load_settings
from core.god_env import GodEnv, QuestDef
from core.player_agent import ScriptedPlayer
from core.world import World
from engine.game_engine import GameEngine
from llm.mock_backend import MockBackend
from models.personality import Personality
from tools import lint_story

ROOT = Path(__file__).resolve().parent.parent
STORY = ROOT / "content"


def settings(tmp_path):
    s = load_settings(env_file=tmp_path / "none.env")
    s.game.data_dir, s.game.seed, s.harness.game_token_budget = tmp_path / "data", 0, 0
    s.game.content_dir, s.game.quest_id, s.game.npc_id = STORY, "duval_main", "ayaka"
    return s


def engine(tmp_path, script=("q",), world=None):
    if world is not None:
        f = tmp_path / "data/saves/d.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"world": world}))
    out = []
    eng = GameEngine(settings(tmp_path), MockBackend(seed=0), ScriptedPlayer(list(script)), out=out.append, save_name="d", new_game=world is None)
    return eng, out


def party_world(**extra):
    return {"day": 3, "phase_idx": 1, "location": "kamisato_estate", "flags": ["party_open", "duval_arrived"], "act_id": "act1", **extra}


def make_god(state):
    q, w = QuestDef.load(STORY / "quests/duval_main.json"), World.load(STORY / "world.json")
    g = GodEnv(q, w.phases, w.start, w.turns_per_phase, state, clock=w.clock())
    g.trust_lookup = lambda n: 50
    return g


def ids(g):
    return {d["id"] for d in g.pending_signals()}


def test_the_story_lints_clean_and_is_the_default(monkeypatch, tmp_path):
    assert lint_story.main([]) == 0
    monkeypatch.delenv("CONTENT_DIR")                                         # conftest 为引擎级测试设了另一个故事
    s = load_settings(env_file=tmp_path / "none.env")
    assert s.game.content_dir == STORY and s.game.quest_id == "duval_main" and s.game.npc_id == "ayaka"
    monkeypatch.setenv("GAME_NPC", "thoma")                                   # 显式写了就以它为准
    assert load_settings(env_file=tmp_path / "none.env").game.npc_id == "thoma"


def test_before_the_party_only_the_host_side_is_around_and_the_guests_are_elsewhere(tmp_path):
    eng, _ = engine(tmp_path)
    g = eng.god
    assert set(g.physical.at("kamisato_estate")) == {"ayaka", "thoma", "paimon"}
    assert g.physical.where("duval") == "ritou" and g.physical.where("fake_duval") is None      # 真画家在离岛；另一位还没出现


def test_the_party_opens_with_one_guest_and_the_other_arrives_half_an_hour_later(tmp_path):
    eng, _ = engine(tmp_path)
    g = eng.god
    g.minute = g.phase_start(3, 1) + 1                                          # 第 3 天午后
    g.agendas.tick()
    assert "party_open" in g.flags and "fake_duval" in g.physical.at("kamisato_estate") and "duval" not in g.physical.at("kamisato_estate")
    g.minute += 31
    notes = g.agendas.tick()
    assert "duval_arrived" in g.flags and set(g.physical.at("kamisato_estate")) >= {"duval", "fake_duval", "ayaka"}
    assert any("自报姓名" in n for n in notes)                                    # 两位杜瓦尔撞了个正着
    g.flags.append("deadline_passed")                                           # 散席之后
    assert "fake_duval" not in g.physical.at("kamisato_estate")


def test_the_five_tells_are_only_visible_once_the_party_is_on(tmp_path):
    eng, _ = engine(tmp_path)
    ex = eng.explorer
    desc, got = ex.examine("o_tea_entrance")
    assert not got and "还没有人" in desc                                         # 茶会前：只是一个半人高的入口
    eng.god.flags += ["party_open"]
    desc, got = ex.examine("o_tea_entrance")
    assert got == ["c_tell_entrance"] and "枫丹淑女从不摘帽" in desc
    for obj, clue in (("o_sweets", "c_tell_sweets"), ("o_easel", "c_tell_sketch"), ("o_cushions", "c_tell_seat"), ("o_outfits", "c_tell_outfit")):
        assert ex.examine(obj)[1] == [clue]


def test_the_two_guests_are_labelled_by_what_they_wear_not_by_who_they_are(tmp_path):
    eng, _ = engine(tmp_path)
    assert {eng.personas[n].name for n in ("duval", "fake_duval")} == {"杜瓦尔女士（旅行装）", "杜瓦尔女士（羽毛帽）"}      # 标签只描述外观


def test_the_hosts_signal_needs_two_tells_not_just_one():
    g = make_god(party_world())
    g.clues = ["c_tell_entrance"]
    g.begin_conversation("ayaka")
    g.conv["turns"] = 2
    assert "tells_noticed" not in ids(g)
    g.clues.append("c_tell_sweets")
    assert "tells_noticed" in ids(g)


def decision_god(clues):
    g = make_god(party_world(act_id="act2"))
    g.clues = list(clues)
    return g


def test_the_ways_to_end_it_only_appear_once_two_tells_are_found():
    assert decision_god(["c_tell_entrance"]).proposals_for("ayaka") == []
    g = decision_god(["c_tell_entrance", "c_tell_seat"])
    assert {p["flag"] for p in g.proposals_for("ayaka")} == {"proposed_expose", "proposed_expose_wrong", "proposed_private"}
    assert {p["flag"] for p in g.proposals_for("paimon")} == {"proposed_paimon"}


@pytest.mark.parametrize("flag,signal,npc,ending", [
    ("proposed_expose", "decided_expose", "ayaka", "expose"),
    ("proposed_private", "decided_private", "ayaka", "private"),
    ("proposed_paimon", "decided_paimon", "paimon", "paimon"),
    ("proposed_expose_wrong", "decided_expose_wrong", "ayaka", "paimon_after_wrong")])
def test_each_choice_reaches_its_own_ending(flag, signal, npc, ending):
    g = decision_god(["c_tell_entrance", "c_tell_seat"])
    g.begin_conversation(npc)
    g.conv["turns"] = 2
    g.propose(flag)
    assert signal in ids(g)
    g.apply_signals([signal])
    assert g.resolved and g.compute_ending(50).id == ending


def test_the_choices_exclude_each_other():
    g = decision_god(["c_tell_entrance", "c_tell_seat"])
    g.begin_conversation("ayaka")
    g.conv["turns"] = 2
    g.propose("proposed_expose")
    g.propose("proposed_private")                                               # 改主意：旧提议作废
    assert "proposed_expose" not in g.flags
    g.apply_signals(["decided_private"])
    assert g.proposals_for("ayaka") == [] and g.proposals_for("paimon") == []


def test_accusing_the_real_guest_turns_into_the_paimon_ending_with_a_line_about_the_mistake():
    g = decision_god(["c_tell_entrance", "c_tell_seat"])
    g.begin_conversation("ayaka")
    g.conv["turns"] = 2
    g.propose("proposed_expose_wrong")
    g.apply_signals(["decided_expose_wrong"])
    e = g.compute_ending(50)
    assert e.id == "paimon_after_wrong" and "帽子底下什么也没有" in e.text and "豆吉" in e.text and "accused_wrong" in g.flags


def test_if_nobody_acts_by_the_end_of_the_party_the_default_ending_is_quiet():
    g = decision_god(["c_tell_entrance"])
    g.act_id = "act1"
    g.advance(2)
    assert g.deadline_reached() and g.compute_ending(30).id == "too_late"


def test_staking_out_the_back_door_on_the_second_night_shows_the_practice_scene(tmp_path):
    eng, _ = engine(tmp_path, world={"location": "kamisato_estate", "act_id": "act1", "minute": 1440 + 17 * 60 + 5})     # 第 2 天黄昏
    g = eng.god
    eng.actions.resolve("stakeout", None, "在后门蹲守")                          # 守到下一个时段（夜晚）
    assert "c_practice" in g.clues and "saw_practice" in g.flags
    assert any("屈膝行礼" in n for n in g.notices)                                # 看到的那一幕会播给玩家


def test_not_being_there_means_not_seeing_it(tmp_path):
    eng, _ = engine(tmp_path, world={"location": "chinju_forest", "act_id": "act1", "minute": 1440 + 17 * 60 + 5})
    g = eng.god
    g.advance(1)
    g.agendas.tick()
    assert "c_practice" not in g.clues and "practice_done" in g.flags


def test_the_old_tanuki_needs_two_pieces_of_evidence_before_he_tells_the_story():
    g = make_god({"act_id": "act1", "location": "chinju_forest", "minute": 17 * 60})
    g.clues = ["c_paw_trail"]
    g.begin_conversation("old_tanuki")
    g.conv["turns"] = 2
    assert "rv_old_tanuki_tale" not in ids(g)
    g.clues.append("c_leaf")
    assert "rv_old_tanuki_tale" in ids(g)


def test_the_hosts_cards_do_not_know_who_is_who():
    personas = Personality.load_all(STORY / "npcs")
    for nid in ("ayaka", "thoma", "paimon"):
        text = json.dumps(personas[nid].__dict__, ensure_ascii=False)
        assert "豆吉" not in text, nid
    assert "豆吉" in personas["fake_duval"].identity and "绝不" in "".join(personas["fake_duval"].speech_style)


def test_a_whole_party_scene_plays_through_the_engine_without_error(tmp_path):
    script = [{"kind": "examine", "obj": "o_tea_entrance"}, {"kind": "examine", "obj": "o_sweets"}, {"kind": "talk", "npc": "ayaka"},
              ("绫华小姐，借一步说话——这两位杜瓦尔女士，我看有一位不太对劲。", "自由发言"), {"kind": "leave"}, "q"]
    eng, out = engine(tmp_path, script, world=party_world(minute=3 * 1440 + 12 * 60))
    eng.run()
    assert {"c_tell_entrance", "c_tell_sweets"} <= set(eng.god.clues)
    assert any("茶室入口只有半人高" in line for line in out)
