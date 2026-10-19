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
    return {"day": 2, "phase_idx": 1, "location": "kamisato_estate", "flags": ["party_open", "duval_arrived"], "act_id": "act1", **extra}


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
    assert g.physical.where("duval") is None and g.physical.where("fake_duval") is None      # 两位杜瓦尔都还没出现：真画家茶会当天才和玩家见面


def test_the_party_opens_with_one_guest_and_the_other_arrives_half_an_hour_later(tmp_path):
    eng, _ = engine(tmp_path)
    g = eng.god
    g.minute = g.phase_start(2, 1) + 1                                          # 第 2 天午后
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
    eng.god.location = "kamisato_estate"                                        # 开局在离岛，茶室在屋敷
    ex = eng.explorer
    desc, got = ex.examine("o_tea_entrance")
    assert not got and "平日的样子" in desc                                       # 茶会前：只是平日里在打扫的茶室，没有摆好席
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


def test_game_starts_at_ritou_with_affinity_35_and_a_countdown(tmp_path):
    eng, _ = engine(tmp_path)
    g = eng.god
    assert g.location == "ritou" and eng.npcs["ayaka"].relationship.affinity == 35
    assert g.milestone_text().startswith("距雪见茶会开席还有约1天4个钟头（明天午后")     # 07:00 → 第 2 天 11:00
    g.minute = g.phase_start(1, 1) + 5
    assert "明天午后" in g.milestone_text() and "还有约" in g.milestone_text()
    g.minute = g.phase_start(2, 1) + 1
    assert g.milestone_text() == "雪见茶会开席已经开始了"


def test_activities_need_affinity_and_the_right_phase(tmp_path):
    eng, _ = engine(tmp_path, world=party_world(day=1, phase_idx=1, flags=[]))
    g, phys = eng.god, eng.god.physical
    g.location = "kamisato_estate"
    assert phys.can_do("ayaka", "sword", 35)[0] is False and "不好意思" in phys.can_do("ayaka", "sword", 35)[1]
    assert phys.can_do("ayaka", "sword", 45) == (True, "")
    assert phys.can_do("ayaka", "moonwalk", 80)[0] is False                      # 午后不是夜晚
    g.minute = g.phase_start(1, 3) + 5
    assert phys.can_do("ayaka", "moonwalk", 54)[0] is False and phys.can_do("ayaka", "moonwalk", 55)[0] is True
    g.flags.append("did:moonwalk:1")
    assert phys.can_do("ayaka", "moonwalk", 99)[0] is False                      # 同一天只一次


def test_inviting_ayaka_to_spar_runs_the_scene_and_raises_affinity(tmp_path):
    world = party_world(day=1, phase_idx=1, flags=[], location="kamisato_estate", minute=None)
    world.pop("minute")
    eng, out = engine(tmp_path, world=world,
                      script=[{"kind": "talk", "npc": "ayaka"}, {"kind": "invite", "activity": "sword"}, {"kind": "leave"}, "q"])
    eng.npcs["ayaka"].relationship.affinity = 50
    before = eng.god.minute
    eng.run()
    text = "\\n".join(out)
    assert "白砂园里，你们各取了一柄练习用的木刀" in text
    assert "did:sword:1" in eng.god.flags and eng.god.minute >= before + 60
    assert eng.npcs["ayaka"].relationship.affinity >= 53


def test_the_player_does_not_know_the_painter_before_the_party(tmp_path):
    eng, _ = engine(tmp_path)
    pv = eng.explorer.player_view("ayaka")
    assert "杜瓦尔" not in pv["known_text"] and "画家" not in pv["known_text"]
    assert "杜瓦尔" in pv["spoiler_terms"]                                      # 台词里要提她，得先听人说起
    assert eng.god.physical.where("duval") is None                              # 离岛上也碰不到她


def test_daily_life_services_cost_time_and_run_once_per_slot(tmp_path):
    eng, _ = engine(tmp_path)
    g, ex = eng.god, eng.explorer
    assert [s["id"] for s in ex.services_here()] == ["ritou_breakfast"]          # 开局：离岛的清晨
    t0 = g.minute
    text = ex.service("ritou_breakfast")
    assert g.minute == t0 + 20 and text
    assert ex.services_here() == []                                              # 同一时段不再给第二次早饭
    g.location = "akisa_yu"
    g.minute = g.phase_start(1, 1) + 5
    assert [s["id"] for s in ex.services_here()] == ["akisa_bath"]
    t1 = g.minute
    ex.service("akisa_bath")
    assert g.minute == t1 + 60



def test_cameos_are_chance_encounters_decided_once_per_slot(tmp_path):
    eng, _ = engine(tmp_path)
    g, phys = eng.god, eng.god.physical
    seen = []
    for day in range(1, 31):
        g.minute = g.phase_start(day, 2) + 1                                     # 每天的黄昏
        a = phys.where("yoimiya")
        assert a == phys.where("yoimiya")                                        # 同一时段里不会忽有忽无
        seen.append(a)
    assert "hanamizaka" in seen and None in seen                                 # 有时在，有时不在
    assert phys.where("aikawa") == "akisa_yu"                                    # 钱汤老板常驻钱汤
    assert phys.where("anna") == "hanamizaka"                                    # 料理店老板娘常驻花见坂


def test_hidden_activity_buttons_need_closeness_and_the_right_time_and_show_no_numbers(tmp_path):
    world = party_world(day=1, phase_idx=1, flags=[], location="kamisato_estate")
    world.pop("day"); world.pop("phase_idx")
    eng, _ = engine(tmp_path, world=world, script=[{"kind": "talk", "npc": "ayaka"}, "q"])
    eng.god.minute = eng.god.phase_start(1, 1) + 5                                # 第 1 天午后
    from core.dialogue_session import DialogueSession
    def labels(aff, minute):
        eng.god.minute = minute
        eng.npcs["ayaka"].relationship.affinity = aff
        eng.god.begin_conversation("ayaka")
        s = DialogueSession(eng.npcs["ayaka"], eng.player, eng.option_gen, eng.god, eng.world, eng.settings, eng.reviewer, eng.budget)
        return [o.text for _, o, _, _ in s._actions()]
    noon, night = eng.god.phase_start(1, 1) + 5, eng.god.phase_start(1, 3) + 5
    assert not any("切磋" in t for t in labels(40, noon))                         # 交情不够：看不到
    assert any("切磋" in t for t in labels(45, noon))                             # 够了、时段也对：出现
    assert not any("月下散步" in t for t in labels(80, noon))                      # 时段不对：看不到
    assert any("月下散步" in t for t in labels(55, night))
    assert not any("好感" in t for t in labels(45, noon))


def test_people_are_introduced_once_in_the_panel_and_the_talk_header(tmp_path):
    eng, out = engine(tmp_path, world={"location": "kamisato_estate", "act_id": "act1"})
    ex = eng.explorer
    first = ex.panel()
    assert "神里绫华（" in first and "托马（" in first
    second = ex.panel()
    assert "这里的人：神里绫华；" in second and "（社奉行" not in second          # 第二次只剩名字


def test_small_talk_with_nothing_at_stake_winds_down_even_while_getting_warmer(tmp_path):
    eng, _ = engine(tmp_path)
    g = eng.god
    g.location = "hanamizaka"
    g.begin_conversation("anna")
    anna = eng.personas["anna"]
    for i in range(g.CHAT_WIND_DOWN):
        assert not g.npc_task_detail(anna, 20)["wind_down"]
        g.conversation_turn()
        g.note_engagement(warm=True)                                             # 越聊越暖，也不算正事
    assert g.npc_task_detail(anna, 20)["wind_down"] and not g.chat_idle(g.CHAT_HARD_STOP)
    for i in range(g.CHAT_HARD_STOP - g.CHAT_WIND_DOWN):
        g.conversation_turn()
    assert g.chat_idle(g.CHAT_HARD_STOP)                                         # 对方没收住，代码兜底
    g.note_engagement()                                                          # 真有进展就重新计数
    assert g.idle_turns == 0
