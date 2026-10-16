"""物理世界：时间按分钟走（说话便宜、走路花时间）、同行、带话、各人只知道自己看到和听说的、结局之后继续。"""
import json

from config.settings import load_settings
from core.explorer import Explorer
from core.physical import PLAYER, PhysicalWorld
from core.player_agent import ScriptedPlayer
from engine.game_engine import GameEngine
from llm.mock_backend import MockBackend
from models.personality import Personality
from models.relationship import Relationship
from tests.helpers import ROOT, WORLD, make_god

PERSONAS = Personality.load_all(ROOT / "content/npcs")


def world(state=None, aff=30):
    g = make_god(state)
    phys = PhysicalWorld(WORLD, PERSONAS, g, player_name="旅行者")
    g.physical = phys
    rels = {n: Relationship(aff, 20) for n in PERSONAS}
    return g, phys, Explorer(g, WORLD, PERSONAS, lambda n: rels[n], out=lambda s: None)


def test_talking_is_cheap_travel_takes_time():
    g, _, ex = world()
    for _ in range(20):
        g.conversation_turn()
    assert g.clock_label() == "07:40"                       # 二十轮话才四十分钟
    ex.move("hanamizaka")
    assert g.clock_label() == "08:25"                       # 屋敷到花见坂走了 45 分钟
    ex.move("komore_teahouse")
    assert g.clock_label() == "08:35"


def test_invite_join_follow_and_who_knows():
    g, phys, ex = world({"act_id": "act2", "phase_idx": 1})
    assert phys.can_join("ayaka", 45) == (True, "")
    phys.join("ayaka", "花见坂")
    assert phys.inbox["thoma"]                               # 出门前告诉了托马（人物卡 outing.tell）
    ex.move("hanamizaka")
    t = phys.belief("thoma", "ayaka")                       # 托马也看着你们出了门
    assert t["how"] == "left" and "花见坂" in t["note"]
    assert phys.where("ayaka") == "hanamizaka"              # 跟着玩家走
    assert phys.belief(PLAYER, "ayaka")["how"] == "with"
    assert phys.belief("aen", "ayaka")["how"] == "usual"    # 阿缘没人告诉，还以为她在屋敷
    assert phys.belief("aen", "ayaka")["location"] == "kamisato_estate"
    assert "ev_walk" not in [e.id for e in g.available_events("hanamizaka", lambda n: 99)]   # 人就在身边，不算偶遇


def test_decline_when_not_close_enough_or_busy():
    g, phys, _ = world({"act_id": "act2"})
    ok, why = phys.can_join("ayaka", 30)
    assert not ok and "不算熟" in why
    ok, why = phys.can_join("thoma", 90)                     # 清晨托马在忙茶会
    assert not ok and "抽不开身" in why
    assert phys.can_join("yohei", 99)[0] is False            # 账房先生不出门


def test_companion_goes_home_when_time_is_up():
    g, phys, ex = world({"act_id": "act2", "phase_idx": 2})
    phys.join("ayaka")
    ex.move("hanamizaka")
    g.advance(1)                                             # 入夜
    assert "ayaka" not in phys.party
    assert any("回屋敷" in line for line in g.notices)
    assert phys.where("ayaka") is None                       # 夜里不在能去的地方（回家路上 / 歇下了）


def test_secret_outing_tells_no_one():
    g, phys, _ = world({"act_id": "act2", "phase_idx": 3, "location": "ritou"})
    assert phys.can_join("soji", 45)[0]
    phys.join("soji")
    assert phys.mail == []


def test_messages_take_time_by_distance():
    g, phys, _ = world({"act_id": "act2", "phase_idx": 1, "location": "ritou", "clues": ["c_workshop"]})
    got = []
    phys.on_deliver = got.append
    msg = phys.send("soji", "kokichi", "有人在查送信的事，你小心。")
    assert msg["deliver"] - msg["sent"] == 60                 # 离岛到花见坂要一个钟头
    g.pass_time(30)
    assert got == [] and phys.peek_inbox("kokichi") == []
    g.pass_time(30)
    assert got and "小心" in phys.npc_view("kokichi")["inbox"][0]
    assert phys.take_inbox("kokichi") and phys.peek_inbox("kokichi") == []


def test_seen_then_guessed():
    g, phys, ex = world({"act_id": "act2", "phase_idx": 1})
    ex.move("hanamizaka")
    ex.move("komore_teahouse")
    assert phys.belief(PLAYER, "aen")["how"] == "here"
    ex.move("hanamizaka")
    assert phys.belief(PLAYER, "aen")["how"] == "seen"       # 刚才在茶室见过
    g.advance(1)
    assert phys.belief(PLAYER, "aen")["how"] == "usual"      # 过了一个时段，只能按平常猜


def test_npc_view_present_and_mentioned():
    g, phys, _ = world({"act_id": "act2", "phase_idx": 1})
    v = phys.npc_view("ayaka", "托马最近在忙什么？宗次呢？")
    assert any("托马" in x for x in v["present"])
    assert any(x.startswith("托马：就在这里") for x in v["whereabouts"])
    assert any(x.startswith("宗次：平常这时候在离岛") for x in v["whereabouts"])


def test_absent_after_story_flag():
    g, phys, _ = world({"act_id": "act3", "location": "print_workshop", "flags": ["expose"]})
    assert phys.where("kokichi") is None                     # 被送官了，不在印坊


def settings(tmp_path):
    s = load_settings(env_file=tmp_path / "none.env")
    s.game.data_dir, s.game.seed, s.harness.game_token_budget = tmp_path / "data", 0, 0
    return s


def engine(tmp_path, state, script, rels=None):
    save = tmp_path / "data/saves/p.json"
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps({"world": state, "relationships": rels or {}}))
    out = []
    eng = GameEngine(settings(tmp_path), MockBackend(seed=0), ScriptedPlayer(script), out=out.append, save_name="p")
    res = eng.run()
    return eng, res, out


def test_typed_invitation_then_walk_together(tmp_path):
    eng, _, out = engine(tmp_path, {"act_id": "act2", "phase_idx": 1, "clues": ["c_case", "c_shops"]},
                         [{"kind": "talk", "npc": "ayaka"}, ("要不要一起去花见坂走走？", "自由发言"), {"kind": "leave"},
                          {"kind": "move", "dest": "hanamizaka"}, "q"],
                         rels={"ayaka": {"affinity": 45, "trust": 30}})
    phys = eng.god.physical
    assert phys.party == ["ayaka"] and phys.where("ayaka") == "hanamizaka"
    assert any("和你同行了" in line for line in out)
    assert phys.belief("thoma", "ayaka")["how"] == "left"


def test_game_goes_on_after_the_ending(tmp_path):
    state = {"act_id": "act3", "phase_idx": 0, "location": "kamisato_estate",
             "clues": ["c_case", "c_shops", "c_messenger", "c_letter", "c_soji", "c_motive"],
             "achieved": ["asked_purpose", "accepted_commission", "reported_findings"]}
    eng, res, out = engine(tmp_path, state, [{"kind": "talk", "npc": "ayaka"}, {"kind": "propose", "flag": "proposed_mercy"},
                                             {"kind": "wait"}, "q"])
    assert res.ending == "mercy" and res.stopped_by_player
    assert any("主线告一段落" in line for line in out)
    assert eng.god.phase == "午后"                             # 结局之后还在玩：等到了午后


def test_deadline_is_an_event_not_game_over(tmp_path):
    eng, res, out = engine(tmp_path, {"day": 3, "phase_idx": 2, "act_id": "act2", "clues": ["c_case"]},
                           [{"kind": "wait"}, {"kind": "view", "view": "clues"}, "q"])
    assert "deadline_passed" in eng.god.flags and eng.god.ending is None
    assert any("雪落无声" in line for line in out) and any("还可以接着查" in line for line in out)
    assert any("线索本" in line for line in out)                # 茶会过后还能继续操作
