"""世界自己会动：硬问（软锁）、NPC 的打算（agenda）、自由行动、由状态拼出的结局。"""
import core.actions as actions_mod
from core.actions import ActionResolver
from core.agendas import Agendas
from tests.helpers import WORLD
from tests.test_physical import PERSONAS
from tests.test_physical import engine, world


def living(state):
    g, phys, ex = world(state)
    g.agendas = Agendas(g, g.quest.agendas)
    g.seed = 0
    return g, phys, ex, ActionResolver(g, WORLD, PERSONAS, ex)


# ------------------------------------------------------------ 软锁

def test_locked_testimony_can_be_pressed_and_odds_move():
    g, *_ = living({"act_id": "act2", "clues": ["c_case", "c_shops"]})
    g.trust_lookup = lambda n: 20
    sig = {s.id: (p, why) for s, p, why in g.pressable("kokichi")}
    assert "rv_kokichi_double" in sig                      # 没有样张也能硬问
    bare = sig["rv_kokichi_double"][0]
    g.clues.append("c_foxfire")
    shown = dict((s.id, p) for s, p, _ in g.pressable("kokichi"))["rv_kokichi_double"]
    assert shown > bare                                    # 手上有宗次的话，硬问更有底气
    g.known["kokichi"] = ["c_foxfire"]
    assert "rv_kokichi_double" not in {s.id for s, *_ in g.pressable("kokichi")}   # 摆出来就不用赌了
    g.known["kokichi"] = []
    g.note_press_miss(g.quest.signal("rv_kokichi_double"), "kokichi")
    assert "alerted:kokichi" in g.flags                    # 问砸了：对方起了戒心
    assert dict((s.id, p) for s, p, _ in g.pressable("kokichi"))["rv_kokichi_double"] < shown


def test_plot_gates_cannot_be_pressed():
    g, *_ = living({"act_id": "act2", "clues": ["c_case", "c_shops"]})
    assert "decided_mercy" not in {s.id for s, *_ in g.pressable("ayaka")}
    assert "reported_findings" not in {s.id for s, *_ in g.pressable("ayaka")}


# ------------------------------------------------------------ NPC 的打算

def test_alerted_apprentice_hides_at_the_shrine():
    g, phys, ex, _ = living({"act_id": "act2", "phase_idx": 0, "clues": ["c_case"], "location": "hanamizaka"})
    g.flags.append("alerted:kokichi")
    g.pass_time(190)
    assert "kokichi_hid" in g.flags and phys.where("kokichi") in (None, "chinju_forest")      # 被吓着的人往神龛去了（路上或已到）


def test_being_there_catches_him_praying():
    g, _, _, _ = living({"act_id": "act2", "phase_idx": 0, "clues": ["c_case"], "location": "chinju_forest"})
    g.flags.append("alerted:kokichi")
    g.pass_time(190)
    assert "caught_praying" in g.flags and "kokichi_hid" in g.flags


def test_clearing_him_stops_the_hiding():
    g, *_ = living({"act_id": "act2", "phase_idx": 0, "clues": ["c_case"], "location": "hanamizaka",
                    "achieved": ["rv_kokichi_double"]})
    g.flags.append("alerted:kokichi")
    g.pass_time(190)
    assert "kokichi_hid" not in g.flags                       # 说开了就不用躲了


def test_unchecked_forger_sends_a_second_wave():
    g, *_ = living({"act_id": "act2", "day": 2, "phase_idx": 1, "clues": ["c_case"], "location": "kamisato_estate"})
    g.advance(1)
    assert "second_wave" in g.flags and "c_round_prints" in g.clues          # 正好在屋敷：撞见狸猫叼走点心盒
    assert "saw_tanuki" in g.flags


# ------------------------------------------------------------ 自由行动

def test_report_to_tenryou_needs_no_evidence_and_ends_the_case(tmp_path):
    # 新案情里报案不要凭据：天领奉行接了案，平藏自己会去查，案子就从绫华手里溜走了
    g, _, _, act = living({"act_id": "act2", "clues": ["c_case"]})
    assert act.parse("我要去天领奉行报案")[0] == "report_authority"
    text, _ = act.resolve("report_authority", None, "报案")
    assert "报了案" in text and "reported_tenryou" in g.flags
    again, _ = act.resolve("report_authority", None, "报案")
    assert "已经去过" in again                                       # 报过一次就不再重复
    eng, _, out = engine(tmp_path, {"act_id": "act2", "phase_idx": 0, "clues": ["c_case"]},
                         [("去天领奉行报案", "自由发言"), {"kind": "wait"}, {"kind": "wait"}, "q"])
    assert "tenryou_arrest" in eng.god.flags and eng.god.ending == "tenryou"
    assert any("侦探的结案书" in line for line in out)


def test_follow_reveals_what_talking_cannot(monkeypatch):
    g, phys, ex, act = living({"act_id": "act2", "phase_idx": 1, "clues": ["c_case"], "location": "ritou"})
    monkeypatch.setattr(actions_mod, "roll", lambda *a: (True, 80))
    assert act.parse("悄悄跟上去看看宗次去哪") == ("follow", "soji")
    _, got = act.resolve("follow", "soji", "跟踪宗次")
    assert got == ["c_foxfire"] and "saw_soji_beach" in g.flags      # 跟着宗次到了海滩，听见他嘟囔那团火


def test_failed_follow_alerts_target(monkeypatch):
    g, phys, ex, act = living({"act_id": "act2", "phase_idx": 1, "clues": ["c_case"], "location": "ritou"})
    monkeypatch.setattr(actions_mod, "roll", lambda *a: (False, 30))
    act.resolve("follow", "soji", "跟踪宗次")
    assert "alerted:soji" in g.flags


def test_warning_the_forger_changes_the_epilogue(tmp_path):
    eng, _, out = engine(tmp_path, {"act_id": "act2", "phase_idx": 0, "location": "print_workshop", "clues": ["c_case"]},
                         [("找个没人的时候通风报信，让他快跑", "自由发言"), {"kind": "wait"}, {"kind": "wait"}, "q"])
    g = eng.god
    assert {"warned_kokichi", "kokichi_hid"} <= set(g.flags)
    assert "绫华始终不知道" in "".join(eng.outcome_lines())


def _heizou_state():
    return {"act_id": "act2", "phase_idx": 1, "location": "hanamizaka", "clues": ["c_case", "c_tanuki_hair", "c_sill"],
            "achieved": ["asked_purpose", "accepted_commission", "rv_heizou_sill"]}


def test_concession_said_out_loud_counts_and_removes_the_dice(tmp_path):
    # trace：平藏嘴上已经认了输，却没算达成，玩家还被摆出一个低概率的硬问按钮
    from llm.decider import ScriptedDecider
    from tests.test_decider import engine as eng_with
    d = ScriptedDecider({"conceded": {"rv_heizou_shapeshift": (True, 0.9)}})
    eng, _, _ = eng_with(tmp_path, _heizou_state(),
                         [{"kind": "talk", "npc": "heizou"}, ("样张只印了一半，阿缘说那人的耳朵是尖的，人和狐其实是同一个。", "自由发言"), "q"], d)
    assert "rv_heizou_shapeshift" in eng.god.achieved and "heizou_conceded" in eng.god.flags


def test_good_argument_raises_press_odds():
    g, *_ = living(_heizou_state())
    g.trust_lookup = lambda n: 22
    s = g.quest.signal("rv_heizou_shapeshift")
    base, _ = g.press_chance(s, "heizou")
    strong, why = g.press_chance(s, "heizou", 2)
    assert strong >= base + 0.39 and any("理由很充分" in w for w in why)
