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
    assert "rv_kokichi_motive" in sig                      # 没有拓纸也能硬问
    bare = sig["rv_kokichi_motive"][0]
    g.clues.append("c_soji")
    shown = dict((s.id, p) for s, p, _ in g.pressable("kokichi"))["rv_kokichi_motive"]
    assert shown > bare                                    # 手上有宗次的话，硬问更有底气
    g.known["kokichi"] = ["c_soji"]
    assert "rv_kokichi_motive" not in {s.id for s, *_ in g.pressable("kokichi")}   # 摆出来就不用赌了
    g.known["kokichi"] = []
    g.note_press_miss(g.quest.signal("rv_kokichi_motive"), "kokichi")
    assert "alerted:kokichi" in g.flags                    # 问砸了：对方起了戒心
    assert dict((s.id, p) for s, p, _ in g.pressable("kokichi"))["rv_kokichi_motive"] < shown


def test_plot_gates_cannot_be_pressed():
    g, *_ = living({"act_id": "act2", "clues": ["c_case", "c_shops"]})
    assert "decided_mercy" not in {s.id for s, *_ in g.pressable("ayaka")}
    assert "reported_findings" not in {s.id for s, *_ in g.pressable("ayaka")}


# ------------------------------------------------------------ NPC 的打算

def test_alerted_forger_burns_evidence_then_flees():
    g, phys, ex, _ = living({"act_id": "act2", "phase_idx": 0, "clues": ["c_case"], "location": "hanamizaka"})
    g.flags.append("alerted:kokichi")
    g.pass_time(70)
    assert "evidence_burned" in g.flags
    g.location = "print_workshop"
    desc, got = ex.examine("o_scrap_box")
    assert got == [] and "纸灰" in desc                    # 拓纸已经没了，得另找证据
    g.pass_time(200)
    assert "kokichi_fled" in g.flags and phys.where("kokichi") in (None, "ritou")


def test_being_there_catches_him_burning():
    g, _, _, _ = living({"act_id": "act2", "phase_idx": 0, "clues": ["c_case"], "location": "print_workshop"})
    g.flags.append("alerted:kokichi")
    g.pass_time(70)
    assert "caught_burning" in g.flags and "evidence_burned" not in g.flags
    assert "c_seal_rubbing" in g.clues


def test_unchecked_forger_sends_a_second_wave():
    g, *_ = living({"act_id": "act2", "day": 2, "phase_idx": 1, "clues": ["c_case"], "location": "komore_teahouse"})
    g.advance(1)
    assert "second_wave" in g.flags and "c_messenger" in g.clues    # 正好在茶室：撞见了送信人


# ------------------------------------------------------------ 自由行动

def test_report_to_tenryou_needs_evidence_and_ends_the_case(tmp_path):
    g, _, _, act = living({"act_id": "act2", "clues": ["c_case"]})
    assert act.parse("我要去天领奉行告发")[0] == "report_authority"
    text, _ = act.resolve("report_authority", None, "告发")
    assert "无凭无据" in text and "reported_tenryou" not in g.flags
    eng, _, out = engine(tmp_path, {"act_id": "act2", "phase_idx": 0, "clues": ["c_case", "c_soji"]},
                         [("去天领奉行告发小吉", "自由发言"), {"kind": "wait"}, {"kind": "wait"}, "q"])
    assert eng.god.ending == "tenryou"
    assert any("奉行所的封条" in line for line in out)


def test_follow_reveals_what_talking_cannot(monkeypatch):
    g, phys, ex, act = living({"act_id": "act2", "phase_idx": 1, "clues": ["c_case"], "location": "ritou"})
    monkeypatch.setattr(actions_mod, "roll", lambda *a: (True, 80))
    assert act.parse("悄悄跟上去看看宗次去哪") == ("follow", "soji")
    _, got = act.resolve("follow", "soji", "跟踪宗次")
    assert got == ["c_workshop"] and "saw_soji_kokichi" in g.flags


def test_failed_follow_alerts_target(monkeypatch):
    g, phys, ex, act = living({"act_id": "act2", "phase_idx": 1, "clues": ["c_case"], "location": "ritou"})
    monkeypatch.setattr(actions_mod, "roll", lambda *a: (False, 30))
    act.resolve("follow", "soji", "跟踪宗次")
    assert "alerted:soji" in g.flags


def test_warning_the_forger_changes_the_epilogue(tmp_path):
    eng, _, out = engine(tmp_path, {"act_id": "act2", "phase_idx": 0, "location": "print_workshop", "clues": ["c_case"]},
                         [("找个没人的时候通风报信，让他快跑", "自由发言"), {"kind": "wait"}, {"kind": "wait"}, "q"])
    g = eng.god
    assert {"warned_kokichi", "kokichi_fled"} <= set(g.flags)
    assert "绫华始终不知道" in "".join(eng.outcome_lines())
