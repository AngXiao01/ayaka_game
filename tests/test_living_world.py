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
    assert "rv_kokichi_motive" in sig                      # 没有碎瓷片也能硬问
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
    assert got == [] and "扫得干干净净" in desc            # 碎瓷片已经被扫走了，得另找证据
    g.pass_time(200)
    assert "kokichi_fled" in g.flags and phys.where("kokichi") in (None, "ritou")


def test_being_there_catches_him_burning():
    g, _, _, _ = living({"act_id": "act2", "phase_idx": 0, "clues": ["c_case"], "location": "print_workshop"})
    g.flags.append("alerted:kokichi")
    g.pass_time(70)
    assert "caught_burning" in g.flags and "evidence_burned" not in g.flags
    assert "c_seal_rubbing" in g.clues


def test_unchecked_forger_sends_a_second_wave():
    g, *_ = living({"act_id": "act2", "day": 2, "phase_idx": 1, "clues": ["c_case"], "location": "kamisato_estate"})
    g.advance(1)
    assert "second_wave" in g.flags and "c_date" in g.clues          # 正好在屋敷：撞见狸猫叼走点心盒
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
    assert got == ["c_soji"] and "saw_soji_ship" in g.flags       # 跟着宗次到了补碗匠的船边


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


def _heizou_state():
    return {"act_id": "act2", "phase_idx": 1, "location": "hanamizaka", "clues": ["c_case", "c_letter", "c_window"],
            "achieved": ["asked_purpose", "accepted_commission", "rv_heizou_window"]}


def test_concession_said_out_loud_counts_and_removes_the_dice(tmp_path):
    # trace：平藏嘴上已经说「这一局算你赢」，却没算达成，玩家还被摆出 17% 的硬问按钮
    from llm.decider import ScriptedDecider
    from tests.test_decider import engine as eng_with
    d = ScriptedDecider({"conceded": {"rv_heizou_concede": (True, 0.9)}})
    eng, _, _ = eng_with(tmp_path, _heizou_state(),
                         [{"kind": "talk", "npc": "heizou"}, ("野猫可解不开锁，也搬不动茶碗，说明不止一个怪盗。", "自由发言"), "q"], d)
    assert "rv_heizou_concede" in eng.god.achieved and "heizou_conceded" in eng.god.flags


def test_good_argument_raises_press_odds():
    g, *_ = living(_heizou_state())
    g.trust_lookup = lambda n: 22
    s = g.quest.signal("rv_heizou_concede")
    base, _ = g.press_chance(s, "heizou")
    strong, why = g.press_chance(s, "heizou", 2)
    assert strong >= base + 0.39 and any("理由很充分" in w for w in why)


def test_crime_scene_is_open_after_the_commission():
    # 请人来查案，就得让人看现场：接下委托后能进茶室，查到瓷渣和窗台的痕迹，拿给托马看能让他想起那天谁等过
    g, phys, ex, _ = living({"act_id": "act1", "location": "kamisato_estate"})
    assert not WORLD.unlocked("tearoom", g.clues)                     # 还没接委托，不便进主人家的茶室
    g.clues += ["c_case", "c_shops"]
    assert "tearoom" in [o.meta["dest"] for o in ex.menu() if o.kind == "move"]
    ex.move("tearoom")
    g.misses.update({"o_tatami": 2, "o_tearoom_window": 2})          # 判定必成，这里只测规则
    assert ex.examine("o_tatami")[1] == ["c_chip"] and ex.examine("o_tearoom_window")[1] == ["c_window"]
    g.act_id = "act2"
    g.known["thoma"] = ["c_chip"]
    assert g.signal_status(g.quest.signal("rv_thoma_workshop"), "thoma", None, 99)[0] == "available"
