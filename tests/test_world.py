"""开放世界：信号解锁条件、出示线索、幕推进、互斥抉择、时间与截止、结局、事件、菜单。"""
from core.explorer import Explorer
from core.player_option_generator import PlayerOptionGenerator
from core.context import TurnContext
from models.personality import Personality
from models.relationship import Relationship
from tests.helpers import QUEST, ROOT, WORLD, make_god

PERSONAS = Personality.load_all(ROOT / "tests/fixtures/yuebai/npcs")


def ids(g):
    return [d["id"] for d in g.pending_signals()]


def test_yohei_needs_to_see_the_commission():
    g = make_god(trust={"yohei": 20})
    g.begin_conversation("yohei")
    assert ids(g) == [] and g.locked_signals()[0][1]          # 不知道失窃的事：锁住
    g.give_clue("c_case")
    assert "rv_yohei_hair" not in ids(g)                      # 没看到凭据，不知道对方的来路
    assert g.useful_clues_for("yohei") == ["c_case"]
    g.mark_known("yohei", ["c_case"])                           # 出示绫华的委托
    assert ids(g) == ["rv_yohei_hair"]


def test_showing_clue_unlocks_thoma():
    g = make_god({"act_id": "act2"})
    g.clues = ["c_case", "c_bundle"]
    g.begin_conversation("thoma")
    assert ids(g) == ["rv_thoma_gate"]                       # 托马是自己人：问到就说，不必先拿凭据来换
    g.mark_known("thoma", ["c_bundle"])                      # 看了布包，也不改变他会说这件事
    assert ids(g) == ["rv_thoma_gate"]
    newly = g.apply_signals(["rv_thoma_gate"])
    assert newly[0]["clues"] == ["夜里折回来的小吉"] and "c_gate_visitor" in g.known["thoma"]


def test_act_progression_and_exclusive_decision():
    g = make_god()
    g.begin_conversation("ayaka")
    g.conv["turns"] = 2
    g.apply_signals(["asked_purpose", "accepted_commission"])
    assert g.act_id == "act2" and {"c_case", "c_shops"} <= set(g.clues) and g.narrations
    g.clues += ["c_bundle", "c_tanuki_hair", "c_round_prints"]
    g.clues += ["c_gate_visitor", "c_sample"]
    assert "reported_findings" not in ids(g)                    # 还没找到碗，不能结案
    g.clues += ["c_bowl_on_altar"]
    assert "reported_findings" in ids(g)
    g.apply_signals(["reported_findings"])
    assert g.act_id == "act3"
    g.conv["turns"] = 3
    assert ids(g) == [] and len(g.proposals_for("ayaka")) == 2   # 关键抉择要玩家先明确提出
    g.propose("proposed_expose")
    assert ids(g) == ["decided_expose"]
    g.propose("proposed_mercy")                                  # 改主意：旧提议作废
    assert ids(g) == ["decided_mercy"] and "proposed_expose" not in g.flags
    g.apply_signals(["decided_expose", "decided_mercy"])
    assert g.resolved and "mercy" in g.flags and "expose" not in g.flags
    assert g.compute_ending(50).id == "mercy" and g.proposals_for("ayaka") == []


def test_private_event_only_has_private_signals():
    g = make_god()
    g.act_id = "act2"
    g.begin_conversation("ayaka", "ev_sword")
    assert ids(g) == ["ev_sword_shared"]


def test_time_deadline_and_endings():
    g = make_god()
    assert g.slots_left() == 11
    for _ in range(20):
        g.conversation_turn()
    assert g.phase == "清晨" and g.clock_label() == "07:40"         # 说一轮话只花两分钟
    g.advance(1)
    assert g.phase == "午后" and g.clock_label() == "11:00"
    g.advance(10)
    assert g.deadline_reached() and g.compute_ending(30).id == "too_late"
    g2 = make_god()
    g2.resolved, g2.flags, g2.clues = True, ["mercy", "ev_heart"], ["c_warm"]
    assert g2.compute_ending(70).id == "bond" and g2.compute_ending(50).id == "mercy"


def test_events_gated_by_place_time_affinity():
    g = make_god({"act_id": "act2", "phase_idx": 3})
    assert [e.id for e in g.available_events("kamisato_estate", lambda n: 45)] == ["ev_sword"]
    assert g.available_events("kamisato_estate", lambda n: 30) == []
    assert g.available_events("hanamizaka", lambda n: 99) == []


def test_schedule():
    a = PERSONAS["ayaka"]
    assert a.location_at("黄昏", "act2") == "hanamizaka"
    assert a.location_at("黄昏", "act1") == "kamisato_estate"
    assert a.location_at("夜晚", "act1") is None


def explorer(g, aff=30):
    rels = {n: Relationship(aff, 20) for n in PERSONAS}
    return Explorer(g, WORLD, PERSONAS, lambda n: rels[n], out=lambda s: None)


def test_menu_and_actions():
    g = make_god()
    ex = explorer(g)
    menu = ex.menu()
    talk = [o for o in menu if o.kind == "talk"]
    assert {o.meta["npc"] for o in talk} == {"ayaka", "thoma"}
    assert next(o for o in talk if o.meta["npc"] == "ayaka").meta["pending"] > 0
    assert "print_workshop" not in [o.meta.get("dest") for o in menu]
    ex.move("hanamizaka")
    assert g.location == "hanamizaka" and g.clock_label() == "07:45"             # 走过去 45 分钟
    assert "print_workshop" not in [o.meta.get("dest") for o in ex.menu()]     # 没线索，找不到印坊
    g.clues.append("c_gate_visitor")
    assert "print_workshop" in [o.meta.get("dest") for o in ex.menu()]
    ex.move("print_workshop")
    assert g.clock_label() == "07:55"                                            # 同一片区，走几步就到
    g.misses["o_spoiled_proofs"] = 2                                             # 判定必成，这里只测规则
    _, gained = ex.examine("o_spoiled_proofs")
    assert gained == ["c_sample"] and g.journal


def test_shrine_needs_the_prints():
    g = make_god({"location": "chinju_forest"})
    ex = explorer(g)
    assert ex.examine("o_shrine")[1] == []                   # 不知道脚印通向哪儿，不知道该翻哪个树洞
    g.clues.append("c_tanuki_hair")
    assert ex.examine("o_shrine")[1] == []                   # 货架上的棕毛帮不上忙
    g.clues.append("c_round_prints")
    g.misses["o_shrine"] = 2                                  # 判定必成，这里只测规则
    assert ex.examine("o_shrine")[1] == ["c_den"]


def test_option_generator_drops_repeats():
    class Stub:
        def generate(self, ctx, npc, n):
            return [{"text": "那晚您看见的人长什么样？"},
                    {"text": "这茶真香。"},
                    {"text": "今天天气不错呢。"}]
    ctx = TurnContext(player_name="旅行者", scene={}, world={}, relationship={},
                      history=[{"speaker": "player", "text": "那晚您看见的人长什么样"}])
    opts, _ = PlayerOptionGenerator(Stub()).generate(ctx, "阿缘", [])
    assert [o.text for o in opts] == ["这茶真香。", "今天天气不错呢。"]


def test_every_place_has_a_layout_sheet():
    """NPC 说到门窗、位置、出入口时以格局为准：每个地点都要写，且都带「数字只以这里为准」的提醒。"""
    for lid, loc in WORLD.locations.items():
        assert loc.get("layout"), f"{lid} 没写格局"
        assert any("数字" in x for x in loc["layout"]), f"{lid} 的格局缺少数字提醒"
