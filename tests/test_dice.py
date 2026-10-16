"""概率判定：读档不能重掷、越试越容易、同伴帮忙、关系够深才有机会直接信。"""
from core.dice import roll
from tests.helpers import make_god
from tests.test_world import explorer


def test_same_moment_same_result():
    g = make_god({})
    g.seed = 7
    a = [roll(g, "x", 0.5)[0] for _ in range(3)]
    assert len(set(a)) == 1                                   # 同一时刻反复掷，结果不变
    assert len(g.rolls) == 3


def test_failed_search_gets_easier_until_found():
    g = make_god({"location": "ritou", "clues": ["c_messenger"]})
    g.seed = 1
    ex = explorer(g)
    for _ in range(3):                                         # 0.5 → 0.75 → 1.0，最多三次一定找到
        if ex.examine("o_manifest")[1] == ["c_paper"]:
            break
    assert "c_paper" in g.clues
    assert any(not r["ok"] for r in g.rolls) or len(g.rolls) == 1


def test_companion_helps_search(monkeypatch):
    g = make_god({"location": "ritou", "clues": ["c_messenger"]})
    g.seed = 1
    ex = explorer(g)
    monkeypatch.setattr(type(ex), "party", property(lambda self: ["thoma"]))
    desc, _ = ex.examine("o_manifest")
    assert "托马帮忙" in desc
    assert g.rolls[-1]["p"] == 0.8


def test_belief_needs_relationship_and_moves_with_context():
    g = make_god({"act_id": "act2", "clues": ["c_case", "c_shops"]})
    say = "茶碗是辰巳印坊的学徒小吉弄坏后带走的"
    assert g.belief_chance("ayaka", say, 40, 20)[0] == 0
    lo, hi = g.belief_chance("ayaka", say, 55, 40)[0], g.belief_chance("ayaka", say, 70, 55)[0]
    assert 0 < lo < hi <= 0.9
    g.note_belief_miss("ayaka", say, 55, 40)
    again, why = g.belief_chance("ayaka", say, 55, 40)
    assert again < lo and any("又说一遍" in w for w in why)        # 刚被婉拒、什么都没变，再说只会更难
    g.known["ayaka"] = ["c_messenger"]
    more, why = g.belief_chance("ayaka", say, 55, 40)
    assert more > lo and any("新东西" in w for w in why)          # 拿出了新证据，更容易信
    g.known["ayaka"] = []
    g.pass_time(180)
    assert g.belief_chance("ayaka", say, 55, 40)[0] == lo         # 隔了些时候，回到平常
