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
    g = make_god({"location": "ritou", "clues": ["c_letter"]})
    g.seed = 1
    ex = explorer(g)
    for _ in range(3):                                         # 0.5 → 0.75 → 1.0，最多三次一定找到
        if ex.examine("o_manifest")[1] == ["c_paper"]:
            break
    assert "c_paper" in g.clues
    assert any(not r["ok"] for r in g.rolls) or len(g.rolls) == 1


def test_companion_helps_search(monkeypatch):
    g = make_god({"location": "ritou", "clues": ["c_letter"]})
    g.seed = 1
    ex = explorer(g)
    monkeypatch.setattr(type(ex), "party", property(lambda self: ["thoma"]))
    desc, _ = ex.examine("o_manifest")
    assert "托马帮你一起找" in desc
    assert g.rolls[-1]["p"] == 0.8


def test_belief_needs_relationship_and_scales():
    g = make_god({"act_id": "act2", "clues": ["c_case", "c_shops"]})
    say = "伪信是辰巳印坊的学徒小吉做的"
    assert g.belief_chance("ayaka", say, 40, 20) == 0
    lo, hi = g.belief_chance("ayaka", say, 55, 40), g.belief_chance("ayaka", say, 70, 55)
    assert 0 < lo < hi <= 0.9
    g.misses["belief"] = g.day
    assert g.belief_chance("ayaka", say, 70, 55) == 0          # 当天被婉拒过，不再掷
