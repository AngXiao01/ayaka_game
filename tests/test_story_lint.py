"""故事内容的静态校验：真内容不能有 error；故意弄坏的内容要能被指出来。"""
import copy

import pytest

from core.story_lint import lint
from tools import lint_story


@pytest.fixture(scope="module")
def content():
    return lint_story.load()


def broken(content, mutate):
    q, w, p = (copy.deepcopy(x) for x in content)
    mutate(q, w, p)
    return lint(q, w, p)


def codes(findings, level=None):
    return {(f.code) for f in findings if level is None or f.level == level}


def test_real_content_has_no_structural_errors(content):
    findings = lint(*content)
    assert not [str(f) for f in findings if f.level == "error"]
    assert codes(findings) <= {"witness_absent"}                      # 剩下的只能是“说不通、要作者决定”的疑点


def test_cli_exit_code(content, capsys):
    assert lint_story.main([]) == 0
    assert "错误" in capsys.readouterr().out


def test_an_npc_who_is_never_anywhere_breaks_the_chain(content):
    out = broken(content, lambda q, w, p: setattr(p["fox"], "schedule", []))
    assert {"unreachable_signal", "unreachable_clue", "unreachable_act", "npc_never_met"} <= codes(out)
    assert any("c_bowl_on_altar" in f.msg for f in out if f.code == "unreachable_clue")


def test_a_location_locked_behind_an_impossible_clue(content):
    out = broken(content, lambda q, w, p: w.locations["print_workshop"].update(locked_until_any_clue=["c_double"]))
    assert "unreachable_location" in codes(out) and any("c_sample" in f.msg for f in out)


def test_prerequisite_cycles_are_reported_with_the_reason(content):
    out = broken(content, lambda q, w, p: setattr(next(s for s in q.signals if s.id == "rv_thoma_gate"), "requires", ["rv_thoma_ping"]))
    msg = next(f.msg for f in out if f.code == "unreachable_signal" and "rv_thoma_gate" in f.msg)
    assert "前置信号" in msg and "rv_thoma_ping" in msg


def test_dangling_references(content):
    out = broken(content, lambda q, w, p: q.secrets[0].update(unlocked_by_clues=["c_nope"]))
    assert any(f.code == "bad_ref" and "c_nope" in f.msg for f in out)
    out = broken(content, lambda q, w, p: q.clues["c_ping"].update(witness={"npc": "thoma", "phase": "半夜", "at": "nowhere"}))
    assert sum(f.code == "bad_ref" for f in out) == 2


def test_a_witness_must_be_where_they_usually_are_unless_explained(content):
    def move(q, w, p):
        q.clues["c_foxfire"]["witness"]["at"] = "hanamizaka"                  # 宗次夜里平时在离岛，不在花见坂
    out = broken(content, move)
    assert any(f.code == "witness_absent" and "c_foxfire" in f.msg for f in out)

    def explain(q, w, p):
        move(q, w, p)
        q.clues["c_foxfire"]["witness"]["off_routine"] = "那天加班送货"
    assert not any(f.code == "witness_absent" and "c_foxfire" in f.msg for f in broken(content, explain))


def test_the_witness_has_to_be_the_one_who_tells_it(content):
    out = broken(content, lambda q, w, p: q.clues["c_gate_visitor"]["witness"].update(npc="aen"))
    assert any(f.code == "witness_not_the_teller" and "c_gate_visitor" in f.msg for f in out)
