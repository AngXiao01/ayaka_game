"""硬问失败时的“漏口风”：只给方向，只漏一次，不算达成，也不能自己把事说完。"""
import copy
import json

from core.player_agent import ScriptedPlayer
from core.story_lint import lint
from engine.game_engine import GameEngine
from llm.mock_backend import MockBackend
from tests.helpers import QUEST
from tests.test_physical import settings
from tools import lint_story

SLIP = "那晚海边……有件事我一直没敢提，说出来怕被你们笑话。"
AT_RITOU = {"act_id": "act2", "phase_idx": 1, "location": "ritou", "clues": ["c_case", "c_shops"]}


def run_presses(tmp_path, monkeypatch, n, ok=False):
    monkeypatch.setattr("core.dialogue_session.roll", lambda g, key, p, attempt=0: (ok, 15))
    f = tmp_path / "data/saves/c.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"world": AT_RITOU}))
    script = [{"kind": "talk", "npc": "soji"}] + [{"kind": "press", "signal": "rv_soji_fire"}] * n + [{"kind": "leave"}, "q"]
    eng = GameEngine(settings(tmp_path), MockBackend(seed=0), ScriptedPlayer(script), out=lambda s: None, save_name="c")
    eng.run()
    events = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    prompts = [e["messages"][-1]["content"] for e in events if e["event"] == "llm" and e["task"] == "npc_step"]
    return eng, [p for p in prompts if "漏了一点口风" in p], prompts


def test_first_failed_press_lets_a_hint_slip_into_the_prompt(tmp_path, monkeypatch):
    eng, slipped, prompts = run_presses(tmp_path, monkeypatch, 1)
    assert len(slipped) >= 1 and all(SLIP in p for p in slipped)
    assert "这次你没有松口" in slipped[0] and "不要展开细节" in slipped[0]
    assert "rv_soji_fire" not in eng.god.achieved                                  # 漏口风不等于说出来


def test_the_second_failed_press_does_not_slip_again(tmp_path, monkeypatch):
    _, slipped, prompts = run_presses(tmp_path, monkeypatch, 2)
    after_second = [p for p in prompts if "对方在逼问" in p]
    assert len(after_second) >= 2 and sum("漏了一点口风" in p for p in after_second) < len(after_second)
    assert "漏了一点口风" not in after_second[-1]


def test_a_successful_press_has_no_slip(tmp_path, monkeypatch):
    _, slipped, _ = run_presses(tmp_path, monkeypatch, 1, ok=True)
    assert not slipped


def test_real_slips_do_not_leak_and_the_lint_checks_it():
    assert {s.id for s in QUEST.signals if s.partial} == {"rv_yohei_hair", "rv_soji_fire", "rv_heizou_sill", "rv_fox_truth"}
    q, w, p = (copy.deepcopy(x) for x in lint_story.load())
    soji = next(s for s in q.signals if s.id == "rv_soji_fire")
    soji.partial = "那晚海边有一团青白色的灯火，还有个人影蹲着拾东西。"          # 把内容说完了
    msgs = [f.msg for f in lint(q, w, p) if f.code == "partial_leaks"]
    assert any("内容词" in m for m in msgs) and any("evidence" in m for m in msgs)
    soji.partial, soji.ask = "随便说点什么", None
    assert any(f.code == "partial_unused" for f in lint(q, w, p))
