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
    return eng, [p for p in prompts if "漏出一点口风" in p], prompts


def test_first_failed_press_lets_a_hint_slip_into_the_prompt(tmp_path, monkeypatch):
    eng, slipped, prompts = run_presses(tmp_path, monkeypatch, 1)
    assert len(slipped) >= 1 and all(SLIP in p for p in slipped)
    assert "这次你没有松口" in slipped[0] and "不要展开细节" in slipped[0] and "必须漏出一点口风" in slipped[0]
    assert "推说不知道" not in slipped[0]                                          # 漏口风时不再给“推说不知道”这个自相矛盾的选项
    assert "rv_soji_fire" not in eng.god.achieved                                  # 漏口风不等于说出来


def test_the_second_failed_press_does_not_slip_again(tmp_path, monkeypatch):
    _, slipped, prompts = run_presses(tmp_path, monkeypatch, 2)
    after_second = [p for p in prompts if "对方在逼问" in p]
    assert len(after_second) >= 2 and sum("漏出一点口风" in p for p in after_second) < len(after_second)
    assert "漏出一点口风" not in after_second[-1]


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


def test_when_a_hint_slips_the_canned_lie_is_not_offered(tmp_path, monkeypatch):
    from tools import preview_prompt as pp
    _, _, prompts = run_presses(tmp_path, monkeypatch, 1)
    pressed = [p for p in prompts if "对方在逼问" in p]
    assert pressed and all("你可能会说的谎" not in p for p in pressed)               # 又说“早睡了”又漏“有件事没敢提”，自相矛盾
    f = tmp_path / "data/saves/p.json"
    f.write_text(json.dumps({"world": AT_RITOU}))
    plain = pp.capture("p", "soji", say="那晚海边有没有什么怪事？", intent="打听情报", settings=settings(tmp_path))
    assert "你可能会说的谎" in plain.requests[0].messages[-1]["content"]              # 普通提问时他还是有那条谎可说


def test_a_reply_without_the_slip_is_sent_back_with_the_reason():
    from core.guardrails import slip_missing
    slip = "那晚海边……有件事我一直没敢提，说出来怕被你们笑话。"
    for bad in ("……不、不寻常？我就是个扛货的，天黑就睡了，哪知道那些。你问这个做什么？", "……没、没看见。那晚我下工就睡了，什么都没瞧见。"):
        assert slip_missing(bad, slip) and "漏出一点口风" in slip_missing(bad, slip)[0]
    for good in ("……我、我一直没敢说，那晚海边的事。", "……这、这儿人多眼杂。其实……那晚海边有点怪事，我没敢说，怕被笑话。",
                 "海边确实有件事，我一直没敢提。"):
        assert slip_missing(good, slip) == []
    assert slip_missing("随便什么", "") == []                                       # 没有口风要求就不查


class Stubborn(MockBackend):
    """每次说话都推说早睡了、不带口风：看代码会不会把它拦下来。"""
    def __init__(self):
        super().__init__(seed=0)

    def _reply(self, meta):
        r = super()._reply(meta)
        r["speech"] = "……我就是个扛货的，那晚早睡了。"
        return r


def test_a_reply_that_leaves_the_slip_out_is_stopped_and_then_completed_by_code(tmp_path, monkeypatch):
    monkeypatch.setattr("core.dialogue_session.roll", lambda g, key, p, attempt=0: (False, 15))
    f = tmp_path / "data/saves/c.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"world": AT_RITOU}))
    eng = GameEngine(settings(tmp_path), Stubborn(), ScriptedPlayer([{"kind": "talk", "npc": "soji"}, {"kind": "press", "signal": "rv_soji_fire"},
                                                                      {"kind": "leave"}, "q"]), out=lambda s: None, save_name="c")
    eng.run()
    events = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    audits = [e for e in events if e["event"] == "audit" and "漏出一点口风" in " ".join(e.get("problems", []))]
    assert audits and "早睡" in audits[0]["speech"]                                  # 第一版没带口风，被代码拦下并记了理由
    assert any(e["event"] == "slip_forced" for e in events)                         # 模型重说还是不肯，代码兜底
    press_turn = next(e for e in events if e["event"] == "turn" and e["turn"] > 0)          # 硬问那一回合（turn 0 是开场）
    assert "早睡" in press_turn["reply"]["speech"] and press_turn["reply"]["speech"].endswith(SLIP)
    assert "rv_soji_fire" not in eng.god.achieved                                    # 漏口风不等于说出来
