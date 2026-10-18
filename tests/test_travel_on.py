"""同行中选“接着走”是继续赶路，不是告辞。"""
import json

from core.player_agent import ScriptedPlayer
from engine.game_engine import GameEngine
from llm.mock_backend import MockBackend
from tests.test_physical import settings


def run(tmp_path, script, name="c"):
    f = tmp_path / f"data/saves/{name}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"world": {"act_id": "act2", "phase_idx": 1, "location": "kamisato_estate", "clues": ["c_case", "c_shops"]},
                             "relationships": {"ayaka": {"affinity": 80, "trust": 60}}}))
    out = []
    eng = GameEngine(settings(tmp_path), MockBackend(seed=0), ScriptedPlayer(script), out=out.append, save_name=name)
    res = eng.run()
    return eng, res, "\n".join(out)


def test_walking_on_with_a_companion_is_not_a_farewell(tmp_path):
    eng, res, text = run(tmp_path, [{"kind": "talk", "npc": "ayaka"}, {"kind": "invite"}, {"kind": "leave"}, "q"])
    assert "ayaka" in eng.god.physical.party                                       # 邀请被接受了
    first = res.sessions[0]
    assert first.end_reason.name == "TRAVEL_ON" and "—— 你们接着往前走 ——" in text
    assert "你起身告辞" not in text
    ev = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    mem = next(e for e in ev if e["event"] == "llm" and e["task"] == "summary")
    assert "你们接着往前走" in mem["messages"][1]["content"] and "你起身告辞" not in mem["messages"][1]["content"]   # 摘要看到的结束方式也对


def test_a_real_farewell_is_still_a_farewell(tmp_path):
    _, res, text = run(tmp_path, [{"kind": "talk", "npc": "ayaka"}, {"kind": "leave"}, "q"])
    assert res.sessions[0].end_reason.name == "PLAYER_LEFT" and "—— 你起身告辞 ——" in text
