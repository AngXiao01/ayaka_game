"""事件溯源：状态差异、重建、整局重放。"""
import json

from core import eventlog
from core.player_agent import AutoPlayer
from engine.game_engine import GameEngine
from llm.mock_backend import MockBackend
from tests.test_game_flow import make_settings
from tools import replay_game


def test_diff_and_apply_roundtrip():
    old = {"a": 1, "l": [1, 2], "d": {"x": 1, "y": [1]}, "gone": True}
    new = {"a": 2, "l": [1, 2, 3], "d": {"x": 1, "y": [1, 2], "z": 9}, "n": "new"}
    ops = eventlog.diff(old, new)
    assert {"p": ["l"], "append": [3]} in ops and {"p": ["gone"], "del": True} in ops       # 只追加的列表只记追加
    assert eventlog.apply(old, ops) == new
    assert eventlog.diff(new, new) == [] and old == {"a": 1, "l": [1, 2], "d": {"x": 1, "y": [1]}, "gone": True}


def test_lists_diff_by_index_when_mostly_unchanged():
    old = {"m": [{"t": "a", "n": 1}, {"t": "b", "n": 1}, {"t": "c", "n": 1}, {"t": "d", "n": 1}]}
    new = {"m": [{"t": "a", "n": 1}, {"t": "b", "n": 2}, {"t": "c", "n": 1}, {"t": "d", "n": 1}, {"t": "e", "n": 1}]}
    ops = eventlog.diff(old, new)
    assert {"p": ["m", 1, "n"], "v": 2} in ops and len(ops) == 2                      # 改了一项 + 追加一项，不是整表重写
    assert eventlog.apply(old, ops) == new
    shrunk = {"m": new["m"][:2]}
    assert eventlog.apply(new, eventlog.diff(new, shrunk)) == shrunk                  # 变短走整体替换


def _game(tmp_path):
    eng = GameEngine(make_settings(tmp_path), MockBackend(seed=1), AutoPlayer(seed=1, out=lambda s: None),
                     out=lambda s: None, save_name="t")
    res = eng.run()
    return eng, res


def test_trace_rebuilds_the_world_at_any_step(tmp_path):
    eng, res = _game(tmp_path)
    ev = replay_game.load(eng.tracer.path)
    assert [e["event"] for e in ev if e["event"] in ("state_init", "input")][0] == "state_init"
    final, n = eventlog.reconstruct(ev)
    saved = json.loads((tmp_path / "data/saves/t.json").read_text())
    assert final["world"] == json.loads(json.dumps(saved["world"], default=str))             # 重建到最后一步 = 存档里的世界
    early, _ = eventlog.reconstruct(ev, 3)
    assert early["world"]["minute"] <= final["world"]["minute"] and len(early["world"]["clues"]) <= len(final["world"]["clues"])
    assert n > 10


def test_a_whole_game_replays_identically(tmp_path):
    eng, res = _game(tmp_path)
    out = replay_game.replay(eng.tracer.path)
    assert out["ok"], out
    assert out["inputs"] > 10 and out["llm_calls"] > 20 and out["ending"] == res.ending


def test_replay_notices_when_behaviour_changes(tmp_path):
    eng, _ = _game(tmp_path)
    lines = eng.tracer.path.read_text().splitlines()
    tampered = [l for l in lines]
    for i, l in enumerate(tampered):
        e = json.loads(l)
        if e.get("event") == "input" and e["kind"] == "move":                                # 改掉一次移动：世界就走偏了
            e["meta"]["dest"] = "chinju_forest" if e["meta"].get("dest") != "chinju_forest" else "hanamizaka"
            tampered[i] = json.dumps(e, ensure_ascii=False)
            break
    bad = tmp_path / "tampered.jsonl"
    bad.write_text("\n".join(tampered) + "\n")
    assert replay_game.replay(bad)["ok"] is False
