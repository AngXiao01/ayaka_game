"""金标准局：仓库里存的一局固定对局，每次都重放一遍。

红了 = 世界的走向变了（规则、内容、流程里有什么改了结果）。有意改的，确认无误后重录：
    python3 tools/replay_game.py --make-golden tests/golden/mock_game.jsonl.gz
"""
import gzip
import json
from pathlib import Path

from tools import replay_game

GOLDEN = Path(__file__).parent / "golden" / "mock_game.jsonl.gz"
HOW = "（有意改了行为的话，重录：python3 tools/replay_game.py --make-golden tests/golden/mock_game.jsonl.gz）"


def test_the_golden_game_still_plays_out_the_same():
    out = replay_game.replay(GOLDEN)
    assert out["ok"], f"{out.get('problem')} {out.get('differences', [])[:3]} {HOW}"
    assert out["inputs"] >= 60 and out["llm_calls"] >= 200 and out["ending"] == "mercy"
    assert out["reordered"] == 0                                             # 串行录的局，调用顺序必须一模一样


def test_the_golden_file_is_small_and_complete():
    events = replay_game.load(GOLDEN)
    kinds = {e["event"] for e in events}
    assert kinds == {"state_init", "input", "state_delta", "llm"}
    assert not any("messages" in e for e in events)                          # 提示词不进仓库，只留回答
    assert GOLDEN.stat().st_size < 50_000


def test_recording_itself_is_deterministic(tmp_path):
    """同样的代码录两次必须一字不差——否则金标准局会无缘无故变红。（和金标准比较的是上面第一条。）"""
    a, b = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"
    replay_game.record_mock_game(a)
    replay_game.record_mock_game(b)
    assert gzip.decompress(a.read_bytes()) == gzip.decompress(b.read_bytes())


def test_a_changed_decision_is_noticed(tmp_path):
    events = replay_game.load(GOLDEN)
    for e in events:
        if e["event"] == "input" and e["kind"] == "move":
            e["meta"]["dest"] = "chinju_forest" if e["meta"].get("dest") != "chinju_forest" else "hanamizaka"
            break
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events))
    assert replay_game.replay(bad)["ok"] is False


def _swap_two_different_calls(events, kind="llm", field="task"):
    idx = [i for i, e in enumerate(events) if e["event"] == kind]
    for a, b in zip(idx, idx[1:]):
        if events[a][field] != events[b][field]:
            events[a], events[b] = events[b], events[a]
            return events
    raise AssertionError("没找到相邻且不同的两次调用")


def test_calls_recorded_out_of_order_by_a_parallel_run_still_replay(tmp_path):
    """并行发出的几次调用是按“谁先回来”记的，和重放时的发出顺序可能不同：对得上的就取，并记一次乱序。"""
    events = _swap_two_different_calls(replay_game.load(GOLDEN))
    f = tmp_path / "swapped.jsonl"
    f.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events))
    out = replay_game.replay(f)
    assert out["ok"] and out["reordered"] >= 1


def test_a_call_that_was_never_recorded_is_still_a_mismatch(tmp_path):
    events = replay_game.load(GOLDEN)
    first = next(i for i, e in enumerate(events) if e["event"] == "llm")
    events[first] = {**events[first], "task": "no_such_task"}              # 记录里没有这个任务，引擎发出的那次对不上
    f = tmp_path / "bad.jsonl"
    f.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events))
    out = replay_game.replay(f)
    assert out["ok"] is False and "行为对不上" in out["problem"]
