"""整局重放与时间旅行（事件溯源，见 core/eventlog.py）。

  python3 tools/replay_game.py data/traces/xxx.jsonl                 用记下的输入和模型回答把整局重放一遍，逐步核对状态
  python3 tools/replay_game.py data/traces/xxx.jsonl --state-at 12   不跑引擎：重建第 12 次输入之前的世界状态
  python3 tools/replay_game.py data/traces/xxx.jsonl --diff-at 12    看第 12 次输入之前世界发生了哪些变化
  python3 tools/replay_game.py --make-golden tests/golden/mock_game.jsonl.gz   重录“金标准局”（见下）

金标准局：仓库里存了一局固定的 mock 对局（只留重放需要的事件，gzip 后十几 KB），测试每次都把它重放一遍。
改了规则、内容或流程之后，只要世界的走向变了，重放就对不上、测试变红——这是有意的：它在告诉你“行为变了”。
确认变得对，就用上面那条命令重录一份。

重放一致 = 代码对同样的输入给出同样的世界；不一致会指出是第几次输入、哪些字段不同——
改了规则之后拿几局老 trace 跑一遍，就知道有没有改坏别的地方。
只有这个版本之后录的 trace（含 state_init / input / decide_io）才能整局重放。
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import eventlog  # noqa: E402


def load(path) -> list[dict]:
    path = Path(path)
    text = gzip.decompress(path.read_bytes()).decode("utf-8") if path.suffix == ".gz" else path.read_text(encoding="utf-8")
    return [json.loads(l) for l in text.splitlines() if l.strip()]


REPLAY_EVENTS = {"state_init", "input", "state_delta", "decide_io", "llm"}
LLM_KEEP = ("event", "task", "output", "finish_reason", "error", "usage", "latency_s")


def compact(events: list[dict]) -> list[dict]:
    """只留重放用得上的：状态、输入、判断、模型回答（不要发出去的提示词，那是 trace 里最大的部分）。"""
    out = []
    for e in events:
        if e.get("event") not in REPLAY_EVENTS:
            continue
        if e["event"] == "llm":
            out.append({k: e[k] for k in LLM_KEEP if k in e})
        else:
            out.append({k: v for k, v in e.items() if k not in ("seq", "ts", "run", "span")})
    return out


def write_events(events: list[dict], path) -> None:
    text = "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0) if path.suffix == ".gz" else text.encode("utf-8"))


def record_mock_game(out_path) -> dict:
    """用固定种子的 mock 后端 + 自动玩家录一局，压缩后写到 out_path。返回 {inputs, ending}。"""
    from config.settings import load_settings
    from core.player_agent import AutoPlayer
    from engine.game_engine import GameEngine
    from llm.mock_backend import MockBackend

    tmp = Path(tempfile.mkdtemp(prefix="ayaka_golden_"))
    try:
        settings = load_settings(env_file=tmp / "none.env")
        settings.game.data_dir, settings.game.seed, settings.harness.game_token_budget = tmp / "data", 1, 0
        eng = GameEngine(settings, MockBackend(seed=1), AutoPlayer(seed=1, out=lambda s: None), out=lambda s: None, save_name="golden")
        res = eng.run()
        events = load(eng.tracer.path)
        write_events(compact(events), out_path)
        return {"inputs": sum(e.get("event") == "input" for e in events), "ending": res.ending}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _deltas(events: list[dict]) -> list[tuple[int, list[dict]]]:
    return [(e["before_input"], e["ops"]) for e in events if e.get("event") == "state_delta"]


def _compare(rec: list[dict], rep: list[dict]) -> list[dict]:
    """两段状态变化里不一样的字段：[{path, recorded, replayed}]。"""
    def index(ops):
        return {".".join(map(str, o["p"])): o.get("v", o.get("append", "(删除)")) for o in ops}
    a, b = index(rec), index(rep)
    return [{"path": k, "recorded": a.get(k, "(无)"), "replayed": b.get(k, "(无)")} for k in sorted({*a, *b}) if a.get(k) != b.get(k)]


def replay(trace_path) -> dict:
    """重放一局，返回 {ok, inputs, llm_calls, decisions, ending, problem}。"""
    from config.settings import load_settings
    from engine.game_engine import GameEngine
    from llm.decider import Decider
    from llm.replay import ReplayBackend, ReplayDecider, ReplayMismatch, ReplayPlayer
    from llm.router import ModelRouter

    events = load(trace_path)
    init = next((e for e in events if e.get("event") == "state_init"), None)
    if init is None:
        return {"ok": False, "problem": "这份 trace 没有 state_init，是录制功能之前的老 trace，不能整局重放"}
    snap, rs = init["state"], init["settings"]
    tmp = Path(tempfile.mkdtemp(prefix="ayaka_replay_"))
    try:
        (tmp / "saves").mkdir(parents=True)
        (tmp / "saves" / "replay.json").write_text(json.dumps({"world": snap["world"], "relationships": snap["relationships"]},
                                                              ensure_ascii=False), encoding="utf-8")
        mem = tmp / "memories" / "replay"
        mem.mkdir(parents=True)
        for nid, items in snap.get("memories", {}).items():
            (mem / f"{nid}.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in items), encoding="utf-8")
        (mem / "_world_canon.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in snap.get("world_canon", [])),
                                                encoding="utf-8")
        settings = load_settings(env_file=tmp / "none.env")
        for k, v in rs["game"].items():
            setattr(settings.game, k, v)
        for k, v in rs["harness"].items():
            setattr(settings.harness, k, v)
        settings.harness.resume_after_crash = False              # 重放不需要行动日志
        settings.game.data_dir = tmp
        settings.harness.game_token_budget = 0
        backend, player = ReplayBackend(events), ReplayPlayer(events)
        player.stops_at_ending = rs.get("stops_at_ending", True)
        decider = ReplayDecider(events) if rs.get("decider", "off") != "off" else Decider()      # 录制时没开判断，重放也不能多问
        eng = GameEngine(settings, backend, player, out=lambda s: None, save_name="replay", new_game=False,
                         router=ModelRouter(backend, settings.llm), decider=decider)
        try:
            res = eng.run(max_actions=2000)
        except ReplayMismatch as e:
            return {"ok": False, "problem": f"行为对不上：{e}", "inputs": player.i}
        again = load(eng.tracer.path)
        want, got = _deltas(events), _deltas(again)
        for i in range(max(len(want), len(got))):
            a = want[i] if i < len(want) else None
            b = got[i] if i < len(got) else None
            if a != b:
                return {"ok": False, "problem": f"第 {(a or b)[0]} 次输入之前的状态对不上", "inputs": player.i,
                        "differences": _compare(a[1] if a else [], b[1] if b else [])}
        return {"ok": True, "inputs": player.i, "llm_calls": backend.i, "decisions": getattr(decider, "i", 0), "ending": res.ending,
                "deltas": len(want), "reordered": backend.reordered + getattr(decider, "reordered", 0)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def describe(state: dict) -> str:
    w = state.get("world", {})
    return (f"第{w.get('day')}天 分钟={w.get('minute')} 地点={w.get('location')} 幕={w.get('act_id')}\n"
            f"线索：{w.get('clues')}\n达成的信号：{w.get('achieved')}\n标记：{w.get('flags')}\n结局：{w.get('ending')}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="整局重放与时间旅行")
    ap.add_argument("trace", nargs="?")
    ap.add_argument("--make-golden", metavar="OUT", help="录一局固定的 mock 对局并压缩写到 OUT（.jsonl.gz）")
    ap.add_argument("--state-at", type=int)
    ap.add_argument("--diff-at", type=int)
    args = ap.parse_args(argv)
    if args.make_golden:
        r = record_mock_game(args.make_golden)
        print(f"已写入 {args.make_golden}：{r['inputs']} 次输入，结局 {r['ending']}")
        return 0
    if not args.trace:
        ap.error("要给一个 trace 文件（或用 --make-golden）")
    if args.state_at is not None:
        state, n = eventlog.reconstruct(load(args.trace), args.state_at)
        print(f"（共 {n} 次输入；下面是第 {args.state_at} 次输入之前的世界）\n" + describe(state))
        return 0
    if args.diff_at is not None:
        for before, ops in _deltas(load(args.trace)):
            if before == args.diff_at:
                for op in ops:
                    print(".".join(map(str, op["p"])), "←", json.dumps(op.get("v", op.get("append", "(删除)")), ensure_ascii=False)[:120])
                return 0
        print("这一步之前世界没有变化")
        return 0
    r = replay(args.trace)
    if r["ok"]:
        print(f"✔ 重放一致：{r['inputs']} 次输入，{r['llm_calls']} 次模型回答，{r['decisions']} 次判断，{r['deltas']} 段状态变化；结局 {r['ending']}")
        return 0
    print(f"✘ {r['problem']}")
    for d in r.get("differences", [])[:8]:
        cut = lambda v: json.dumps(v, ensure_ascii=False)[:160]   # noqa: E731
        print(f"   {d['path']}\n      记录：{cut(d['recorded'])}\n      重放：{cut(d['replayed'])}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
