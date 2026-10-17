"""抽样监控：从 trace 里统计这一局（或一批局）的健康度。

成熟的 NPC 管线会在线上抽样盯几个比率，越线就报警：设定违规率、重说率、兜底率、延迟。这里同样从 trace 事件里数：

  越界率    被护栏 / 核对员打回去重说的 NPC 回复占比（audit 事件 / 回复数）；里面再分“知识边界”和“事实·物理”
  重说率    复读被检测到、重来了一次的占比（repeat 事件）
  兜底率    模型挂了、用预设台词顶上的占比（degraded 回复 + fallback 事件）
  回落率    免费优先时，免费模型失败改走付费的次数（free_fallback）
  修复率    JSON 不合法、让模型重写的次数（llm 事件里 ok=false 的占比）
  延迟/花费  各类调用的中位延迟、总 token 与花费

用法：python3 tools/monitor.py data/traces/<文件>.jsonl [更多文件]   越线时退出码为 1，可以接在 CI 里。
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

LIMITS = {"out_of_bounds": 0.25, "retry": 0.20, "fallback": 0.05, "repair": 0.15}


def load(paths) -> list[dict]:
    events = []
    for p in paths:
        events += [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    return events


def summarize(events: list[dict]) -> dict:
    turns = [e for e in events if e.get("event") == "turn" and e.get("turn", 0) > 0]
    n = max(1, len(turns))
    llm = [e for e in events if e.get("event") == "llm"]
    audits = [e for e in events if e.get("event") == "audit"]
    boundary = [a for a in audits if any("你并不知道" in p for p in a.get("problems", []))]
    degraded = sum(1 for t in turns if t.get("degraded"))
    fallbacks = [e for e in events if e.get("event") == "fallback"]
    lat: dict[str, list[float]] = defaultdict(list)
    tokens = cost = 0
    for e in llm:
        if e.get("latency_s"):
            lat[e["task"]].append(e["latency_s"])
        u = e.get("usage") or {}
        tokens += int(u.get("prompt_tokens") or 0) + int(u.get("completion_tokens") or 0)
        cost += float(u.get("cost") or 0)
    out = {
        "turns": len(turns),
        "out_of_bounds": round(len(audits) / n, 3),
        "boundary_violations": len(boundary),
        "fact_physics_violations": len(audits) - len(boundary),
        "retry": round(len([e for e in events if e.get("event") == "repeat"]) / n, 3),
        "fallback": round((degraded + len(fallbacks)) / n, 3),
        "repair": round(sum(1 for e in llm if e.get("ok") is False) / max(1, len(llm)), 3),
        "free_fallbacks": len([e for e in events if e.get("event") == "free_fallback"]),
        "beats_offered": len([e for e in events if e.get("event") == "beat" and e.get("picked")]),
        "reflections": len([e for e in events if e.get("event") == "reflection"]),
        "tokens": tokens, "cost_usd": round(cost, 5),
        "median_latency_s": {k: round(statistics.median(v), 2) for k, v in lat.items()},
        "calls": dict(Counter(e["task"] for e in llm)),
    }
    out["alerts"] = [f"{k} {out[k]:.0%} 超过 {lim:.0%}" for k, lim in LIMITS.items() if out[k] > lim]
    return out


def line(s: dict) -> str:
    return (f"监控：{s['turns']} 轮｜越界重说 {s['out_of_bounds']:.0%}（知识边界 {s['boundary_violations']}，事实·物理 {s['fact_physics_violations']}）"
            f"｜复读重说 {s['retry']:.0%}｜兜底 {s['fallback']:.0%}｜JSON 修复 {s['repair']:.0%}｜花费 ${s['cost_usd']}"
            + (f"｜⚠ {'；'.join(s['alerts'])}" if s["alerts"] else ""))


def main(argv=None) -> int:
    paths = (argv if argv is not None else sys.argv[1:])
    if not paths:
        print(__doc__)
        return 2
    s = summarize(load(paths))
    print(json.dumps(s, ensure_ascii=False, indent=2))
    print(line(s))
    return 1 if s["alerts"] else 0


if __name__ == "__main__":
    sys.exit(main())
