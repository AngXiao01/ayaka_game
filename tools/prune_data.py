"""清理 data/：只留网页上的有效对局，其余存档、记忆、trace、事务日志全部删掉。

有效对局 = 存档名是网页的（`web`，或分享模式的 `p_<id>`），并且玩家在对话里至少说了 N 句话（默认 10，数 trace 里
`input` 事件的 at=dialogue、kind=say；一局可以有好几个 trace，累加）。只开了个头、只有几句的都算无效。

  python3 tools/prune_data.py              只列出会删什么（不动文件）
  python3 tools/prune_data.py --apply      真删
  python3 tools/prune_data.py --min 15     换门槛
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACE_RE = re.compile(r"^(?P<save>.+)-\d{8}-\d{6}\.jsonl$")
WEB_RE = re.compile(r"^(web|p_.+)$")


def count_says(trace: Path) -> int:
    """一个 trace 里玩家在对话里说的话数。老 trace 没有 input 事件时，退而数对话轮数。"""
    says = turns = 0
    for line in trace.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue                                    # 崩溃时最后一行可能是半截
        if e.get("event") == "input":
            says += e.get("at") == "dialogue" and e.get("kind") == "say"
        elif e.get("event") == "turn" and e.get("turn", 0) > 0:
            turns += 1
    return says or turns


def scan(data: Path) -> dict[str, dict]:
    """{存档名: {"paths": [它的全部文件 / 目录], "says": 对话里玩家说的话数, "traces": trace 数}}"""
    games: dict[str, dict] = {}

    def game(name: str) -> dict:
        return games.setdefault(name, {"paths": [], "says": 0, "traces": 0})

    for p in (data / "saves").glob("*.json"):
        game(p.stem)["paths"].append(p)
    for sub in ("memories", "uow"):
        for p in (data / sub).glob("*"):
            if p.is_dir():
                game(p.name)["paths"].append(p)
    for p in (data / "traces").glob("*.jsonl"):
        m = TRACE_RE.match(p.name)
        if m:
            g = game(m["save"])
            g["paths"].append(p)
            g["traces"] += 1
            g["says"] += count_says(p)
    return games


def prune(data: Path, min_says: int = 10, apply: bool = False) -> tuple[list[str], list[str]]:
    """返回 (保留的存档名, 删除的存档名)。apply=False 时不动文件。"""
    keep, drop = [], []
    for name, g in sorted(scan(data).items()):
        (keep if WEB_RE.match(name) and g["says"] >= min_says else drop).append(name)
        if name in drop and apply:
            for p in g["paths"]:
                shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)
    return keep, drop


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="只保留网页上的有效对局")
    ap.add_argument("--data", default=str(ROOT / "data"), help="data 目录")
    ap.add_argument("--min", type=int, default=10, help="玩家在对话里至少说几句才算有效（默认 10）")
    ap.add_argument("--apply", action="store_true", help="真删；不加只列出")
    args = ap.parse_args(argv)
    data = Path(args.data)
    games = scan(data)
    keep, drop = prune(data, args.min, apply=args.apply)
    for name in keep:
        print(f"保留  {name:<24} 玩家说了 {games[name]['says']} 句，{games[name]['traces']} 个 trace")
    for name in drop:
        why = "不是网页对局" if not WEB_RE.match(name) else f"只说了 {games[name]['says']} 句" + ("（没有 trace）" if not games[name]["traces"] else "")
        print(f"{'删除' if args.apply else '将删'}  {name:<24} {why}，{len(games[name]['paths'])} 项")
    print(f"\n保留 {len(keep)} 局，{'已删除' if args.apply else '将删除'} {len(drop)} 局" + ("" if args.apply else "（加 --apply 才会真删）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
