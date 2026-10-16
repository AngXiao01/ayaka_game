"""把 trace JSONL 渲染成本地 HTML 并在浏览器打开。

  python3 tools/view_trace.py                 最新的 trace
  python3 tools/view_trace.py data/traces/xxx.jsonl
  python3 tools/view_trace.py --no-open       只生成，不打开

HTML 写在 jsonl 旁边（同名 .html），数据内嵌，离线可看；页面里也能拖入别的 jsonl。
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).resolve().parent / "trace_viewer.html"
PLACEHOLDER = "/*__TRACE_DATA__*/null"


def load_events(path: Path) -> list[dict]:
    events = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({"event": "bad_line", "line": i, "raw": line[:500]})
    return events


def render(path: Path) -> Path:
    data = {"file": path.name, "events": load_events(path)}
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, blob)
    out = path.with_suffix(".html")
    out.write_text(html, encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="trace JSONL → HTML")
    ap.add_argument("trace", nargs="?", help="默认取 data/traces 下最新的 .jsonl")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args(argv)
    if args.trace:
        path = Path(args.trace)
    else:
        traces = sorted((ROOT / "data" / "traces").glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not traces:
            print("data/traces 下没有 trace", file=sys.stderr)
            return 1
        path = traces[-1]
    out = render(path)
    print(out)
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
