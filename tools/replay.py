"""回放 trace 里的某一次 LLM 调用：定位到具体某一圈，原样重发，对比新旧输出。

  python3 tools/replay.py data/traces/xxx.jsonl --list                 列出所有 LLM 调用
  python3 tools/replay.py data/traces/xxx.jsonl --list --problems      只列失败/修复过的
  python3 tools/replay.py data/traces/xxx.jsonl --seq 42               用当前配置的后端重放 seq=42
  python3 tools/replay.py data/traces/xxx.jsonl --seq 42 --temperature 0 --n 3   贪心解码重放 3 次看稳定性
  python3 tools/replay.py data/traces/xxx.jsonl --seq 42 --show-prompt 打印完整 prompt

改了 prompt 想验证效果时，用 --edit-system 文件 替换 system 消息再重放。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import load_settings  # noqa: E402
from llm.base import LLMError, LLMRequest  # noqa: E402
from llm.json_utils import JSONExtractError, extract_json  # noqa: E402
from llm.mock_backend import MockBackend  # noqa: E402


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def span_str(e: dict) -> str:
    sp = e.get("span") or {}
    return "/".join(f"{k[0]}{v}" if k in ("session", "turn") else str(v) for k, v in sp.items()) or "-"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="回放 trace 中的 LLM 调用")
    ap.add_argument("trace")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--problems", action="store_true", help="--list 时只列失败或修复重试过的调用")
    ap.add_argument("--seq", type=int)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--show-prompt", action="store_true")
    ap.add_argument("--edit-system", help="用这个文件的内容替换 system 消息")
    args = ap.parse_args(argv)

    llm_events = [e for e in load(Path(args.trace)) if e.get("event") == "llm"]
    if args.list or args.seq is None:
        for e in llm_events:
            if args.problems and e["ok"] and e["attempt"] == 0:
                continue
            u = e.get("usage") or {}
            print(f"seq={e['seq']:<5} {span_str(e):<22} {e['task']:<15} attempt={e['attempt']} "
                  f"{'ok ' if e['ok'] else 'ERR'} {e.get('latency_s') or 0:>5.1f}s "
                  f"{u.get('prompt_tokens', '-')}→{u.get('completion_tokens', '-')} "
                  f"{(e.get('error') or '')[:60]}")
        return 0

    ev = next((e for e in llm_events if e["seq"] == args.seq), None)
    if ev is None:
        print(f"没有 seq={args.seq} 的 LLM 调用（用 --list 查看）", file=sys.stderr)
        return 1
    messages = ev["messages"]
    if args.edit_system:
        new_sys = Path(args.edit_system).read_text(encoding="utf-8")
        messages = [{"role": "system", "content": new_sys}] + [m for m in messages if m["role"] != "system"]
    req = LLMRequest(task=ev["task"], messages=messages,
                     temperature=ev["temperature"] if args.temperature is None else args.temperature,
                     max_tokens=ev["max_tokens"], json_schema=ev.get("json_schema"),
                     schema_name=ev.get("schema_name") or "output")
    if args.mock:
        backend = MockBackend()
    else:
        from main import build_llm
        backend = build_llm(load_settings().llm)
    print(f"# seq={ev['seq']} {span_str(ev)} task={ev['task']} T={req.temperature}")
    if args.show_prompt:
        for m in messages:
            print(f"\n--- {m['role']} ---\n{m['content']}")
    print(f"\n--- 原输出（{'ok' if ev['ok'] else 'ERR: ' + str(ev.get('error'))}）---\n{ev.get('output')}")
    for i in range(args.n):
        try:
            r = backend.complete(req)
        except LLMError as e:
            print(f"\n--- 重放 #{i + 1} 失败 ---\n{e}")
            return 1
        try:
            extract_json(r.text)
            verdict = "JSON 合法"
        except JSONExtractError as e:
            verdict = f"JSON 不合法：{e}"
        print(f"\n--- 重放 #{i + 1}（{r.latency_s:.1f}s，{verdict}）---\n{r.text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
