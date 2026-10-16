"""列出 OpenRouter 上这个游戏能用的最便宜模型，可选实测并把结果写进 .env。

  python3 tools/pick_model.py                      # 排名前 15（/models 不需要 key）
  python3 tools/pick_model.py --probe 5            # 对前 5 个各跑 3 次真实的结构化 NPC 回复（要 key，合计不到 1 美分）
  python3 tools/pick_model.py --probe 5 --write    # 把 3/3 全过里最便宜的写进 .env 的 OPENROUTER_MODEL（固定下来，评测可复现）
  python3 tools/pick_model.py --probe 5 --max-latency 8 --write   # 再加一条：中位延迟不超过 8 秒
  python3 tools/pick_model.py --free               # 连 :free 一起排（每天 50 次请求，一局要 100~130 次）
  python3 tools/pick_model.py --vendors '*'        # 不限厂商（会排进抽取 / 摘要之类的专用模型）
  python3 tools/pick_model.py --trace data/traces/xxx.jsonl   # 用某局实跑的 token 配比估成本

OPENROUTER_MODEL=cheapest（默认）时游戏启动也按同样的规则自动挑第一名，只是不实测。
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import load_settings  # noqa: E402
from llm import model_picker  # noqa: E402
from llm.base import LLMError  # noqa: E402
from llm.diagnostics import structured_probe  # noqa: E402
from llm.gemma_client import GemmaClient  # noqa: E402


def mix_from_trace(path: Path) -> tuple[int, int]:
    p = c = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e.get("event") == "llm" and e.get("usage"):
            p += int(e["usage"].get("prompt_tokens") or 0)
            c += int(e["usage"].get("completion_tokens") or 0)
    if not p:
        raise SystemExit(f"{path} 里没有带 usage 的 LLM 调用")
    return p, c


def write_env(model_id: str, env_path: Path) -> None:
    """只改 OPENROUTER_MODEL 这一行（没有就追加），其他行原样保留。"""
    lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True) if env_path.is_file() else []
    new = f"OPENROUTER_MODEL={model_id}                # tools/pick_model.py 选的\n"
    for i, l in enumerate(lines):
        if re.match(r"\s*OPENROUTER_MODEL\s*=", l):
            lines[i] = new
            break
    else:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(new)
    with open(env_path, "w", encoding="utf-8") as f:       # 原地覆盖写，不删文件
        f.writelines(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="挑 OpenRouter 上能用的最便宜模型")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--free", action="store_true", help="把 :free 模型也排进来")
    ap.add_argument("--vendors", default="", help="逗号分隔的厂商前缀；* = 不限；默认用 model_picker 的通用对话厂商名单")
    ap.add_argument("--trace", type=Path, help="用这局 trace 的 prompt/completion 配比估成本")
    ap.add_argument("--probe", type=int, default=0, help="对排名前 N 的模型实测结构化 NPC 回复（需要 key）")
    ap.add_argument("--repeat", type=int, default=3, help="每个模型实测几次；全过才算通过（便宜模型常常时好时坏）")
    ap.add_argument("--max-latency", type=float, default=0, help="中位延迟上限（秒），0 = 不限")
    ap.add_argument("--write", action="store_true", help="把第一个通过的（不实测时是第一名）写进 .env")
    ap.add_argument("--env", type=Path, default=ROOT / ".env")
    args = ap.parse_args(argv)

    cfg = load_settings(env_file=args.env).llm
    base = cfg.base_url if cfg.backend == "openrouter" else "https://openrouter.ai/api/v1"
    lister = GemmaClient(base, "", timeout=cfg.timeout, max_retries=0, name="openrouter", headers=cfg.headers)
    try:
        models = lister.list_models()
    except LLMError as e:
        print(f"拿不到模型列表：{e}", file=sys.stderr)
        return 1
    vendors = tuple(v.strip() for v in args.vendors.split(",") if v.strip()) or cfg.cheapest_vendors \
        or model_picker.GENERAL_VENDORS
    mix = mix_from_trace(args.trace) if args.trace else model_picker.GAME_MIX
    ranked = model_picker.rank(models, allow_free=args.free, vendors=vendors, mix=mix)
    reasons = Counter(model_picker.why_not(m, args.free, vendors) or "合格" for m in models)
    print(f"共 {len(models)} 个模型：" + "，".join(f"{k} {v}" for k, v in reasons.most_common()))
    print(f"成本按一局 {mix[0]:,} prompt + {mix[1]:,} completion tokens 估（不算缓存折扣）\n")
    print("     " + model_picker.HEADER)
    for i, c in enumerate(ranked[:args.top], 1):
        print(f"{i:>3}. {c.row(mix)}")
    if not ranked:
        return 1

    chosen = ranked[0]
    if args.probe:
        if not cfg.api_key or cfg.backend != "openrouter":
            print("\n--probe 需要 LLM_BACKEND=openrouter 和 OpenRouter key", file=sys.stderr)
            return 1
        print(f"\n实测前 {args.probe} 个，每个 {args.repeat} 次（关推理，参数同游戏里的 NPC：T={cfg.npc.temperature}，"
              f"max_tokens={cfg.npc.max_tokens}）：")
        summary = []
        for c in ranked[:args.probe]:
            cl = GemmaClient(cfg.base_url, c.id, api_key=cfg.api_key, timeout=cfg.timeout, max_retries=1,
                             structured=cfg.structured, name="openrouter", headers=cfg.headers,
                             key_hint=cfg.key_source, reasoning_effort=cfg.reasoning_effort)
            rs = []
            for _ in range(args.repeat):
                r = structured_probe(cl, max_tokens=cfg.npc.max_tokens, temperature=cfg.npc.temperature)
                rs.append(r)
                print(f"  {'✓' if r.ok else '✗'} {c.id}：{r.detail}")
                if "401" in r.detail:
                    print("  key 无效，停止实测（先跑 python3 main.py --check 看 key 的诊断）", file=sys.stderr)
                    return 1
            lat = statistics.median(r.latency for r in rs)
            summary.append((c, sum(r.ok for r in rs), lat, sum(r.cost for r in rs)))
        print(f"\n{'模型':<46} 通过   中位延迟  一局模型耗时  估算成本")
        for c, ok, lat, _ in summary:
            print(f"{c.id:<48} {ok}/{args.repeat}  {lat:>6.1f}s  {lat * 130 / 60:>7.0f} 分钟  ${c.game_cost(mix):.3f}/局")
        good = [(c, lat) for c, ok, lat, _ in summary
                if ok == args.repeat and (not args.max_latency or lat <= args.max_latency)]
        if not good:
            print(f"没有 {args.repeat}/{args.repeat} 全过"
                  f"{'且延迟 ≤ %.0fs ' % args.max_latency if args.max_latency else ''}的；加大 --probe 或换 --vendors",
                  file=sys.stderr)
            return 1
        chosen = good[0][0]                      # ranked 本来就按成本排好了
        print(f"实测共花费 ${sum(x[3] for x in summary):.5f}")
    print(f"\n选中：{chosen.id}（约 ${chosen.game_cost(mix):.3f}/局）")
    if args.write:
        write_env(chosen.id, args.env)
        print(f"已写入 {args.env}：OPENROUTER_MODEL={chosen.id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
