"""神里绫华 · 叙事 Agent 游戏入口。

  python main.py                         连 LLM_BACKEND 指定的后端（默认 OpenRouter，需要 OPENROUTER_API_KEY）
  python main.py --mock                  离线跑（不联网）
  python main.py --check                 只检查后端连通性与结构化输出
  python main.py --auto cooperative      AI 扮演玩家，无人值守跑一遍
  LLM_BACKEND=gemma python main.py       连自建 vLLM 上的 Gemma（默认 http://127.0.0.1:8000/v1）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config.settings import load_settings
from core.player_agent import AutoPlayer, HumanPlayer
from engine.game_engine import GameEngine
from llm.base import LLMError
from llm.diagnostics import run_check, start_hint
from llm.gemma_client import GemmaClient
from llm import model_picker
from llm.mock_backend import MockBackend
from llm.router import ModelRouter


_RESOLVED: dict[tuple, str] = {}


def build_llm(cfg, model: str = "") -> GemmaClient:
    """按 LLM_BACKEND 建客户端（OpenRouter 与自建 vLLM 都是 OpenAI 兼容接口）。
    model 是 cheapest / cheapest-free 时，查一次 /models 挑最便宜的可用模型，结果缓存并写回 cfg.model。"""
    if cfg.backend == "openrouter" and not cfg.api_key:
        raise LLMError(f"OpenRouter 需要 API key：{cfg.key_source}")
    client = GemmaClient(cfg.base_url, model or cfg.model, api_key=cfg.api_key, timeout=cfg.timeout,
                         max_retries=cfg.max_retries, merge_system=cfg.merge_system,
                         structured=cfg.structured, strip_thinking=cfg.strip_thinking,
                         chat_template_kwargs=cfg.chat_template_kwargs, name=cfg.backend,
                         headers=cfg.headers, key_hint=cfg.key_source, reasoning_effort=cfg.reasoning_effort)
    if client.model in model_picker.SENTINELS:
        sentinel, k = client.model, (cfg.base_url, client.model, cfg.cheapest_vendors)
        if k not in _RESOLVED:
            try:
                c = model_picker.resolve(sentinel, client.list_models(),
                                         cfg.cheapest_vendors or model_picker.GENERAL_VENDORS)
            except LookupError as e:
                raise LLMError(str(e)) from e
            _RESOLVED[k] = c.id
            cfg.model_note = (f"{sentinel} → {c.id}（输入 ${c.prompt * 1e6:.3f} / 输出 ${c.completion * 1e6:.3f} 每百万 token，"
                              f"按一整局估约 ${c.game_cost():.3f}）")
        client.model = _RESOLVED[k]
        if not model or model == cfg.model:
            cfg.model = client.model
    return client


build_gemma = build_llm          # 旧名字


def with_free_first(cfg, backend, tracer=None):
    """FREE_FIRST=1 时给付费后端套一层“免费优先”：先试 FREE_MODELS 里第一个模型列表里真有的免费模型，
    失败这一次回落付费。挑不到免费模型、或没开，原样返回。"""
    from llm.free_first import FreeFirstBackend
    if not getattr(cfg, "free_first", False) or cfg.backend != "openrouter":
        return backend
    try:
        have = {m.get("id") for m in backend.list_models()}
    except LLMError:
        have = set()
    pick = next((m for m in cfg.free_models if m in have), "")
    if not pick:
        return backend
    free = build_llm(cfg, pick)
    free.max_retries = 0                 # 免费那条线失败要快：不退避不重试，直接回落付费
    cfg.model_note = (cfg.model_note + "；" if cfg.model_note else "") + f"免费优先：{pick}（失败回落 {backend.model}）"
    return FreeFirstBackend(free, backend, tracer)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="神里绫华 · 叙事 Agent 游戏")
    ap.add_argument("--mock", action="store_true", help="用离线 Mock 后端")
    ap.add_argument("--check", action="store_true", help="只检查后端连通性与结构化输出")
    ap.add_argument("--auto", choices=["cooperative", "chaotic", "idler", "injector", "steer"],
                    help="自动玩家：cooperative 配合 / chaotic 乱选 / idler 只闲聊 / injector 夹带注入 / steer 带节奏")
    ap.add_argument("--save", default="default", help="存档名")
    ap.add_argument("--new", action="store_true", help="清掉该存档（含记忆）重新开始")
    ap.add_argument("--max-actions", type=int, default=400, help="地点层最多行动次数（防失控）")
    ap.add_argument("--max-turns", type=int, help="每次交谈最多几轮")
    ap.add_argument("--no-hints", action="store_true", help="不在菜单里标出可能有进展的人和地方")
    ap.add_argument("--options", choices=["llm", "static"], help="玩家选项由 LLM 生成还是用固定兜底")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--debug", action="store_true", help="显示好感/信任/张力变化")
    ap.add_argument("--incognito", action="store_true", help="无痕：存档、记忆放临时目录，不写 trace，退出即删")
    args = ap.parse_args(argv)

    settings = load_settings()
    g = settings.game
    if args.max_turns:
        g.max_turns_per_session = args.max_turns
    if args.options:
        g.options_mode = args.options
    if args.seed is not None:
        g.seed = args.seed
    g.debug = g.debug or args.debug
    g.hints = g.hints and not args.no_hints

    if args.mock or settings.llm.backend == "mock":
        backend = MockBackend(seed=g.seed)
        router = ModelRouter(backend, settings.llm)
    else:
        try:
            backend = build_llm(settings.llm)
        except LLMError as e:
            print(f"{e}\n{start_hint(settings.llm.backend)}", file=sys.stderr)
            return 1
        if args.check:
            if settings.llm.model_note:
                print(f"自动选模型：{settings.llm.model_note}")
            return 0 if run_check(backend, max_tokens=settings.llm.npc.max_tokens) else 1
        try:
            backend.list_models()
        except LLMError as e:
            print(f"连不上 {settings.llm.backend}（{settings.llm.base_url}）：{e}\n{start_hint(settings.llm.backend)}",
                  file=sys.stderr)
            return 1
        if settings.llm.model_note:
            print(f"[{settings.llm.backend}] 自动选模型：{settings.llm.model_note}")
        backend = with_free_first(settings.llm, backend)
        router = ModelRouter(backend, settings.llm, lambda m: build_llm(settings.llm, m))

    if args.auto in ("idler", "injector", "steer"):
        from evals.personas import make_player
        player = make_player(args.auto, g.seed or 0, 6)
    elif args.auto:
        player = AutoPlayer(args.auto, seed=g.seed)
    else:
        player = HumanPlayer()
    tmpdir = None
    if args.incognito:
        import shutil
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="ayaka_incognito_")
        g.data_dir = Path(tmpdir)
    engine = GameEngine(settings, backend, player, save_name=args.save, new_game=args.new or args.incognito,
                        router=router, trace=not args.incognito)
    if hasattr(backend, "free"):
        backend.tracer = engine.tracer                      # 免费回落的时候写进 trace
    try:
        engine.run(max_actions=args.max_actions)
    except KeyboardInterrupt:
        if not args.incognito:
            engine.save()
        print("\n（已中断" + ("，无痕模式不保存）" if args.incognito else "，进度已保存）"))
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"\n[{backend.name}] {engine.stats_line()}")
    if hasattr(backend, "free"):
        print("免费优先：" + backend.summary())
    if engine.tracer.path:
        print(f"trace：{engine.tracer.path}\n可视化：python3 tools/view_trace.py {engine.tracer.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
