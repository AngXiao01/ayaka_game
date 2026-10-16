"""在 OpenRouter 的模型列表里挑「这个游戏能用的最便宜的模型」。

按标价排第一的往往是专用模型（网页抽取、摘要、视觉），对角色扮演没用，所以先过硬条件再比价：
  1. 文本进文本出，上下文 ≥ 32k（游戏自己的窗口）
  2. 支持 response_format / structured_outputs（整个 harness 靠结构化 JSON）
  3. 不是强制推理（推理 token 按输出计费，还会吃掉 200~700 的 max_tokens，导致正文为空）
  4. 排除 :free（免费档每天 50 次请求，一局就要 100~130 次）、:batch（异步批处理）、~ 开头的浮动别名
  5. 只看通用对话模型的厂商（中文和角色扮演靠得住的那几家），OPENROUTER_CHEAPEST_VENDORS=* 可放开
比价按「一整局」算：此前实跑一局约 37 万 prompt + 2 万 completion tokens，prompt 占 95%，
所以输入价比输出价重要得多——只看输出价排序会选错。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

GAME_MIX = (366_000, 20_000)        # 一整局 prompt / completion tokens（2026-10-10 web 局实跑 trace，131 次调用）
MIN_CONTEXT = 32_768
GENERAL_VENDORS = ("deepseek", "qwen", "z-ai", "moonshotai", "xiaomi", "minimax", "inclusionai", "baidu",
                   "tencent", "bytedance", "stepfun-ai", "openai", "anthropic", "google", "x-ai", "mistralai",
                   "meta-llama")
SENTINELS = ("cheapest", "cheapest-free")


@dataclass
class Candidate:
    id: str
    prompt: float                    # 美元 / token
    completion: float
    context: int
    reasoning: bool                  # 可选推理（会用 effort=none 关掉）
    free: bool = False

    def game_cost(self, mix: tuple[int, int] = GAME_MIX) -> float:
        return mix[0] * self.prompt + mix[1] * self.completion

    def row(self, mix: tuple[int, int] = GAME_MIX) -> str:
        return (f"{self.id:<48} ${self.prompt * 1e6:>6.3f} / ${self.completion * 1e6:>6.3f}  "
                f"{self.context // 1000:>5}k  {'推理可关' if self.reasoning else '—':<4}  ${self.game_cost(mix):.3f}/局")


HEADER = f"{'模型':<46} 输入 / 输出（每百万）  上下文  推理    估算"


def _price(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def why_not(m: dict, allow_free: bool = False, vendors: Iterable[str] = GENERAL_VENDORS) -> str:
    """不合格的理由；合格返回空串。"""
    mid = m.get("id") or ""
    if mid.startswith("~"):
        return "浮动别名"
    if mid.endswith(":batch"):
        return "批处理"
    free = mid.endswith(":free")
    if free and not allow_free:
        return "免费档"
    vendors = tuple(vendors)
    if "*" not in vendors and mid.split("/")[0] not in vendors:
        return "不是通用对话厂商"
    arch = m.get("architecture") or {}
    if "text" not in (arch.get("input_modalities") or ["text"]) or "text" not in (arch.get("output_modalities") or ["text"]):
        return "不是文本模型"
    if int(m.get("context_length") or 0) < MIN_CONTEXT:
        return "上下文不够"
    params = set(m.get("supported_parameters") or [])
    if not params & {"response_format", "structured_outputs"}:
        return "不支持结构化输出"
    if (m.get("reasoning") or {}).get("mandatory"):
        return "强制推理"
    p = m.get("pricing") or {}
    pin, pout = _price(p.get("prompt")), _price(p.get("completion"))
    if pin is None or pout is None or pin < 0 or pout < 0 or (not free and pin + pout == 0):
        return "价格异常"
    return ""


def rank(models: list[dict], allow_free: bool = False, vendors: Iterable[str] = GENERAL_VENDORS,
         mix: tuple[int, int] = GAME_MIX) -> list[Candidate]:
    out = []
    for m in models:
        if why_not(m, allow_free, vendors):
            continue
        p = m["pricing"]
        out.append(Candidate(m["id"], float(p["prompt"]), float(p["completion"]), int(m["context_length"]),
                             "reasoning" in (m.get("supported_parameters") or []), m["id"].endswith(":free")))
    return sorted(out, key=lambda c: (c.game_cost(mix), c.id))


def resolve(sentinel: str, models: list[dict], vendors: Iterable[str] = GENERAL_VENDORS) -> Candidate:
    ranked = rank(models, allow_free=sentinel == "cheapest-free", vendors=vendors)
    if not ranked:
        raise LookupError("模型列表里没有满足条件的模型（结构化输出 / 上下文 ≥ 32k / 非强制推理）")
    return ranked[0]
