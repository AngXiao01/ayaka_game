"""--check：不进游戏，只检查后端（OpenRouter / 自建 Gemma）能不能用、结构化输出走的哪一档。"""
from __future__ import annotations

import re
import statistics
import time
from dataclasses import dataclass
from typing import Callable

from llm.base import LLMError, LLMRequest
from llm.gemma_client import GemmaClient
from llm.json_utils import JSONExtractError, extract_json
from core.agent_loop import step_schema, validate_step
from llm.schemas import ValidationError, npc_reply_schema, validate_npc_reply

START_HINT = ("服务没起来的话：~/Desktop/agentrt/scripts/up51.sh 会在 51 起 vLLM 并把 8000 隧道到本机；"
              "或者先用 --mock 离线跑。")
HINTS = {
    "gemma": START_HINT,
    "openrouter": "检查网络能否访问 openrouter.ai，以及 OPENROUTER_API_KEY / ~/.openrouter_key；或者先用 --mock 离线跑。",
}


def start_hint(backend: str) -> str:
    return HINTS.get(backend, START_HINT)


def run_check(client: GemmaClient, out: Callable[[str], None] = print, repeat: int = 3, max_tokens: int = 700) -> bool:
    out(f"检查 [{client.name}] {client.base_url}（model={client.model}，structured={client.structured}，"
        f"key：{getattr(client, 'key_hint', '?')}）")
    try:
        models = client.list_models()
    except LLMError as e:
        out(f"✗ 连不上：{e}\n  {start_hint(client.name)}")
        return False
    ids = [m.get("id") for m in models]
    var = f"{client.name.upper()}_MODEL"
    if len(ids) > 12:                    # OpenRouter 一次返回几百个
        near = [i for i in ids if i and client.model.split("/")[-1].split("-")[0] in i][:8]
        out(f"✓ /models：共 {len(ids)} 个，{'包含' if client.model in ids else '不包含'} {client.model}")
        if client.model not in ids:
            out(f"✗ 没有 {client.model}；把 {var} 设成存在的 id，相近的有：{near}")
            return False
    else:
        out(f"✓ /models：{ids}")
        if client.model not in ids:
            out(f"✗ 服务端没有 {client.model}；把 {var} 设成上面之一")
            return False

    if client.name == "openrouter":
        try:
            info = client.key_info()
        except LLMError as e:
            out(f"✗ key 校验失败：{e}\n  /models 不需要 key 所以上面能过；这里 401 说明 key 本身无效"
                f"（粘贴错、已删除或已轮换）。去 https://openrouter.ai/settings/keys 新建一把，重新写进 key 文件。")
            return False
        free = info.get("free_model_daily_requests") or {}
        left = info.get("limit_remaining")
        out(f"✓ key 有效：{'账户没充值过（free tier）' if info.get('is_free_tier') else '账户充值过'}，"
            f"这把 key 累计已用 ${float(info.get('usage') or 0):.4f}，"
            f"key 的花费上限剩余 {'不限' if left is None else '$%.2f' % float(left)}"
            + (f"，今日免费模型请求剩 {free.get('remaining')}/{free.get('limit')}" if free else ""))

    t0 = time.monotonic()
    try:
        r = client.complete(LLMRequest(task="ping", messages=[{"role": "user", "content": "用一句话问好。"}],
                                       temperature=0.2, max_tokens=64))
        out(f"✓ 普通对话（{time.monotonic() - t0:.1f}s）：{r.text[:60]!r}")
    except LLMError as e:
        out(f"✗ 普通对话失败：{e}")
        return False

    results = []
    for i in range(repeat):                  # 同一请求跑几次：便宜模型的结构化输出常常时好时坏，一次说明不了问题
        r = structured_probe(client, max_tokens=max_tokens)
        results.append(r)
        out(f"{'✓' if r.ok else '✗'} 结构化 NPC 回复 {i + 1}/{repeat}：{r.detail}")
        if "401" in r.detail or "402" in r.detail:
            break
    passed = sum(r.ok for r in results)
    lat = statistics.median(r.latency for r in results) if results else 0
    out(f"{'✓' if passed == len(results) else '✗'} 结构化通过 {passed}/{len(results)}，中位延迟 {lat:.1f}s"
        f"（一局约 130 次调用，单是模型耗时就约 {lat * 130 / 60:.0f} 分钟）")
    return passed == len(results)


_CJK = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class ProbeResult:
    ok: bool
    detail: str
    latency: float = 0.0
    cost: float = 0.0
    speech: str = ""


def structured_probe(client: GemmaClient, max_tokens: int = 700, temperature: float = 0.8) -> ProbeResult:
    """跑一步真实的 NPC ReAct 结构化输出（参数与游戏里 NPC 一致）：JSON 能解析、过 schema 校验、台词是中文。
    失败时带上 finish_reason、输出 token 数和输出结尾，区分「写太长被截断」「空白死循环」「格式不对」。"""
    ids_ = ["greeted_properly"]
    msgs = [{"role": "system", "content": "你扮演神里绫华。每一步只输出一个 JSON：{\"thought\": \"简短思考\", \"tool\": \"工具名\", "
                                          "\"args\": {...}}。工具：lookup_lore(topic)、respond(action, speech, emotion, intent, "
                                          "affinity_delta, trust_delta, tension_delta, task_signals, wants_to_end)。"},
            {"role": "user", "content": '<player_input intent="礼貌问候">打扰了，多谢您的邀请。</player_input>\n调用 respond 回应。'}]
    before = client.structured
    t0 = time.monotonic()
    try:
        r = client.complete(LLMRequest(task="npc_step", messages=msgs, temperature=temperature, max_tokens=max_tokens,
                                       json_schema=step_schema(["respond"], "respond", npc_reply_schema(ids_)),
                                       schema_name="agent_step"))
    except LLMError as e:
        return ProbeResult(False, f"请求失败（{time.monotonic() - t0:.1f}s）：{str(e)[:300]}", time.monotonic() - t0)
    dt = time.monotonic() - t0
    u = r.usage or {}
    cost = float(u.get("cost") or 0)
    try:
        st = validate_step(extract_json(r.text))
        obj = validate_npc_reply(st["args"], ids_)
    except (JSONExtractError, ValidationError) as e:
        tail = r.text[-60:] if r.text else ""
        ctoks = int(u.get("completion_tokens") or 0)
        rtoks = int(((u.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
        visible = len((r.text or "").strip())
        why = ""
        if r.finish_reason == "length":
            why = ("；token 用满但正文只有几个字：大概率是结构化解码下的空白死循环，或推理吃掉了预算"
                   if visible < ctoks / 4 else "；正文写太长被截断")
        return ProbeResult(False, f"{e}（{dt:.1f}s，finish_reason={r.finish_reason or '?'}，"
                                  f"输出 {ctoks or '?'}/{max_tokens} tokens"
                                  f"{'（其中推理 %d）' % rtoks if rtoks else ''}，正文 {visible} 字，"
                                  f"结尾 {tail!r}{why}）", dt, cost)
    if not _CJK.search(obj["speech"]):
        return ProbeResult(False, f"台词不是中文：{obj['speech'][:60]!r}", dt, cost)
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    return ProbeResult(True, (f"ReAct 步骤结构化输出（{dt:.1f}s，档位 {client.structured}"
                              f"{'，已从 ' + before + ' 降级' if client.structured != before else ''}"
                              f"{'' if cached is None else '，返回了 cached_tokens'}"
                              f"{'，这次花费 $%.6f' % cost if u.get('cost') is not None else ''}）：{obj['speech'][:60]!r}"),
                       dt, cost, obj["speech"])
