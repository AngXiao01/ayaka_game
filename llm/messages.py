"""消息编排的不变量与 token 粗估。

所有发给模型的会话在出门前都过一遍 check_conversation：
  [system] → user → assistant → user → … → user
多 agent、多步循环、工具观察回填都在同一套规则下拼接，乱序直接抛错而不是让模型去猜。
"""
from __future__ import annotations

import re

_CJK_RE = re.compile(r"[　-〿㐀-鿿＀-￯]")


class ConversationError(ValueError):
    pass


def check_conversation(messages: list[dict]) -> None:
    if not messages:
        raise ConversationError("会话为空")
    body = messages[1:] if messages[0].get("role") == "system" else messages
    if any(m.get("role") == "system" for m in body):
        raise ConversationError("system 消息只能出现在第一条")
    if not body:
        raise ConversationError("缺少 user 消息")
    for i, m in enumerate(body):
        want = "user" if i % 2 == 0 else "assistant"
        if m.get("role") != want:
            raise ConversationError(f"第 {i + 1} 条非 system 消息应为 {want}，实际是 {m.get('role')}")
        if not str(m.get("content") or "").strip():
            raise ConversationError(f"第 {i + 1} 条非 system 消息内容为空")
    if body[-1]["role"] != "user":
        raise ConversationError("会话必须以 user 消息结尾，模型才知道该轮到自己说话")


def estimate_tokens(text: str) -> int:
    """中文按 1 字 ≈ 1 token，其余按 4 字符 ≈ 1 token，偏保守。"""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    return cjk + (len(text) - cjk + 3) // 4


def messages_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(m.get("content") or "") + 4 for m in messages)
