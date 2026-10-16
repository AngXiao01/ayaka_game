"""结构化调用：请求 → 抠 JSON → 校验；失败就把错误回填给模型让它重写。"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Optional, TypeVar

from llm.base import LLMBackend, LLMError, LLMRequest, StructuredOutputError
from llm.json_utils import JSONExtractError, extract_json
from llm.schemas import ValidationError

T = TypeVar("T")
REPAIR_PREFIX = "【格式错误】"


def call_json(backend: LLMBackend, req: LLMRequest, validate: Callable[[Any], T],
              tracer: Optional[Any] = None, repair_attempts: int = 2) -> T:
    return call_json_ex(backend, req, validate, tracer, repair_attempts)[0]


def call_json_ex(backend: LLMBackend, req: LLMRequest, validate: Callable[[Any], T],
                 tracer: Optional[Any] = None, repair_attempts: int = 2) -> tuple[T, int]:
    """同 call_json，另返回本次（含修复重试）消耗的 token 数，供循环做预算。"""
    messages = list(req.messages)
    tokens = 0
    last_err: Optional[Exception] = None
    for attempt in range(repair_attempts + 1):
        r = replace(req, messages=messages)
        try:
            resp = backend.complete(r)
        except LLMError as e:
            if tracer:
                tracer.log_llm(r, None, attempt, ok=False, error=f"{type(e).__name__}: {e}")
            raise
        tokens += int(resp.usage.get("prompt_tokens") or 0) + int(resp.usage.get("completion_tokens") or 0)
        try:
            result = validate(extract_json(resp.text))
        except (JSONExtractError, ValidationError) as e:
            last_err = e
            if tracer:
                tracer.log_llm(r, resp, attempt, ok=False, error=str(e))
            hint = "上一次输出被截断了，请把内容写得更简短。" if resp.finish_reason == "length" else ""
            messages = messages + [
                {"role": "assistant", "content": resp.text[:2000] or "（空）"},
                {"role": "user", "content": f"{REPAIR_PREFIX}你的上一条输出无法使用：{e}。{hint}"
                                            "请严格只输出一个符合要求的 JSON 对象，不要任何解释，不要代码块。"},
            ]
            continue
        if tracer:
            tracer.log_llm(r, resp, attempt, ok=True)
        return result, tokens
    raise StructuredOutputError(f"{req.task}：{repair_attempts + 1} 次尝试仍未得到合法 JSON（{last_err}）")
