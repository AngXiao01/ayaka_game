"""LLM 后端的公共接口。游戏逻辑只依赖这里，不直接碰 HTTP。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol


class LLMError(Exception):
    """不可重试的模型错误。"""


class RetryableError(LLMError):
    """网络抖动、5xx、429 等可重试错误。"""


class ContextLengthError(LLMError):
    """prompt 超出服务端 max_model_len。"""


class StructuredOutputError(LLMError):
    """多次修复后仍拿不到合法 JSON。"""


@dataclass
class LLMRequest:
    task: str                                   # npc_reply | player_options | summary | ...
    messages: list[dict]
    temperature: float = 0.7
    max_tokens: int = 512
    json_schema: Optional[dict] = None          # 有就请求结构化输出
    schema_name: str = "output"
    meta: dict = field(default_factory=dict)    # 只给 mock/trace 用，绝不发给模型


@dataclass
class LLMResponse:
    text: str
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    reasoning: str = ""                         # 服务端单独返回的思考内容（若有）


class LLMBackend(Protocol):
    name: str

    def complete(self, req: LLMRequest) -> LLMResponse: ...
