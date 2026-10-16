"""OpenAI 兼容接口客户端（OpenRouter / 自建 vLLM 上的 Gemma），纯标准库。

类名沿用 GemmaClient（最早只连 vLLM 上的 Gemma），现在 OpenRouter 也走这里，区别只在配置：
base_url、key、额外请求头、name（写进 trace 的后端名）。

通用处理：
- 消息整形：Gemma 模板要求 user/assistant 严格交替、首条非 system 必须是 user；
  连续同角色消息合并，必要时补一条 user。merge_system=True 时把 system 并进首条 user。
- 结构化输出：优先 response_format=json_schema（vLLM 结构化解码），服务端不支持时
  自动降级 json_object → off，降级结果对后续请求生效。
- 去掉 <think> 等思考块；vLLM 单独返回的 reasoning_content 另存，不进正文。
- 网络错误 / 5xx / 429 指数退避重试；上下文超长单独抛 ContextLengthError。
- OpenRouter 上游出错时可能回 HTTP 200 + {"error": {...}}、没有 choices：按 error.code 归到可重试或不可重试。
"""
from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request
from typing import Callable, Optional

from llm.base import ContextLengthError, LLMError, LLMRequest, LLMResponse, RetryableError
from llm.json_utils import strip_thinking as _strip_thinking

STRUCTURED_LEVELS = ["json_schema", "json_object", "off"]
_FORMAT_UNSUPPORTED_RE = re.compile(r"response_format|json_schema|json_object|guided|grammar|structured", re.I)
_CTX_RE = re.compile(r"maximum context length|context length|too long|max_model_len", re.I)
_REASONING_RE = re.compile(r"reasoning|thinking|effort", re.I)


def describe_key(key: str) -> str:
    """描述 key 的「形状」帮助排查 401，不泄露内容：长度、前缀对不对、有没有空白 / 引号 / 非 ASCII。"""
    if not key:
        return "空"
    parts = [f"长度 {len(key)}"]
    if key.startswith("sk-or-v1-"):
        parts.append("sk-or-v1- 开头")
    elif key.startswith("sk-"):
        parts.append("sk- 开头但不是 sk-or-（可能是别家的 key）")
    else:
        parts.append(f"开头是 {key[:3]!r}")
    if re.search(r"\s", key):
        parts.append("中间有空白")
    if any(c in key for c in "\"'"):
        parts.append("含引号")
    if not key.isascii():
        parts.append("含非 ASCII 字符")
    return "，".join(parts)


def to_wire(messages: list[dict], merge_system: bool = False) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        role, content = m["role"], m.get("content") or ""
        if out and role == out[-1]["role"] and role in ("user", "assistant"):
            out[-1] = {"role": role, "content": out[-1]["content"] + "\n\n" + content}
        else:
            out.append({"role": role, "content": content})
    head = [m for m in out if m["role"] == "system"]
    body = [m for m in out if m["role"] != "system"]
    if not body or body[0]["role"] != "user":
        body.insert(0, {"role": "user", "content": "（对话开始）"})
    if head:
        sys_text = "\n\n".join(m["content"] for m in head)
        if merge_system:
            body[0] = {"role": "user", "content": sys_text + "\n\n" + body[0]["content"]}
            return body
        return [{"role": "system", "content": sys_text}] + body
    return body


def _retryable(code) -> bool:
    try:
        code = int(code)
    except (TypeError, ValueError):
        return False
    return code in (408, 429) or code >= 500


class GemmaClient:
    name = "gemma"

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 120.0,
                 max_retries: int = 3, merge_system: bool = False, structured: str = "json_schema",
                 strip_thinking: bool = True, chat_template_kwargs: Optional[dict] = None,
                 sleep: Callable[[float], None] = time.sleep, name: str = "gemma",
                 headers: Optional[dict] = None, key_hint: str = "GEMMA_API_KEY / ~/.agentrt/api_key",
                 reasoning_effort: str = ""):
        if structured not in STRUCTURED_LEVELS:
            raise ValueError(f"structured 只能是 {STRUCTURED_LEVELS}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.merge_system = merge_system
        self.structured = structured
        self.strip_thinking = strip_thinking
        self.chat_template_kwargs = chat_template_kwargs
        self._sleep = sleep
        self.name = name
        self.headers = dict(headers or {})
        self.key_hint = key_hint
        self.reasoning_effort = reasoning_effort      # OpenRouter：reasoning.effort；模型不认时自动去掉

    # ------------------------------------------------------------ HTTP

    def _request(self, method: str, path: str, payload: Optional[dict] = None) -> dict:
        data = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")
            if e.code == 400 and _CTX_RE.search(body):
                raise ContextLengthError(body[:500]) from e
            if _retryable(e.code):
                raise RetryableError(f"HTTP {e.code}: {body[:300]}") from e
            if e.code == 401:
                raise LLMError(f"HTTP 401：API key 无效或没带（来源 {self.key_hint}；key 形状：{describe_key(self.api_key)}）"
                               f"：{body[:200]}") from e
            if e.code == 402:
                raise LLMError(f"HTTP 402：余额不足，或这把 key 设了额度上限：{body[:300]}") from e
            raise LLMError(f"HTTP {e.code}: {body[:500]}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise RetryableError(f"{type(e).__name__}: {e}") from e

    def _with_retry(self, fn):
        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except RetryableError:
                if attempt == self.max_retries:
                    raise
                self._sleep(min(20.0, 1.0 * 2 ** attempt) * (0.5 + random.random()))

    # ------------------------------------------------------------ API

    def list_models(self) -> list[dict]:
        """不重试，用来快速判断服务在不在。"""
        return self._request("GET", "/models").get("data", []) or []

    def key_info(self) -> dict:
        """OpenRouter：GET /key 校验 key 本身（/models 不需要 key，测不出来）。"""
        return self._request("GET", "/key").get("data", {}) or {}

    def _response_format(self, req: LLMRequest) -> Optional[dict]:
        if req.json_schema is None or self.structured == "off":
            return None
        if self.structured == "json_schema":
            return {"type": "json_schema",
                    "json_schema": {"name": req.schema_name, "schema": req.json_schema}}
        return {"type": "json_object"}

    def _chat(self, payload: dict) -> dict:
        data = self._request("POST", "/chat/completions", payload)
        err = data.get("error") if isinstance(data, dict) else None
        if err and not data.get("choices"):
            code = err.get("code") if isinstance(err, dict) else None
            msg = err.get("message", "") if isinstance(err, dict) else str(err)
            meta = (err.get("metadata") or {}) if isinstance(err, dict) else {}
            text = f"上游错误 {code}: {msg} {json.dumps(meta, ensure_ascii=False)[:300] if meta else ''}".strip()
            if _CTX_RE.search(msg):
                raise ContextLengthError(text)
            if _retryable(code):
                raise RetryableError(text)
            raise LLMError(text)
        return data

    def complete(self, req: LLMRequest) -> LLMResponse:
        payload: dict = {
            "model": self.model,
            "messages": to_wire(req.messages, self.merge_system),
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        rf = self._response_format(req)
        if rf:
            payload["response_format"] = rf
        t0 = time.monotonic()
        try:
            data = self._with_retry(lambda: self._chat(payload))
        except LLMError as e:
            if "reasoning" in payload and not isinstance(e, (RetryableError, ContextLengthError)) \
                    and _REASONING_RE.search(str(e)):
                self.reasoning_effort = ""                # 模型是强制推理、或不认这个档位：之后都不再发
                return self.complete(req)
            if rf and not isinstance(e, (RetryableError, ContextLengthError)) and _FORMAT_UNSUPPORTED_RE.search(str(e)):
                self.structured = STRUCTURED_LEVELS[STRUCTURED_LEVELS.index(self.structured) + 1]
                return self.complete(req)
            raise
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"响应里没有 choices：{str(data)[:300]}") from e
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        if self.strip_thinking:
            text = _strip_thinking(text)
        return LLMResponse(
            text=text,
            finish_reason=choice.get("finish_reason") or "",
            usage=data.get("usage") or {},
            latency_s=round(time.monotonic() - t0, 3),
            reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
        )
