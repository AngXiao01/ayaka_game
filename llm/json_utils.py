"""从模型文本里稳健地抠出 JSON。

Gemma 常见的不规范输出：
- 用 ```json 代码块包起来，或前后带解释文字
- 输出前带思考块 <think>...</think>
- 漂到原生工具调用格式 call:name{...}
- max_tokens 不够被截断
- 末尾多余逗号
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterator, Union

_THINK_RES = [
    re.compile(r"<think>.*?</think>", re.S | re.I),
    re.compile(r"<thinking>.*?</thinking>", re.S | re.I),
    re.compile(r"<\|?channel\|?>\s*thought.*?<\|?channel\|?>", re.S | re.I),
]
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)
_NATIVE_CALL_RE = re.compile(r"^\s*call:([A-Za-z_]\w*)\s*\{")
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


class JSONExtractError(ValueError):
    pass


def strip_thinking(text: str) -> str:
    for r in _THINK_RES:
        text = r.sub("", text)
    return text.strip()


def _balanced_objects(text: str) -> Iterator[str]:
    """依次产出顶层 {...} 片段；感知字符串与转义，括号在字符串里不算。"""
    depth, start, in_str, esc = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                yield text[start:i + 1]


def _loads_lenient(s: str) -> Any:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return json.loads(_TRAILING_COMMA_RE.sub(r"\1", s))


def extract_json(text: str) -> Union[dict, list]:
    if text is None or not text.strip():
        raise JSONExtractError("输出为空")
    t = strip_thinking(text)
    candidates = [t] + [m.strip() for m in _FENCE_RE.findall(t)] + list(_balanced_objects(t))
    for c in candidates:
        try:
            obj = _loads_lenient(c)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, (dict, list)):
            return obj
    native = _NATIVE_CALL_RE.match(t)
    if native:
        raise JSONExtractError(f"输出成了 call:{native.group(1)}{{...}} 形式的工具调用，这里只要一个纯 JSON 对象")
    if t.count("{") > t.count("}"):
        raise JSONExtractError("JSON 不完整（大括号没有闭合，可能被截断）")
    raise JSONExtractError("没有找到合法的 JSON 对象")
