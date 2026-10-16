"""工具注册表：名字 → 描述、参数、执行函数。

- 工具名严格校验：解码层用 enum 约束（json_schema），运行时再查一遍注册表，未知工具回填错误而不是猜
- 参数校验：缺参、类型不对、超长都在执行前拦下
- 结果缓存：可缓存的工具同参数直接返回（并在 trace 里标 cached）
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from llm.schemas import ValidationError


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, str] = field(default_factory=dict)       # 参数名 -> 说明（字符串参数）
    fn: Optional[Callable[[dict], str]] = None                 # None 表示终止工具（如 respond）
    terminal: bool = False
    cacheable: bool = False
    max_arg_chars: int = 60

    def clean_args(self, args: dict) -> dict:
        if not isinstance(args, dict):
            raise ValidationError("args 必须是 JSON 对象")
        out = {}
        for k in self.params:
            v = args.get(k)
            if not isinstance(v, str) or not v.strip():
                raise ValidationError(f"{self.name} 缺少字符串参数 {k}")
            out[k] = v.strip()[: self.max_arg_chars]
        return out

    def signature(self) -> str:
        ps = ", ".join(f"{k}: {v}" for k, v in self.params.items())
        return f"- {self.name}({ps})：{self.description}"


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}
        self.cache: dict[str, str] = {}

    def names(self) -> list[str]:
        return list(self.tools)

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def describe(self) -> str:
        return "\n".join(t.signature() for t in self.tools.values())

    @staticmethod
    def key(name: str, args: dict) -> str:
        return name + json.dumps(args, ensure_ascii=False, sort_keys=True)

    def execute(self, name: str, args: dict) -> tuple[str, bool]:
        """返回 (观察结果, 是否命中缓存)。调用方负责截断与异常兜底。"""
        tool = self.tools[name]
        if tool.fn is None:
            raise ValidationError(f"{name} 是终止工具，不能当普通工具执行")
        k = self.key(name, args)
        if tool.cacheable and k in self.cache:
            return self.cache[k], True
        obs = tool.fn(args)
        if tool.cacheable:
            self.cache[k] = obs
        return obs, False
