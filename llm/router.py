"""模型路由：按任务选后端与采样参数。

不同 agent 对模型的要求不同：NPC 要文采和角色一致性；审核、压缩、选项要稳和快。
默认全部走同一个模型；设了 MODEL_<TASK>（如 MODEL_REVIEW=openai/gpt-4o-mini）就把该任务
路由到同一后端下的另一个模型——小任务用小模型，省钱也降延迟。
"""
from __future__ import annotations

from typing import Callable

from config.settings import TASKS, LLMSettings, TaskParams
from llm.base import LLMBackend


class ModelRouter:
    def __init__(self, default: LLMBackend, cfg: LLMSettings,
                 make_backend: Callable[[str], LLMBackend] | None = None):
        self.default = default
        self.cfg = cfg
        self.backends: dict[str, LLMBackend] = {}
        for task, model in cfg.routes.items():
            if make_backend and model:
                self.backends[task] = make_backend(model)

    def backend(self, task: str) -> LLMBackend:
        if task not in TASKS:
            raise KeyError(f"未知任务：{task}")
        return self.backends.get(task, self.default)

    def params(self, task: str) -> TaskParams:
        return getattr(self.cfg, task)

    def describe(self) -> dict[str, str]:
        return {t: getattr(self.backend(t), "model", self.backend(t).name) for t in TASKS}
