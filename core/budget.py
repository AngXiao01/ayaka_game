"""整局 token 预算与降级策略。

预算按 tracer 统计的真实 usage 扣减（所有 LLM 调用都经过 tracer）。
- 剩余比例 < degrade_ratio：降级——玩家选项改静态模板、审核只走规则、NPC 循环只允许直接 respond
- 用尽：本次见面礼貌收尾，游戏停止（存档保留，补预算后可继续）
"""
from __future__ import annotations

from typing import Any


class Budget:
    def __init__(self, stats: dict, max_tokens: int = 0, degrade_ratio: float = 0.15):
        self.stats = stats            # Tracer.stats 的引用
        self.max_tokens = max_tokens
        self.degrade_ratio = degrade_ratio
        self._announced = False

    def used(self) -> int:
        return int(self.stats.get("prompt_tokens", 0)) + int(self.stats.get("completion_tokens", 0))

    def remaining_ratio(self) -> float:
        if self.max_tokens <= 0:
            return 1.0
        return max(0.0, 1 - self.used() / self.max_tokens)

    def degraded(self) -> bool:
        return self.max_tokens > 0 and self.remaining_ratio() < self.degrade_ratio

    def exhausted(self) -> bool:
        return self.max_tokens > 0 and self.used() >= self.max_tokens

    def snapshot(self) -> dict[str, Any]:
        return {"used": self.used(), "max": self.max_tokens, "remaining_ratio": round(self.remaining_ratio(), 3),
                "degraded": self.degraded(), "exhausted": self.exhausted()}

    def first_degrade(self) -> bool:
        """第一次进入降级时返回 True（用来只提示一次）。"""
        if self.degraded() and not self._announced:
            self._announced = True
            return True
        return False
