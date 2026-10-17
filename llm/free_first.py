"""免费优先：先试免费模型，失败了（限流、日额度用尽、上游挂了、超时）这一次改走付费模型。

为什么要有这一层：OpenRouter 的 :free 模型不花钱，但常被上游限流（429），每天也有请求数上限；
游戏一轮要调好几次模型，不能因为免费那边卡一下就让这一轮没话说。所以：
  · 免费那条线只试一次，不重试（失败要快，别等退避），失败就当次回落付费；
  · 连着失败 trip_after 次就“断路”一段时间，这段时间直接走付费，不再浪费时间去撞免费那边；
    日额度用尽（报错里带 per-day）断得更久；
  · JSON 不合法时的“修复重写”请求直接走付费——免费模型已经写坏一次了，别让它再碰；
  · 回落发生的时候写 trace（free_fallback），局末能看到免费接住了多少、付费补了多少。
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

from llm.base import LLMError, LLMRequest, LLMResponse
from llm.structured import REPAIR_PREFIX

DAILY_HINTS = ("per-day", "per day", "daily", "free-models-per-day")


class FreeFirstBackend:
    def __init__(self, free: Any, paid: Any, tracer: Any = None, trip_after: int = 3, cooldown_s: float = 300.0,
                 daily_cooldown_s: float = 6 * 3600.0, repair_to_paid: bool = True,
                 clock: Callable[[], float] = time.monotonic):
        self.free, self.paid, self.tracer = free, paid, tracer
        self.trip_after, self.cooldown_s, self.daily_cooldown_s = trip_after, cooldown_s, daily_cooldown_s
        self.repair_to_paid = repair_to_paid
        self.clock = clock
        self.name = paid.name
        self.fails = 0
        self.open_until = 0.0
        self.stats = {"free_ok": 0, "free_failed": 0, "paid": 0, "skipped_open": 0}

    @property
    def model(self) -> str:
        return f"{getattr(self.free, 'model', '免费')} → {getattr(self.paid, 'model', '付费')}"

    def __getattr__(self, item: str) -> Any:        # list_models 等其余接口都交给付费那条线
        return getattr(self.paid, item)

    def _log(self, **kw) -> None:
        if self.tracer:
            self.tracer.log("free_fallback", **kw)

    def complete(self, req: LLMRequest) -> LLMResponse:
        is_repair = bool(req.messages) and str((req.messages[-1] or {}).get("content", "")).startswith(REPAIR_PREFIX)
        if self.clock() < self.open_until:
            self.stats["skipped_open"] += 1
        elif self.repair_to_paid and is_repair:
            pass
        else:
            try:
                resp = self.free.complete(req)
            except LLMError as e:
                self.fails += 1
                self.stats["free_failed"] += 1
                text = str(e).lower()
                if any(h in text for h in DAILY_HINTS):
                    self.open_until = self.clock() + self.daily_cooldown_s
                elif self.fails >= self.trip_after:
                    self.open_until = self.clock() + self.cooldown_s
                self._log(task=req.task, error=str(e)[:160], fails=self.fails,
                          open_for_s=round(max(0.0, self.open_until - self.clock())))
            else:
                self.fails = 0
                self.stats["free_ok"] += 1
                return resp
        self.stats["paid"] += 1
        return self.paid.complete(req)

    def summary(self) -> str:
        s = self.stats
        return f"免费接住 {s['free_ok']} 次，回落付费 {s['paid']} 次（免费失败 {s['free_failed']}，断路期间直接付费 {s['skipped_open']}）"
