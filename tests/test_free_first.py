"""免费优先：先试免费，失败回落付费；连败断路；日额度用尽久断；修复重写直接走付费。"""
from llm.base import LLMError, LLMRequest, LLMResponse
from llm.free_first import FreeFirstBackend
from llm.structured import REPAIR_PREFIX


class Fake:
    def __init__(self, name, fail=None):
        self.name, self.model, self.fail, self.calls = "openrouter", name, fail, 0

    def complete(self, req):
        self.calls += 1
        if self.fail:
            raise LLMError(self.fail)
        return LLMResponse(text=self.model)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def req(content="你好"):
    return LLMRequest(task="npc_step", messages=[{"role": "user", "content": content}])


def test_free_is_used_first():
    free, paid = Fake("free"), Fake("paid")
    b = FreeFirstBackend(free, paid)
    assert b.complete(req()).text == "free" and paid.calls == 0 and b.stats["free_ok"] == 1


def test_a_failure_falls_back_for_that_request_only():
    free, paid = Fake("free", fail="HTTP 429 rate-limited"), Fake("paid")
    b = FreeFirstBackend(free, paid)
    assert b.complete(req()).text == "paid" and free.calls == 1
    free.fail = None
    assert b.complete(req()).text == "free"                        # 一次失败不影响下一次
    assert b.fails == 0


def test_repeated_failures_trip_the_breaker_then_recover():
    clock = Clock()
    free, paid = Fake("free", fail="HTTP 502"), Fake("paid")
    b = FreeFirstBackend(free, paid, trip_after=3, cooldown_s=300, clock=clock)
    for _ in range(3):
        b.complete(req())
    assert free.calls == 3
    b.complete(req())
    assert free.calls == 3 and b.stats["skipped_open"] == 1        # 断路期间不再去撞免费那边
    clock.t = 301
    free.fail = None
    assert b.complete(req()).text == "free"                        # 过了冷却再试，恢复


def test_daily_quota_opens_the_breaker_for_long():
    clock = Clock()
    free, paid = Fake("free", fail="HTTP 429: free-models-per-day limit exceeded"), Fake("paid")
    b = FreeFirstBackend(free, paid, clock=clock)
    b.complete(req())
    clock.t = 3600
    b.complete(req())
    assert free.calls == 1                                          # 一小时后仍然不试：今天的免费额度用完了


def test_repair_requests_go_straight_to_paid():
    free, paid = Fake("free"), Fake("paid")
    b = FreeFirstBackend(free, paid)
    assert b.complete(req(REPAIR_PREFIX + "你的上一条输出无法使用")).text == "paid" and free.calls == 0
