"""预取：把互不依赖的几次模型调用提前一起发出去，用到的时候再取结果。

为什么是“预取 + 用时核对”，而不是把节点本身改成并行：
  · 状态的修改仍然严格按原来的顺序、在原来的节点里做——并行的只有“问模型”这一步（纯读、不改世界）；
  · 取结果时核对参数：预取时的参数和用到那一刻要问的完全相同才用预取的答案，否则当场串行重问。
    所以预取只可能“省时间”，不会改变结果；条件判错了最多白发一次请求；
  · 关掉（或者后端是 mock / 重放）时，start 什么都不做，get 就是直接调用——代码路径只有一条。
一个 Prefetch 对象只管一轮对话。每一项（judge / claims / branch）有自己固定的一条线程，线程里沿用提交时的 trace 标签（谁在问）。
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional

_LANES: dict[str, ThreadPoolExecutor] = {}


def _lane(key: str) -> ThreadPoolExecutor:
    """每一项固定用自己的一条线程：连接是按线程保活的（llm/http_pool.py），同一项每一轮走同一条线程才能复用连接，
    不然每次换个线程就多一次握手——在网络不好时握手比判定本身还慢。"""
    if key not in _LANES:
        _LANES[key] = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"prefetch-{key}")
    return _LANES[key]


class Prefetch:
    def __init__(self, enabled: bool = False, tracer: Any = None):
        self.enabled, self.tracer = enabled, tracer
        self._started: dict[str, tuple[tuple, Future]] = {}
        self.hits = self.misses = 0

    def start(self, key: str, fn: Callable, args: Optional[tuple]) -> None:
        """args=None 表示这一项这一轮不会问；没开预取时什么都不做。"""
        if not self.enabled or args is None or key in self._started:
            return
        span = self.tracer.snapshot_span() if self.tracer else {}

        def work():
            if not self.tracer:
                return fn(*args)
            with self.tracer.span(**span, prefetch=key):
                return fn(*args)
        self._started[key] = (args, _lane(key).submit(work))

    def get(self, key: str, fn: Callable, args: tuple) -> Any:
        """要的就是 fn(*args)：有参数一致的预取就等它的结果，没有就当场调。异常和直接调用时一样在这里抛出。"""
        got = self._started.pop(key, None)
        if got is not None and got[0] == args:
            self.hits += 1
            return got[1].result()
        if got is not None:
            self.misses += 1                       # 预取时的参数变了：那次的结果作废
        return fn(*args)
