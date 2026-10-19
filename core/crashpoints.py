"""崩溃注入点：测试时在命名的位置“拔电源”，看恢复之后状态对不对。

用法（与游戏无关，任何有副作用的 agent 都能用）：
  在副作用的前后放 `hit("memory.after_append")` 这样的点——平时什么都不做，开销是一次字典查询；
  测试里 `arm("memory.after_append", nth=2)`，第 2 次经过这个点时抛 SimulatedCrash。

SimulatedCrash 继承 BaseException，而不是 Exception：真实的进程被杀不会被 `except Exception` 吞掉，
模拟的也不该被吞掉（否则代码里随手一个 except Exception 就把崩溃“恢复”了，测试就成了摆设）。
"""
from __future__ import annotations

from collections import Counter
from typing import Optional


class SimulatedCrash(BaseException):
    pass


_armed: dict[str, int] = {}
_hits: Counter = Counter()


def hit(name: str) -> None:
    _hits[name] += 1
    want: Optional[int] = _armed.get(name)
    if want is not None and _hits[name] == want:
        _armed.clear()
        raise SimulatedCrash(f"{name}#{want}")


def arm(name: str, nth: int = 1) -> None:
    """让第 nth 次经过 name 时崩。次数从现在起重新数。"""
    _hits.clear()
    _armed.clear()
    _armed[name] = nth


def disarm() -> None:
    _armed.clear()
    _hits.clear()


def seen() -> dict[str, int]:
    """这一轮里各个点被经过了几次——用来枚举“有哪些崩溃点、各经过几次”，生成崩溃矩阵。"""
    return dict(_hits)
