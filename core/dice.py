"""概率判定。掷骰结果由（存档种子, 事项, 当前分钟）决定：读档重来同一时刻结果不变，换个时间再试才会重掷。"""
import random
from typing import Any


def roll(god: Any, key: str, p: float) -> tuple[bool, int]:
    """返回 (成功?, 显示用的成功率百分比)。每次判定都记进 god.rolls，trace 和调试面板能看到。"""
    p = max(0.0, min(1.0, p))
    r = random.Random(f"{god.seed}|{key}|{god.minute}").random()
    ok = r < p
    god.rolls.append({"key": key, "p": round(p, 2), "roll": round(r, 3), "ok": ok, "minute": god.minute})
    del god.rolls[:-50]
    return ok, round(p * 100)


def tag(pct: int, ok: bool, why: list[str] = ()) -> str:
    return f"🎲 {pct}%{'（' + '，'.join(why) + '）' if why else ''} → {'成功' if ok else '失败'}"
