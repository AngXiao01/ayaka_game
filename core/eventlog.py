"""事件溯源：把“世界状态怎么一步步变成现在这样”记成可以重放的东西。

AI Town 的做法是：对世界的改动只通过输入提交，每一步存状态差异。我们的世界状态散在 GodEnv、物理世界、关系、各 NPC 的长期记忆里，
改动入口太多，没法逐个改成事件；所以用一个对所有入口都管用的办法——**在每次玩家输入之前，把状态快照和上一次比一比，记下差异**：

  state_init   这局开始时的完整快照（世界、关系、长期记忆、关键设置）
  input        玩家的每一次输入（做了什么、说了什么）
  state_delta  两次输入之间世界发生的所有变化（路径 + 新值；列表只改了几项 / 只追加，就只记那几项和追加的部分）
  llm / decide_io  每一次模型回答、每一次便宜判断的结果（原样记下，重放时按顺序喂回去）

有了这四样，一局就能：
  · 不跑引擎，直接重建任意一步之后的世界（reconstruct）——时间旅行调试；
  · 用同样的输入 + 记下的模型回答把整局原样重放一遍，逐步核对每一步的状态差异（tools/replay_game.py）——
    代码改动后跑一遍，就知道逻辑有没有改坏。
"""
from __future__ import annotations

import copy
from typing import Any


def diff(old: Any, new: Any, path: tuple = ()) -> list[dict]:
    """返回把 old 变成 new 的操作列表。dict 逐键递归；列表大部分不动就按下标记改动、追加；其余整体替换。"""
    if isinstance(old, dict) and isinstance(new, dict):
        ops: list[dict] = []
        for k in new:
            if k not in old:
                ops.append({"p": [*path, k], "v": new[k]})
            else:
                ops += diff(old[k], new[k], (*path, k))
        for k in old:
            if k not in new:
                ops.append({"p": [*path, k], "del": True})
        return ops
    if isinstance(old, list) and isinstance(new, list) and len(new) >= len(old):
        changed = [i for i in range(len(old)) if old[i] != new[i]]
        if len(changed) * 2 <= len(old):                       # 大部分原样不动：只记改动的几项和末尾追加的
            ops = []
            for i in changed:
                ops += diff(old[i], new[i], (*path, i))
            if len(new) > len(old):
                ops.append({"p": list(path), "append": new[len(old):]})
            return ops
    if old != new:
        return [{"p": list(path), "v": new}]
    return []


def apply(state: Any, ops: list[dict]) -> Any:
    """把操作列表套到 state 上，返回新的状态（不改原对象）。"""
    out = copy.deepcopy(state)
    for op in ops:
        path = op["p"]
        if not path:
            if "append" in op:
                out = out + copy.deepcopy(op["append"])
            else:
                out = copy.deepcopy(op["v"])
            continue
        cur = out
        for k in path[:-1]:
            cur = cur[k]
        last = path[-1]
        if op.get("del"):
            cur.pop(last, None)
        elif "append" in op:
            cur[last] = cur[last] + copy.deepcopy(op["append"])
        else:
            cur[last] = copy.deepcopy(op["v"])
    return out


def reconstruct(events: list[dict], upto_input: int | None = None) -> tuple[dict, int]:
    """从 trace 事件重建世界：state_init 开头，依次套 state_delta，直到第 upto_input 次输入之前（None = 到最后）。
    返回 (状态, 实际套用到的输入序号)。"""
    state: dict = {}
    n_inputs = 0
    for e in events:
        ev = e.get("event")
        if ev == "state_init":
            state = copy.deepcopy(e["state"])
        elif ev == "input":
            n_inputs += 1
        elif ev == "state_delta":
            if upto_input is not None and e.get("before_input", 0) > upto_input:
                break
            state = apply(state, e["ops"])
    return state, n_inputs
