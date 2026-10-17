"""玩家「说自己做过什么」的核对：真假由代码对着玩家的真实足迹判，对方知不知道由代码按对方的所在判，
模型只负责抽取（llm/claim_llm_adapter.py）和把结果用自己的口吻说出来。

对方看到的只有两种结果之一，绝不会看到「真相」本身：
  能核实  对方就在那个地方 / 平常就在那儿 / 当事人就是对方——他知道这件事是真是假
  不能核实  对方无从知道，当作玩家的一面之词（没法验证，也没法证伪）
"""
from __future__ import annotations

from typing import Any, Callable, Optional


class ClaimChecker:
    def __init__(self, god: Any, world: Any, personas: dict, extract: Callable[..., list[dict]]):
        self.god, self.world, self.personas, self.extract = god, world, personas, extract

    # ------------------------------------------------------------ 真相（只有代码知道）

    def _place_of_object(self, oid: str) -> Optional[str]:
        return next((lid for lid in self.world.locations if any(o["id"] == oid for o in self.world.objects(lid))), None)

    def truth(self, kind: str, target: str) -> bool:
        g = self.god
        if kind == "talked":
            return target in g.met
        if kind == "visited":
            return target in g.visited or target == g.location
        if target in self.world.locations:                 # 调查过某个地方：那儿至少有一样东西亲手查看过
            return any(o["id"] in g.examined for o in self.world.objects(target))
        return target in g.examined

    # ------------------------------------------------------------ 对方知不知道

    def could_know(self, npc: str, kind: str, target: str) -> bool:
        if kind == "talked":
            return target == npc                           # 谈没谈过话，只有当事人自己清楚
        place = target if target in self.world.locations else self._place_of_object(target)
        if place is None:
            return False
        phys = self.god.physical
        here = phys.where(npc) if phys is not None else None
        routine = phys.routine(npc) if phys is not None else None
        return place in (here, routine)

    # ------------------------------------------------------------ 入口

    def check(self, npc: str, text: str) -> list[dict]:
        if len((text or "").strip()) < 4:
            return []
        w = self.world
        places = {lid: w.name(lid) for lid in w.locations}
        objects = {o["id"]: o["name"] for lid in w.locations for o in w.objects(lid)}
        people = {nid: p.name for nid, p in self.personas.items()}
        out = []
        for c in self.extract(text, places, objects, people):
            if c["kind"] == "talked" and c["target"] == npc:
                continue                                   # 正在和他说话，不存在「说自己和他谈过」
            known = self.could_know(npc, c["kind"], c["target"])
            real = self.truth(c["kind"], c["target"])
            name = places.get(c["target"]) or objects.get(c["target"]) or people.get(c["target"], "")
            out.append({**c, "name": name, "verifiable": known,
                        "verdict": ("true" if real else "false") if known else "unknown"})
        return out
