"""World：地图与场景描写。地点、出口、旅行耗时、可查看的物件、需要线索才解锁的地点。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class World:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.phases: list[str] = cfg["phases"]
        self.start: dict = cfg["start"]
        self.turns_per_phase: int = cfg.get("turns_per_phase", 6)
        self.locations: dict[str, dict] = cfg["locations"]

    @classmethod
    def load(cls, path: Path) -> "World":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def loc(self, lid: str) -> dict:
        if lid not in self.locations:
            raise KeyError(f"未定义的地点：{lid}")
        return self.locations[lid]

    def name(self, lid: str) -> str:
        return self.loc(lid)["name"]

    def unlocked(self, lid: str, clues: list[str]) -> bool:
        need = self.loc(lid).get("locked_until_any_clue")
        return not need or bool(set(need) & set(clues))

    def travel_cost(self, a: str, b: str) -> int:
        return 0 if self.loc(a)["area"] == self.loc(b)["area"] else 1

    def exits(self, lid: str) -> list[str]:
        return list(self.loc(lid).get("exits", []))

    def objects(self, lid: str) -> list[dict]:
        return list(self.loc(lid).get("objects", []))

    def examine(self, lid: str, obj_id: str, clues: list[str]) -> tuple[str, Optional[str]]:
        """返回 (描写, 可获得的线索 id)。条件不满足时给 locked_desc，不给线索。"""
        obj = next(o for o in self.objects(lid) if o["id"] == obj_id)
        if obj.get("requires_clues") and not set(obj["requires_clues"]) <= set(clues):
            return obj.get("locked_desc", obj["desc"]), None
        return obj["desc"], obj.get("clue")

    def scene_context(self, lid: str, phase: str) -> dict:
        sc = self.loc(lid)
        return {"id": lid, "name": sc["name"], "desc": sc["desc"],
                "atmosphere": sc.get("atmosphere", {}).get(phase, ""),
                "notable_elements": sc.get("notable_elements", []),
                "public": bool(sc.get("public", False))}

    def describe(self, lid: str, phase: str, tension: float = 0.3) -> str:
        ctx = self.scene_context(lid, phase)
        text = f"【{ctx['name']}】\n{ctx['desc']}"
        if ctx["atmosphere"]:
            text += "\n" + ctx["atmosphere"]
        if tension >= 0.75:
            text += "\n空气里隐约有种说不清的紧绷感。"
        return text
