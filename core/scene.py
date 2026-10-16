"""场景 agent：ENV 层里的模型部分。

地图、物件、哪个物件给哪条线索都写在 world.json 里，查看由 Explorer 按规则处理（ENV 层的代码部分）。
原来只能从菜单里点「查看：xx」；现在玩家可以直接写想看什么（「看看桌上的茶杯」）：
- 对得上已有物件 → 照常走 Explorer.examine，线索规则一条不变
- 对不上 → 模型即兴写一两句描写，不能给线索，不能出现案情关键词；
  新确定的细节写进共享的世界设定（和 NPC 即兴编的设定同一个库），之后描写和 NPC 的话都要和它一致
"""
from __future__ import annotations

from typing import Any

from core.god_env import GodEnv
from core.world import World
from llm.base import LLMError
from models.memory import MemoryItem, MemoryStore
from models.retrieval import coverage

NOTHING = "你仔细看了看，没有发现什么特别的。"
FACT_GROUNDING = 0.4


class SceneAgent:
    def __init__(self, god: GodEnv, world: World, explorer: Any, canon: MemoryStore, guard_terms: list[str],
                 adapter: Any = None, mode: str = "llm", tracer: Any = None):
        self.god = god
        self.world = world
        self.explorer = explorer
        self.canon = canon
        self.guard_terms = guard_terms
        self.adapter = adapter
        self.mode = mode
        self.tracer = tracer

    def known_details(self, lid: str, text: str, k: int = 4) -> list[str]:
        name = self.world.name(lid)
        items = self.canon.retrieve(name + text, k=k * 3, now=self.god.total_turns)
        return [m.text for m in items if m.extra.get("location") == lid][:k]

    def look(self, text: str, degraded: bool = False) -> tuple[str, list[str]]:
        g, lid = self.god, self.god.location
        text = text.strip()[:60]
        if self.mode == "off":
            return "想查看什么，从菜单里选吧。", []
        objects = [{"id": o["id"], "name": o["name"]} for o in self.world.objects(lid)]
        used, out = ("rules" if degraded else self.mode), None
        if used == "llm" and self.adapter is not None:
            sc = self.world.scene_context(lid, g.phase)
            try:
                out = self.adapter.describe(text, sc, objects, self.known_details(lid, text), self.guard_terms)
            except LLMError as e:
                used = "rules"
                if self.tracer:
                    self.tracer.log("fallback", where="scene", error=str(e))
        if out is None:
            used = "rules"
            obj = next((o for o in objects if o["name"] in text or text in o["name"]), None)
            out = {"object": obj["id"] if obj else "none", "description": "" if obj else NOTHING, "new_fact": ""}
        kept, dropped, gained = [], [], []
        if out["object"] in {o["id"] for o in objects}:
            desc, gained = self.explorer.examine(out["object"])          # 线索规则不变
        else:
            desc = out["description"] or NOTHING
            hit = [t for t in self.guard_terms if t in desc]
            if hit:
                dropped.append({"text": desc, "reason": f"描写涉及案情（{'、'.join(hit)}）"})
                desc = NOTHING
            fact = out.get("new_fact", "")
            if fact and desc != NOTHING:
                fhit = [t for t in self.guard_terms if t in fact]
                if fhit:
                    dropped.append({"text": fact, "reason": f"涉及案情（{'、'.join(fhit)}）"})
                elif coverage(fact, desc) < FACT_GROUNDING:
                    dropped.append({"text": fact, "reason": "描写里并没有写出这个细节"})
                else:
                    item, merged = self.canon.add(MemoryItem(
                        text=f"{self.world.name(lid)}：{fact}", kind="canon", importance=3,
                        world_time=g.total_turns, extra={"by": "", "about": "world", "location": lid,
                                                         "time": g.time_label()}))
                    kept.append({"fact": item.text, "merged": merged})
        if self.tracer:
            self.tracer.log("scene", mode=used, location=lid, text=text, object=out["object"], description=desc,
                            gained=gained, kept=kept, dropped=dropped)
        return desc, gained
