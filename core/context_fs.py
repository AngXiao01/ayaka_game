"""NPC 的「文件系统」：按需取用的上下文。

每轮塞进提示的只有眼下这一轮必需的东西（短期记忆 = 这次谈话 + 情境）；其余的放成一棵只读的目录树，
NPC 需要时用 ls / read 去翻，翻到的内容只在这一轮的工具观察里出现，不常驻上下文：

  /地点/<名字>     这个地方是什么样、离这儿多远、你觉得谁在那儿；也包括海外的国度（离这儿有多远）
  /人物/<名字>     你和这个人的关系、你对他/她的看法、你知道的去向
  /我/记忆         你记得的往事（长期记忆，最近的在前）
  /我/今天         你今天的事、日程
  /我/说过的事     你已经对对方说出口的事实、随口编过的设定、撒过的谎（说话要和它们一致）
  /案子/我知道的   别人给你看过、告诉过你的线索（你没见过的，这里没有）
  /稻妻/<条目>     设定资料（和 lookup_lore 同源）

权限即视角：每个 NPC 的目录树由代码按「这个人知道什么」现生成，真相、别人没告诉过他的事一律不在树里。
"""
from __future__ import annotations

from typing import Any, Callable

ROOTS = {"地点": "各个地方（含海外的国度）", "人物": "你认识或听说过的人", "我": "你自己的记忆、今天的事",
         "案子": "你知道的案情", "稻妻": "设定资料"}


class ContextFS:
    def __init__(self, npc_id: str, god: Any, world: Any, personas: dict, lore: Any,
                 memories: Callable[[int], list[str]], canon: Callable[[], list[str]] = lambda: []):
        self.npc, self.god, self.world, self.personas, self.lore = npc_id, god, world, personas, lore
        self.memories = memories
        self.canon = canon

    # ------------------------------------------------------------ 目录

    def ls(self, path: str = "/") -> str:
        parts = [p for p in path.strip().strip("/").split("/") if p]
        if not parts:
            return "\n".join(f"/{k}/  {v}" for k, v in ROOTS.items())
        root = parts[0]
        if root == "地点":
            names = [self.world.name(l) for l in self.world.locations] + list(self.far_places())
            return "\n".join(f"/地点/{n}" for n in names)
        if root == "人物":
            return "\n".join(f"/人物/{p.name}" for nid, p in self.personas.items() if nid != self.npc)
        if root == "我":
            return "/我/记忆\n/我/今天\n/我/说过的事"
        if root == "案子":
            return "/案子/我知道的"
        if root == "稻妻":
            return "\n".join(f"/稻妻/{c.title}" for c in self.lore.chunks)
        return f"（没有这个目录：{path}。可以先 ls /）"

    # ------------------------------------------------------------ 文件

    def read(self, path: str) -> str:
        parts = [p for p in path.strip().strip("/").split("/") if p]
        if len(parts) < 2:
            return self.ls(path)
        root, name = parts[0], "/".join(parts[1:])
        fn = {"地点": self._place, "人物": self._person, "我": self._me, "案子": self._case, "稻妻": self._lore}.get(root)
        return fn(name) if fn else f"（没有这个文件：{path}）"

    def places_in(self, text: str, k: int = 2) -> list[str]:
        """话里提到了哪些地方，就把这个 NPC 对它们的了解附上（距离、他觉得谁在那儿）。代码按名字认，不靠模型自己想起来去翻。"""
        names = [self.world.name(l) for l in self.world.locations] + list(self.far_places())
        hit = [n for n in dict.fromkeys(names) if n and n in (text or "")]
        return [self._place(n).replace("\n", " ")[:160] for n in hit[:k]]

    def far_places(self) -> dict:
        return self.world.cfg.get("far_places", {})

    def _place(self, name: str) -> str:
        far = self.far_places()
        if name in far:
            return f"{name}：{far[name]}"
        lid = next((l for l in self.world.locations if self.world.name(l) == name or l == name), None)
        if lid is None:
            return f"（你不知道「{name}」这个地方。）"
        g, loc = self.god, self.world.loc(lid)
        here = g.physical.where(self.npc) if g.physical is not None else None
        lines = [f"{loc['name']}：{loc['desc']}"]
        if here and here != lid:
            lines.append(f"从你这儿过去，走路约 {self.world.travel_minutes(here, lid)} 分钟。")
        if g.physical is not None:
            seen = [self.personas[n].name for n in self.personas
                    if n != self.npc and (g.physical.belief(self.npc, n) or {}).get("location") == lid]
            if seen:
                lines.append("你觉得在那儿的人：" + "、".join(seen))
        return "\n".join(lines)

    def _person(self, name: str) -> str:
        nid = next((n for n, p in self.personas.items() if p.name == name or name in getattr(p, "aliases", [])), None)
        me = self.personas[self.npc]
        if nid is None:
            rel = me.relations.get(name)
            return f"{name}：{rel}" if rel else f"（你不太了解「{name}」。）"
        p = self.personas[nid]
        lines = [f"{p.name}（{p.title}）"]
        rel = me.relations.get(p.name)
        if rel:
            lines.append(f"你们的关系：{rel}")
        op = (me.opinions or {}).get(p.name)
        if op:
            lines.append(f"你的看法：{op}")
        if self.god.physical is not None:
            b = self.god.physical.belief(self.npc, nid)
            if b and b.get("location"):
                lines.append(f"你以为对方在：{self.world.name(b['location'])}（{b.get('how', '')}）")
        return "\n".join(lines)

    def _me(self, name: str) -> str:
        if name == "记忆":
            return "\n".join(f"- {m}" for m in self.memories(8)) or "（还没有和对方相关的记忆）"
        if name == "今天":
            p = self.personas[self.npc]
            today = (p.days or {}).get(str(self.god.day), "")
            sched = "；".join(f"{'/'.join(s['phases'])}在{self.world.name(s['location'])}" for s in p.schedule)
            return "\n".join(x for x in (today, f"平常的日程：{sched}" if sched else "") if x) or "（今天没什么特别的）"
        if name == "说过的事":
            return "\n".join(f"- {x}" for x in self.said()) or "（还没对对方说过什么要紧的事）"
        return f"（没有这个文件：/我/{name}）"

    def said(self) -> list[str]:
        """这个 NPC 已经说出口、之后必须保持一致的事：达成的信号里说出的事实 + 随口编的设定 + 撒过的谎。"""
        g = self.god
        facts = [s.reveal or s.desc for s in g.quest.signals if s.id in g.achieved and s.npc == self.npc]
        return facts + self.canon() + [f"（谎）{t['said']}——真相：{t['truth']}" for t in g.lies_told(self.npc)]

    def _case(self, name: str) -> str:
        g = self.god
        known = [c for c in g.known.get(self.npc, []) if c in g.quest.clues]
        return "\n".join(f"- {g.quest.clue_name(c)}：{g.quest.clues[c]['desc']}" for c in known) or \
            "（还没有人给你看过、告诉过你什么线索）"

    def _lore(self, name: str) -> str:
        c = next((c for c in self.lore.chunks if c.title == name), None) or next(iter(self.lore.search(name, k=1)), None)
        return f"【{c.title}】{c.text}" if c else "（资料里没有这一条，不要编造）"
