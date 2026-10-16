"""地点层的行动：在哪、能和谁说话、能查看什么、能去哪、等待、查看线索本 / 人物 / 日志。

这些动作都是确定性的（不调模型），只有「交谈」会进入 DialogueSession。
菜单项带 meta，给提示标记和自动玩家用。
"""
from __future__ import annotations

from typing import Callable, Optional

from core.god_env import GodEnv
from core.player_option_generator import PlayerOption
from core.world import World
from models.personality import Personality
from models.relationship import Relationship


class Explorer:
    def __init__(self, god: GodEnv, world: World, personas: dict[str, Personality],
                 relationships: Callable[[str], Relationship], out: Callable[[str], None] = print,
                 hints: bool = True):
        self.god = god
        self.world = world
        self.personas = personas
        self.rel = relationships
        self.out = out
        self.hints = hints

    # ------------------------------------------------------------ 查询

    def npcs_at(self, lid: str, phase: str) -> list[str]:
        return [pid for pid, p in self.personas.items() if self.god.where(p, phase) == lid]

    def npcs_here(self) -> list[str]:
        return self.npcs_at(self.god.location, self.god.phase)

    def _grantable(self, lid: str, obj: dict) -> bool:
        _, clue = self.world.examine(lid, obj["id"], self.god.clues)
        return bool(clue) and clue not in self.god.clues

    def _useful_here(self, lid: str, phase: str, cost: int) -> bool:
        if any(self.god.preview_pending(n) for n in self.npcs_at(lid, phase)):
            return True
        if any(self._grantable(lid, o) for o in self.world.objects(lid)):
            return True
        return cost == 0 and bool(self.god.available_events(lid, lambda n: self.rel(n).affinity))

    def _distance_to_progress(self, dest: str, max_hops: int = 3) -> Optional[tuple[int, int]]:
        """从 dest 出发沿出口往外搜：到最近一个「有进展」的地方要花 (时段, 步数)，没有则 None。"""
        g, w = self.god, self.world
        frontier = [(w.travel_cost(g.location, dest), 1, dest)]
        seen = {g.location}
        while frontier:
            frontier.sort()
            cost, hops, lid = frontier.pop(0)
            if lid in seen:
                continue
            seen.add(lid)
            _, phase = g.phase_after(cost)
            if self._useful_here(lid, phase, cost):
                return cost, hops
            if hops < max_hops:
                frontier += [(cost + w.travel_cost(lid, n), hops + 1, n)
                             for n in w.exits(lid) if n not in seen and w.unlocked(n, g.clues)]
        return None

    def _mark(self, text: str, on: bool, tip: str) -> str:
        return f"{text}  💡{tip}" if (self.hints and on) else text

    # ------------------------------------------------------------ 菜单

    def menu(self) -> list[PlayerOption]:
        g, w, here = self.god, self.world, self.god.location
        opts: list[PlayerOption] = []
        for ev in g.available_events(here, lambda n: self.rel(n).affinity):
            opts.append(PlayerOption(f"✨【{ev.title}】", "事件", kind="event", meta={"event": ev.id, "npc": ev.npc}))
        for nid in self.npcs_here():
            p = self.personas[nid]
            pending = g.preview_pending(nid)
            opts.append(PlayerOption(self._mark(f"和{p.name}交谈（{p.title}）", pending > 0, "似乎有话可谈"),
                                     "交谈", kind="talk", meta={"npc": nid, "pending": pending,
                                                               "companion": p.role == "companion"}))
        for o in w.objects(here):
            grant = self._grantable(here, o)
            seen = o["id"] in g.examined
            opts.append(PlayerOption(self._mark(f"查看：{o['name']}{'（看过）' if seen and not grant else ''}",
                                                grant, "也许有发现"),
                                     "查看", kind="examine", meta={"obj": o["id"], "clue": grant, "seen": seen}))
        dests = [d for d in w.exits(here) if w.unlocked(d, g.clues)]
        dist = {d: self._distance_to_progress(d) for d in dests}
        best = min((v for v in dist.values() if v), default=None)
        for dest in dests:
            cost = w.travel_cost(here, dest)
            useful = dist[dest] is not None and dist[dest] == best      # 只标最近的方向
            label = f"前往{w.name(dest)}（{'约一个时段' if cost else '就在附近'}）"
            opts.append(PlayerOption(self._mark(label, useful, "那边也许有进展"), "移动", kind="move",
                                     meta={"dest": dest, "cost": cost, "useful": useful, "distance": dist[dest]}))
        opts.append(PlayerOption("在这里待一会儿（时间推进一个时段）", "等待", kind="wait"))
        opts.append(PlayerOption("翻看线索本", "查看", kind="view", meta={"view": "clues"}))
        opts.append(PlayerOption("证词本", "查看", kind="view", meta={"view": "testimony"}))
        opts.append(PlayerOption("人物与关系", "查看", kind="view", meta={"view": "people"}))
        opts.append(PlayerOption("调查日志", "查看", kind="view", meta={"view": "journal"}))
        opts.append(PlayerOption("保存并退出", "退出", kind="quit"))
        return opts

    def panel(self) -> str:
        g, w = self.god, self.world
        lines = ["", "─" * 40, g.status_line(), "",
                 w.describe(g.location, g.phase, g.tension)]
        people = [f"{self.personas[n].name}（{self.personas[n].brief}）" for n in self.npcs_here()]
        lines.append("这里的人：" + ("；".join(people) if people else "没有能说上话的人"))
        if self.hints:
            lines.append(f"提示：{g.act.hint}")
        return "\n".join(lines)

    # ------------------------------------------------------------ 动作

    def move(self, dest: str) -> str:
        cost = self.world.travel_cost(self.god.location, dest)
        self.god.location = dest
        self.god.advance(cost)
        return f"你来到了{self.world.name(dest)}。" + ("路上花了些时间。" if cost else "")

    def examine(self, obj_id: str) -> tuple[str, list[str]]:
        g = self.god
        desc, clue = self.world.examine(g.location, obj_id, g.clues)
        if obj_id not in g.examined:
            g.examined.append(obj_id)
        gained = []
        if clue and g.give_clue(clue):
            gained.append(clue)
            name = next(o["name"] for o in self.world.objects(g.location) if o["id"] == obj_id)
            g.journal.append(f"{g.time_label()}·{self.world.name(g.location)}　查看{name}，发现线索「{g.quest.clue_name(clue)}」")
        return desc, gained

    def wait(self) -> str:
        self.god.advance(1)
        return f"你在{self.world.name(self.god.location)}待了一阵。"

    def view(self, which: str) -> str:
        g = self.god
        if which == "clues":
            if not g.clues:
                return "线索本还是空的。"
            return "【线索本】\n" + "\n".join(f"· {g.quest.clue_name(c)}：{g.quest.clues[c]['desc']}" for c in g.clues)
        if which == "testimony":
            if not g.testimonies:
                return "证词本还是空的。"
            rows = ["【证词本】（谁说了什么——不一定都是真话）"]
            for t in g.testimonies:
                rows.append(f"· {t['time']} {self.personas[t['npc']].name}：「{t['said']}」" + ("  ✗ 已被拆穿" if t["caught"] else ""))
            return "\n".join(rows)
        if which == "people":
            rows = ["【人物与关系】"]
            for nid, p in self.personas.items():
                if not g.met.get(nid) and p.role != "companion":
                    continue
                r = self.rel(nid)
                where = g.where(p)
                rows.append(f"· {p.name}（{p.title}）{p.brief}\n  关系：{r.tier()[0]}（好感 {r.affinity}，信任 {r.trust}）"
                            f"｜此刻大概在：{self.world.name(where) if where else '不知去向'}")
            return "\n".join(rows)
        return "【调查日志】\n" + ("\n".join(g.journal) if g.journal else "还没有记录。")
