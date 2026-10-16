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
        self._shown_leads: Optional[tuple] = None

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

    # ------------------------------------------------------------ 下一步（给不熟悉设定的玩家：该找谁、去哪、出示什么）

    def whereabouts(self, p: Personality) -> str:
        """某人这一幕的作息，写成「木漏茶室·清晨、午后、黄昏」；人就在眼前时写「就在这里」。"""
        g = self.god
        if g.where(p) == g.location:
            return "就在这里"
        by: dict[str, list[str]] = {}
        for ph in g.phases:
            lid = p.location_at(ph, g.act_id)
            if lid:
                by.setdefault(lid, []).append(ph)
        return "；".join(f"{self.world.name(lid)}·{'、'.join(phs)}" for lid, phs in by.items()) or "不知去向"

    def lead_items(self) -> list[dict]:
        """此刻有路可走的几件事，全由世界状态算出来（不调模型）：
        谁那里有话可问（以及要不要先出示什么）、哪里有东西可查、关系够了会在哪里遇见谁。
        每项 {key, text, here}：key 用来判断「下一步」变没变，here 表示就在眼前。"""
        g, w, q = self.god, self.world, self.god.quest
        if g.resolved or g.ending:
            return []
        out: list[dict] = []
        for s in q.signals:
            if not s.lead or s.event or s.id in g.achieved or g.missing_prereqs(s.id):
                continue
            if g.signal_status(s, s.npc, None, 99, check_turns=False, optimistic=True)[0] != "available":
                continue
            p = self.personas[s.npc]
            text = s.lead
            if s.requires_known_any and not set(s.requires_known_any) & set(g.known.get(s.npc, [])):
                show = [q.clue_name(c) for c in s.requires_known_any if c in g.clues]
                text += f"——交谈时出示「{'」或「'.join(show)}」"
            if s.min_trust and self.rel(s.npc).trust < s.min_trust:
                text += f"（{p.name}还不够信任你：信任 {self.rel(s.npc).trust}/{s.min_trust}，先客气地多聊几句）"
            where = self.whereabouts(p)
            out.append({"key": s.id, "text": f"{text}（{where}）", "here": where == "就在这里"})
        for lid in w.locations:
            if not w.unlocked(lid, g.clues):
                continue
            for o in w.objects(lid):
                if o.get("lead") and self._grantable(lid, o):
                    here = lid == g.location
                    out.append({"key": o["id"], "text": f"{o['lead']}（{'就在这里' if here else w.name(lid)}）",
                                "here": here})
        return out + self._event_leads()

    def leads(self) -> list[str]:
        return [x["text"] for x in self.lead_items()]

    def _event_leads(self) -> list[dict]:
        g, out, waiting = self.god, [], []
        for e in g.quest.events:
            if e.id in g.events_done or (e.acts and g.act_id not in e.acts):
                continue
            aff = self.rel(e.npc).affinity
            name = self.personas[e.npc].name
            if aff >= e.affinity_min:
                out.append({"key": e.id, "here": g.location == e.location and g.phase in e.phases,
                            "text": f"{'、'.join(e.phases)}去{self.world.name(e.location)}走走，也许会遇见{name}"})
            else:
                waiting.append((e.affinity_min, name, aff))
        if waiting and not out:
            need, name, aff = min(waiting)
            out.append({"key": f"affinity:{need}", "here": False,
                        "text": f"和{name}多聊聊、真诚相待，关系更近时也许会有意外的相遇（好感 {aff}/{need}）"})
        return out

    def player_view(self, npc_id: str, event: Optional[str] = None) -> dict:
        """玩家视角：玩家此刻真正知道的事。替玩家想台词的选项 agent 只能看这些——
        NPC 心里的秘密、还没解锁的信号内容（里面写着答案）一律不给，不然选项会替玩家「先知道」。
          clues          线索本
          directions     可以朝哪里问（关键问题按钮的说法 / 该出示什么），不含答案
          known_text     玩家看到过的文字（开场、线索、证词、日志、见过的人、到过的地方……），过滤剧透用
          spoiler_terms  玩家没听说过就不该出现在台词里的词（案情关键词 + 人名）"""
        g, w, q = self.god, self.world, self.god.quest
        directions = []
        for s in q.signals:
            if s.npc != npc_id or s.id in g.achieved or g.missing_prereqs(s.id):
                continue
            if g.signal_status(s, npc_id, event, 99, check_turns=False, optimistic=True)[0] != "available":
                continue
            if s.requires_known_any and not set(s.requires_known_any) & set(g.known.get(npc_id, [])):
                show = [q.clue_name(c) for c in s.requires_known_any if c in g.clues]
                directions.append(f"把「{'」或「'.join(show)}」的事说给对方听，看对方什么反应")
            elif s.ask or s.lead:
                directions.append((s.ask or {}).get("label", "").lstrip("❓🤝💬 ") or s.lead)
        parts = [q.title, *q.primer, *q.how_to_play, *q.prologue, *(k + v for k, v in q.glossary.items()), *directions]
        for a in q.acts:                                   # 到过的幕
            parts += [a.objective, a.hint]
            if a.id == g.act_id:
                break
            parts.append(a.narration)
        parts += [q.clue_name(c) + q.clues[c]["desc"] for c in g.clues]
        parts += [t["said"] for t in g.testimonies] + g.journal + (self.leads() if self.hints else [])
        here = set(self.npcs_here())
        parts += [p.name + p.brief for nid, p in self.personas.items() if g.met.get(nid) or nid in here]
        parts += [w.name(lid) for lid in w.locations if w.unlocked(lid, g.clues)]
        loc = w.loc(g.location)
        parts += [loc["desc"], *loc.get("atmosphere", {}).values(), *(o["name"] for o in w.objects(g.location))]
        parts += [w.examine(lid, o["id"], g.clues)[0] for lid in w.locations for o in w.objects(lid)
                  if o["id"] in g.examined]
        if event and q.event(event):
            parts.append(q.event(event).premise)
        terms = list(dict.fromkeys(q.spoiler_terms + [p.name for p in self.personas.values()]))
        return {"clues": [f"{q.clue_name(c)}：{q.clues[c]['desc']}" for c in g.clues],
                "directions": directions, "known_text": "\n".join(parts), "spoiler_terms": terms}

    def checklist_line(self) -> str:
        items = self.god.checklist()
        if not items:
            return ""
        return "🎯 要查清的事：" + "　".join(f"{'✔' if ok else '✘'}{label}" for label, ok in items)

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
        if self.hints:
            opts.append(PlayerOption("🧭 下一步可以做什么", "查看", kind="view", meta={"view": "leads"}))
        opts.append(PlayerOption("翻看线索本", "查看", kind="view", meta={"view": "clues"}))
        opts.append(PlayerOption("证词本", "查看", kind="view", meta={"view": "testimony"}))
        opts.append(PlayerOption("人物与关系", "查看", kind="view", meta={"view": "people"}))
        opts.append(PlayerOption("调查日志", "查看", kind="view", meta={"view": "journal"}))
        if g.quest.glossary:
            opts.append(PlayerOption("名词速查（不熟悉稻妻的设定时看看）", "查看", kind="view", meta={"view": "glossary"}))
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
            items = self.lead_items()
            keys = (g.act_id, tuple(g.checklist()), tuple(x["key"] for x in items))
            if keys != self._shown_leads:            # 下一步变了才整张列出来，没变就只提眼前的
                self._shown_leads = keys
                if self.checklist_line():
                    lines.append(self.checklist_line())
                lines += [f"🧭 {x['text']}" for x in items]
            else:
                lines += [f"🧭 {x['text']}" for x in items if x["here"]]
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
        if which == "leads":
            rows = ["【下一步】", f"目标：{g.act.objective}"] + ([self.checklist_line()] if self.checklist_line() else [])
            rows += [f"🧭 {x}" for x in self.leads()] or [f"提示：{g.act.hint}"]
            return "\n".join(rows)
        if which == "glossary":
            return "【名词速查】\n" + "\n".join(f"· {k}：{v}" for k, v in g.quest.glossary.items())
        return "【调查日志】\n" + ("\n".join(g.journal) if g.journal else "还没有记录。")
