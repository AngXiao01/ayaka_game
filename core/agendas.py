"""NPC 自己的打算：到了时候、或者条件满足了，他们会自己行动，不等玩家。

一条 agenda（写在 quests/*.json 的 agendas 里）：
  trigger  什么时候生效：flags_all / flags_none / signals_all / signals_none / clues_none / after（{"day", "phase"}）/ delay（条件满足后再过几分钟）
  do       做什么：go_to（地点）+ stay（待多久）+ instant（true = 不算路上的时间，到点就在那儿）、set_flags、tension、rumor、send（{"from","to","text","clues"}）、
           notice（玩家知道的事，不管在不在场）、resolve（主线就此了结）
  witness  玩家恰好在场会看到什么：{"at", "text", "grants", "set_flags"}；at 省略时以 go_to 为准
代码按时间结算，模型不参与；NPC 之后的对话会从世界状态里知道这些事发生过。
"""
from __future__ import annotations

from typing import Any


class Agendas:
    def __init__(self, god: Any, specs: list[dict]):
        self.god = god
        self.specs = specs
        # 每条 agenda 的状态：armed（条件第一次满足的分钟）、done
        self.state: dict[str, dict] = god.agenda_state

    def _ready(self, a: dict) -> bool:
        g, t = self.god, a.get("trigger", {})
        if not set(t.get("flags_all", [])) <= set(g.flags):
            return False
        if t.get("flags_any") and not set(t["flags_any"]) & set(g.flags):
            return False
        if set(t.get("flags_none", [])) & set(g.flags):
            return False
        if not set(t.get("signals_all", [])) <= set(g.achieved):
            return False
        if set(t.get("signals_none", [])) & set(g.achieved):
            return False
        if set(t.get("clues_none", [])) & set(g.clues):
            return False
        if t.get("after"):
            day, ph = t["after"]["day"], g.phases.index(t["after"].get("phase", g.phases[0]))
            if g.minute < g.phase_start(day, ph):
                return False
        return not g.resolved

    def tick(self) -> list[str]:
        g, lines = self.god, []
        for a in self.specs:
            st = self.state.setdefault(a["id"], {})
            if st.get("done"):
                continue
            if not self._ready(a):
                st.pop("armed", None)                 # 条件又不成立了（比如被玩家抢先一步），重新等
                continue
            st.setdefault("armed", g.minute)
            if g.minute - st["armed"] < a.get("trigger", {}).get("delay", 0):
                continue
            st["done"] = g.minute
            lines += self._fire(a)
        return lines

    def _fire(self, a: dict) -> list[str]:
        g, d, phys, lines = self.god, a.get("do", {}), self.god.physical, []
        npc = a.get("npc")
        if d.get("go_to") and phys is not None and npc:
            if npc in phys.party:
                phys.leave(npc)
            phys.set_plan(npc, d["go_to"], g.minute + d.get("stay", 600), "agenda", reason=a["id"], travel=not d.get("instant"))
        w = a.get("witness")
        here = w and g.location == (w.get("at") or d.get("go_to"))
        flags = list(d.get("set_flags", [])) + (list(w.get("set_flags", [])) if here else [])
        flags = [f for f in flags if not (here and f in w.get("cancels", []))]
        for f in flags:
            if f not in g.flags:
                g.flags.append(f)
        if d.get("tension"):
            g.tension = max(0.0, min(1.0, g.tension + d["tension"]))
        if d.get("rumor"):
            g.add_rumor(d["rumor"], {"agenda": a["id"]})
        if d.get("send") and phys is not None:
            s = d["send"]
            phys.send(s.get("from", npc), s["to"], s["text"], kind="report" if s["to"] == "player" else "note",
                      clues=s.get("clues"))
        if here:
            lines.append(f"👁 {w['text']}")
            for c in w.get("grants", []):
                if g.give_clue(c):
                    lines.append(f"    📜 获得线索「{g.quest.clue_name(c)}」")
            g.journal.append(f"{g.time_label()}　{w['text']}")
        elif d.get("notice"):
            lines.append(f"📣 {d['notice']}")
            if not d.get("rumor"):
                g.add_rumor(d["notice"], {"agenda": a["id"]})     # 公开的事：城里的人（NPC）也会听说
        if d.get("resolve") and not g.resolved:
            g.resolved = True
        if g.tracer_log:
            g.tracer_log("agenda", id=a["id"], npc=npc, witnessed=bool(here), flags=flags)
        return lines
