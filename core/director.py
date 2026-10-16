"""导演 agent：GOD 层里的模型部分。

最初的设计里 GOD agent 由模型决定剧情走向，问题是会被玩家带偏。现在拆成两半：
  GodEnv（代码）  唯一的真相：时间、线索、信号、幕、结局，只有它能改
  Director（这里） 只管节奏：在时段切换时看一眼全局，提议「谁该主动提什么、谁该出现在哪、要不要放传闻、
                  气氛要不要松紧」，每条提议都由代码校验后才执行

三条约束：
1. 只提议，不改状态：动作是白名单（nudge / rumor / move_npc / tension），不能发线索、不能达成信号、不能改结局
2. 不看玩家原话：输入是 view() 生成的结构化视图（GodEnv 状态、关系、作息、代码写的调查日志），
   玩家打的字进不来——想通过对话「带节奏」的路径在信息流上就断了
3. 低频：只在时段切换时跑（一局最多十几次），预算降级或模型出错时退回规则
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from core.god_env import GodEnv
from core.world import World
from llm.base import LLMError
from models.personality import Personality

KINDS = ("nudge", "rumor", "move_npc", "tension")
RUMOR_AFTER_STALL = 2          # 连续几个时段没有进展，才允许放传闻（不然玩家还没卡住就被喂答案）
TENSION_STEP = 0.1             # 导演每次最多把气氛调多少


@dataclass
class Decision:
    action: dict
    ok: bool
    reason: str
    line: str = ""             # 给玩家看的一行（传闻）；空 = 不显示


class Director:
    def __init__(self, god: GodEnv, world: World, personas: dict[str, Personality], explorer: Any,
                 affinity_trust: Callable[[str], tuple[int, int]], adapter: Any = None, mode: str = "llm",
                 max_actions: int = 2, tracer: Any = None):
        self.god = god
        self.world = world
        self.personas = personas
        self.explorer = explorer
        self.rel = affinity_trust
        self.adapter = adapter
        self.mode = mode
        self.max_actions = max(0, max_actions)
        self.tracer = tracer

    # ------------------------------------------------------------ 视图：只给结构化状态，不给玩家原话

    def rumor_targets(self) -> list[dict]:
        """此刻能带来进展的人和地方（代码算出来的，导演只能从这里面挑）。"""
        g, out = self.god, []
        for nid, p in self.personas.items():
            if g.preview_pending(nid) > 0 and g.where(p):
                out.append({"type": "npc", "id": nid, "name": p.name, "location": g.where(p)})
        for lid in self.world.locations:
            if lid != g.location and self.world.unlocked(lid, g.clues) \
                    and any(self.explorer._grantable(lid, o) for o in self.world.objects(lid)):
                out.append({"type": "location", "id": lid, "name": self.world.name(lid)})
        return out

    def view(self) -> dict:
        g = self.god
        npcs = []
        for nid, p in self.personas.items():
            aff, trust = self.rel(nid)
            sigs = []
            for s in g.quest.signals:
                st, why = g.signal_status(s, nid, None, 99, check_turns=False)
                if st != "hidden":
                    sigs.append({"id": s.id, "desc": s.desc, "status": st, "why": why})
            npcs.append({"id": nid, "name": p.name, "role": p.role, "location_now": g.where(p),
                         "schedule_locations": sorted({x["location"] for x in p.schedule}),
                         "met": g.met.get(nid, 0), "affinity": aff, "trust": trust,
                         "promising": g.preview_pending(nid), "signals": sigs})
        return {
            "time": g.time_label(), "slots_left": g.slots_left(), "deadline": g.quest.deadline["label"],
            "act": {"id": g.act_id, "title": g.act.title, "objective": g.act.objective},
            "tension": g.tension, "clues": [g.quest.clue_name(c) for c in g.clues], "clue_total": len(g.quest.clues),
            "slots_since_progress": g.slots_since_progress(), "player_location": g.location,
            "locations": {lid: self.world.name(lid) for lid in self.world.locations
                          if self.world.unlocked(lid, g.clues)},
            "npcs": npcs, "rumor_targets": self.rumor_targets(),
            "recent_journal": g.journal[-5:], "active_nudges": dict(g.nudges),
        }

    # ------------------------------------------------------------ 运行

    def run(self, degraded: bool = False) -> list[str]:
        """每次地点层动作之后调用；只有时段切换了才真的跑。返回要给玩家看的行（传闻）。"""
        g = self.god
        if g.director_slot is None:            # 开局 / 读档：只记下基准，不跑
            g.director_slot = g.slot_index
            return []
        if g.slot_index == g.director_slot:
            return []
        g.director_slot = g.slot_index
        g.expire_overrides()
        if self.mode == "off" or self.max_actions == 0:
            return []
        view = self.view()
        used = "rules" if self.mode == "rules" or degraded or self.adapter is None else "llm"
        note, error = "", None
        if used == "llm":
            try:
                out = self.adapter.propose(view, self.max_actions)
                proposed, note = out["actions"], out.get("note", "")
            except LLMError as e:
                error, used = f"{type(e).__name__}: {e}", "rules"
                if self.tracer:
                    self.tracer.log("fallback", where="director", error=error)
        if used == "rules":
            proposed = self.propose_rules(view)
        decisions = self.apply_all(proposed)
        if self.tracer:
            self.tracer.log("director", mode=used, time=g.time_label(), note=note, error=error,
                            proposed=proposed,
                            applied=[d.action for d in decisions if d.ok],
                            rejected=[{**d.action, "reason": d.reason} for d in decisions if not d.ok])
        return [d.line for d in decisions if d.ok and d.line]

    def propose_rules(self, view: dict) -> list[dict]:
        """不调模型的基线：玩家卡住就放传闻；身边有能谈的人就让他主动开口。"""
        out: list[dict] = []
        if view["slots_since_progress"] >= RUMOR_AFTER_STALL and view["rumor_targets"]:
            t = view["rumor_targets"][0]
            out.append({"kind": "rumor", "npc": t["id"] if t["type"] == "npc" else "",
                        "location": t["id"] if t["type"] == "location" else "", "reason": "玩家卡住了"})
        for n in view["npcs"]:
            if n["location_now"] == view["player_location"] and n["id"] not in view["active_nudges"]:
                avail = [s for s in n["signals"] if s["status"] == "available"]
                if avail:
                    out.append({"kind": "nudge", "npc": n["id"], "signal": avail[0]["id"], "reason": "人就在身边"})
                    break
        return out

    # ------------------------------------------------------------ 校验与执行（代码说了算）

    def apply_all(self, actions: list[dict]) -> list[Decision]:
        decisions, used_kinds = [], set()
        for a in actions:
            if len([d for d in decisions if d.ok]) >= self.max_actions:
                decisions.append(Decision(a, False, f"超过每次 {self.max_actions} 个动作的上限"))
                continue
            if a.get("kind") in used_kinds:
                decisions.append(Decision(a, False, "同类动作每次只做一个"))
                continue
            d = self.apply(a)
            if d.ok:
                used_kinds.add(a["kind"])
            decisions.append(d)
        return decisions

    def apply(self, a: dict) -> Decision:
        kind = a.get("kind")
        if kind not in KINDS:
            return Decision(a, False, f"未知动作 {kind}")
        return getattr(self, f"_apply_{kind}")(a)

    def _apply_nudge(self, a: dict) -> Decision:
        g, nid, sid = self.god, a.get("npc", ""), a.get("signal", "")
        if nid not in self.personas:
            return Decision(a, False, f"没有这个人：{nid}")
        s = g.quest.signal(sid)
        if s is None:
            return Decision(a, False, f"没有这个信号：{sid}")
        if s.npc != nid:
            return Decision(a, False, f"{sid} 不是 {nid} 能说的事")
        if s.id in g.achieved:
            return Decision(a, False, "这件事已经发生过了")
        if g.signal_status(s, nid, None, 99, check_turns=False, optimistic=True)[0] != "available":
            return Decision(a, False, "眼下还没有让这件事发生的途径")
        g.set_nudge(nid, sid)
        return Decision(a, True, "ok")

    def _apply_rumor(self, a: dict) -> Decision:
        g = self.god
        if g.slots_since_progress() < RUMOR_AFTER_STALL:
            return Decision(a, False, f"玩家还没卡住（距上次进展才 {g.slots_since_progress()} 个时段）")
        targets = self.rumor_targets()
        nid, lid = a.get("npc") or "", a.get("location") or ""
        hit = next((t for t in targets if (nid and t["type"] == "npc" and t["id"] == nid)
                    or (lid and t["type"] == "location" and t["id"] == lid)), None)
        if hit is None:
            return Decision(a, False, "传闻指向的人或地方此刻并不能带来进展")
        target = {"type": hit["type"], "id": hit["id"]}
        if any(r["target"] == target and r.get("slot", 0) >= g.last_progress_slot for r in g.rumors):
            return Decision(a, False, "同一个方向的传闻已经放过，玩家之后还没有进展，不重复")
        if hit["type"] == "npc":
            text = f"你隐约听人说起，{hit['name']}这几天像是有话想找人说（大概在{self.world.name(hit['location'])}）。"
        else:
            text = f"你听路人议论，{hit['name']}那边最近有些不寻常的动静。"
        g.add_rumor(text, target)
        return Decision(a, True, "ok", line=f"💬 {text}")

    def _apply_move_npc(self, a: dict) -> Decision:
        g, nid, lid = self.god, a.get("npc", ""), a.get("location", "")
        p = self.personas.get(nid)
        if p is None:
            return Decision(a, False, f"没有这个人：{nid}")
        if lid not in self.world.locations:
            return Decision(a, False, f"没有这个地方：{lid}")
        if not self.world.unlocked(lid, g.clues):
            return Decision(a, False, "那个地方玩家还去不了")
        if lid not in {x["location"] for x in p.schedule}:
            return Decision(a, False, f"{p.name}平时不会去{self.world.name(lid)}")
        g.set_override(nid, lid)
        return Decision(a, True, "ok")

    def _apply_tension(self, a: dict) -> Decision:
        try:
            delta = float(a.get("delta", 0))
        except (TypeError, ValueError):
            return Decision(a, False, "delta 不是数字")
        if delta == 0:
            return Decision(a, False, "delta 为 0")
        clamped = max(-TENSION_STEP, min(TENSION_STEP, delta))
        self.god.adjust_tension(clamped)
        return Decision({**a, "delta": clamped}, True, "ok" if clamped == delta else f"幅度截到 {clamped:+.2f}")
