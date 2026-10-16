"""PhysicalWorld：物理世界——谁在哪、跟谁在一起、要去哪；人和人之间带话；每个人只知道自己看到的和听说的。

真相和认知分开：
  真相  where(npc)：作息（人物卡 schedule）+ 计划（和玩家同行、回家路上、自己要去的地方、导演的安排）。
        只存「计划」，不逐分钟模拟；计划到点自动失效，回到作息。GOD（GodEnv / 导演）看真相。
  认知  belief(observer, target)：observer 亲眼看到的（同处一地时自动记下）、别人带话告诉的；
        都没有、或者已经过了一个时段，就按平常的作息去猜——猜错是正常的。NPC 和玩家只看认知。

同行：玩家邀请、代码按好感和对方此刻有没有空判定（人物卡 outing），答应了就跟着玩家走；
     出门前告诉某些人（outing.tell），有的人出门谁也不说（outing.secret）；天色到了自己回去，回去也要走路时间。
带话：口信要有人跑腿，送达时间按两人相隔多远算；同处一地立刻送到。送到后进对方的记忆和下次对话的情境，
     只传话，不替玩家解锁证据（证据仍要玩家自己出示）。
"""
from __future__ import annotations

from typing import Any, Callable, Optional

PLAYER = "player"
MAIL_KEEP = 200


class PhysicalWorld:
    def __init__(self, world: Any, personas: dict, god: Any, state: Optional[dict] = None,
                 player_name: str = "旅行者"):
        self.world = world
        self.personas = personas
        self.god = god
        self.player_name = player_name
        s = state or {}
        self.party: list[str] = s.get("party", [])                       # 和玩家同行的人
        self.party_info: dict[str, dict] = s.get("party_info", {})       # npc -> {since, from, dest}
        self.plans: dict[str, dict] = s.get("plans", {})                 # npc -> {kind, location, arrive, until, reason}
        self.sightings: dict[str, dict[str, dict]] = s.get("sightings", {})   # observer -> target -> 最近一次认知
        self.mail: list[dict] = s.get("mail", [])
        self.inbox: dict[str, list[str]] = s.get("inbox", {})            # 送到了、还没在对话里提起的口信 id
        self.errands: list[dict] = s.get("errands", [])                  # 派出去查事的人：到地方、查完、托人回话
        self.on_deliver: Optional[Callable[[dict], None]] = None

    # ------------------------------------------------------------ 名字

    def name(self, who: str) -> str:
        return self.player_name if who == PLAYER else self.personas[who].name

    def aliases(self, npc: str) -> list[str]:
        p = self.personas[npc]
        out = [p.name] + ([p.name[-2:]] if len(p.name) >= 3 else []) + list(getattr(p, "aliases", []) or [])
        return list(dict.fromkeys(a for a in out if a))

    def resolve(self, text: str) -> Optional[str]:
        """「托马」「绫华」「大小姐」→ npc id。"""
        text = (text or "").strip()
        if text in self.personas:
            return text
        return next((nid for nid in self.personas if any(a == text for a in self.aliases(nid))), None) \
            or next((nid for nid in self.personas if any(a in text for a in self.aliases(nid))), None)

    def mentioned(self, text: str) -> list[str]:
        return [nid for nid in self.personas if any(a in (text or "") for a in self.aliases(nid))]

    # ------------------------------------------------------------ 真相：谁在哪

    def routine(self, npc: str, phase: Optional[str] = None) -> Optional[str]:
        p = self.personas[npc]
        if set(getattr(p, "absent_when_flags", []) or []) & set(self.god.flags):
            return None
        return p.location_at(phase or self.god.phase, self.god.act_id)

    def where(self, npc: str, phase: Optional[str] = None) -> Optional[str]:
        """npc 此刻在哪（None = 在路上 / 不在任何能去的地方）。问的是别的时段时只能按作息估。"""
        g = self.god
        if phase is not None and phase != g.phase:
            return self.routine(npc, phase)
        if npc in self.party:
            return g.location
        plan = self.plans.get(npc)
        if plan and g.minute < plan["until"]:
            return None if g.minute < plan.get("arrive", 0) else plan.get("location")
        return self.routine(npc)

    def at(self, lid: str) -> list[str]:
        return [nid for nid in self.personas if self.where(nid) == lid]

    def set_plan(self, npc: str, location: Optional[str], until: int, kind: str, reason: str = "",
                 travel: bool = True) -> dict:
        frm = self.where(npc)
        arrive = self.god.minute + (self.world.travel_minutes(frm, location) if travel and location else 0)
        self.plans[npc] = {"kind": kind, "location": location, "arrive": arrive, "until": max(until, arrive),
                           "reason": reason}
        return self.plans[npc]

    # ------------------------------------------------------------ 同行

    def can_join(self, npc: str, affinity: int) -> tuple[bool, str]:
        """玩家邀请 npc 同行，此刻答不答应——代码说了算，模型只负责用自己的语气说出来。"""
        g, o = self.god, getattr(self.personas[npc], "outing", None) or {}
        if not o:
            return False, "你有自己的活要干，走不开"
        if npc in self.party:
            return False, "你们已经在一起了"
        if self.where(npc) != g.location:
            return False, "你不在这里"
        acts = o.get("acts")
        if acts and g.act_id not in acts and not (g.resolved and "after" in acts):
            return False, o.get("busy") or "眼下有事走不开"
        if affinity < o.get("min_affinity", 0):
            return False, o.get("decline_low") or "和对方还不够熟，不方便一起出门"
        if o.get("phases") and g.phase not in o["phases"]:
            return False, o.get("busy") or "这会儿有事走不开"
        return True, ""

    def join(self, npc: str, dest: str = "") -> None:
        g, o = self.god, getattr(self.personas[npc], "outing", None) or {}
        if npc not in self.party:
            self.party.append(npc)
        self.party_info[npc] = {"since": g.minute, "from": g.location, "dest": dest}
        self.plans.pop(npc, None)
        if not o.get("secret"):
            where_to = f"，去{dest}" if dest else ""
            for c in o.get("tell", []):
                if c in self.personas and c != npc:
                    self.send(npc, c, f"我陪{self.player_name}出去一趟{where_to}。", kind="whereabouts",
                              about=npc, location=None)

    def leave(self, npc: str) -> Optional[str]:
        """离开队伍，走回作息里该在的地方（路上要花时间）。返回要回去的地方。"""
        if npc not in self.party:
            return None
        self.party.remove(npc)
        self.party_info.pop(npc, None)
        home = self.routine(npc)
        arrive = self.god.minute + self.world.travel_minutes(self.god.location, home)
        self.plans[npc] = {"kind": "return", "location": None, "arrive": arrive, "until": arrive, "reason": "回去"}
        note = f"回{self.world.name(home)}去了" if home else "回去了"
        self._record(PLAYER, npc, None, "left", note=note)
        return home

    def outing_note(self, npc: str) -> str:
        info = self.party_info.get(npc)
        if npc not in self.party or not info:
            return ""
        o = getattr(self.personas[npc], "outing", None) or {}
        frm = self.world.name(info["from"]) if info.get("from") else "别处"
        dest = f"，说好去{info['dest']}" if info.get("dest") else ""
        return (f"你正和{self.player_name}一起在外面（{self.god.clock_label(info['since'])}从{frm}出来{dest}）。"
                + (o.get("note") or ""))

    # ------------------------------------------------------------ 认知

    def _record(self, observer: str, target: str, location: Optional[str], how: str, by: str = "",
                note: str = "", t: Optional[int] = None) -> None:
        self.sightings.setdefault(observer, {})[target] = {
            "location": location, "t": self.god.minute if t is None else t, "how": how, "by": by, "note": note}

    def observe(self) -> None:
        """同处一地的人互相看见（包括玩家）。"""
        locs = {nid: self.where(nid) for nid in self.personas}
        locs[PLAYER] = self.god.location
        for a, la in locs.items():
            for b, lb in locs.items():
                if a != b and la and la == lb:
                    self._record(a, b, la, "seen")

    def depart(self, dest: str) -> None:
        """玩家（和同行的人）要离开这里：留在原地的人看见他们往哪儿去了。"""
        here, name = self.god.location, self.world.name(dest)
        movers = [PLAYER] + list(self.party)
        for w in self.personas:
            if w not in self.party and self.where(w) == here:
                for m in movers:
                    note = f"往{name}去了" if m == PLAYER else f"和{self.player_name}一起往{name}去了"
                    self._record(w, m, None, "left", note=note)

    def belief(self, observer: str, target: str) -> dict:
        """observer 以为 target 在哪。how：here 就在眼前 / with 同行 / seen 刚看见 / told 听说 / left 刚走 /
        usual 按平常的作息猜 / stale 很久以前的消息 / unknown 不知道。"""
        g = self.god
        mine = g.location if observer == PLAYER else self.where(observer)
        theirs = g.location if target == PLAYER else self.where(target)
        if PLAYER in (observer, target) and (target in self.party or observer in self.party):
            return {"how": "with", "location": g.location}
        if mine and mine == theirs:
            return {"how": "here", "location": mine}
        rec = self.sightings.get(observer, {}).get(target)
        if rec and rec["t"] >= g.phase_start(g.day, g.phase_idx):
            return rec
        usual = None if target == PLAYER else self.routine(target)
        if usual:
            return {"how": "usual", "location": usual}
        return {**rec, "how": "stale"} if rec else {"how": "unknown", "location": None}

    def describe(self, b: dict) -> str:
        loc = self.world.name(b["location"]) if b.get("location") else ""
        when = self.god.clock_label(b["t"]) if b.get("t") is not None else ""
        how = b["how"]
        if how == "here":
            return "就在这里"
        if how == "with":
            return "和你同行"
        if how == "seen":
            return f"{when}在{loc}见过"
        if how == "told":
            by = self.name(b["by"]) if b.get("by") else "别人"
            return f"听{by}说（{when}）：{b.get('note') or ('在' + loc)}"
        if how == "left":
            return f"{when}{b.get('note') or '走了'}"
        if how == "usual":
            return f"平常这时候在{loc}"
        if how == "stale":
            return f"上次知道是在{loc or '外面'}（{when}）"
        return "不知去向"

    # ------------------------------------------------------------ 带话

    def send(self, frm: str, to: str, text: str, kind: str = "note", about: Optional[str] = None,
             location: Optional[str] = None, clues: Optional[list[str]] = None) -> dict:
        g = self.god
        src = g.location if frm == PLAYER else self.where(frm)
        dst = g.location if to == PLAYER else self.where(to)
        delay = 0 if src and src == dst else self.world.travel_minutes(src, dst)
        msg = {"id": f"m{len(self.mail) + 1}-{g.minute}", "from": frm, "to": to, "kind": kind, "text": text[:80],
               "about": about, "location": location, "sent": g.minute, "deliver": g.minute + delay,
               "delivered": False, "clues": list(clues or [])}
        self.mail.append(msg)
        self.mail = self.mail[-MAIL_KEEP:]
        if delay == 0:
            self._deliver(msg)
        return msg

    def _deliver(self, msg: dict) -> None:
        msg["delivered"] = True
        if msg.get("about"):
            self._record(msg["to"], msg["about"], msg.get("location"), "told", by=msg["from"], note=msg["text"],
                         t=msg["sent"])
        self.inbox.setdefault(msg["to"], []).append(msg["id"])
        if self.on_deliver:
            self.on_deliver(msg)

    def peek_inbox(self, npc: str) -> list[dict]:
        ids = set(self.inbox.get(npc, []))
        return [m for m in self.mail if m["id"] in ids]

    def take_inbox(self, npc: str) -> list[dict]:
        msgs = self.peek_inbox(npc)
        self.inbox.pop(npc, None)
        return msgs

    # ------------------------------------------------------------ 时间推进时结算

    def dispatch(self, agent: str, location: str, search_minutes: int, report_to: list[str], text: str,
                 clues: list[str], tag: str) -> dict:
        """派 agent 去 location 查一件事：走过去、查 search_minutes 分钟，然后托人把结果带给 report_to。"""
        g = self.god
        plan = self.set_plan(agent, location, 0, "errand", reason=tag)
        done = plan["arrive"] + search_minutes
        plan["until"] = done + 30                       # 查完再待一会儿才回去
        e = {"agent": agent, "location": location, "done": done, "report_to": report_to, "text": text,
             "clues": clues, "tag": tag, "reported": False, "sent": g.minute}
        self.errands.append(e)
        return e

    def dispatch_chain(self, agent: str, stops: list[dict], report_to: list[str]) -> list[dict]:
        """派 agent 依次跑几个地方查事（stops: {location, search_minutes, text, clues, tag}），每查完一处就托人回话。"""
        if len(stops) == 1:
            s = stops[0]
            return [self.dispatch(agent, s["location"], s["search_minutes"], report_to, s["text"], s["clues"], s["tag"])]
        g, here, t, out = self.god, self.where(agent), self.god.minute, []
        for s in stops:
            t += self.world.travel_minutes(here, s["location"]) + s["search_minutes"]
            here = s["location"]
            e = {"agent": agent, "location": here, "done": t, "report_to": report_to, "text": s["text"],
                 "clues": s["clues"], "tag": s["tag"], "reported": False, "sent": g.minute}
            self.errands.append(e)
            out.append(e)
        self.plans[agent] = {"kind": "errand", "location": here, "arrive": t - stops[-1]["search_minutes"],
                             "until": t + 30, "reason": "、".join(s["tag"] for s in stops)}
        return out

    def tick(self) -> list[str]:
        """时间往前走之后调用：过期的计划失效、同行者到点告辞、同处一地的人互相看见、到点的口信送达。
        返回玩家看得到的事。"""
        g, lines = self.god, []
        self.plans = {k: v for k, v in self.plans.items() if v["until"] > g.minute}
        for npc in list(self.party):
            o = getattr(self.personas[npc], "outing", None) or {}
            if o.get("phases") and g.phase not in o["phases"]:
                lines.append(f"（{self.name(npc)}看了看天色：「{o.get('farewell') or '时候不早了，我得回去了。'}」）")
                self.leave(npc)
        self.observe()
        for e in self.errands:
            if not e["reported"] and e["done"] <= g.minute:
                e["reported"] = True
                for to in e["report_to"]:
                    self.send(e["agent"], to, e["text"], kind="report",
                              clues=e["clues"] if to == PLAYER else None)
        for m in self.mail:
            if not m["delivered"] and m["deliver"] <= g.minute:
                self._deliver(m)
        return lines

    # ------------------------------------------------------------ 给 NPC / 玩家看的视图

    def npc_view(self, npc: str, text: str = "") -> dict:
        """某个 NPC 这一轮该知道的：身边有谁、是不是在和玩家同行、对方提到的人在哪（它以为的）、别人带来的话。"""
        g = self.god
        here = self.where(npc)
        present = [f"{self.name(o)}（{self.personas[o].brief}）" for o in self.personas
                   if o != npc and here and self.where(o) == here]
        targets = [t for t in self.mentioned(text) if t != npc]
        whereabouts = [f"{self.name(t)}：{self.describe(self.belief(npc, t))}" for t in targets]
        inbox = [f"{self.name(m['from'])}（{g.clock_label(m['sent'])}）：{m['text']}" for m in self.peek_inbox(npc)]
        return {"present": present, "outing": self.outing_note(npc), "whereabouts": whereabouts, "inbox": inbox}

    def player_belief(self, npc: str) -> str:
        return self.describe(self.belief(PLAYER, npc))

    def to_dict(self) -> dict:
        return {"party": self.party, "party_info": self.party_info, "plans": self.plans,
                "sightings": self.sightings, "mail": self.mail, "inbox": self.inbox, "errands": self.errands}
