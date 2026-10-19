"""故事内容的静态校验：不跑游戏，只看内容数据本身有没有漏洞。

灵感来自 Holmes：它先生成一条时间线，再从时间线导出每个嫌疑人“经历过什么”，所以每个人知道的事都有来源。
我们的剧情是手写的，没有这条线，所以改成**事后核对**：

一、可达性（乐观地推演一遍“玩家最顺的情况”）
  从开局的地点、没有线索、第一幕出发，反复问：现在能去哪、能找到谁、哪些信号可能达成、能拿到哪些线索、哪一幕能结束。
  推到不再有新东西为止。推不到的就是漏洞：永远拿不到的线索、永远达成不了的信号、永远走不进的地点和幕、见不到的人。
  “乐观”指：信任、好感、时间够不够都不管，只看**结构上有没有路**——结构上没路的，玩家怎么玩都没用。

二、在场（线索可以写 witness：谁、哪个时段、在哪里亲眼见到的）
  作证的人平时那个时段是不是真的在那个地方？不在就必须写明为什么（witness.off_routine，比如“偷偷溜出去的”）。
  再核对：这条线索是不是由 witness.npc 自己说出口的（给出线索的信号挂在同一个人身上）。

三、引用：秘密、谎话、会毁掉的证据、议程里引用的线索 / 信号 / 人 / 地点都存在。

结果分两级：error（结构上走不通，必须改）和 warn（说不通，要作者决定怎么圆）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Finding:
    level: str            # "error" | "warn"
    code: str
    msg: str

    def __str__(self) -> str:
        return f"{'✘' if self.level == 'error' else '！'} [{self.code}] {self.msg}"


def lint(quest: Any, world: Any, personas: dict) -> list[Finding]:
    out: list[Finding] = []
    out += _references(quest, world, personas)
    out += _reachability(quest, world, personas)
    out += _witnesses(quest, world, personas)
    return out


# ------------------------------------------------------------ 三、引用

def _references(q: Any, w: Any, personas: dict) -> list[Finding]:
    out = []
    sig_ids = {s.id for s in q.signals}
    clue_ids = set(q.clues)
    known = lambda kind, ids, pool, where: [  # noqa: E731
        Finding("error", "bad_ref", f"{where} 引用了不存在的{kind} {x}") for x in ids if x not in pool]
    for s in q.secrets:
        where = f"秘密 {s.get('id')}"
        out += known("人", s.get("known_by", []), personas, where)
        out += known("线索", s.get("unlocked_by_clues", []), clue_ids, where)
        out += known("信号", s.get("unlocked_by_signals", []), sig_ids, where)
    from core.god_env import LOCK_STYLES
    for sg in q.signals:
        if sg.lock_style and sg.lock_style not in LOCK_STYLES:
            out.append(Finding("error", "bad_ref", f"信号 {sg.id} 的 lock_style「{sg.lock_style}」不存在，可选：{'、'.join(LOCK_STYLES)}"))
    for sg in q.signals:                                           # 漏的口风只能是方向，不能自己就把事说完了
        if not sg.partial:
            continue
        said = [t for sec in q.secrets for t in sec.get("terms", []) if t in sg.partial]
        said += [k for c in sg.grants for k in q.clues.get(c, {}).get("keywords", []) if k in sg.partial]
        if said:
            out.append(Finding("error", "partial_leaks", f"信号 {sg.id} 的 partial 里带着内容词 {sorted(set(said))}，口风不能把事说出来"))
        if sg.evidence and all(any(w in sg.partial for w in grp) for grp in sg.evidence):
            out.append(Finding("error", "partial_leaks", f"信号 {sg.id} 的 partial 满足了它自己的 evidence，会被当成已经说出口"))
        if not sg.ask:
            out.append(Finding("warn", "partial_unused", f"信号 {sg.id} 写了 partial，但没有 ask，玩家没法硬问，用不上"))
    for lie in q.lies:
        out += known("人", [lie.npc], personas, f"谎话 {lie.id}")
        out += known("线索", lie.contradicted_by, clue_ids, f"谎话 {lie.id}")
    out += known("线索", list(q.fragile), clue_ids, "会毁掉的证据 fragile")
    for a in q.agendas:
        where = f"议程 {a.get('id')}"
        if a.get("npc"):
            out += known("人", [a["npc"]], personas, where)
        for key in ("do", "witness"):
            part = a.get(key) or {}
            for loc in (part.get("go_to"), part.get("at")):
                if loc and loc not in w.locations:
                    out.append(Finding("error", "bad_ref", f"{where} 的 {key} 引用了不存在的地点 {loc}"))
            out += known("线索", part.get("grants", []), clue_ids, where)
    for p in q.proposals:
        out += known("人", [p["npc"]] if p.get("npc") else [], personas, f"提议 {p.get('flag')}")
    for c, spec in q.clues.items():
        wit = spec.get("witness")
        if wit:
            out += known("人", [wit.get("npc")], personas, f"线索 {c} 的 witness")
            if wit.get("at") not in w.locations:
                out.append(Finding("error", "bad_ref", f"线索 {c} 的 witness.at 不是地点：{wit.get('at')}"))
            if wit.get("phase") not in w.phases:
                out.append(Finding("error", "bad_ref", f"线索 {c} 的 witness.phase 不是时段：{wit.get('phase')}"))
    return out


# ------------------------------------------------------------ 一、可达性

def _reachability(q: Any, w: Any, personas: dict) -> list[Finding]:
    clues: set[str] = set()
    done: set[str] = set()
    flags: set[str] = set()
    acts: set[str] = {q.acts[0].id}
    all_acts = [a.id for a in q.acts]
    # 玩家能自己提出的方案会立起这些标记；议程 / 发现 / 信号也会
    proposal_flags = {p["flag"]: p.get("acts") or all_acts for p in q.proposals if p.get("flag")}

    def locations() -> set[str]:
        seen, todo = {w.start["location"]}, [w.start["location"]]
        while todo:
            for d in w.exits(todo.pop()):
                if d not in seen and w.unlocked(d, list(clues)):
                    seen.add(d)
                    todo.append(d)
        return seen

    def meet(npc: str, in_acts: list[str], locs: set[str]) -> bool:
        p = personas.get(npc)
        return bool(p) and any(p.location_at(ph, a) in locs for a in in_acts for ph in w.phases)

    changed = True
    while changed:
        changed = False
        locs = locations()
        before = (len(clues), len(done), len(flags), len(acts), len(locs))
        for pf, pacts in proposal_flags.items():
            if set(pacts) & acts:
                flags.add(pf)
        for lid in locs:
            for o in w.objects(lid):
                if o.get("clue") and set(o.get("requires_clues", [])) <= clues and set(o.get("requires_flags", [])) <= flags:
                    clues.add(o["clue"])
        for s in q.signals:
            if s.id in done:
                continue
            ok_acts = [a for a in (s.acts or all_acts) if a in acts]
            if not ok_acts:
                continue
            if s.event:
                ev = q.event(s.event)
                present = bool(ev) and ev.location in locs and bool(set(ev.acts or all_acts) & set(ok_acts))
            else:
                present = meet(s.npc, ok_acts, locs)
            p = personas.get(s.npc)
            known = clues | set(getattr(p, "knows_clues", []) or [])
            if (present and set(s.requires) <= done and set(s.requires_clues) <= clues
                    and (not s.requires_clues_any or len(set(s.requires_clues_any) & clues) >= max(1, s.any_min))
                    and len(clues) >= s.clues_min
                    and (not s.requires_known_any or set(s.requires_known_any) & known)
                    and set(s.requires_flags) <= flags):
                done.add(s.id)
                clues |= set(s.grants)
                flags |= set(s.flags)
        for d in (q.actions or {}).get("discoveries", []):
            there = (meet(d["target"], list(acts), locs) if d.get("target")
                     else d.get("at") in locs if d.get("at") else True)
            if there:
                clues |= set(d.get("clues", []))
                flags |= set(d.get("set_flags", []))
        for a in q.agendas:
            trig = a.get("trigger") or {}
            if trig.get("flags_any") and not set(trig["flags_any"]) & flags:
                continue
            flags |= set((a.get("do") or {}).get("set_flags", []))
            wit = a.get("witness") or {}
            if wit.get("at") in locs:
                clues |= set(wit.get("grants", []))
                flags |= set(wit.get("set_flags", []))
        for act in q.acts:
            if act.id in acts and act.next and set(act.complete_when) <= done:
                acts.add(act.next)
        locs = locations()
        changed = before != (len(clues), len(done), len(flags), len(acts), len(locs))

    out = []
    for lid in w.locations:
        if lid not in locs:
            out.append(Finding("error", "unreachable_location", f"地点 {lid} 永远去不了（没有出口通向它，或解锁线索永远拿不到）"))
    for a in q.acts:
        if a.id not in acts:
            out.append(Finding("error", "unreachable_act", f"第 {a.id} 幕永远进不去（上一幕的完成信号达成不了）"))
    for nid in personas:
        if not meet(nid, all_acts, locs) and not any(s.npc == nid and s.event for s in q.signals):
            out.append(Finding("warn", "npc_never_met", f"{nid} 在任何时段都不在玩家能到的地方，见不到"))
    for s in q.signals:
        if s.id not in done:
            out.append(Finding("error", "unreachable_signal", f"信号 {s.id}（{s.npc}）永远达成不了：{_why(s, q, personas, w, clues, done, flags, acts, locs)}"))
    for c in q.clues:
        if c not in clues:
            out.append(Finding("error", "unreachable_clue", f"线索 {c}（{q.clues[c].get('name', '')}）永远拿不到"))
    return out


def _why(s, q, personas, w, clues, done, flags, acts, locs) -> str:
    all_acts = [a.id for a in q.acts]
    if not [a for a in (s.acts or all_acts) if a in acts]:
        return f"它只在 {s.acts} 幕可达成，而这些幕进不去"
    if s.event:
        return f"事件 {s.event} 发生不了（地点去不了或幕进不去）"
    p = personas.get(s.npc)
    if not p or not any(p.location_at(ph, a) in locs for a in all_acts if a in acts for ph in w.phases):
        return f"{s.npc} 在能去的地方一直见不到"
    miss = [("前置信号", sorted(set(s.requires) - done)), ("需要的线索", sorted(set(s.requires_clues) - clues)),
            ("需要的标记", sorted(set(s.requires_flags) - flags))]
    miss = [f"{k} {v}" for k, v in miss if v]
    if s.requires_clues_any and len(set(s.requires_clues_any) & clues) < max(1, s.any_min):
        miss.append(f"需要其中 {max(1, s.any_min)} 条线索 {s.requires_clues_any}")
    if s.requires_known_any and not set(s.requires_known_any) & (clues | set(getattr(p, "knows_clues", []) or [])):
        miss.append(f"{s.npc} 需要知道其中之一 {s.requires_known_any}")
    return "；".join(miss) or "条件互相等待"


# ------------------------------------------------------------ 二、在场

def _routine(p: Any, phase: str) -> str | None:
    """案发前的日常作息：只看不限幕的那几条（限定某一幕的是接了委托之后才有的安排）。"""
    for slot in p.schedule:
        if phase in slot.get("phases", []) and not slot.get("acts"):
            return slot["location"]
    return None


def _witnesses(q: Any, w: Any, personas: dict) -> list[Finding]:
    out = []
    giver = {}
    for s in q.signals:
        for c in s.grants:
            giver.setdefault(c, set()).add(s.npc)
    for c, spec in q.clues.items():
        wit = spec.get("witness")
        if not wit or wit.get("npc") not in personas:
            continue
        npc, phase, at = wit["npc"], wit.get("phase"), wit.get("at")
        if at not in w.locations or phase not in w.phases:
            continue                                                  # 地点 / 时段写错了，引用检查已经报过
        if giver.get(c) and npc not in giver[c]:
            out.append(Finding("error", "witness_not_the_teller",
                               f"线索 {c} 的目击者是 {npc}，可给出它的信号挂在 {'、'.join(sorted(giver[c]))} 身上"))
        if wit.get("off_routine"):
            continue
        here = _routine(personas[npc], phase)
        if here != at:
            where = w.name(here) if here else "不在任何地点"
            out.append(Finding("warn", "witness_absent",
                               f"线索 {c}：{personas[npc].name}平时{phase}{where}，却说自己在{w.name(at)}见到了——"
                               f"要么调整作息，要么给线索的 witness 写上 off_routine 说明为什么这天不一样"))
    return out
