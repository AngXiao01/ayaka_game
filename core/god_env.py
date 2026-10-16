"""GodEnv：世界状态的唯一真相——时间与截止、所在地点、线索、谁知道什么、主线幕、事件与结局。

模型只能「上报」信号（task_signals），导演 agent 只能「提议」动作（core/director.py 校验后才调用这里），这里决定：
- 哪些信号此刻可以达成（玩家掌握的线索、NPC 被出示过什么、信任、聊了几轮、所处的幕、互斥组）
- 达成后发生什么（获得线索、设置标记、推进幕、解决主线）
- 何时结束（截止时间到、主线有了决定）以及走向哪个结局
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional


@dataclass
class SignalDef:
    id: str
    npc: str
    desc: str
    acts: list[str] = field(default_factory=list)
    event: Optional[str] = None                 # 只在某个私人事件里可达成
    requires: list[str] = field(default_factory=list)            # 前置信号（DAG，审核层按拓扑序校验）
    requires_clues: list[str] = field(default_factory=list)      # 玩家必须已掌握的线索
    requires_known_any: list[str] = field(default_factory=list)  # 该 NPC 必须已经知道其中之一（被出示 / 汇报过）
    requires_flags: list[str] = field(default_factory=list)      # 必须已有的标记（如玩家明确提出的方案）
    requires_clues_any: list[str] = field(default_factory=list)  # 玩家至少掌握其中之一
    clues_min: int = 0
    min_trust: int = 0
    min_turns: int = 0
    group: Optional[str] = None                 # 互斥组：组内只能达成一个
    grants: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    lock_hint: str = ""
    # 给玩家的引导（不懂设定也能玩）：
    lead: str = ""                              # 「下一步」里的一句：去找谁、做什么
    ask: Optional[dict] = None                  # {label, text}：条件满足时对话里出现的关键问题按钮
    reveal: str = ""                            # 由动作直接达成时，告诉 NPC 这一轮要说出口的事
    evidence: list[list[str]] = field(default_factory=list)   # NPC 自己说出口就算数：每组至少一个词出现在 NPC 的话里


@dataclass
class ActDef:
    id: str
    title: str
    objective: str
    hint: str
    complete_when: list[str]
    next: Optional[str]
    narration: str = ""
    checklist: list[dict] = field(default_factory=list)   # [{label, clues_any}]：这一幕要弄清的几件事


@dataclass
class EventDef:
    id: str
    title: str
    npc: str
    location: str
    phases: list[str]
    acts: list[str]
    affinity_min: int
    premise: str
    stance: str


@dataclass
class LieDef:
    id: str
    npc: str
    when: str                     # 什么情境下会说这个谎
    said: str
    truth: str
    contradicted_by: list[str] = field(default_factory=list)   # 能戳穿它的线索


@dataclass
class EndingDef:
    id: str
    title: str
    when: dict
    text: str


@dataclass
class QuestDef:
    id: str
    title: str
    companion: str
    era: str
    premise: str
    prologue: list[str]
    deadline: dict
    acts: list[ActDef]
    clues: dict[str, dict]
    signals: list[SignalDef]
    events: list[EventDef]
    endings: list[EndingDef]
    epilogue_by_tier: dict[str, str] = field(default_factory=dict)
    proposals: list[dict] = field(default_factory=list)      # 玩家可以明确提出的方案（关键抉择不交给模型揣测）
    canon_guard_terms: list[str] = field(default_factory=list)  # 即兴设定里不许出现的案情关键词
    lies: list[LieDef] = field(default_factory=list)
    initial_tension: float = 0.3
    primer: list[str] = field(default_factory=list)           # 开场前给不熟悉设定的玩家的背景速览
    how_to_play: list[str] = field(default_factory=list)
    glossary: dict[str, str] = field(default_factory=dict)    # 名词速查
    spoiler_terms: list[str] = field(default_factory=list)    # 玩家没听说过就不该出现在玩家台词选项里的词
    public_premise: str = ""                                  # 城里人都知道的版本（不知道绫华托人查访的 NPC 用这个）
    after_hint: str = ""                                      # 结局之后继续玩时的一句提示
    late_note: str = ""                                       # 截止时间过了才解决时，结局前面加的一段

    @classmethod
    def load(cls, path: Path) -> "QuestDef":
        d = json.loads(path.read_text(encoding="utf-8"))
        d["acts"] = [ActDef(**a) for a in d["acts"]]
        d["signals"] = [SignalDef(**s) for s in d["signals"]]
        d["events"] = [EventDef(**e) for e in d["events"]]
        d["endings"] = [EndingDef(**e) for e in d["endings"]]
        d["lies"] = [LieDef(**x) for x in d.get("lies", [])]
        return cls(**d)

    def lie(self, lid: str) -> Optional[LieDef]:
        return next((x for x in self.lies if x.id == lid), None)

    def act(self, act_id: str) -> ActDef:
        return next(a for a in self.acts if a.id == act_id)

    def signal(self, sid: str) -> Optional[SignalDef]:
        return next((s for s in self.signals if s.id == sid), None)

    def event(self, eid: str) -> Optional[EventDef]:
        return next((e for e in self.events if e.id == eid), None)

    def clue_name(self, cid: str) -> str:
        return self.clues.get(cid, {}).get("name", cid)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


DEFAULT_CLOCK = {"phase_starts": [360, 660, 1020, 1170], "minutes_per_turn": 2, "start_time": None}


class GodEnv:
    def __init__(self, quest: QuestDef, phases: list[str], start: dict, turns_per_phase: int = 6,
                 state: Optional[dict] = None, clock: Optional[dict] = None):
        self.quest = quest
        self.phases = phases
        self.turns_per_phase = max(1, turns_per_phase)       # 旧接口，时间改按分钟算后不再使用
        c = {**DEFAULT_CLOCK, **(clock or {})}
        self.day_start: int = c["phase_starts"][0]          # 一天从第一个时段开始算（清晨 06:00）
        self.phase_offsets: list[int] = [x - self.day_start for x in c["phase_starts"]]
        self.minutes_per_turn: int = c["minutes_per_turn"]
        s = state or {}
        if "minute" in s:
            self.minute: int = s["minute"]
        elif "day" in s or "phase_idx" in s:              # 旧存档只记到时段
            self.minute = self.phase_start(s.get("day", start["day"]), s.get("phase_idx", start["phase"])) + 30
        else:
            t0 = c.get("start_time")
            self.minute = self.phase_start(start["day"], start["phase"]) + (
                max(0, t0 - c["phase_starts"][start["phase"]]) if t0 is not None else 0)
        self.location: str = s.get("location", start["location"])
        self.act_id: str = s.get("act_id", quest.acts[0].id)
        self.clues: list[str] = s.get("clues", [])
        self.known: dict[str, list[str]] = s.get("known", {})
        self.achieved: list[str] = s.get("achieved", [])
        self.flags: list[str] = s.get("flags", [])
        self.examined: list[str] = s.get("examined", [])
        self.events_done: list[str] = s.get("events_done", [])
        self.met: dict[str, int] = s.get("met", {})
        self.tension: float = s.get("tension", quest.initial_tension)
        self.session_count: int = s.get("session_count", 0)
        self.total_turns: int = s.get("total_turns", 0)
        self.resolved: bool = s.get("resolved", False)
        self.ending: Optional[str] = s.get("ending")
        self.journal: list[str] = s.get("journal", [])
        self.testimonies: list[dict] = s.get("testimonies", [])   # 证词：真话与谎话同一格式，玩家分不出
        # 导演层的状态（同样外化进存档）：都只影响「节奏」，不影响线索、信号与结局
        self.director_slot: Optional[int] = s.get("director_slot")          # 导演上次运行时的时段序号
        self.last_progress_slot: int = s.get("last_progress_slot", self._index(self.day, self.phase_idx))
        self.nudges: dict[str, str] = s.get("nudges", {})                   # npc -> 想让他主动提起的信号
        self.npc_overrides: dict[str, dict] = s.get("npc_overrides", {})    # npc -> {location, slot}，只在当前时段有效
        self.rumors: list[dict] = s.get("rumors", [])
        self.conv: Optional[dict] = None            # 当前对话（不存档）
        self.narrations: list[str] = []             # 待播放的剧情旁白
        self.notices: list[str] = []                # 时间推进时世界里发生的、玩家看得到的事（同行者告辞等）
        self.physical = None                        # PhysicalWorld：谁在哪、跟谁在一起（引擎装配；单测里可以没有）
        self.trust_lookup: Callable[[str], int] = lambda npc: 0

    # ------------------------------------------------------------ 时间（按分钟走；时段由钟点推出来）

    def phase_start(self, day: int, phase_idx: int) -> int:
        """第 day 天第 phase_idx 个时段开始的那一分钟（从第 1 天 00:00 起算）。"""
        return self.day_start + (day - 1) * 1440 + self.phase_offsets[phase_idx]

    def _split(self, minute: int) -> tuple[int, int]:
        rel = minute - self.day_start
        day, in_day = rel // 1440 + 1, rel % 1440
        idx = max(i for i, off in enumerate(self.phase_offsets) if off <= in_day)
        return day, idx

    @property
    def day(self) -> int:
        return self._split(self.minute)[0]

    @property
    def phase_idx(self) -> int:
        return self._split(self.minute)[1]

    @property
    def phase(self) -> str:
        return self.phases[self.phase_idx]

    @property
    def day_phase(self) -> str:
        return self.phase

    def phase_at(self, minute: int) -> str:
        return self.phases[self._split(minute)[1]]

    def clock_label(self, minute: Optional[int] = None) -> str:
        m = (self.minute if minute is None else minute) % 1440
        return f"{m // 60:02d}:{m % 60:02d}"

    def time_label(self) -> str:
        return f"第{self.day}天·{self.phase}"

    def next_phase_start(self) -> int:
        day, idx = self._split(self.minute)
        return self.phase_start(day + 1, 0) if idx + 1 >= len(self.phases) else self.phase_start(day, idx + 1)

    def pass_time(self, minutes: int) -> int:
        """时间往前走 minutes 分钟；返回跨过了几个时段。物理世界跟着结算（到点的口信、同行者告辞……）。"""
        if minutes <= 0:
            return 0
        before = self.slot_index
        self.minute += minutes
        crossed = self.slot_index - before
        for _ in range(crossed):
            self.tension = round(self.tension + (self.quest.initial_tension - self.tension) * 0.2, 3)
        if self.physical is not None:
            self.notices += self.physical.tick()
        return crossed

    def _index(self, day: int, phase_idx: int) -> int:
        return (day - 1) * len(self.phases) + phase_idx

    def slots_left(self) -> int:
        dl = self.quest.deadline
        return self._index(dl["day"], dl["phase"]) - self._index(self.day, self.phase_idx)

    @property
    def slot_index(self) -> int:
        """从开局算起的第几个时段，导演按它判断「时段是否切换了」。"""
        return self._index(self.day, self.phase_idx)

    def slots_since_progress(self) -> int:
        return self.slot_index - self.last_progress_slot

    def deadline_reached(self) -> bool:
        return self.slots_left() <= 0

    def phase_after(self, n: int) -> tuple[int, str]:
        idx = self.phase_idx + n
        return self.day + idx // len(self.phases), self.phases[idx % len(self.phases)]

    def advance(self, n: int = 1) -> None:
        """跳到 n 个时段之后的开头（等待 / 休息）。"""
        target = self.minute
        for _ in range(n):
            saved, self.minute = self.minute, target
            target = self.next_phase_start()
            self.minute = saved
        self.pass_time(target - self.minute)

    # ------------------------------------------------------------ 主线

    @property
    def act(self) -> ActDef:
        return self.quest.act(self.act_id)

    @property
    def stage(self) -> ActDef:          # 兼容旧接口
        return self.act

    def adjust_tension(self, delta: float) -> None:
        self.tension = round(_clamp01(self.tension + delta), 3)

    def tension_label(self) -> str:
        if self.tension < 0.25:
            return "轻松"
        if self.tension < 0.5:
            return "平稳"
        if self.tension < 0.75:
            return "紧绷"
        return "一触即发"

    def give_clue(self, cid: str, npc: Optional[str] = None) -> bool:
        if npc:
            self.mark_known(npc, [cid])
        if cid in self.clues:
            return False
        self.clues.append(cid)
        self.last_progress_slot = self.slot_index
        self.note_engagement()
        return True

    def propose(self, flag: str) -> None:
        """玩家明确提出某个方案；同一组里之前的提议作废（可以改主意）。"""
        prop = next(p for p in self.quest.proposals if p["flag"] == flag)
        same_group = {p["flag"] for p in self.quest.proposals if p.get("group") == prop.get("group")}
        self.flags = [f for f in self.flags if f not in same_group] + [flag]

    def proposals_for(self, npc: str) -> list[dict]:
        return [p for p in self.quest.proposals if p["npc"] == npc and self.act_id in p.get("acts", [self.act_id])
                and not any(s.group == p.get("group") and s.id in self.achieved for s in self.quest.signals)]

    # ------------------------------------------------------------ 证词与说谎

    def add_testimony(self, npc: str, said: str, kind: str, truth: str = "", lie_id: str = "",
                      contradicted_by: Optional[list[str]] = None) -> dict:
        for t in self.testimonies:          # 同一个谎只记一次
            if t["npc"] == npc and (lie_id and t.get("lie_id") == lie_id or t["said"] == said):
                return t
        t = {"id": f"t{len(self.testimonies) + 1}", "npc": npc, "said": said, "kind": kind, "truth": truth,
             "lie_id": lie_id, "contradicted_by": list(contradicted_by or []), "caught": False,
             "time": self.time_label()}
        self.testimonies.append(t)
        return t

    def testimonies_of(self, npc: str, open_only: bool = True) -> list[dict]:
        return [t for t in self.testimonies if t["npc"] == npc and not (open_only and t["caught"])]

    def lies_told(self, npc: str) -> list[dict]:
        return [t for t in self.testimonies if t["npc"] == npc and t["kind"] == "lie" and not t["caught"]]

    def lies_available(self, npc: str) -> list[LieDef]:
        """这个人可能会说、但还没说过也没被拆穿的预设谎言。"""
        used = {t.get("lie_id") for t in self.testimonies if t["npc"] == npc}
        return [x for x in self.quest.lies if x.npc == npc and x.id not in used]

    def confront(self, tid: str, clue: Optional[str], trust: int) -> dict:
        """对质：结果由规则判定，不交给模型。
        caught    —— 谎言 + 能戳穿它的证据（或：临时编的谎，没证据但对方够信任你、心虚认了）
        unproven  —— 确实是谎，但证据不对 / 没证据，对方可以嘴硬
        truth     —— 对方说的是真话，你冤枉了人"""
        t = next(x for x in self.testimonies if x["id"] == tid)
        if t["kind"] != "lie":
            outcome = "truth"
        elif clue and clue in t["contradicted_by"]:
            outcome = "caught"
        elif not t["contradicted_by"] and trust >= 40:
            outcome = "caught"
        else:
            outcome = "unproven"
        if outcome == "caught":
            t["caught"] = True
            self.flags.append(f"caught:{t.get('lie_id') or t['id']}")
            self.note_engagement()
        res = {"testimony": t, "clue": clue, "outcome": outcome}
        if self.conv is not None:
            self.conv["confrontation"] = res
        return res

    def mark_known(self, npc: str, cids: list[str]) -> None:
        k = self.known.setdefault(npc, [])
        for c in cids:
            if c not in k:
                k.append(c)

    # ------------------------------------------------------------ 对话

    def begin_conversation(self, npc: str, event: Optional[str] = None) -> None:
        self.session_count += 1
        self.met[npc] = self.met.get(npc, 0) + 1
        self.conv = {"npc": npc, "event": event, "turns": 0, "turns_since_progress": 0}
        if event and event not in self.events_done:
            self.events_done.append(event)

    def end_conversation(self) -> None:
        self.conv = None

    def conversation_turn(self, spend_time: bool = True) -> bool:
        """一轮对话结束时调用；说一轮话只花几分钟（minutes_per_turn）。返回这一轮是否跨进了新的时段。
        spend_time=False：私人时光占着那个时段本身，聊多久都不再额外花时间。"""
        self.total_turns += 1
        if self.conv:
            self.conv["turns"] += 1
            self.conv["turns_since_progress"] += 1
        return bool(spend_time and self.pass_time(self.minutes_per_turn))

    @property
    def turns_in_stage(self) -> int:
        return self.conv["turns"] if self.conv else 0

    @property
    def turns_since_progress(self) -> int:
        return self.conv["turns_since_progress"] if self.conv else 0

    def note_engagement(self) -> None:
        """有进展或关系升温都算对话有意义，停滞计数清零。"""
        if self.conv:
            self.conv["turns_since_progress"] = 0

    def stalled(self, limit: int) -> bool:
        return limit > 0 and self.turns_since_progress >= limit

    # ------------------------------------------------------------ 信号

    def signal_def(self, sid: str) -> Optional[dict]:
        s = self.quest.signal(sid)
        return asdict(s) if s else None

    def _group_taken(self, s: SignalDef) -> bool:
        return bool(s.group) and any(o.group == s.group and o.id in self.achieved for o in self.quest.signals)

    def signal_status(self, s: SignalDef, npc: str, event: Optional[str], turns: int,
                      check_turns: bool = True, optimistic: bool = False) -> tuple[str, str]:
        """optimistic：只看玩家「有没有办法」让它发生——信任可以再聊，手上的线索可以出示。"""
        """返回 (状态, 原因)：available / locked / hidden。hidden 的连提都不该提。"""
        if s.id in self.achieved or s.npc != npc or self._group_taken(s):
            return "hidden", ""
        if s.event:
            if event != s.event:
                return "hidden", ""
        elif event:
            return "hidden", ""               # 私人时间只谈私事
        if s.acts and self.act_id not in s.acts:
            return "hidden", ""
        if s.requires_flags and not set(s.requires_flags) <= set(self.flags):
            return "locked", s.lock_hint or "要等对方明确提出来"
        if s.requires_clues and not set(s.requires_clues) <= set(self.clues):
            return "locked", s.lock_hint or "对方还没掌握相关的情况，你不会主动提"
        if s.requires_clues_any and not set(s.requires_clues_any) & set(self.clues):
            return "locked", s.lock_hint or "对方还没掌握关键的证据"
        if s.clues_min and len(self.clues) < s.clues_min:
            return "locked", s.lock_hint or "对方带来的线索还不够"
        known = set(self.clues) if optimistic else set(self.known.get(npc, []))
        if s.requires_known_any and not set(s.requires_known_any) & known:
            return "locked", s.lock_hint or "除非对方拿出相关的证据"
        if s.min_trust and not optimistic and self.trust_lookup(npc) < s.min_trust:
            return "locked", s.lock_hint or "你还不够信任对方"
        if check_turns and s.min_turns and turns < s.min_turns:
            return "locked", "时机未到，先多聊几句再说"
        return "available", ""

    def _conv_signals(self, want: str, check_turns: bool = True) -> list[tuple[SignalDef, str]]:
        if not self.conv:
            return []
        out = []
        for s in self.quest.signals:
            st, why = self.signal_status(s, self.conv["npc"], self.conv["event"], self.conv["turns"], check_turns)
            if st == want:
                out.append((s, why))
        return out

    def available_ids(self, check_turns: bool = True) -> set[str]:
        return {s.id for s, _ in self._conv_signals("available", check_turns)}

    def askable(self) -> list[SignalDef]:
        """当前对话里可以用「关键问题」按钮直接问出来的：条件都满足了、前置也达成了（不看聊了几轮）。"""
        return [s for s, _ in self._conv_signals("available", check_turns=False)
                if s.ask and not self.missing_prereqs(s.id)]

    def mentioned_clues(self, npc: str, text: str) -> list[str]:
        """玩家在对话里提到了手上的哪条线索（线索的 keywords），而对方还不知道：说出来就等于出示了。"""
        known = set(self.known.get(npc, []))
        return [c for c in self.clues if c not in known
                and any(k and k in text for k in self.quest.clues[c].get("keywords", []))]

    def spoken_signals(self, speech: str, player_text: str = "") -> list[str]:
        """NPC 这句话里自己说出了哪件事（信号的 evidence：每组至少一个词在 NPC 的话里、且不是玩家刚说过的）。
        只看当前对话里没有被隐藏的信号（别人的、别的幕的、私人时光以外的都不算）。"""
        if not self.conv:
            return []
        out = []
        for s in self.quest.signals:
            if not s.evidence or s.id in self.achieved or self.missing_prereqs(s.id):
                continue
            st, _ = self.signal_status(s, self.conv["npc"], self.conv["event"], self.conv["turns"], check_turns=False)
            if st == "hidden":
                continue
            if all(any(k in speech and k not in player_text for k in group) for group in s.evidence):
                out.append(s.id)
        return out

    def checklist(self) -> list[tuple[str, bool]]:
        """这一幕要弄清的几件事，各自弄清了没有（给玩家看的目标进度）。"""
        return [(c["label"], bool(set(c.get("clues_any", [])) & set(self.clues))) for c in self.act.checklist]

    def pending_signals(self) -> list[dict]:
        return [{"id": s.id, "desc": s.desc, "requires": s.requires} for s, _ in self._conv_signals("available")]

    def locked_signals(self) -> list[tuple[dict, str]]:
        return [({"id": s.id, "desc": s.desc}, why) for s, why in self._conv_signals("locked")]

    def preview_pending(self, npc: str, event: Optional[str] = None) -> int:
        """不在对话中时，估计和某人聊有几件事「有戏」（给菜单提示与自动玩家用）：
        此刻就能谈的，加上多聊几句建立信任、或出示手上的线索就能谈的。"""
        ok = {s.id for s in self.quest.signals
              if self.signal_status(s, npc, event, 99, check_turns=False, optimistic=True)[0] == "available"}
        # 前置信号既没达成、本身也谈不了的，不算有戏
        return sum(1 for sid in ok if all(r in self.achieved or r in ok for r in self.quest.signal(sid).requires))

    def useful_clues_for(self, npc: str) -> list[str]:
        """出示哪些线索能让这个人开口。"""
        out: list[str] = []
        for s in self.quest.signals:
            if s.npc == npc and s.id not in self.achieved and s.requires_known_any \
                    and not set(s.requires_known_any) & set(self.known.get(npc, [])):
                out += [c for c in s.requires_known_any if c in self.clues and c not in out]
        return out

    def missing_prereqs(self, sid: str, also_done: set[str] = frozenset()) -> list[str]:
        s = self.quest.signal(sid)
        done = set(self.achieved) | set(also_done)
        return [r for r in (s.requires if s else []) if r not in done]

    def topo_order(self, ids: list[str]) -> list[str]:
        """按依赖关系排序（Kahn），同一轮里多个信号时先审前置。环依赖视为内容配置错误。"""
        ids = list(dict.fromkeys(ids))
        pending, out = set(ids), []
        while pending:
            ready = [i for i in ids if i in pending
                     and not (set(self.quest.signal(i).requires if self.quest.signal(i) else []) & pending)]
            if not ready:
                raise ValueError(f"信号依赖有环：{sorted(pending)}")
            for i in ready:
                out.append(i)
                pending.discard(i)
        return out

    def apply_signals(self, ids: list[str], forced: bool = False, heard: bool = False) -> list[dict]:
        """只接受当前对话里此刻可达成的信号；返回新达成的（含获得的线索）。
        forced：玩家用明确的动作（关键问题、出示证据、汇报、提议）满足了条件，由代码直接判定达成——
        不经审核，所以这里自己查前置信号；也不看聊了几轮（轮数门槛是防模型抢跑的，动作不存在抢跑）。
        heard：NPC 已经亲口说出来了（spoken_signals），玩家听到了就算——条件没满足（locked）也认，
        不然剧情里说过的事线索本里没有，前后对不上。"""
        forced = forced or heard
        available = self.available_ids(check_turns=not forced)
        if heard:
            available |= {s.id for s, _ in self._conv_signals("locked", check_turns=False)}
        newly = []
        for sid in ids:
            s = self.quest.signal(sid)
            if s is None or sid not in available or sid in self.achieved or self._group_taken(s):
                continue
            if forced and self.missing_prereqs(sid):
                continue
            self.achieved.append(sid)
            self.last_progress_slot = self.slot_index
            if self.nudges.get(s.npc) == sid:
                self.nudges.pop(s.npc)
            gained = [c for c in s.grants if self.give_clue(c, s.npc)]
            for f in s.flags:
                if f not in self.flags:
                    self.flags.append(f)
            self.note_engagement()
            newly.append({"id": sid, "desc": s.desc, "clues": [self.quest.clue_name(c) for c in gained]})
            if sid in self.act.complete_when:
                self._complete_act()
        return newly

    def _complete_act(self) -> None:
        act = self.act
        if act.narration:
            self.narrations.append(act.narration)
        if act.next:
            self.act_id = act.next
        else:
            self.resolved = True

    # ------------------------------------------------------------ 导演层（只改节奏：谁在哪、谁想提什么、传闻、气氛）

    def where(self, persona, phase: Optional[str] = None) -> Optional[str]:
        """某个 NPC 此刻（或当天某个时段）在哪。有物理世界时以它为准（作息 + 同行 + 计划）；
        单测里没有物理世界时按作息 + 导演的临时安排（只在生效的那个时段内有效）。"""
        if self.physical is not None:
            return self.physical.where(persona.id, phase)
        ov = self.npc_overrides.get(persona.id)
        if ov and ov.get("slot") == self.slot_index and (phase is None or phase == self.phase):
            return ov["location"]
        return persona.location_at(phase or self.phase, self.act_id)

    def set_nudge(self, npc: str, sid: str) -> None:
        self.nudges[npc] = sid

    def set_override(self, npc: str, location: str) -> None:
        self.npc_overrides[npc] = {"location": location, "slot": self.slot_index}
        if self.physical is not None:            # 导演的安排也是一种计划：这个时段结束就回到作息
            self.physical.set_plan(npc, location, self.next_phase_start(), "director", "导演安排")

    def expire_overrides(self) -> None:
        self.npc_overrides = {k: v for k, v in self.npc_overrides.items() if v.get("slot") == self.slot_index}

    def add_rumor(self, text: str, target: dict) -> None:
        self.rumors.append({"text": text, "target": target, "time": self.time_label(), "slot": self.slot_index})
        self.journal.append(f"{self.time_label()}　（传闻）{text}")

    # ------------------------------------------------------------ 事件与结局

    def available_events(self, location: str, affinity_of: Callable[[str], int]) -> list[EventDef]:
        party = self.physical.party if self.physical is not None else []
        return [e for e in self.quest.events
                if e.id not in self.events_done and e.location == location and self.phase in e.phases
                and e.npc not in party                      # 人已经和你在一起了，就不会「偶遇」
                and (not e.acts or self.act_id in e.acts) and affinity_of(e.npc) >= e.affinity_min]

    def compute_ending(self, affinity: int) -> EndingDef:
        for e in self.quest.endings:
            w = e.when
            if w.get("deadline_passed"):
                if self.resolved or not self.deadline_reached():
                    continue
            elif not self.resolved:
                continue
            if not set(w.get("flags", [])) <= set(self.flags):
                continue
            if not set(w.get("clues", [])) <= set(self.clues):
                continue
            if affinity < w.get("affinity_min", 0):
                continue
            return e
        return self.quest.endings[0]

    # ------------------------------------------------------------ 给 LLM / 玩家看的视图

    def narrative_guidance(self, scene_public: bool = False, pending: Optional[list[dict]] = None) -> list[str]:
        g = []
        if self.turns_since_progress >= 4 and pending:
            g.append(f"对话有些原地打转了。可以自然地把话题引向与「{pending[0]['desc']}」有关的方向，但不要生硬。")
        if self.tension >= 0.75:
            g.append("气氛很紧张，言辞会更克制谨慎，必要时会礼貌地结束谈话。")
        elif self.tension >= 0.5:
            g.append("气氛有些紧绷，说话会更加斟酌。")
        if scene_public:
            g.append("这里人来人往，谈到奉行所的事务时要隐晦，避免被旁人听见。")
        if 0 < self.slots_left() <= 2:
            g.append(f"{self.quest.deadline['label']}就在眼前，时间不多了。")
        return g

    def stance_of(self, persona) -> str:
        """主线结束后用后日谈的立场（可以按结局细分：after:mercy）。"""
        if self.resolved:
            for key in (f"after:{self.ending}", "after"):
                if persona.stances.get(key):
                    return persona.stances[key]
        return persona.stance(self.act_id)

    def npc_task_detail(self, persona, trust: int) -> dict:
        npc = persona.id
        ev = self.quest.event(self.conv["event"]) if self.conv and self.conv.get("event") else None
        pending = self.pending_signals()
        locked = self.locked_signals()
        met = self.met.get(npc, 1)
        forced = [self.quest.signal(sid) for sid in (self.conv or {}).get("forced", [])]
        must_reveal = [s.reveal or s.desc for s in forced if s]
        in_party = self.physical is not None and npc in self.physical.party
        if ev:
            opening = ev.premise
        elif in_party:
            opening = "和你一路同行，到了这里，转过头来跟你说话"
        elif met <= 1:
            opening = "第一次来见你"
        else:
            opening = f"又来找你了（这是你们第 {met} 次见面）"
        return {
            "npc_id": npc,
            "act_id": self.act_id,
            "stage_title": self.act.title,
            "objective": self.act.objective if persona.role == "companion" else "",
            "opening_hint": opening,
            "npc_stance": ev.stance if ev else self.stance_of(persona),
            "event": {"title": ev.title, "premise": ev.premise} if ev else None,
            "info_goals": [d["desc"] for d in pending],
            "locked_info": [f"{d['desc']}（{why}）" for d, why in locked],
            "secrets_unlocked": [x["text"] for x in persona.secrets if trust >= x.get("min_trust", 0)],
            "known_clues": [f"{self.quest.clue_name(c)}：{self.quest.clues[c]['desc']}" for c in self.known.get(npc, [])],
            "player_clue_count": len(self.clues),
            "achieved_signals": [{"id": s.id, "desc": s.desc} for s in self.quest.signals
                                 if s.id in self.achieved and s.npc == npc],
            "pending_signals": pending,
            "turns_in_stage": self.turns_in_stage,
            "first_meeting": met <= 1,
            "must_reveal": must_reveal,
            "invite": (self.conv or {}).get("invite"),          # 对方邀你同行：代码判定的答应 / 婉拒
            "dismiss": bool((self.conv or {}).get("dismiss")),  # 对方请你先回去
            # 这一轮要说出实情，就不再给可以说的谎（不然模型可能用谎话把该说的事搪塞过去）
            "lies_available": [] if must_reveal else [{"id": x.id, "when": x.when, "said": x.said, "truth": x.truth}
                                                      for x in self.lies_available(npc)],
            "lies_told": [{"said": t["said"], "truth": t["truth"]} for t in self.lies_told(npc)],
            "confrontation": (self.conv or {}).get("confrontation"),
            "today": persona.days.get(str(self.day), ""),
            "curiosity": (persona.curiosities[(self.session_count + self.turns_in_stage) % len(persona.curiosities)]
                          if persona.curiosities else ""),
            "director_note": next((f"这次可以找个自然的时机，主动提起与「{d['desc']}」有关的事，不要硬塞。"
                                   for d in pending if self.nudges.get(npc) == d["id"]), ""),
        }

    def world_state_summary(self, persona, trust: int, scene_public: bool = False) -> dict:
        task = self.npc_task_detail(persona, trust)
        return {
            "quest_title": self.quest.title,
            # 绫华托人暗中查访这件事，只有绫华自己和被告知过（看过线索）的人知道；其余的人只知道城里的传言
            "premise": self.quest.premise if persona.role == "companion" or self.known.get(persona.id)
                       else (self.quest.public_premise or self.quest.premise),
            "era": self.quest.era,
            "time_label": self.time_label(),
            "day_phase": self.phase,
            "deadline_label": self.quest.deadline["label"],
            "deadline_left": self.slots_left(),
            "stage_no": [a.id for a in self.quest.acts].index(self.act_id) + 1,
            "stage_total": len(self.quest.acts),
            "tension": self.tension,
            "tension_label": self.tension_label(),
            "narrative_guidance": self.narrative_guidance(scene_public, task["pending_signals"])
                                  + ([task["director_note"]] if task["director_note"] and not task["must_reveal"] else []),
            "npc_task_detail": task,
        }

    def status_line(self) -> str:
        due = (f"距{self.quest.deadline['label']}还有 {max(0, self.slots_left())} 个时段" if self.slots_left() > 0
               else f"{self.quest.deadline['label']}已经过去了")
        return (f"{self.time_label()} {self.clock_label()} ｜ {due} ｜ "
                f"线索 {len(self.clues)}/{len(self.quest.clues)} ｜ 【{self.act.title}】{self.act.objective}")

    # ------------------------------------------------------------ 存档

    def to_dict(self) -> dict:
        return {"quest_id": self.quest.id, "minute": self.minute, "day": self.day, "phase_idx": self.phase_idx,
                "location": self.location,
                "act_id": self.act_id, "clues": self.clues, "known": self.known, "achieved": self.achieved,
                "flags": self.flags, "examined": self.examined, "events_done": self.events_done, "met": self.met,
                "tension": self.tension, "session_count": self.session_count, "total_turns": self.total_turns,
                "resolved": self.resolved, "ending": self.ending, "journal": self.journal,
                "testimonies": self.testimonies, "director_slot": self.director_slot,
                "last_progress_slot": self.last_progress_slot, "nudges": self.nudges,
                "npc_overrides": self.npc_overrides, "rumors": self.rumors,
                **({"physical": self.physical.to_dict()} if self.physical is not None else {})}
