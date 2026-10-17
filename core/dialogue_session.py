"""一次交谈 = 一个 DialogueSession（外层循环），适用于任何 NPC，也用于私人事件。

每一轮：预算检查 → 台词选项 + 动作（出示线索 / 汇报 / 离开）→ 玩家输入 → 输入防护 → 技能路由
      → NPC ReAct 循环（内层）→ 进度审核 → 结算（线索、关系、幕）→ 时间流逝 / 截止 / 对方离场 / 停滞
      → 工作记忆压缩
结束：写该 NPC 的长期记忆（带来源与事实校验），记一条调查日志。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from config.settings import Settings
from core.budget import Budget
from core.context import TurnContext
from core.god_env import EventDef, GodEnv
from core import judge
from core.beats import pick_beat
from models.retrieval import bigram_jaccard
from core.dice import roll, tag
from core.guard import inspect_player_input
from core.npc_agent import NPCAgent, NPCReply
from core.player_option_generator import PlayerOption, PlayerOptionGenerator
from core.reviewer import SignalReviewer
from core.working_memory import WorkingMemory
from core.world import World


# 打字邀请同行：「一起去茶会吧」「要不要去镇守之森转转」「和你一起」
INVITE_RE = re.compile(r"一起去|一起走|一起出去|一起逛|陪我去|陪我走|陪我逛|跟我来|跟我走|和我一起|和你一起|出门走走|"
                       r"去.{0,6}?(?:走走|转转|逛逛)")


class DialogueEndReason(str, Enum):
    PLAYER_LEFT = "你起身告辞"
    PLAYER_QUIT = "保存并退出"
    NPC_ENDED = "对方结束了谈话"
    NPC_LEFT = "对方有事先走了"
    MAX_TURNS = "聊了很久，自然收尾"
    STALLED = "话题渐渐冷了下来"
    BUDGET = "预算用尽，暂停"
    DEADLINE = "时间到了"
    RESOLVED = "事情有了决断"


@dataclass
class SessionResult:
    end_reason: DialogueEndReason
    turns: int
    npc: str = ""
    transcript: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    clues_gained: list[str] = field(default_factory=list)
    degraded_turns: int = 0
    summary: Optional[dict] = None


class _NullSpan:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class DialogueSession:
    def __init__(self, npc: NPCAgent, player: Any, option_gen: PlayerOptionGenerator, god: GodEnv,
                 world: World, settings: Settings, reviewer: SignalReviewer, budget: Budget,
                 event: Optional[EventDef] = None, compressor: Optional[Callable[[str, list[str]], str]] = None,
                 out: Callable[[str], None] = print, tracer: Any = None, branch: Any = None,
                 player_view: Optional[Callable[[str, Optional[str]], dict]] = None, claims: Any = None):
        self.npc = npc
        self.player_view = player_view  # 玩家视角（Explorer.player_view）：选项 agent 只能看这个，看不到 NPC 的心思
        self.npc_names = set(god.physical.personas) if god.physical is not None else set()
        self.player = player
        self.option_gen = option_gen
        self.god = god
        self.world = world
        self.cfg = settings.game
        self.h = settings.harness
        self.reviewer = reviewer
        self.budget = budget
        self.event = event
        self.out = out
        self.tracer = tracer
        self.claims = claims            # 声明核对（core/claims.py）：玩家说自己做过什么，对着真实足迹核对
        self.branch = branch            # 分支 agent：自由输入里的处置方案 → 预设方案（可为 None）
        self.wm = WorkingMemory(self.cfg.history_turns, self.cfg.player_name, npc.name, compressor,
                                self.h.compress_mode, tracer)
        self.transcript: list[str] = []
        self.scene_id = god.location
        self._last_ctx: Optional[TurnContext] = None
        self._ended_by_player = False
        self._clues_before = list(god.clues)

    def _span(self, **kv):
        return self.tracer.span(**kv) if self.tracer else _NullSpan()

    def _log(self, event: str, /, **data) -> None:
        if self.tracer:
            self.tracer.log(event, **data)

    # ------------------------------------------------------------ 上下文

    def build_ctx(self, turn: int, query_text: str = "") -> TurnContext:
        p = self.npc.p
        scene = self.world.scene_context(self.scene_id, self.god.phase)
        world = self.god.world_state_summary(p, self.npc.relationship.trust, scene["public"])
        last_player = next((e["text"] for e in reversed(self.wm.entries) if e["speaker"] == "player"), "")
        topic = query_text or last_player
        memories = [m for m in self.npc.recall(self.god.act.objective + topic, self.cfg.memory_top_k, self.god.total_turns)]
        canon = self.npc.recall_canon(topic or self.god.act.objective, 3, self.god.total_turns)
        memories = [m for m in memories if not any(m in c for c in canon)]
        handoff = self.npc.handoff()
        reflection = self.npc.latest_reflection(self.god.total_turns)
        if handoff.get("minute") is not None:
            handoff["ago"] = self.god.elapsed_since(handoff["minute"])
            handoff["same_place"] = handoff.get("where") == self.scene_id
        memories = [m for m in memories if m != handoff.get("summary")]
        place = self.god.physical.npc_view(p.id, topic) if self.god.physical is not None else {}
        if self.npc.fs is not None:
            place = {**place, "places": self.npc.fs.places_in(topic + " " + " ".join(h["text"] for h in self.wm.entries[-2:]))}
        return TurnContext(player_name=self.cfg.player_name, scene=scene, world=world,
                           relationship=self.npc.relationship.summary(), memories=memories,
                           history=list(self.wm.entries), history_summary=self.wm.summary, turn=turn, canon=canon,
                           handoff=handoff, place=place, reflection=reflection)

    def _flush_notices(self) -> None:
        for line in self.god.notices:
            self.out("\n" + line)
        self.god.notices.clear()

    def _place_name(self, text: str) -> tuple[str, str]:
        """话里提到的地方：(地点 id, 名字)。"""
        for lid, loc in self.world.locations.items():
            if loc["name"] in (text or ""):
                return lid, loc["name"]
        return "", ""

    # ------------------------------------------------------------ 展示

    def _show_npc(self, r: NPCReply) -> None:
        self._show_speech(r)
        self._show_debug(r)

    def _show_speech(self, r: NPCReply) -> None:
        act = f"（{r.action}）" if r.action else ""
        self.out(f"\n{self.npc.name}{act}：「{r.speech}」")
        self.transcript.append(f"{self.npc.name}{act}：「{r.speech}」")

    def _show_debug(self, r: NPCReply) -> None:
        if self.cfg.debug:
            rel = self.npc.relationship
            loop = r.loop or {}
            tools = f" | 工具 {'→'.join(loop['tools'])}" if loop.get("tools") else ""
            forced = f" | 被 {loop['forced_by']} 收敛" if loop.get("forced_by") else ""
            self.out(f"    · {r.emotion}{'' if r.fit in ('接得上', '换了话题') else ' | 觉得突兀：' + r.fit} | 好感 {r.affinity_delta:+d}→{rel.affinity} 信任 {r.trust_delta:+d}→{rel.trust}"
                     f"{tools}{forced}{' | 兜底' if r.degraded else ''}")

    # ------------------------------------------------------------ 选项

    def _actions(self) -> list[tuple[str, PlayerOption, str, str]]:
        """此刻能做的事：(编号, 按钮, 给选项 agent 看的说明, 台词上标的小字)。选项 agent 可以把其中一件「顺带」进一句台词里。"""
        g, nid = self.god, self.npc.p.id
        acts: list[tuple[str, PlayerOption, str, str]] = []
        for s in g.askable():               # 关键问题：条件都满足了，问出口就由代码判定达成，不赌模型
            label = s.ask["label"]
            acts.append((f"ask:{s.id}", PlayerOption(label, "表达关心" if s.event else "推进任务", kind="ask",
                                                     meta={"signal": s.id}),
                         f"问出这个问题：{label.lstrip('❓💬🤝 ')}", label[:1] + " 关键问题"))
        if any(s["id"] in g.act.complete_when and s["id"] == "reported_findings" for s in g.pending_signals()):
            acts.append(("report", PlayerOption(f"📣 把调查到的一切告诉{self.npc.name}", "汇报", kind="report"),
                         f"把查到的事一五一十告诉{self.npc.name}", "📣 汇报"))
        if not self.event:
            for prop in g.proposals_for(nid):
                chosen = prop["flag"] in g.flags
                acts.append((f"propose:{prop['flag']}",
                             PlayerOption(prop["label"] + ("（已提出）" if chosen else ""), "推进任务", kind="propose",
                                          meta={"flag": prop["flag"], "chosen": chosen}),
                             f"提出主张：{prop['label'].replace('⚖️ 提议：', '')}", "⚖️ 提议"))
        phys = g.physical
        together = phys is not None and nid in phys.party
        if phys is not None and not self.event:
            if together:
                acts.append(("dismiss", PlayerOption(f"👋 请{self.npc.name}先回去", "告别", kind="dismiss"),
                             f"请{self.npc.name}先回去", "👋 请对方先回去"))
            elif self.npc.p.outing and not (g.conv or {}).get("invite_declined"):
                acts.append(("invite", PlayerOption(f"🚶 邀{self.npc.name}一起走走", "邀请", kind="invite"),
                             f"邀{self.npc.name}一起出去走走", "🚶 邀请同行"))
        acts.append(("leave", PlayerOption("🚶 接着走" if together else "🚪 离开", "告别", kind="leave"),
                     "结束这次谈话，离开" if not together else "先不聊了，接着赶路", "🚪 告辞" if not together else "🚶 接着走"))
        return acts

    def _options(self, ctx: TurnContext) -> list[PlayerOption]:
        g, nid = self.god, self.npc.p.id
        acts = self._actions()
        if self.player_view is not None:
            ctx.player_view = self.player_view(nid, self.event.id if self.event else None)
            ctx.player_view["actions"] = [{"id": i, "desc": d, "kind": o.kind, "meta": o.meta, "tag": t}
                                          for i, o, d, t in acts]
        with self._span(agent="options"):
            says, degraded = self.option_gen.generate(ctx, self.npc.name, self.npc.p.fallback_options,
                                                      force_static=self.budget.degraded())
        if degraded and self.cfg.debug:
            self.out("    · 选项生成失败，已用兜底选项")
        covered = {o.meta.get("act") for o in says if o.meta.get("line")}
        opts = list(says)
        tail = [o for i, o, _, _ in acts if i not in covered]    # 台词已经带上了这件事，就不再另给一个按钮
        if not self.event:                   # 软锁：条件不够也能硬问，由代码掷骰定松不松口
            for s, p, _ in g.pressable(nid):
                opts.append(PlayerOption(f"🎲 {s.ask['label']}（硬问，约 {round(p * 100)}%）", "推进任务", kind="press",
                                         meta={"signal": s.id}))
        unknown = [c for c in g.clues if c not in g.known.get(nid, [])]
        if unknown and not self.event:
            useful = g.useful_clues_for(nid)
            opts.append(PlayerOption("🔍 出示线索……" + ("  💡" if useful and self.cfg.hints else ""), "出示线索",
                                     kind="show_menu", meta={"useful": bool(useful)}))
        claims = g.testimonies_of(nid)
        if claims and not self.event:
            useful = any(t["kind"] == "lie" and set(t["contradicted_by"]) & set(g.clues) for t in claims)
            opts.append(PlayerOption("⚠ 对质：指出对方话里的破绽……" + ("  💡" if useful and self.cfg.hints else ""),
                                     "对质", kind="confront_menu", meta={"useful": useful}))
        asks = [o for o in tail if o.kind == "ask"]
        return [*opts[:len(says)], *asks, *opts[len(says):], *[o for o in tail if o.kind != "ask"]]

    def _resolve_action(self, ctx: TurnContext, options: list[PlayerOption]):
        """处理二级菜单（出示线索），返回最终动作。"""
        g, nid = self.god, self.npc.p.id
        while True:
            action = self.player.choose(options, ctx)
            if action.kind == "confront_menu":
                picked = self._confront_menu(ctx)
                if picked:
                    return picked
                continue
            if action.kind != "show_menu":
                return action
            useful = set(g.useful_clues_for(nid))
            sub = [PlayerOption(f"{g.quest.clue_name(c)}——{g.quest.clues[c]['desc'][:30]}…"
                                + ("  💡" if c in useful and self.cfg.hints else ""), "出示线索", kind="clue",
                                meta={"clue": c, "useful": c in useful})
                   for c in g.clues if c not in g.known.get(nid, [])]
            sub.append(PlayerOption("算了，不拿出来了", "", kind="cancel"))
            pick = self.player.choose(sub, ctx)
            if pick.kind == "clue":
                return pick
            if pick.kind == "quit":
                return pick

    def _apply_confrontation(self, res: dict) -> None:
        """对质的后果按规则结算：拆穿 / 嘴硬 / 冤枉好人。"""
        g, rel, t = self.god, self.npc.relationship, res["testimony"]
        if res["outcome"] == "caught":
            rel.apply(-1, 0)
            g.adjust_tension(0.1)
            self.out(f"    ⚡ 拆穿了{self.npc.name}的谎！")
            g.journal.append(f"{g.time_label()}·{self.world.name(self.scene_id)}　拆穿了{self.npc.name}的谎：「{t['said']}」")
        elif res["outcome"] == "unproven":
            g.adjust_tension(0.05)
            self.out("    …… 这份证据似乎说明不了什么。")
        else:
            rel.apply(-1, -2)
            self.out(f"    😣 {self.npc.name}看起来被冤枉了。")
        self._log("confront", npc=self.npc.p.id, testimony=t["id"], said=t["said"], clue=res["clue"],
                  outcome=res["outcome"], relationship=rel.to_dict())

    def _confront_menu(self, ctx: TurnContext):
        """对质：先挑一句对方说过的话，再挑一条线索当证据（或不拿证据直接质疑）。"""
        from core.player_agent import PlayerAction
        g = self.god
        claims = g.testimonies_of(self.npc.p.id)
        tip = lambda on: "  💡" if on and self.cfg.hints else ""    # noqa: E731
        sub = []
        for t in claims:
            useful = t["kind"] == "lie" and bool(set(t["contradicted_by"]) & set(g.clues))
            sub.append(PlayerOption(f"「{t['said']}」（{t['time']}）{tip(useful)}", "对质", kind="testimony",
                                    meta={"tid": t["id"], "useful": useful}))
        sub.append(PlayerOption("算了", "", kind="cancel"))
        pick = self.player.choose(sub, ctx)
        if pick.kind != "testimony":
            return pick if pick.kind == "quit" else None
        t = next(x for x in claims if x["id"] == pick.meta["tid"])
        ev = [PlayerOption(f"拿出「{g.quest.clue_name(c)}」{tip(c in t['contradicted_by'])}", "对质", kind="evidence",
                           meta={"clue": c, "useful": c in t["contradicted_by"]}) for c in g.clues]
        ev.append(PlayerOption("不拿证据，直接说我不信", "对质", kind="evidence", meta={"clue": None, "useful": False}))
        ev.append(PlayerOption("算了", "", kind="cancel"))
        e = self.player.choose(ev, ctx)
        if e.kind != "evidence":
            return e if e.kind == "quit" else None
        return PlayerAction("confront", intent="对质", meta={"tid": t["id"], "clue": e.meta["clue"]})

    # ------------------------------------------------------------ 结算

    def _settle(self, r: NPCReply, turn: int, *, opening: bool, player: Optional[dict] = None,
                options: Optional[list] = None, flags: Optional[list[str]] = None,
                context_lines: Optional[list[str]] = None, forced: Optional[list[str]] = None) -> list[dict]:
        g = self.god
        forced = list(forced or [])
        self.npc.apply_reply(r)
        g.adjust_tension(r.tension_delta)
        if r.affinity_delta > 0 or r.trust_delta > 0:
            g.note_engagement()
        # 玩家用明确的动作满足了条件：代码直接判定（和「提议」一样不交给模型揣测），其余的上报照常过审核
        newly = g.apply_signals(forced, forced=True) if forced else []
        if forced:
            self._log("action_signal", npc=self.npc.p.id, action=(player or {}).get("kind"), forced=forced,
                      accepted=[s["id"] for s in newly])
        # NPC 自己说出口了（不管有没有上报、条件满没满足），玩家听到了就算：线索本和剧情不能对不上
        heard = [] if flags else g.spoken_signals(r.speech + r.action, (player or {}).get("evidence_text") or (player or {}).get("text", ""))
        if not flags and not opening:
            # 关键词没认出来的说法，交给模型判：他是不是已经亲口承认了还没达成的事
            nid = self.npc.p.id
            open_ = [s_ for s_ in g.quest.signals
                     if s_.npc == nid and s_.id not in g.achieved and s_.id not in heard + forced and not s_.event
                     and not g.missing_prereqs(s_.id)
                     and g.signal_status(s_, nid, None, 99, check_turns=False)[0] != "hidden"
                     and (s_.ask or s_.evidence)]
            heard += judge.conceded(g, self.npc.name, r.speech + r.action,
                                    (player or {}).get("evidence_text") or (player or {}).get("text", ""), open_) or []
        if heard:
            got = g.apply_signals(heard, heard=True)
            newly += got
            self._log("heard_signal", npc=self.npc.p.id, heard=heard, accepted=[s["id"] for s in got])
            forced = forced + heard
        review = self.reviewer.review([x for x in r.task_signals if x not in forced], opening=opening,
                                      injection_flags=flags or [],
                                      player_text=(player or {}).get("evidence_text") or (player or {}).get("text", ""), npc_speech=r.speech,
                                      context_lines=context_lines or [], degraded=self.budget.degraded())
        newly += g.apply_signals(review.accepted)
        for s_ in [] if self.npc.p.role == "companion" else newly:
            if s_["clues"]:                  # 吐露线索的那句话记进证词本（和谎话同一格式）
                g.add_testimony(self.npc.p.id, r.speech[:80], "truth")
        self._world_moves(r, flags or [])
        lies = self.npc.record_lies(r, g, g.total_turns)
        if self.cfg.debug:
            for k in lies["kept"]:
                self.out(f"    🤥（调试）{self.npc.name}说了谎：「{k['said']}」")
        canon = self.npc.remember_improvisations(r, g.total_turns, {"time": g.time_label(), "turn": turn})
        if self.cfg.debug:
            for k in canon["kept"]:
                self.out(f"    📝 记下即兴设定：{k['fact']}{'（共享）' if k['about'] == 'world' else ''}")
            for d in canon["dropped"]:
                self.out(f"    ✗ 不记：{d['fact']}（{d['reason']}）")
        for s in newly:
            self.out(f"    ✦ {s['desc']}")
            for c in s["clues"]:
                self.out(f"    📜 获得线索「{c}」")
        if self.cfg.debug:
            for v in review.rejected:
                self.out(f"    ✗ 未采纳 {v.id}（{v.layer}：{v.reason}）")
        self._log("turn", turn=turn, npc=self.npc.p.id, private_event=self.event.id if self.event else None,
                  stage_id=g.act_id, stage_title=g.act.title, scene=self.scene_id, time=g.time_label(),
                  player=player, options=[{"text": o.text, "intent": o.intent, "kind": o.kind, "tone": o.tone}
                                          for o in options] if options else None,
                  reply=r.to_dict(), degraded=r.degraded, loop=r.loop,
                  claimed=r.task_signals, new_signals=[s["id"] for s in newly],
                  clues=list(g.clues), relationship=self.npc.relationship.to_dict(), tension=g.tension,
                  turns_in_stage=g.turns_in_stage, turns_since_progress=g.turns_since_progress,
                  budget=self.budget.snapshot())
        return newly

    def _repeated_question(self, text: str, kind: str) -> Optional[dict]:
        """玩家把已经问过的话又问了一遍：找到上次的问法和 NPC 当时的回答（PLAYER* 的“重复提问收益递减”）。"""
        if kind in ("clue", "report", "leave", "invite", "dismiss", "propose") or len(text) < 6:
            return None
        entries = list(self.wm.entries)
        for i in range(len(entries) - 1, -1, -1):
            e = entries[i]
            if e["speaker"] == "player" and bigram_jaccard(e["text"], text) >= 0.55:
                nxt = next((x for x in entries[i + 1:] if x["speaker"] == "npc"), None)
                return {"asked": e["text"], "answered": nxt["text"] if nxt else ""}
        return None

    def _world_moves(self, r: NPCReply, flags: list[str]) -> None:
        """NPC 在回复里说要托人带话（tell）、说完要去别处（go_to）：代码校验后落到物理世界。"""
        phys, nid = self.god.physical, self.npc.p.id
        if phys is None or flags:
            return
        for t in r.tell[:1]:
            to = phys.resolve(t["to"])
            if to and to != nid:
                msg = phys.send(nid, to, t["text"], kind="note")
                self._log("message", action="sent", **{k: msg[k] for k in ("id", "from", "to", "kind", "text")},
                          deliver=msg["deliver"])
                if self.cfg.debug:
                    self.out(f"    📨 {self.npc.name}托人给{phys.name(to)}带话：{t['text']}")
        if (r.go_to and r.wants_to_end and not self.event and nid not in phys.party and self.npc.p.outing
                and not getattr(self, "_ended_by_player", False) and self.npc.p.role != "companion"):   # 是玩家离开的、或她是委托人（留在屋敷）：不去
            dest, name = self._place_name(r.go_to)
            if dest and dest != phys.where(nid):
                phys.set_plan(nid, dest, self.god.next_phase_start(), "go", reason=r.go_to)
                self.out(f"    （{self.npc.name}说完，往{name}去了。）")
                self._log("npc_move", npc=nid, dest=dest)

    # ------------------------------------------------------------ 主循环

    def run(self) -> SessionResult:
        self.god.begin_conversation(self.npc.p.id, self.event.id if self.event else None)
        self.reviewer.npc_name = self.npc.name
        try:
            with self._span(session=self.god.session_count, npc=self.npc.p.id):
                return self._run()
        finally:
            self.god.end_conversation()

    def _run(self) -> SessionResult:
        g, p = self.god, self.npc.p
        rel = self.npc.relationship
        last = self.npc.handoff().get("minute")
        met = g.met.get(p.id, 1)
        tag = "初次见面" if last is None or met <= 1 else f"第 {met} 次见面 · {g.ago_short(last)}"
        self.out(f"\n—— 与{p.name}交谈 ——  {p.brief}  （{tag}）")
        if self.event:
            self.out(f"\n✨【{self.event.title}】\n{self.event.premise}")
        elif self.cfg.hints:
            useful = g.useful_clues_for(p.id)
            if useful:
                self.out(f"提示：试试用「🔍 出示线索」把「{'」或「'.join(g.quest.clue_name(c) for c in useful)}」拿给{p.name}看。")
        self._log("session_start", npc=p.id, private_event=self.event.id if self.event else None, scene=self.scene_id,
                  day_phase=g.phase, time=g.time_label(), stage_id=g.act_id, stage_title=g.act.title,
                  relationship=rel.to_dict(), tension=g.tension, clues=list(g.clues), budget=self.budget.snapshot())

        signals: list[str] = []
        degraded = 0
        with self._span(turn=0, agent="npc"):
            self._last_ctx = self.build_ctx(0)
            skills = self.npc.skills.route(None, "", g.act_id, False, npc=self.npc.p.id)
            opening = self.npc.opening(self._last_ctx, skills, allow_tools=not self.budget.degraded())
            self.wm.add({"speaker": "npc", "text": opening.speech, "reply": opening.to_dict()})
            self._show_speech(opening)
            signals += [s["id"] for s in self._settle(opening, 0, opening=True)]
            self._show_debug(opening)
            if g.physical is not None:
                g.physical.take_inbox(p.id)
            degraded += opening.degraded

        end: Optional[DialogueEndReason] = None
        turn = 0
        while end is None and turn < self.cfg.max_turns_per_session:
            if self.budget.exhausted():
                end = DialogueEndReason.BUDGET
                self._log("budget", action="stop", **self.budget.snapshot())
                break
            if self.budget.first_degrade():
                self._log("budget", action="degrade", **self.budget.snapshot())
                if self.cfg.debug:
                    self.out("    · 预算不足，已降级：静态选项 / 规则审核 / NPC 不调用工具")
            turn += 1
            with self._span(turn=turn):
                end, newly, deg = self._turn(turn)
                signals += newly
                degraded += deg

        if end is None:
            end = DialogueEndReason.MAX_TURNS
        if end in (DialogueEndReason.MAX_TURNS, DialogueEndReason.STALLED, DialogueEndReason.BUDGET,
                   DialogueEndReason.DEADLINE, DialogueEndReason.NPC_LEFT):
            self._show_npc(self.npc.farewell(self._last_ctx))

        for n in g.narrations:
            self.out("\n" + n)
        g.narrations.clear()
        self.out(f"\n—— {end.value} ——")
        gained = [c for c in g.clues if c not in self._clues_before]
        g.journal.append(f"{g.time_label()}·{self.world.name(self.scene_id)}　"
                         f"{'【' + self.event.title + '】' if self.event else ''}与{p.name}交谈 {turn} 轮"
                         + (f"，获得线索：{'、'.join(g.quest.clue_name(c) for c in gained)}" if gained else ""))
        with self._span(agent="memory"):
            summary = self.npc.remember_session(
                self._last_ctx, "\n".join(self.transcript), end.value, turn, g.total_turns,
                provenance={"session": g.session_count, "time": g.time_label(), "turns": turn,
                            "minute": g.minute, "where": self.scene_id})
        if self.cfg.debug:
            self.out(f"    · {p.name}记住了：{summary['summary']}")
        return SessionResult(end_reason=end, turns=turn, npc=p.id, transcript=self.transcript, signals=signals,
                             clues_gained=gained, degraded_turns=degraded, summary=summary)

    def _turn(self, turn: int) -> tuple[Optional[DialogueEndReason], list[str], int]:
        g = self.god
        ctx = self._last_ctx = self.build_ctx(turn)
        options = self._options(ctx)
        self.out("")
        action = self._resolve_action(ctx, options)
        if action.kind == "quit":
            self._show_npc(self.npc.farewell(ctx))
            return DialogueEndReason.PLAYER_QUIT, [], 0

        nid = self.npc.p.id
        before = g.available_ids(check_turns=False)
        forced: list[str] = []
        shown: list[tuple[str, str]] = []          # 这一轮玩家递给对方看的东西（名字、内容）
        press = None
        if action.kind == "ask":
            s = g.quest.signal(action.meta["signal"])
            text, intent = (action.text if action.meta.get("line") else s.ask["text"]), action.intent or "推进任务"
            forced = [s.id]
        elif action.kind == "press":
            s = g.quest.signal(action.meta["signal"])
            text, intent = s.ask["text"], "推进任务"
            lines = [f"{'玩家' if h['speaker'] == 'player' else self.npc.name}：{h['text']}" for h in ctx.history[-6:]]
            strength = judge.persuasion(g, self.npc.name, s.desc, lines) or 0
            p, why = g.press_chance(s, nid, strength)
            prev = g.misses.get(f"press:{s.id}")
            ok, pct = roll(g, f"press:{s.id}", p, attempt=prev["n"] if isinstance(prev, dict) else 0)
            press = {"ok": ok, "desc": s.desc, "line": tag(pct, ok, why),
                     "slip": s.partial if not ok and not isinstance(prev, dict) else ""}      # 第一次问崩了才漏口风，再逼也不会重复漏
            if ok:
                forced = [s.id]
            else:
                g.note_press_miss(s, nid)
            self._log("press", npc=nid, signal=s.id, p=p, ok=ok)
        elif action.kind == "clue":
            c = action.meta["clue"]
            g.mark_known(nid, [c])
            # 递东西是个动作，不是一段台词：玩家这边只记一句动作，内容作为「对方亲眼看到的东西」交给 NPC 的情境
            text, intent = f"（把「{g.quest.clue_name(c)}」递给{self.npc.name}看）", "出示线索"
            shown = [(g.quest.clue_name(c), g.quest.clues[c]["desc"])]
            ctx = self._last_ctx = self.build_ctx(turn)          # 出示之后对方知道的事变了，重建情境
        elif action.kind == "report":
            g.mark_known(nid, list(g.clues))
            text = (action.text if action.meta.get("line") else "") + "（把这几天查到的事一五一十地说了出来）"
            shown = [(g.quest.clue_name(c), g.quest.clues[c]["desc"].split("。")[0]) for c in g.clues]
            intent = "汇报"
            ctx = self._last_ctx = self.build_ctx(turn)
        elif action.kind == "propose":
            prop = next(p for p in g.quest.proposals if p["flag"] == action.meta["flag"])
            g.propose(prop["flag"])
            text, intent = (action.text if action.meta.get("line") else prop["text"]), "推进任务"
            ctx = self._last_ctx = self.build_ctx(turn)
        elif action.kind == "confront":
            t = next(x for x in g.testimonies if x["id"] == action.meta["tid"])
            c = action.meta["clue"]
            res = g.confront(t["id"], c, self.npc.relationship.trust)
            if c:
                g.mark_known(nid, [c])
                text = f"（拿出「{g.quest.clue_name(c)}」）你之前那句话，和这个对不上。"
                shown = [(g.quest.clue_name(c), g.quest.clues[c]["desc"])]
            else:
                text = "你之前那句话，说实话，我不太信。"
            intent = "对质"
            self._apply_confrontation(res)
            ctx = self._last_ctx = self.build_ctx(turn, query_text=t["said"])
        elif action.kind == "leave":
            together = g.physical is not None and nid in g.physical.party
            # 按钮只是个动作，不替玩家写台词：想说什么由选项 agent 写（带 act）或玩家自己打字；
            # 这里只记一句不带话的动作，对方怎么送、说不说话，交给对方自己
            text, intent = (("（没再多说，继续往前走）", "闲聊") if together else ("（转身离开）", "告别"))
            if action.meta.get("line"):
                text = action.text
        elif action.kind == "invite":
            text, intent = (action.text if action.meta.get("line") else "难得有空，要不要和我一起出去走走？"), "邀请"
        elif action.kind == "dismiss":
            text, intent = (action.text if action.meta.get("line")
                            else "今天就到这儿吧，你先回去歇着，改天再一起走走。"), "告别"
        else:
            text, intent = action.text, action.intent
        if action.kind in ("clue", "report", "propose", "confront"):
            # 这个动作让哪些事此刻能谈了（出示了对方要看的证据 / 提出了方案），就由这个动作直接达成
            after = g.available_ids(check_turns=False)
            forced = [s.id for s in g.quest.signals if s.id in after
                      and (s.id not in before or (action.kind == "report" and s.id in g.act.complete_when))]
        forced = [sid for sid in forced if not g.missing_prereqs(sid)]

        guard = inspect_player_input(text, self.h.player_input_max_chars if action.kind == "say" else 2000)
        if guard.suspicious:
            forced = []
        elif action.kind == "say":
            # 打字说出手上线索的内容，和点「出示线索」一样：对方知道了，因此能开口的事就由这句话直接达成
            told = g.mentioned_clues(nid, guard.text)
            if told:
                g.mark_known(nid, told)
                after = g.available_ids(check_turns=False)
                forced = [s.id for s in g.quest.signals
                          if s.id in after and s.id not in before and not g.missing_prereqs(s.id)]
                self._log("mention", npc=nid, clues=told, forced=forced)
        phys, invite, felt = g.physical, None, {}
        if phys is not None and action.kind == "say" and not guard.suspicious and not self.event:
            # 一句自由发言只判一次（Jev 或大模型）：语气、接不接得上、是不是在邀请、说中了哪几条推测。
            # 判定器没开或出错时 felt 为空，下面各处退回关键词规则
            open_hs = [] if g.hypotheses_for(nid, guard.text) else \
                [h for h in g.quest.hypotheses if nid in h["listeners"]
                 and f"dispatched:{h['id']}" not in g.flags and not set(h["clues"]) <= set(g.clues)]
            last_line = next((h["text"] for h in reversed(ctx.history) if h["speaker"] == "npc"), "")
            felt = judge.turn(g, self.npc.name, guard.text, open_hs, last_line, g.act.objective or "") or {}
        # 玩家打字说「我该走了」：走不走是玩家的事，不用等对方点头。是不是在告别由判定模型答，结束会话由代码执行
        leaving = bool(felt.get("leaving")) and action.kind == "say"
        self._ended_by_player = action.kind in ("leave", "dismiss") or leaving
        if leaving:
            intent = "告别"
            self._log("leaving", npc=nid, text=guard.text)
        asked_out = felt["invite"] if "invite" in felt else bool(INVITE_RE.search(guard.text))
        if (phys is not None and not self.event and not guard.suspicious and nid not in phys.party and self.npc.p.outing
                and (action.kind == "invite" or (action.kind == "say" and asked_out))):
            # 同行答不答应由代码按好感和对方此刻有没有空判定，模型只负责用自己的口吻说出来
            ok, why = phys.can_join(nid, self.npc.relationship.affinity)
            dest_id, dest = self._place_name(guard.text)
            invite = {"accept": ok, "reason": why, "dest": dest}
            self._log("invite", npc=nid, accept=ok, reason=why, dest=dest_id, text=guard.text)
        dispatch, believe, threatened, odd = None, False, False, []
        self._roll_line = None
        if phys is not None and action.kind == "say" and not guard.suspicious and not self.event:
            rel = self.npc.relationship
            # 便宜的判定（Jev）：语气、说中了哪几条推测。没开或拿不准时为 None，下面照旧只看关键词
            if felt.get("abrupt"):                      # Jev 的第二意见；NPC 自己也会在回复里判一次（fit）
                odd.append("对方这话来得有点突然，和刚才聊的、眼下要做的事接不上。")
            if odd:
                self._log("incongruous", npc=nid, notes=odd, text=guard.text)
            if felt.get("threat"):
                threatened = True
                if f"alerted:{nid}" not in g.flags:
                    g.flags.append(f"alerted:{nid}")         # 被威胁的人会戒备，也可能去报信
                g.tension = min(1.0, g.tension + 0.05)
                self._log("threat", npc=nid, text=guard.text)
            judged = felt.get("hypotheses", [])
            if judged:
                self._log("hypotheses_judged", npc=nid, ids=[h["id"] for h in judged], text=guard.text)
            p, why = g.belief_chance(nid, guard.text, rel.affinity, rel.trust, extra={h["id"] for h in judged},
                                     specific=felt.get("specific", 0))
            if p:
                # 关系够深：掷一次，成了就直接信，跳过查证进入处置（结局会记得「没查实」）；没成就照常派人去查
                believe, pct = roll(g, "belief", p)
                self._roll_line = tag(pct, believe, why)
                if not believe:
                    g.note_belief_miss(nid, guard.text, rel.affinity, rel.trust)
                self._log("believe", npc=nid, affinity=rel.affinity, trust=rel.trust, p=p, ok=believe, text=guard.text)
            if not believe:
                # 玩家直接说出了推测（比如玩过一遍、知道真相）：不当真也不置之不理——派人去查，查到了托人带话回来
                hs = g.hypotheses_for(nid, guard.text) or judged
                if hs:
                    agent = hs[0]["agent"]
                    agent = agent if agent in self.npc_names and agent not in phys.party else nid
                    dispatch = {"ids": [h["id"] for h in hs], "agent": agent, "self": agent == nid,
                                "plan": "；".join(h["plan"] for h in hs), "desc": "；".join(h["desc"] for h in hs),
                                "location": "、".join(self.world.name(h["location"]) for h in hs)}
                    self._log("dispatch", npc=nid, hypotheses=dispatch["ids"], agent=agent)
        if g.conv is not None:
            g.conv["dispatch"] = dispatch
            g.conv["believe"] = believe
            g.conv["press"] = press
            g.conv["threatened"] = threatened
            g.conv["incongruous"] = odd
            g.conv["forced"] = forced                  # NPC 这一轮会被告知要说出口的事
            g.conv["invite"] = invite
            g.conv["dismiss"] = action.kind == "dismiss" and phys is not None and nid in phys.party
        ctx = self._last_ctx = self.build_ctx(turn, query_text=text)   # 按这句话检索相关的往事与即兴设定
        ctx.task["shown"] = [{"name": n, "desc": d} for n, d in shown]
        ctx.task["unverified"] = g.unverified_claims(nid, guard.text) if action.kind == "say" and not guard.suspicious else []
        beat, scored = pick_beat(g, ctx.task["pending_signals"], guard.text, intent, g.turns_since_progress)
        ctx.task["beat"] = beat                               # None = 这一轮什么都不提；不设 beat（开场等）则沿用“列出全部”
        if scored:
            self._log("beat", npc=nid, picked=beat["id"] if beat else None, scored=scored[:4])
        ctx.task["repeated"] = self._repeated_question(guard.text, action.kind) if not guard.suspicious else None
        if ctx.task["repeated"]:
            self._log("repeat_question", npc=nid, text=guard.text, earlier=ctx.task["repeated"]["asked"])
        ctx.task["claims"] = []
        if self.claims is not None and action.kind == "say" and not guard.suspicious and not self.event:
            ctx.task["claims"] = self.claims.check(nid, guard.text)       # 抽取归模型，真假和知不知情归代码
            if ctx.task["claims"]:
                self._log("claim_check", npc=nid, text=guard.text, claims=[
                    {k: c[k] for k in ("kind", "target", "name", "verifiable", "verdict")} for c in ctx.task["claims"]])
        self._log("guard", raw=text, text=guard.text, flags=guard.flags, truncated=guard.truncated)
        if guard.suspicious and self.cfg.debug:
            self.out(f"    · 输入防护：{'；'.join(guard.flags)}（本轮不计进展）")
        if action.kind == "say" and not self.event and self.branch is not None:
            with self._span(agent="branch"):
                prop = self.branch.classify(nid, guard.text, guard.suspicious, self.budget.degraded())
            if prop:                              # 和点菜单提出方案走同一条路
                g.propose(prop["flag"])
                intent = "推进任务"
                ctx = self._last_ctx = self.build_ctx(turn, query_text=guard.text)
                if self.cfg.debug:
                    self.out(f"    · 分支：识别为{prop['label']}")
        if action.kind in ("ask", "press", "clue", "report", "propose", "confront", "invite", "dismiss") \
                or action.meta.get("line"):
            self.out(f"\n{self.cfg.player_name}：{guard.text}")
        if press:
            self.out(f"    {press['line']}")
        skills = self.npc.skills.route(intent, guard.text, g.act_id, guard.suspicious,
                                       has_locked=bool(ctx.task["locked_info"]), npc=nid)
        context_lines = self.wm.lines()
        with self._span(agent="npc"):
            reply = self.npc.respond(ctx, guard.text, intent, skills, guard.flags,
                                     allow_tools=not self.budget.degraded())
        self.wm.add({"speaker": "player", "text": guard.text, "intent": intent})
        self.wm.add({"speaker": "npc", "text": reply.speech, "reply": reply.to_dict()})
        self.transcript.append(f"{self.cfg.player_name}：「{guard.text}」")
        self._show_speech(reply)                  # 先让对方把话说完，再显示获得的线索
        with self._span(agent="reviewer"):
            newly = self._settle(reply, turn, opening=False,
                                 player={"text": guard.text, "intent": intent, "kind": action.kind,
                                         "evidence_text": guard.text + "".join(f"{n}：{d}" for n, d in shown),
                                         "flags": guard.flags, "skills": [s.name for s in skills]},
                                 options=options, flags=guard.flags, context_lines=context_lines, forced=forced)
        self._show_debug(reply)
        dismissed = bool(g.conv and g.conv.get("dismiss"))
        went = False
        dsp = g.conv.pop("dispatch", None) if g.conv else None
        if dsp and phys is not None:
            hs = [next(x for x in g.quest.hypotheses if x["id"] == i) for i in dsp["ids"]]
            g.flags += [f"dispatched:{h['id']}" for h in hs]
            phys.dispatch_chain(dsp["agent"], [{"location": h["location"], "search_minutes": h.get("search_minutes", 30),
                                                "text": h["report"], "clues": h["clues"], "tag": h["id"]} for h in hs],
                                [x for x in ("player", nid) if x != dsp["agent"]])
            who = "自己" if dsp["self"] else phys.name(dsp["agent"])
            if getattr(self, "_roll_line", None):
                self.out(f"    {self._roll_line}　{self.npc.name}犹豫了一下，还是想先查实再说。（拿出更多证据、或过些时候再跟她说，她可能会改主意）")
            self.out(f"    🔎 {self.npc.name}{'决定亲自' if dsp['self'] else '派' + who}去{dsp['location']}查一查，有消息会托人告诉你。")
            g.journal.append(f"{g.time_label()}　你说出了推测：{dsp['desc']}。{self.npc.name}{'亲自' if dsp['self'] else '让' + who}去{dsp['location']}查了")
            went = dsp["self"]
        if g.conv and g.conv.pop("believe", None):
            g.accept_on_trust()
            if self._roll_line:
                self.out(f"    {self._roll_line}")
            self.out(f"    🤝 {self.npc.name}没有等查证，选择了相信你。")
            g.journal.append(f"{g.time_label()}　{self.npc.name}凭着对你的信任，相信了你说的真相（没有查实）")
        if g.conv:
            g.conv.pop("confrontation", None)
            g.conv.pop("forced", None)
            g.conv.pop("dismiss", None)
            inv = g.conv.pop("invite", None)
            if inv and inv["accept"] and phys is not None:
                phys.join(nid, inv["dest"])
                self.out(f"    🚶 {self.npc.name}和你同行了" + (f"（说好去{inv['dest']}）" if inv["dest"] else "")
                         + "。之后走到哪儿，都可以直接和对方说话。")
            elif inv:
                g.conv["invite_declined"] = True        # 刚被婉拒：这次谈话里不再摆出「邀请」按钮（打字邀请仍然可以）
                self.out(f"    （{self.npc.name}这次没有答应同行。）")
        if phys is not None:
            phys.take_inbox(nid)                       # 别人带来的话，这一轮已经在情境里给过了
            if dismissed:
                home = phys.leave(nid)
                self.out(f"    👋 {self.npc.name}往{self.world.name(home) if home else '回'}去了。")
        for n in g.narrations:
            self.out("\n" + n)
        g.narrations.clear()
        ids = [s["id"] for s in newly]
        deg = int(reply.degraded)

        self._flush_notices()
        if g.resolved:
            return DialogueEndReason.RESOLVED, ids, deg
        if action.kind == "leave" or dismissed or leaving:
            return DialogueEndReason.PLAYER_LEFT, ids, deg
        if went:
            return DialogueEndReason.NPC_LEFT, ids, deg
        if reply.wants_to_end:
            return DialogueEndReason.NPC_ENDED, ids, deg
        if g.conversation_turn(spend_time=self.event is None):
            self.out(f"\n    （不知不觉，已是{g.time_label()}）")
            if g.deadline_reached():
                return DialogueEndReason.DEADLINE, ids, deg
            if not self.event and g.where(self.npc.p) != g.location:
                return DialogueEndReason.NPC_LEFT, ids, deg
        if g.stalled(self.h.stall_limit):
            self._log("stall", turns_since_progress=g.turns_since_progress, limit=self.h.stall_limit)
            return DialogueEndReason.STALLED, ids, deg
        with self._span(agent="compress"):
            self.wm.maybe_compress()
        return None, ids, deg
