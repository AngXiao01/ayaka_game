"""GameEngine（Harness 外壳）：加载内容 → 组装 NPC / 工具 / 记忆 / 审核 / 预算 / 观测 → 地点层主循环 → 结局与存档。

存档布局（按 --save 名字隔离）：
  data/saves/<save>.json              世界状态 + 各 NPC 关系（外化状态：上下文重置后从这里恢复）
  data/memories/<save>/<npc>.jsonl    每个 NPC 各自的长期记忆
  data/traces/<save>-<时间>.jsonl     本次运行的结构化 trace
"""
from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from config.settings import Settings
from core.agent_loop import LoopLimits
from core.branch import BranchAgent
from core.budget import Budget
from core.director import Director
from core.dialogue_session import DialogueEndReason, DialogueSession, SessionResult
from core.explorer import Explorer
from core.god_env import GodEnv, QuestDef
from core.npc_agent import NPCAgent
from core.physical import PhysicalWorld
from core.player_option_generator import PlayerOptionGenerator
from core.reviewer import SignalReviewer
from core.scene import SceneAgent
from core.skills import SkillLibrary
from core.world import World
from engine.trace import Tracer
from llm.base import LLMBackend
from llm.branch_llm_adapter import BranchLLMAdapter
from llm.director_llm_adapter import DirectorLLMAdapter
from llm.npc_llm_adapter import NPCLLMAdapter
from llm.player_option_llm_adapter import PlayerOptionLLMAdapter
from llm.reviewer_llm_adapter import ReviewerLLMAdapter
from llm.scene_llm_adapter import SceneLLMAdapter
from llm.router import ModelRouter
from llm.summary_llm_adapter import CompressLLMAdapter, SummaryLLMAdapter
from models.knowledge import LoreBase
from models.memory import MemoryItem, MemoryStore
from models.personality import Personality
from models.relationship import Relationship

GOOD_ENDINGS = {"bond", "mercy", "expose"}


@dataclass
class GameResult:
    sessions: list[SessionResult] = field(default_factory=list)
    ending: Optional[str] = None
    actions: int = 0
    stopped_by_player: bool = False
    stopped_by_budget: bool = False

    @property
    def quest_complete(self) -> bool:
        return self.ending in GOOD_ENDINGS


class GameEngine:
    def __init__(self, settings: Settings, backend: LLMBackend, player: Any,
                 out: Callable[[str], None] = print, save_name: str = "default", new_game: bool = False,
                 trace: bool = True, router: Optional[ModelRouter] = None):
        self.settings = settings
        g, h, lp = settings.game, settings.harness, settings.llm
        self.backend = backend
        self.router = router or ModelRouter(backend, lp)
        self.player = player
        self.out = out
        data = Path(g.data_dir)
        self.save_path = data / "saves" / f"{save_name}.json"
        mem_dir = data / "memories" / save_name
        if new_game:
            self.save_path.unlink(missing_ok=True)
            shutil.rmtree(mem_dir, ignore_errors=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.tracer = Tracer(data / "traces" / f"{save_name}-{stamp}.jsonl" if trace else None,
                             secrets=(lp.api_key,))
        self.budget = Budget(self.tracer.stats, h.game_token_budget, h.degrade_ratio)

        content = Path(g.content_dir)
        self.quest = QuestDef.load(content / "quests" / f"{g.quest_id}.json")
        self.world = World.load(content / "world.json")
        self.personas = Personality.load_all(content / "npcs")
        self.lore = LoreBase.load(content / "lore")
        self.skills = SkillLibrary.load(content / "skills")
        self._validate_content()

        state = json.loads(self.save_path.read_text(encoding="utf-8")) if self.save_path.is_file() else {}
        self.resumed = bool(state)
        self.god = GodEnv(self.quest, self.world.phases, self.world.start, self.world.turns_per_phase,
                          state.get("world"), clock=self.world.clock())
        # 物理世界：谁在哪、跟谁同行、口信在路上——GodEnv 持有，位置以它为准
        self.god.physical = PhysicalWorld(self.world, self.personas, self.god, (state.get("world") or {}).get("physical"),
                                          player_name=g.player_name)
        self.god.physical.on_deliver = self._on_message
        # 主线结束（结局）、截止时间到了之后还能继续玩；自动玩家 / 评测跑到结局就停
        self.continue_after_ending = g.continue_after_ending and not getattr(player, "stops_at_ending", False)
        rels = state.get("relationships", {})

        r, t, rep = self.router, self.tracer, lp.json_repair_attempts
        limits = LoopLimits(max_steps=h.loop_max_steps, max_repeat=h.max_repeat, token_budget=h.turn_token_budget,
                            timeout_s=h.turn_timeout_s, obs_max_chars=h.obs_max_chars)
        npc_adapter = NPCLLMAdapter(r.backend("npc"), r.params("npc"), limits, t, rep, lp.max_context_tokens,
                                    persona_mode=h.persona_mode)
        summarizer = SummaryLLMAdapter(r.backend("summary"), r.params("summary"), t, rep)
        self.world_canon = MemoryStore(mem_dir / "_world_canon.jsonl")     # 共享的即兴世界设定
        self.npcs: dict[str, NPCAgent] = {}
        for nid, p in self.personas.items():
            rel = Relationship.from_dict(rels.get(nid, p.initial))
            self.npcs[nid] = NPCAgent(p, rel, MemoryStore(mem_dir / f"{nid}.jsonl"), npc_adapter, summarizer,
                                      self.lore, self.skills, rng=random.Random(g.seed), tracer=t,
                                      world_canon=self.world_canon, guard_terms=self.quest.canon_guard_terms,
                                      persona_mode=h.persona_mode)
        self.god.trust_lookup = lambda nid: self.npcs[nid].relationship.trust
        self.companion = self.npcs[self.quest.companion]
        self.option_gen = PlayerOptionGenerator(
            PlayerOptionLLMAdapter(r.backend("options"), r.params("options"), t, rep), mode=g.options_mode, tracer=t)
        self.reviewer = SignalReviewer(
            self.god, ReviewerLLMAdapter(r.backend("review"), r.params("review"), t, min(rep, 1)),
            self.companion.name, g.player_name, mode=h.reviewer_mode, fail_policy=h.review_fail_policy, tracer=t)
        self.compressor = CompressLLMAdapter(r.backend("compress"), r.params("compress"), t, min(rep, 1)).compress
        self.explorer = Explorer(self.god, self.world, self.personas, lambda nid: self.npcs[nid].relationship,
                                 out, hints=g.hints)
        self.director = Director(
            self.god, self.world, self.personas, self.explorer,
            lambda nid: (self.npcs[nid].relationship.affinity, self.npcs[nid].relationship.trust),
            DirectorLLMAdapter(r.backend("director"), r.params("director"), t, min(rep, 1)),
            mode=h.director_mode, max_actions=h.director_max_actions, tracer=t)
        self.branch = BranchAgent(self.god, BranchLLMAdapter(r.backend("branch"), r.params("branch"), t, min(rep, 1)),
                                  mode=h.branch_mode, tracer=t)
        self.scene = SceneAgent(self.god, self.world, self.explorer, self.world_canon, self.quest.canon_guard_terms,
                                SceneLLMAdapter(r.backend("scene"), r.params("scene"), t, min(rep, 1)),
                                mode=h.scene_mode, tracer=t)

    def _validate_content(self) -> None:
        """内容配置错误尽早暴露，而不是跑到第三幕才崩。"""
        q, w = self.quest, self.world
        sig_ids = {s.id for s in q.signals}
        for s in q.signals:
            if s.npc not in self.personas:
                raise ValueError(f"信号 {s.id} 指向不存在的角色 {s.npc}")
            for c in s.grants + s.requires_clues + s.requires_known_any + s.requires_clues_any:
                if c not in q.clues:
                    raise ValueError(f"信号 {s.id} 引用了不存在的线索 {c}")
            if set(s.requires) - sig_ids:
                raise ValueError(f"信号 {s.id} 的前置信号不存在：{set(s.requires) - sig_ids}")
            if s.event and not q.event(s.event):
                raise ValueError(f"信号 {s.id} 指向不存在的事件 {s.event}")
            if s.ask is not None and not (s.ask.get("label") and s.ask.get("text")):
                raise ValueError(f"信号 {s.id} 的 ask 要同时写 label 和 text")
        for a in q.acts:
            if set(a.complete_when) - sig_ids:
                raise ValueError(f"{a.id} 的完成条件不存在：{set(a.complete_when) - sig_ids}")
            for item in a.checklist:
                if not item.get("label") or set(item.get("clues_any", [])) - set(q.clues):
                    raise ValueError(f"{a.id} 的 checklist 项有误：{item}")
        for p in self.personas.values():
            for slot in p.schedule:
                w.loc(slot["location"])
            for other in list(p.confidants) + list((p.outing or {}).get("tell", [])):
                if other not in self.personas:
                    raise ValueError(f"{p.id} 的 confidants / outing.tell 里有不存在的人：{other}")
        for e in q.events:
            w.loc(e.location)
        for lid in w.locations:
            for dest in w.exits(lid):
                w.loc(dest)
            for o in w.objects(lid):
                if o.get("clue") and o["clue"] not in q.clues:
                    raise ValueError(f"{lid}/{o['id']} 引用了不存在的线索 {o['clue']}")
        GodEnv(q, w.phases, w.start).topo_order(list(sig_ids))
        if q.companion not in self.personas:
            raise ValueError(f"主角色 {q.companion} 没有人物卡")

    def save(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.save_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"world": self.god.to_dict(),
                                   "relationships": {nid: a.relationship.to_dict() for nid, a in self.npcs.items()}},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.save_path)

    def intro(self) -> None:
        self.out(f"\n════ {self.quest.title} ════\n")
        if self.resumed:
            self.out(f"（读取存档：{self.god.status_line()}）")
        else:
            for block in (self.quest.primer, self.quest.how_to_play):
                if block:
                    self.out("\n".join(block) + "\n")
            for para in self.quest.prologue:
                self.out(para + "\n")

    # ------------------------------------------------------------ 主循环

    def _on_message(self, msg: dict) -> None:
        """口信送到：进收信人的记忆（之后检索得到），下次对话时出现在它的情境里。"""
        to, phys = msg["to"], self.god.physical
        if to in self.npcs:
            self.npcs[to].memory.add(MemoryItem(
                text=f"（{self.god.time_label()}）{phys.name(msg['from'])}托人带话：{msg['text']}", kind="message",
                importance=3, world_time=self.god.total_turns, extra={"from": msg["from"], "sent": msg["sent"]}))
        self.tracer.log("message", action="delivered", **{k: msg[k] for k in ("id", "from", "to", "kind", "text")})

    def _flush_notices(self) -> None:
        for line in self.god.notices:
            self.out("\n" + line)
        self.god.notices.clear()

    def run(self, max_actions: int = 400) -> GameResult:
        g = self.god
        result = GameResult(ending=g.ending)
        if g.ending and not self.continue_after_ending:
            self.out("这条主线已经结束了。用 --new 重新开始。")
            return result
        self.intro()
        g.notices += g.physical.tick()            # 开局 / 读档：先看一眼谁在身边
        h = self.settings.harness
        self.tracer.log("game_start", backend=self.backend.name, routes=self.router.describe(),
                        resumed=self.resumed, world=g.to_dict(),
                        harness={k: getattr(h, k) for k in h.__dataclass_fields__},
                        lore_chunks=len(self.lore.chunks), skills=list(self.skills.skills), npcs=list(self.npcs))
        show_panel = True
        self.director.run()                       # 记下开局的时段，之后每到新时段才跑
        for _ in range(max_actions):
            if g.resolved and not g.ending:
                self._finish(result)
                if not self.continue_after_ending:
                    break
                self._after_ending()
                show_panel = True
            elif g.deadline_reached() and not g.resolved and "deadline_passed" not in g.flags:
                if not self.continue_after_ending:
                    self._finish(result)
                    break
                self._deadline_event()
                show_panel = True
            if self.budget.exhausted():
                result.stopped_by_budget = True
                if not any(sr.end_reason is DialogueEndReason.BUDGET for sr in result.sessions[-1:]):
                    self.tracer.log("budget", action="stop", **self.budget.snapshot())
                self.out("\n（本局 token 预算已用完，进度已保存。调高 GAME_TOKEN_BUDGET 后可继续。）")
                break
            if show_panel:
                self.out(self.explorer.panel())
            show_panel = False
            menu = self.explorer.menu()
            self.out("")
            action = self.player.choose(menu, None)
            result.actions += 1
            self.tracer.log("action", kind=action.kind, text=action.text, meta=action.meta, time=g.time_label(),
                            location=g.location)
            if action.kind == "quit":
                result.stopped_by_player = True
                break
            if action.kind in ("talk", "event"):
                npc = self.npcs[action.meta["npc"]]
                ev = self.quest.event(action.meta["event"]) if action.kind == "event" else None
                r = DialogueSession(npc, self.player, self.option_gen, g, self.world, self.settings, self.reviewer,
                                    self.budget, event=ev, compressor=self.compressor, out=self.out,
                                    tracer=self.tracer, branch=self.branch,
                                    player_view=self.explorer.player_view).run()
                result.sessions.append(r)
                self._pass_word(npc, r, ev)
                self.tracer.log("session_end", npc=r.npc, end_reason=r.end_reason.name, turns=r.turns,
                                signals=r.signals, clues_gained=r.clues_gained, degraded_turns=r.degraded_turns,
                                world=g.to_dict(), budget=self.budget.snapshot())
                show_panel = True
                if r.end_reason is DialogueEndReason.PLAYER_QUIT:
                    result.stopped_by_player = True
                    self.save()
                    break
                if r.end_reason is DialogueEndReason.BUDGET:
                    result.stopped_by_budget = True
            elif action.kind == "move":
                self.out("\n" + self.explorer.move(action.meta["dest"]))
                show_panel = True
            elif action.kind == "examine":
                desc, gained = self.explorer.examine(action.meta["obj"])
                self.out("\n" + desc)
                for c in gained:
                    self.out(f"    📜 获得线索「{self.quest.clue_name(c)}」")
                show_panel = bool(gained)                  # 有了新线索，下一步也跟着变了
            elif action.kind in ("look", "say"):          # 地点层的自由输入：想查看什么
                with self.tracer.span(agent="scene"):
                    desc, gained = self.scene.look(action.text, degraded=self.budget.degraded())
                self.out("\n" + desc)
                for c in gained:
                    self.out(f"    📜 获得线索「{self.quest.clue_name(c)}」")
                show_panel = bool(gained)
            elif action.kind == "wait":
                self.out("\n" + self.explorer.wait())
                show_panel = True
            elif action.kind == "dismiss":
                self.out("\n" + self.explorer.dismiss(action.meta["npc"]))
                show_panel = True
            elif action.kind == "view":
                self.out("\n" + self.explorer.view(action.meta["view"]))
            self._flush_notices()
            if not (g.resolved or g.deadline_reached()):
                with self.tracer.span(agent="director"):
                    for line in self.director.run(degraded=self.budget.degraded()):
                        self.out("\n" + line)
            self.save()
        self.save()
        result.ending = g.ending
        self.out("\n（进度已保存）")
        return result

    def _pass_word(self, npc: NPCAgent, r: SessionResult, ev) -> None:
        """见过玩家之后，NPC 会托人跟自己信得过的人说一声（人物卡 confidants）：只传话，不替玩家解锁证据。"""
        if ev or not r.summary or r.turns == 0:
            return
        text = f"{self.settings.game.player_name}来找过我。{r.summary.get('summary', '')}"
        for c in npc.p.confidants:
            if c in self.npcs and c != npc.p.id:
                msg = self.god.physical.send(npc.p.id, c, text, kind="note")
                self.tracer.log("message", action="sent", **{k: msg[k] for k in ("id", "from", "to", "kind", "text")},
                                deliver=msg["deliver"])

    def _deadline_event(self) -> None:
        """截止时间到了、主线还没解决：那件事照样发生（茶会照开），作为剧情播出来，然后继续——真相还可以接着查。"""
        g = self.god
        g.flags.append("deadline_passed")
        e = next((x for x in self.quest.endings if x.when.get("deadline_passed")), None)
        if e:
            self.out(f"\n════ {self.quest.deadline['label']} · {e.title} ════\n\n{e.text}\n")
        self.out("（主线还没有结束：真相还可以接着查下去。）")
        g.journal.append(f"{g.time_label()}　{self.quest.deadline['label']}已经过去了，伪信的事还没有查清")
        self.tracer.log("deadline_event", time=g.time_label())

    def _after_ending(self) -> None:
        self.out(self.quest.after_hint or "（主线告一段落。你可以继续在稻妻走走：约人同行、拜访旧识……想结束时选「保存并退出」。）")
        self.tracer.log("after_ending", ending=self.god.ending)

    def _finish(self, result: GameResult) -> None:
        g = self.god
        aff = self.companion.relationship.affinity
        e = g.compute_ending(aff)
        g.ending = e.id
        result.ending = e.id
        tier = self.companion.relationship.tier()[0]
        late = "deadline_passed" in g.flags and not e.when.get("deadline_passed")
        self.out(f"\n════ 结局：{e.title} ════\n\n" + (self.quest.late_note + "\n\n" if late and self.quest.late_note else "")
                 + f"{e.text}\n")
        if self.quest.epilogue_by_tier.get(tier):
            self.out(self.quest.epilogue_by_tier[tier] + "\n")
        self.out(f"（{g.time_label()} ｜ 线索 {len(g.clues)}/{len(self.quest.clues)} ｜ "
                 f"与{self.companion.name}：{tier}，好感 {aff}，信任 {self.companion.relationship.trust}）")
        self.tracer.log("ending", ending=e.id, title=e.title, flags=g.flags, clues=g.clues, affinity=aff, tier=tier)

    def stats_line(self) -> str:
        s = self.tracer.summary()
        stops = ",".join(f"{k}:{v}" for k, v in s["loop_stops"].items()) or "-"
        return (f"LLM 调用 {s['llm_calls']}｜JSON 修复 {s['json_repairs']}｜兜底 {s['fallbacks']}｜"
                f"导演 {s['director_runs']} 次（执行 {s['director_applied']}，驳回 {s['director_rejected']}）｜"
                f"错误 {s['llm_errors']}｜工具 {s['tool_calls']}（失败 {s['tool_errors']}，缓存 {s['tool_cache_hits']}）｜"
                f"循环结束 {stops}｜进度 声称{s['signals_claimed']}/采纳{s['signals_accepted']}，动作直达{s['signals_by_action']}｜"
                f"tokens {s['prompt_tokens']}+{s['completion_tokens']}（前缀缓存 {s['cache_hit_ratio']:.0%}）｜"
                + (f"花费 ${s['cost_usd']:.3f}｜" if s["cost_usd"] else "") +
                f"p50 {s['latency_p50']:.1f}s p95 {s['latency_p95']:.1f}s")
