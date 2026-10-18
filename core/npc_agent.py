"""NPCAgent：角色卡 + 关系 + 三层记忆 + 工具 + ReAct 循环。

三层记忆：
  工作记忆  WorkingMemory —— 本次见面的对话窗口 + 滚动摘要（短期，会话结束即重置）
  长期记忆  MemoryStore   —— 每次见面的摘要与事实（情节记忆，按需 recall_memory 检索）
  外部记忆  GodEnv 存档 + LoreBase 知识库 —— 世界状态与设定的权威来源（不在上下文里「记」，用时读取）
"""
from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from core.agent_loop import LoopResult, StopReason
from core.context import TurnContext
from core.skills import Skill, SkillLibrary
from core.tools import Tool, ToolRegistry
from llm.base import LLMError
from llm.npc_llm_adapter import NPCLLMAdapter
from llm.summary_llm_adapter import SummaryLLMAdapter
from models.knowledge import LoreBase
from models.memory import MemoryItem, MemoryStore
from models.personality import Personality
from models.relationship import Relationship
from models.retrieval import bigram_jaccard, coverage

REFLECT_AT = 6                    # 上次反思以来新记忆的重要度累计到这个数才反思（一条摘要约 3，一条事实约 4）
FACT_SIMILAR = 0.3                # 旧事实和新事实字面相似度到这个程度，才拿去问“新增 / 更新 / 作废 / 不用存”
REFLECTION_TTL = 80               # 反思的保鲜期（世界回合数）：太久以前琢磨出的看法不再带进上下文，也不再当“之前的看法”给模型看
SAME_QUESTION = 0.5               # 两个焦点问题字面相似到这个程度，就算同一个问题：新的结论替换旧的，而不是越攒越多
MAX_REFLECTIONS = 3               # 同一个人最多留几条反思
FACT_GROUNDING = 0.5        # 事实至少一半的词要能在对话原文里找到
HANDOFF_GROUNDING = 0.4     # 交接里的「没聊完的话题 / 约定」也要在原文里有依据，防止凭空多出一个承诺
REPEAT_THRESHOLD = 0.7     # 和自己最近几句话相似到这个程度，就算复读
REPEAT_WINDOW = 4
CANON_GROUNDING = 0.4       # 即兴设定要能在她这句台词里找到依据（允许改写）


@dataclass
class NPCReply:
    speech: str
    fit: str = "接得上"                                       # 对方这句话接不接得上（NPC 自己判的，见 FITS）
    action: str = ""
    emotion: str = "平静"
    intent: str = ""
    affinity_delta: int = 0
    trust_delta: int = 0
    tension_delta: float = 0.0
    task_signals: list[str] = field(default_factory=list)
    wants_to_end: bool = False
    new_facts: list[dict] = field(default_factory=list)     # 这一轮即兴编出、需要记住的设定
    lies: list[dict] = field(default_factory=list)          # 这一轮说的谎 [{said, truth, lie_id}]
    tell: list[dict] = field(default_factory=list)          # 托人带话 [{to, text}]（物理世界里按距离送达）
    go_to: str = ""                                         # 说完要去哪（地点名），代码校验后才动
    degraded: bool = False          # True = 兜底台词，不是模型生成的
    loop: Optional[dict] = None     # 本轮循环摘要：stop / forced_by / steps / tools / tokens

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("degraded")
        d.pop("loop")
        return d


class NPCAgent:
    def __init__(self, personality: Personality, relationship: Relationship, memory: MemoryStore,
                 adapter: NPCLLMAdapter, summarizer: SummaryLLMAdapter, lore: LoreBase, skills: SkillLibrary,
                 rng: Optional[random.Random] = None, tracer: Any = None,
                 world_canon: Optional[MemoryStore] = None, guard_terms: Optional[list[str]] = None,
                 persona_mode: str = "full"):
        self.p = personality
        self.relationship = relationship
        self.memory = memory
        self.adapter = adapter
        self.summarizer = summarizer
        self.lore = lore
        self.skills = skills
        self.rng = rng or random.Random()
        self.tracer = tracer
        self.world_canon = world_canon          # 所有角色共享：关于别人和稻妻的即兴设定
        self.guard_terms = guard_terms or []
        self.persona_mode = persona_mode
        self.updater = None                     # MemoryUpdateLLMAdapter：新事实和相近旧事实的取舍（引擎装配；没有就只按字面去重）
        self.reflector = None                   # ReflectionLLMAdapter：谈话结束后合成“我对对方的看法”（引擎装配）
        self.fs = None                          # ContextFS：按需翻看的上下文（引擎装配；没有就不提供 ls / read）
        self.registry = self._build_tools()

    @property
    def name(self) -> str:
        return self.p.name

    # ------------------------------------------------------------ 工具

    def _build_tools(self) -> ToolRegistry:
        def recall(args: dict) -> str:
            items = self.memory.retrieve(args["query"], k=3)
            shared = self.world_canon.retrieve(args["query"], k=2) if self.world_canon else []
            if not items and not shared:
                return "（没有相关的记忆）"
            return "\n".join([f"- {m.text}" for m in items] + [f"- （听{m.extra.get('by', '别人')}说过）{m.text}" for m in shared])

        def lore(args: dict) -> str:
            hits = self.lore.search(args["topic"], k=2)
            return "\n".join(f"【{c.title}】{c.text}" for c in hits) or "（知识库里没有相关条目，不要编造）"

        def load_skill(args: dict) -> str:
            s = self.skills.get(args["name"])
            return s.body if s else f"没有这个技能。可用技能：{', '.join(self.skills.skills)}"

        def load_persona(args: dict) -> str:
            from models.personality import DETAIL_SECTIONS
            body = self.p.detail(args["section"])
            return body or f"没有这一类。可查：{'、'.join(DETAIL_SECTIONS)}"

        def fs_ls(args: dict) -> str:
            return self.fs.ls(args.get("path", "/")) if self.fs else "（不可用）"

        def fs_read(args: dict) -> str:
            return self.fs.read(args.get("path", "/")) if self.fs else "（不可用）"

        files = [Tool("ls", "列出可以翻看的上下文目录（/地点 /人物 /我 /案子 /稻妻）", {"path": "目录，如 /地点"}, fs_ls, cacheable=True),
                 Tool("read", "翻看一个上下文文件：某个地方离这儿多远、你和某人的关系和你知道的去向、你的记忆、你知道的案情",
                      {"path": "文件路径，如 /地点/璃月"}, fs_read, cacheable=False)]
        extra = [Tool("load_persona", "查看你自己人物细节的某一类全文（喜好 / 人际 / 看法 / 往事）",
                      {"section": "类别名"}, load_persona, cacheable=True)] if self.persona_mode == "index" else []
        return ToolRegistry(files + extra + [
            Tool("recall_memory", "回忆你与对方过去见面的经历", {"query": "想回忆的内容"}, recall, cacheable=True),
            Tool("lookup_lore", "查阅稻妻的风物、人物与制度设定，避免说错", {"topic": "要查的主题"}, lore,
                 cacheable=True),
            Tool("load_skill", "加载某个技能的完整做法", {"name": "技能名"}, load_skill, cacheable=True),
            Tool("respond", "对对方说话并结束本步（参数见下方示例）", {}, None, terminal=True),
        ])

    def recall(self, query: str, k: int, now: int, exclude_kinds: tuple = ()) -> list[str]:
        return [m.text for m in self.memory.retrieve(query, k=k, now=now, exclude_kinds=exclude_kinds)]

    def recall_by_impression(self, now: int, k: int = 2) -> list[str]:
        """用“我对这个人的整体印象”（最近一条反思）当检索焦点，翻出和这份印象有关、但玩家这句话没提到的往事
        （Generative Agents 对话前的做法：先总结和对方的关系，再拿它去检索）。还没有反思就什么都不翻。"""
        text = self.latest_reflection(now)
        if not text:
            return []
        focus = re.sub(r"^我对[^：]{1,12}的看法：", "", text.split("；下次我打算")[0])[:100]
        return self.recall(focus, k, now, exclude_kinds=("reflection",))

    def recall_canon(self, query: str, k: int, now: int) -> list[str]:
        """之前即兴编出的设定：自己的 + 共享世界的，按当前话题检索。"""
        own = [m for m in self.memory.retrieve(query, k=k * 3, now=now) if m.kind == "canon"]
        shared = self.world_canon.retrieve(query, k=k, now=now) if self.world_canon else []
        items = sorted(own[:k] + shared, key=lambda m: m.world_time)
        return [m.text + (f"（{m.extra.get('by')}说过）" if m.extra.get("by") and m.extra.get("by") != self.name else "")
                for m in items]

    def remember_improvisations(self, reply: "NPCReply", now: int, provenance: dict) -> dict:
        """即兴设定立刻入库（不等交谈结束），这样同一次交谈里后面就能保持一致。
        三道关：必须真的说出口了（防止只写进记忆却没说）、不许涉及案情（防止即兴编出线索绕过调查）、相似合并。"""
        said = reply.speech + reply.action
        kept, dropped = [], []
        for f in reply.new_facts:
            fact = f["fact"]
            hit = [t for t in self.guard_terms if t in fact]
            if hit:
                dropped.append({"fact": fact, "reason": f"涉及案情（{'、'.join(hit)}），不允许即兴编造"})
                continue
            if coverage(fact, said) < CANON_GROUNDING:
                dropped.append({"fact": fact, "reason": "这句话里并没有说出这件事"})
                continue
            store = self.world_canon if f["about"] == "world" and self.world_canon else self.memory
            item, merged = store.add(MemoryItem(text=fact, kind="canon", importance=4, world_time=now,
                                                extra={**provenance, "by": self.name, "about": f["about"]}))
            kept.append({"fact": fact, "about": f["about"], "merged": merged})
        if self.tracer and (kept or dropped):
            self.tracer.log("canon_write", npc=self.p.id, kept=kept, dropped=dropped)
        if kept:
            self.registry.cache.clear()
        return {"kept": kept, "dropped": dropped}

    def record_lies(self, reply: "NPCReply", god, now: int) -> dict:
        """说出口的谎才记：进证词本（玩家看到的和真话一样），也进她自己的记忆（之后要圆住）。"""
        kept, dropped = [], []
        blocked = god.blocked_terms(self.p.id)
        for lie in reply.lies:
            if coverage(lie["said"], reply.speech) < CANON_GROUNDING:
                dropped.append({**lie, "reason": "这句话里并没有说出这个谎"})
                continue
            unknown = [t for t in blocked if t in lie["truth"] or t in lie["said"]]
            if unknown:      # “真相”是模型写的：里面带着他此刻不可能知道的东西，就是编出来的，不能存进他的记忆和证词本
                dropped.append({**lie, "reason": f"真相里带着你不该知道的内容（{'、'.join(unknown[:3])}）"})
                continue
            pre = god.quest.lie(lie["lie_id"]) if lie["lie_id"] != "improvised" else None
            t = god.add_testimony(self.p.id, pre.said if pre else lie["said"], "lie", truth=lie["truth"],
                                  lie_id=lie["lie_id"] if pre else "",
                                  contradicted_by=pre.contradicted_by if pre else [])
            self.memory.add(MemoryItem(text=f"我对对方撒过谎：「{t['said']}」（真相：{lie['truth']}）",
                                       kind="lie", importance=4, world_time=now, extra={"testimony": t["id"]}))
            kept.append({"said": t["said"], "lie_id": t.get("lie_id"), "testimony": t["id"]})
        if self.tracer and (kept or dropped):
            self.tracer.log("lie", npc=self.p.id, kept=kept, dropped=dropped)
        return {"kept": kept, "dropped": dropped}

    def last_meeting(self) -> str:
        """上一次见面的摘要——确定性注入，保证跨见面的连续性不依赖检索命中。"""
        items = [m for m in self.memory.items if m.kind == "dialogue_summary"]
        return items[-1].text if items else ""

    def handoff(self) -> dict:
        """上下文重置后的结构化交接：每次见面工作记忆清零，接上次的只有这份交接（不是整段对话摘要）。"""
        items = [m for m in self.memory.items if m.kind == "dialogue_summary"]
        if not items:
            return {}
        m = items[-1]
        return {"summary": m.text, "time": m.extra.get("time", ""), "end_reason": m.extra.get("end_reason", ""),
                "minute": m.extra.get("minute"), "where": m.extra.get("where"),
                "open_threads": m.extra.get("open_threads", []), "promises": m.extra.get("promises", []),
                "feeling": m.extra.get("npc_feeling", "")}

    # ------------------------------------------------------------ 发言

    def style_tics(self, ctx: TurnContext) -> list[str]:
        """最近几句话里的固定套路：只数、不评判。数出来的事实交给模型，换不换、怎么换由它决定。"""
        mine = [h["text"] for h in ctx.history if h["speaker"] == "npc"][-4:]
        if len(mine) < 3:
            return []
        notes = []
        q = sum(1 for m in mine if m.rstrip("」』 ").endswith(("？", "?")))
        if q >= 3:
            notes.append(f"最近 {len(mine)} 句里有 {q} 句是以反问/提问收尾的——这一句用陈述收，把话说完，别又抛回一个问题。")
        heads = [m.lstrip("（…。，、 ")[:2] for m in mine]
        top = max(set(heads), key=heads.count)
        if len(top) == 2 and heads.count(top) >= 3:
            notes.append(f"最近几句都以「{top}」开头——换个开头，或者直接说事。")
        terms = [t for line in self.p.speech_style for t in re.findall(r"「([^」]{2,8})」", line)]
        terms = [t for t in terms if not t.startswith("{")]
        for t in dict.fromkeys(terms):
            n = sum(1 for m in mine[-3:] if t in m)
            if n >= 2:
                notes.append(f"「{t}」最近 3 句里用了 {n} 次——这一句不用。")
        runs: dict[str, int] = {}
        for m in mine[-4:]:
            cjk = re.sub(r"[^㐀-鿿]", "", m)
            for g in {cjk[i:i + 4] for i in range(len(cjk) - 3)}:
                runs[g] = runs.get(g, 0) + 1
        rep = [g for g, n in runs.items() if n >= 3]
        if rep:
            notes.append(f"「{rep[0]}」这样的说法在最近几句里反复出现——换一种。")
        return notes

    def _echoes(self, ctx: TurnContext, speech: str) -> list[str]:
        mine = [h["text"] for h in ctx.history if h["speaker"] == "npc"][-REPEAT_WINDOW:]
        return [m for m in mine if bigram_jaccard(m, speech) >= REPEAT_THRESHOLD]

    def _call(self, ctx: TurnContext, text: Optional[str], intent: Optional[str], skills: list[Skill],
              flags: list[str], allow_tools: bool) -> NPCReply:
        ctx.tics = self.style_tics(ctx) if text is not None else []
        blocked = ctx.task.get("blocked_terms") or {}
        # 名字本身就是他此刻“还不该知道”的词的人（比如守夜狐），不告诉记录员：他不认识，也不会替对方带话给这个人
        names = [] if self.fs is None else [q.name for n, q in self.fs.personas.items()
                                            if n != self.p.id and not any(t == q.name or t in q.name for t in blocked)]
        places = [] if self.fs is None else [self.fs.world.name(l) for l in self.fs.world.locations]
        res: LoopResult = self.adapter.reply(self.p, ctx, text, intent, self.registry, self.skills.index(),
                                             skills, flags, allow_tools, names, places)
        info = {"stop": res.stop.value, "forced_by": res.forced_by.value if res.forced_by else None,
                "steps": len(res.steps), "tools": [s.tool for s in res.steps if s.tool != "respond"],
                "tokens": res.tokens, "elapsed_s": res.elapsed_s}
        if res.stop is StopReason.FINISHED and res.final:
            reply = NPCReply(**res.final, loop=info)
            echoed = self._echoes(ctx, reply.speech) if text is not None and not ctx.avoid else []
            if echoed:                  # 复读由代码比对出来，怎么换说法交给模型：带着「刚说过」的提示重来一次
                if self.tracer:
                    self.tracer.log("repeat", npc=self.p.id, speech=reply.speech, echoes=echoed)
                ctx.avoid = echoed
                try:
                    return self._call(ctx, text, intent, skills, flags, allow_tools)
                finally:
                    ctx.avoid = []
            return reply
        if self.tracer:
            self.tracer.log("fallback", where="npc_loop", stop=res.stop.value, error=res.error)
        r = self._fallback(ctx, intent)
        r.loop = info
        return r

    def opening(self, ctx: TurnContext, skills: list[Skill], allow_tools: bool = True) -> NPCReply:
        return self._call(ctx, None, None, skills, [], allow_tools)

    def respond(self, ctx: TurnContext, text: str, intent: Optional[str], skills: list[Skill],
                flags: list[str], allow_tools: bool = True) -> NPCReply:
        return self._call(ctx, text, intent, skills, flags, allow_tools)

    def farewell(self, ctx: TurnContext) -> NPCReply:
        """玩家退出 / 硬兜底收束时的礼貌告别，不调 LLM。"""
        line = self.rng.choice(self.p.farewell_lines or ["那么，改日再会。"])
        return NPCReply(speech=line.format(player_name=ctx.player_name), action="欠身行礼",
                        intent="道别", wants_to_end=True)

    def _fallback(self, ctx: TurnContext, intent: Optional[str]) -> NPCReply:
        emotion = "不悦" if intent == "冒犯" else "平静"
        pool = self.p.fallback_lines.get(emotion) or self.p.fallback_lines.get("平静") or ["……"]
        return NPCReply(speech=self.rng.choice(pool).format(player_name=ctx.player_name),
                        action="略微沉吟", emotion=emotion, intent="缓和", degraded=True,
                        wants_to_end=intent == "告别")

    def apply_reply(self, reply: NPCReply) -> None:
        self.relationship.apply(reply.affinity_delta, reply.trust_delta)

    # ------------------------------------------------------------ 长期记忆

    def remember_session(self, ctx: TurnContext, transcript: str, end_reason: str, turns: int,
                         world_time: int, provenance: dict) -> dict:
        try:
            s = self.summarizer.summarize(self.name, ctx.player_name, transcript, end_reason, turns)
        except LLMError as e:
            if self.tracer:
                self.tracer.log("fallback", where="summary", error=str(e))
            s = {"summary": f"我在{ctx.scene['name']}与{ctx.player_name}谈了{turns}轮，"
                            f"谈的是「{ctx.task['stage_title']}」相关的事。",
                 "topics": [ctx.task["stage_title"]], "player_intent": "", "npc_feeling": "",
                 "facts": [], "importance": 2}
        rel = self.relationship.summary()
        handoff = {k: [x for x in s.get(k, []) if coverage(x, transcript) >= HANDOFF_GROUNDING]
                   for k in ("open_threads", "promises")}
        handoff_dropped = {k: [x for x in s.get(k, []) if x not in handoff[k]] for k in handoff}
        extra = {**provenance, **handoff, "player_intent": s["player_intent"], "npc_feeling": s["npc_feeling"],
                 "end_reason": end_reason, "scene": ctx.scene["id"], "stage": ctx.task["stage_title"],
                 "affinity": rel["affinity"], "trust": rel["trust"]}
        _, merged = self.memory.add(MemoryItem(text=s["summary"], kind="dialogue_summary",
                                               importance=s["importance"], world_time=world_time,
                                               topics=s["topics"], extra=extra))
        kept, dropped = [], []
        for fact in s["facts"]:
            (kept if coverage(fact, transcript) >= FACT_GROUNDING else dropped).append(fact)
        fact_updates = self._store_facts(kept, s, world_time, provenance)
        if self.tracer:
            self.tracer.log("memory_write", summary=s["summary"], merged=merged, facts_kept=kept,
                            facts_dropped=dropped, handoff=handoff, handoff_dropped=handoff_dropped,
                            provenance=provenance, fact_updates=fact_updates)
        self._reflect(ctx, transcript, turns, world_time, provenance)
        self.registry.cache.clear()            # 记忆变了，recall 的缓存作废
        return s

    def _store_facts(self, facts: list[str], s: dict, world_time: int, provenance: dict) -> list[dict]:
        """新事实入库。有相近旧事实的，问一次小模型：新增 / 更新旧的 / 旧的作废 / 不用存（Mem0）。
        模型只在给定的几条旧事实里选；更新后的句子要有依据；没有更新器或调用失败，就退回原来的字面去重。"""
        olds = [m for m in self.memory.items if m.kind == "fact"]
        near = [[(m.id, m.text) for m in sorted(olds, key=lambda m: -bigram_jaccard(m.text, f))[:3]
                 if bigram_jaccard(m.text, f) >= FACT_SIMILAR] for f in facts]
        asked = [i for i, nb in enumerate(near) if nb]
        decisions: dict[int, dict] = {}
        if asked and self.updater is not None:
            got = self.updater.decide(self.name, [facts[i] for i in asked], [near[i] for i in asked])
            decisions = {asked[k]: v for k, v in got.items()}
        log: list[dict] = []
        for i, fact in enumerate(facts):
            item = MemoryItem(text=fact, kind="fact", importance=min(5, s["importance"] + 1), world_time=world_time,
                              topics=s["topics"], extra=dict(provenance))
            d = decisions.get(i)
            if d is None:                                   # 没有相近旧事实 / 没有更新器 / 调用失败：原来的做法
                self.memory.add(item)
                continue
            target = next((m for m in olds if m.id == d["id"]), None)
            if d["event"] == "NONE":
                log.append({"fact": fact, "event": "NONE"})
            elif d["event"] == "UPDATE" and target and coverage(d["text"], fact + target.text) >= FACT_GROUNDING:
                self.memory.revise(target, d["text"], world_time)
                log.append({"fact": fact, "event": "UPDATE", "id": target.id, "text": d["text"]})
            elif d["event"] == "DELETE" and target:
                self.memory.drop([target])
                olds.remove(target)
                self.memory.add(item, dedup=False)
                log.append({"fact": fact, "event": "DELETE", "id": target.id, "old": target.text})
            else:                                           # ADD，或没通过依据检查的 UPDATE：新增（不再按字面相似度合并）
                self.memory.add(item, dedup=False)
                log.append({"fact": fact, "event": "ADD"})
        return log

    def _reflect(self, ctx: TurnContext, transcript: str, turns: int, world_time: int, provenance: dict) -> None:
        """谈话够长就反思一次：合成“我对对方的看法 / 疑问”，只留最近 3 条，下次见面放进上下文。"""
        if self.reflector is None or turns < 2:
            return
        # 生成式智能体的做法：记忆的重要度累计到阈值才反思（而不是每次都反思）；反思的结论要挂着依据的记忆编号
        items = self.memory.items
        last = max((i for i, m in enumerate(items) if m.kind == "reflection"), default=-1)       # 库里只追加：位置就是先后
        fresh = [m for m in items[last + 1:] if m.kind in ("dialogue_summary", "fact")]
        if sum(m.importance for m in fresh) < REFLECT_AT:
            if self.tracer:
                self.tracer.log("reflection_skipped", npc=self.p.id, accumulated=sum(m.importance for m in fresh), need=REFLECT_AT)
            return
        previous = [m.text for m in self._live_reflections(world_time)][-2:]
        shown, groups, focus = self._focal_pool(ctx, fresh, transcript, world_time)
        r = self.reflector.reflect(self.name, ctx.player_name, transcript, previous, [m.text for m in shown], groups)
        if not r:
            return
        text = f"我对{ctx.player_name}的看法：{r['impression']}"
        if r["doubts"]:
            text += "；我的疑问：" + "；".join(r["doubts"])
        if r["next"]:
            text += "；下次我打算：" + r["next"]
        item, _ = self.memory.add(MemoryItem(text=text, kind="reflection", importance=4, world_time=world_time,
                                             topics=["看法", "疑问"],
                                             extra={**provenance, "evidence": [shown[i].id for i in r.get("evidence", []) if i < len(shown)],
                                                    **({"focus": focus} if focus else {})}))
        retired = self._retire_reflections(item, focus, world_time)
        if self.tracer:
            self.tracer.log("reflection", npc=self.p.id, text=text, retired=retired)

    def _live_reflections(self, now: int) -> list:
        return [m for m in self.memory.items if m.kind == "reflection" and now - m.world_time <= REFLECTION_TTL]

    def _retire_reflections(self, keep: MemoryItem, focus: list[str], now: int) -> list[dict]:
        """旧反思的去留：问的是同一个问题的，新结论替换旧结论；过了保鲜期的，丢；剩下的最多留 MAX_REFLECTIONS 条。
        （Talemate 的“强化”也是这样：同一个问题的新答案覆盖旧答案，而不是追加。）"""
        gone: list[tuple[MemoryItem, str]] = []
        for m in [m for m in self.memory.items if m.kind == "reflection" and m is not keep]:
            if any(bigram_jaccard(q, o) >= SAME_QUESTION for q in focus for o in m.extra.get("focus", [])):
                gone.append((m, "same_question"))
            elif now - m.world_time > REFLECTION_TTL:
                gone.append((m, "expired"))
        left = [m for m in self.memory.items if m.kind == "reflection" and all(m is not g for g, _ in gone)]
        gone += [(m, "overflow") for m in left[:-MAX_REFLECTIONS]]
        self.memory.drop([m for m, _ in gone])
        return [{"id": m.id, "why": why} for m, why in gone]

    def _focal_pool(self, ctx: TurnContext, fresh: list, transcript: str, world_time: int):
        """焦点问题 → 各自检索 → 合并成一份带编号的记忆池。返回 (记忆池, [(问题, [池里的下标])], 问题列表)。
        模型挑不出问题 / 检索不到东西，就退回只看最近几条（原来的做法）。"""
        recent = fresh[-8:]
        ask = getattr(self.reflector, "focal_questions", None)
        questions = ask(self.name, ctx.player_name, [m.text for m in recent], transcript) if ask else []
        pool: list = []
        groups: list[tuple[str, list[int]]] = []
        for q in questions:
            idx = []
            for m in self.memory.retrieve(q, k=4, now=world_time, exclude_kinds=("reflection",)):
                at = next((j for j, p in enumerate(pool) if p is m), None)
                if at is None:
                    pool.append(m)
                    at = len(pool) - 1
                idx.append(at)
            if idx:
                groups.append((q, idx))
        if not pool:
            return recent, [], []
        pool = pool[:10]
        groups = [(q, [i for i in idx if i < len(pool)]) for q, idx in groups]
        if self.tracer:
            self.tracer.log("reflection_focus", npc=self.p.id, questions=[q for q, _ in groups],
                            retrieved=[{"q": q, "memories": [pool[i].id for i in idx]} for q, idx in groups])
        return pool, [g for g in groups if g[1]], [q for q, _ in groups]

    def latest_reflection(self, now: Optional[int] = None) -> str:
        """最近一次琢磨出的看法。给了 now 就跳过过了保鲜期的。"""
        items = [m.text for m in (self._live_reflections(now) if now is not None else self.memory.items) if m.kind == "reflection"]
        return items[-1] if items else ""
