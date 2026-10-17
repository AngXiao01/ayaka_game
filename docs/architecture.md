# Harness 架构：参考成熟做法后的分层

## 参考的成熟思路

| 思路 | 要点 | 来源 |
|---|---|---|
| 记忆流 + 检索 + 反思（Generative Agents） | 经历进记忆流；取用时按近因、重要度、相关度打分取前几条；定期把零碎经历“反思”成高层结论；计划被新观察推翻时只重做受影响的一段 | [Generative Agents Memory Stream](https://agentpatterns.ai/patterns/agent-design/generative-agents-memory-stream/) |
| 游戏 NPC 对话管线 | 输入安全 → 载入角色 → 三层记忆（近期对话 / 重要事件 / 世界状态）→ **按角色“知道什么”过滤的设定检索（knowledge_scope / excluded_scope）** → 路由 → 生成 → **生成后检查（越界、人设漂移、设定违规）** → 记录与提交；**重要事件由游戏逻辑写，不从聊天里推** | [Building an AI NPC system](https://www.respan.ai/resources/building-ai-npc-system) |
| 回归与线上监控 | 全链路 trace；抽样监控角色完整度、设定违规率；用线上样本和对抗用例建回归集，指标不达标就拦发布 | 同上 |
| Storylet（条件 + 结果） | 事件 = 条件（对世界状态的真假陈述）+ 结果；一个模块判“哪些条件满足了”，另一个模块把待发生的结果织进下一句 | [Dramamancer](https://arxiv.org/html/2601.18785v1)、[Drama Llama](https://alphaxiv.org/abs/2501.09099) |
| Lorebook / World Info | 设定条目按关键词在最近几条消息里触发，有字数预算，超了就丢；常驻条目先进；条目不保证被模型用到 | [SillyTavern World Info](https://docs.sillytavern.app/usage/core-concepts/worldinfo/) |
| 符号 + 模型 | 模型只在代码给出的动作集里选；世界状态是符号的、可审计的；确定性的作者控制和可调试性很重要 | [LLM 游戏 agent 综述](https://arxiv.org/pdf/2404.02039) |

## 我们的分层（对应关系）

```
┌─ 作者层  content/  ───────────────────────────── 剧情图（线索/信号=storylet）、secrets、layout、knows_clues、addressed_as
├─ 世界层  core/god_env.py physical.py agendas.py ─ 唯一真相：时间天色、线索、谁知道什么、谁在哪、秘密的开锁条件
│            事件由代码写（信号达成、传话、移动），不从聊天里推
├─ 上下文层  llm/npc_llm_adapter.py  ───────────── 分层小节（常驻 / 条件 / 检索）+ 字数预算（fit_state），按角色可见范围过滤
├─ 生成层  llm/npc_stages.py ───────────────────── 评估 → 说话 → 记录；每个调用单一职责
├─ 护栏层  core/guardrails.py + npc_audit ───────── 生成后检查：①知识边界（确定性，secrets）②事实/物理核对（模型）→ 带理由重说一次
├─ 执行层  dialogue_session.py ─────────────────── 任何改世界的动作（给线索、移动、传话）由代码校验后才执行
└─ 评测层  evals/regression.py claim_cases.py ──── 对抗式回归集：每个 trace 里的“假”都是一条用例
```

## 本轮新增

1. **secrets 注册表**（`quest.secrets` + `GodEnv.blocked_terms`）：真相里的关键事实，写明谁本来就知道、被出示什么线索后知道、哪个信号达成后知道。
   这是“知识范围 / 排除范围”的落地。
2. **确定性知识边界检查**（`core/guardrails.py`）：NPC 台词里出现他还不该知道的词（对方刚说过的词除外），不调模型就能判，违规带理由重说。
3. **上下文预算**（`fit_state`）：本轮情境块超预算时按层丢：先丢检索来的、再丢条件出现的；常驻和本轮关键小节不丢；丢了什么记进 trace。
4. **回归套件**（`evals/regression.py`）：11 条用例 + 全局不变量（不说不该知道的词、没有括号动作和英文、相邻不重复）。

## 还没做（按价值排序）

- **反思**：把一次谈话里零碎的事实，定期合成“我对旅行者的看法 / 我的疑问”，写进长期记忆（记忆流的第三步）。
- **Beat 选择**（storylet 的另一半）：现在信号 DAG 决定“什么能说”，导演只做轻微提议；可以让代码按“条件满足的待发生结果”挑一个织进下一句。
- **按需分路由**：评估/核对/记录用小模型，说话用好模型。
- **线上监控**：抽样统计越界率、重说率、兜底率，达不到就报警。
