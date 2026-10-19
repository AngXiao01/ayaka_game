# 神里绫华 · 叙事 Agent 游戏

以《原神》稻妻为背景的开放式叙事调查游戏（同人，故事为原创）。你和 NPC 自由对话、四处查看、出示线索，
最后和神里绫华商量怎么收场。运行时只用 Python 标准库；模型走 **OpenRouter**（OpenAI 兼容接口，模型可换），也可以接自建的 vLLM。

项目的重点不是故事，而是模型外面那层 **Harness**：

- **世界的事实归代码，措辞归模型。** 时间、线索、谁知道什么、谁在哪、剧情进度，都是代码维护的状态；模型只能“提议”，校验和落地由代码做。
  玩家几句话带偏不了剧情，模型被骗了也改不了进度。
- 有边界的 ReAct 循环、上下文组装与渐进式披露、三层记忆、外部校验的进度判断、输入防护、预算与降级、
  生成后的知识边界检查、结构化 trace、整局重放、事务与崩溃恢复。

## 故事

**雪见茶会与多出来的客人**（`content/`）：茶会只请了三位客人，结果来了两位“杜瓦尔”；找出破绽，决定怎么收场（剧情和取舍见 `content/_design/plot.md`）。

内容和引擎是分开的：故事是一个目录（地图、人物卡、主线、设定、技能），默认读 `content/`，也可以用 `CONTENT_DIR=…` 指向任意目录。
各字段怎么写，看 `docs/code_map.md`；检查：`python3 tools/lint_story.py`。
旧的“月白失窃案”版本（含 `stories/` 目录里的两个故事、`evals/` 评测脚本）保存在 `archive/two-stories` 分支。

## 快速开始

```bash
python3 main.py --mock --debug              # 离线：Mock 后端，不联网不花钱
python3 main.py --mock --auto cooperative   # 离线 + AI 扮演玩家，自动跑完一局
python3 main.py --debug                     # 连 OpenRouter 玩（先放好 key，见下）
python3 web/server.py                       # 网页版，默认 http://127.0.0.1:8770
python3 main.py --check                     # 检查后端：连通性、模型 id、结构化输出
~/miniforge3/envs/torch_arm/bin/python -m pytest -q    # 测试（全离线）
```

常用参数：`--save 名字`、`--new`（清掉存档重来）、`--incognito`（不留存档和 trace）、`--no-hints`、`--options static`、`--max-turns N`、`--seed N`。

## 模型与后端

默认后端 `LLM_BACKEND=openrouter`，只读 `OPENROUTER_*` 变量；`gemma`（自建 vLLM，`GEMMA_*`）和 `mock` 同理，互不串用变量。所有变量见 `env.example`。

```bash
pbpaste > ~/.openrouter_key && chmod 600 ~/.openrouter_key   # key 放文件里，不进 shell 历史（默认读这个文件）
# 或者 .env 里写 OPENROUTER_API_KEY=...                       # trace 里会自动打码
python3 main.py --check                                      # 依次查：模型列表 → key → 普通对话 → 结构化 NPC 回复
```

- **模型**：`OPENROUTER_MODEL=cheapest`（默认）启动时按 `llm/model_picker.py` 的规则挑能用的最便宜模型；做评测或想固定就直接写模型 id。
  `python3 tools/pick_model.py --probe 5 --write` 列排名并对前几名实测结构化回复，把全过里最便宜的写进 `.env`。
- **按任务分流**：`MODEL_SMALL`（评估、核对、记录等小活）、`MODEL_REVIEW` 等 `MODEL_<TASK>`。
- **免费优先**：`FREE_FIRST=1`，先试免费模型，失败这一次回落到付费模型；实测免费档延迟高，默认关。
- **推理默认关**（`OPENROUTER_REASONING_EFFORT=none`）：推理 token 按输出计费，还会吃掉 `max_tokens`。
- **便宜的判定器**：`DECIDER_MODE=auto|jev|llm|off`，威胁、告别、邀请、哪个动词这类“回答是/否”的判断走 OpenRouter 的 Decisions API（Jev），拿不准再交给大模型；mock 后端永远不联网判定。
- **连接复用**：`llm/http_pool.py` 每线程每主机保持连接。在代理 / VPN 不稳的网络上，TLS 握手会让小请求等十几秒，复用后中位延迟明显下降。
- **并行判定**：`PARALLEL_JUDGES=1` 把一句话的几次判定一起发出，每轮约省 0.3~1 秒，默认关（见 `docs/turn_graph.md`）。
- 401 时报错会写 key 从哪读的和 key 的“形状”（长度、前缀、有没有空白或引号），不打印 key 本身。

## 分享给朋友玩

```bash
python3 web/server.py --share --code 你定的邀请码 --max-games 5
```

一人一局，要邀请码；每个浏览器各玩各的，关掉页面再来接着玩；同时最多 `--max-games` 局，闲置 30 分钟的局自动存档并释放名额；
每局 token 上限 `--budget`（默认 60 万）。服务只听 `127.0.0.1`，公网要经过你自己起的隧道；没有邀请码只能看到登录页，接口一律 401。

## 玩法（引擎提供的，各故事按需使用）

- **时间与地图**：时间按钟点走，说话便宜、走路花时间；清晨 / 午后 / 黄昏 / 夜晚由钟点推出，决定谁在哪。到了截止时间主线不会强行结束，作为一段剧情播出来，之后还能接着查。
- **对话**：选项（0~3 句台词 + 动作）或直接打字。动作有：❓ 关键问题、🔍 出示线索、📣 汇报、⚖️ 提议、🚶 邀请同行、告辞。
  打字说出手上线索的内容就等于出示；NPC 自己说出关键的事，线索当场到手。
- **线索与进度**：问人、出示线索、查看物件拿到线索；NPC 说“达成了”不算数，要过规则、独立审核和证据落地（`docs/architecture.md`）。
- **软锁与硬问**：证据或信任不够，关键问题变成 🎲 硬问，成功率由给对方看过的证据、信任、对方有没有起戒心决定，界面显示每一项加减；问砸了对方会起戒心。
- **自由行动**：在地点上直接打字——跟踪、蹲守、搜查、告发、通风报信……代码归类动词、检查条件、掷骰、写回状态。掷骰结果由存档种子 + 事项 + 当前分钟决定，读档重来同一刻不变。
- **NPC 有自己的打算**：到时候或条件满足了会自己行动（躲起来、去别处、抢先一步），玩家恰好在场会撞见。
- **同行**：答不答应由代码按好感和对方此刻有没有空决定；同行的人到点会自己回去。
- **谁知道谁在哪**：每个人只知道亲眼看到的和别人告诉的，没有消息就按作息猜。
- **带话**：NPC 见过你之后会托人跟信得过的人说一声，口信按路程送达；只传话，不替你解锁证据。
- **说谎与对质**：有人会说谎，真话谎话在证词本里看不出区别；⚠ 对质时挑一句话再拿一条线索当证据，结果由规则判定。
- **即兴与一致**：聊到预设之外可以即兴编细节，但编了就当场记下，之后必须一致；不许即兴编案情。
- **说出推测**：直接说出自己的推测，对方不会就此认定，也不会置之不理，会派人去查，结果托人带回。
- **新手引导**：开场有背景速览和“怎么玩”，菜单和网页侧栏有名词速查；“🧭 下一步”由世界状态算出（找谁、他什么时段在哪、先出示哪条线索），`--no-hints` 关掉。
- `--debug` 显示好感 / 信任变化、工具调用、未被采纳的进度。

## 架构

### 先把几个词说清楚

| 词 | 在这个项目里指什么 |
|---|---|
| **代码部分** | 确定性的规则：`GodEnv`、`PhysicalWorld`、`Explorer`、输入防护、知识边界检查、掷骰、预算……没有模型参与，世界的真相只有它们能改 |
| **模型调用** | 一次“给定输入 → 类型化 JSON 输出”的请求，输出由代码校验后才可能生效。绝大多数“agent”其实都是这一种 |
| **agent** | 习惯叫法：每个单一职责的模型调用都叫 agent（导演、审核员、选项生成器……）。严格说，**只有 NPC 说话那一步**是会自己决定下一步做什么的循环（有界 ReAct，`core/agent_loop.py`），其余都是一问一答 |
| **adapter**（`llm/*_adapter.py`） | agent 的“模型一侧”：拼提示词、选温度和 token 上限、调后端、校验 / 修复 JSON、转成代码能直接用的结构。决策逻辑和过滤在 `core/` 里，两层分开是为了离线测试（换成 `MockBackend`）和整局重放 |
| **GOD / TREE / ENV** | 三层“真相 + 它的模型提议者”，**不是三个 agent**，见下表 |

### 状态分层

```
GodEnv（剧情真相 + 时钟）     线索、信号、幕、结局；按分钟走的时钟
  └─ PhysicalWorld（空间真相）  谁在哪 = 作息 + 计划；同行队伍；路上的口信
       └─ 认知（每个 agent 一份） 亲眼看到的 + 别人带的话；没有就按作息猜

GameEngine（Harness 外壳：组装、存档 = 事务提交点、预算、trace）
 └─ 地点层循环（Explorer，确定性）：交谈 / 查看 / 移动 / 等待 / 线索本 / 人物 / 日志
     └─ DialogueSession（一次交谈）：一轮 = 19 个节点（TURN_NODES）
          玩家输入 → 输入防护 → 判定 → 结算 → 上下文 → NPC 回复（ReAct 循环）→ 进度审核 → 世界结算 → 读人 → 收尾
```

### GOD / TREE / ENV：模型提议，代码裁决

| 层 | 代码部分（拍板） | 模型部分（提议） | 什么时候跑 | 代码怎么管它 |
|---|---|---|---|---|
| GOD | `GodEnv`（`core/god_env.py`） | 导演 `Director`（`core/director.py`）：`nudge` 让谁主动提什么、`rumor` 放传闻、`move_npc`、`tension` | 每到新时段一次（一局十几次） | 动作是白名单；信号必须属于这个人且眼下有途径达成；传闻只在连续两个时段没进展时放；移动只能去他本来会去的地方；每次最多 2 个动作；**没有玩家原话**；预算降级或出错退回规则 |
| TREE | `quests/*.json` + 审核 + 依赖 DAG | 分支 `BranchAgent`（`core/branch.py`）：自由说出的处置方案 → 预设方案 | 每句自由发言（先问便宜判定器） | 只能在此刻可提出的方案里选或判 none；证据必须在原话里；注入轮不判；判定后走和菜单完全相同的那条路 |
| ENV | `World` + `Explorer` | 场景 `SceneAgent`（`core/scene.py`）：自由查看 → 物件，对不上就即兴描写 | 玩家在地点上自己写想看什么 | 即兴描写不给线索、不含案情关键词；新细节写出来才入共享设定 |

### 所有模型调用一览

“路由任务”是 `MODEL_<TASK>` 能单独换模型的粒度（`small` 这一档是一批小活的合称）；括号里是默认温度 / 输出上限。

| 调用 | 代码 | 路由任务 | 什么时候跑 | 输出 | 失败时 |
|---|---|---|---|---|---|
| **NPC 评估** | `llm/npc_stages.py` | small | 每轮，说话之前 | 接不接得上、情绪、好感信任张力的变化、这一轮打算怎么回 | 退回默认心态 |
| **NPC 说话**（唯一的循环） | `core/agent_loop.py` + `llm/npc_llm_adapter.py` | npc（0.8 / 700） | 每轮 | 只读工具若干次，最后 `respond` 交付台词 | 兜底台词 |
| **NPC 核对** | `llm/npc_stages.py` + `core/guardrails.py` | small | 每轮，说话之后 | 台词对照事实表，矛盾 / 编造就带理由重说一次；知识边界、锁住的事、漏口风这几项是**不调模型**的确定性检查 | 不核对 |
| **NPC 记录** | `llm/npc_stages.py` | small | 每轮 | 哪些待说的事说出口了、新设定、谎话、带话、去向 | “什么都没记” |
| **进度审核** `SignalReviewer` | `core/reviewer.py` | review（0.1 / 500） | 每轮，NPC 声称达成信号时 | 逐个信号判定 + 原文证据 | fail-closed：不推进 |
| **读人 ×2** | `llm/mind_llm_adapter.py`、`core/mind.py` | small | 每轮之后各一次 | NPC 以为玩家知道 / 想听什么（含愿不愿意说），玩家以为 NPC 知道什么 | 不更新 |
| **选项生成器** | `core/player_option_generator.py` + `llm/player_option_llm_adapter.py` | options（0.9 / 700） | 轮到玩家选的时候 | 0~3 句台词，**只有话本身**，不分类、不贴语气标签 | 用人物卡里的兜底台词 |
| 　└ 台词标注 | 同上 | small（0.0） | 紧跟选项生成 | 哪句话本身就在做“告辞 / 邀请 / 问关键问题 / 汇报 / 提议” | 都当普通台词 |
| 　└ 经历核对 | `core/judge.py` `unbacked` | 便宜判定器 | 紧跟选项生成 | 丢掉替玩家编了没去过的地方 / 没做过的事的句子 | 不丢 |
| **便宜判定器** | `core/judge.py`、`llm/decider.py` | 默认 Jev，拿不准用 branch 档大模型 | 每句自由发言 | 是 / 否、选哪个、打分：威胁、告别、邀请、哪个动词、说中哪条推测、说服力…… | 退回关键词；同问题有缓存；连续失败 3 次整局停用 |
| **声明抽取** | `core/claims.py` + `llm/claim_llm_adapter.py` | small | 玩家说自己做过什么时 | 抽出“我去过 X / 查过 Y”；**真假由代码对着真实足迹核对** | 不抽 |
| **分支** | `core/branch.py` + `llm/branch_llm_adapter.py` | branch（0.0 / 200） | 见上表 | 对应哪个预设方案 | 退回关键词规则 |
| **压缩** | `llm/summary_llm_adapter.py` `CompressLLMAdapter` | compress（0.2 / 400） | 对话窗口超过 8 轮 | 把最老的一批压成前情提要 | 抽取式 |
| **摘要** | `llm/summary_llm_adapter.py` `SummaryLLMAdapter` | summary（0.3 / 700） | 谈话结束 | 摘要、事实、没聊完的话题、说好的事 | 模板摘要 |
| **记忆更新** | `llm/memory_update_llm_adapter.py` | small | 谈话结束，新事实和旧事实相近时 | ADD / UPDATE / DELETE / NONE，只在给出的旧事实里选 | 退回字面去重 |
| **反思** | `llm/reflection_llm_adapter.py` + `core/npc_agent.py` | small | 谈话结束，且新记忆重要度累计够 | 见下“记忆是怎么写的” | 不反思 |
| **导演 / 场景** | 见上 | director（0.3 / 400）/ scene（0.7 / 300） | 见上 | 见上 | 退回规则 / “没发现什么” |

### 一轮对话的编排

外层是地点层循环（`GameEngine.run` → `Explorer.menu`），每次行动是一个事务，结束时 `save()` 提交；选了“交谈”才进入 `DialogueSession`。
一轮是**固定顺序的 19 个节点**（`core/dialogue_session.py` 的 `TURN_NODES`，执行器 `core/turn_graph.py`，每个节点登记读写哪些字段，`tests/test_turn_graph.py` 用 AST 核对）：

```
pick_action → interpret_action → guard_input → prefetch → judge_say → detect_leaving → decide_invite
→ social_effects → stash_conv → build_turn_ctx → check_claims → log_guard → classify_branch → echo_player
→ npc_reply → settle → world_effects → update_minds → decide_end
```

- **内层 ReAct 不画成图**：它的价值在于模型自己决定下一步用哪个工具。工具全是只读的：`ls` / `read` / `recall_memory` / `lookup_lore` / `load_skill` / `load_persona`，终止工具 `respond`。
  四道闸：最多 4 步、同参数重复 1 次、单轮 4 万 token、90 秒；触发后先把工具集收紧到只剩 `respond`（解码层 enum 约束）逼它交付。
- **进度判断三层**：NPC 上报信号 → 规则（开场白、注入轮不算）→ 独立审核 → 证据必须逐字出现在本轮原文 → 依赖 DAG；审核失败不推进。玩家用**明确的动作**满足条件时，由代码直接判定。
- **谈话怎么收尾**（`decide_end`）：结局、玩家告辞、NPC 走了或要结束、时间到、停滞（连续 10 轮既无进展关系也没升温）、
  没有正事的闲聊聊到第 8 轮让对方自然收尾、第 11 轮代码强制收束、单次见面最多 20 轮。
- **预算与降级**（`core/budget.py`）：整局默认 150 万 token；剩余低于 15% 时选项改静态、审核只走规则、NPC 不用工具；用尽则礼貌收尾，存档保留。
- **模型路由**（`llm/router.py`）：默认所有任务走同一个模型；`MODEL_<TASK>` 可以把某个任务路由到同一后端下的另一个模型。
- **事务与续跑**：状态靠事务（`core/uow.py`）；崩溃后先回滚，再用行动日志（`core/journal.py`）让问过的模型 / 判定 / 玩家选择原样复用（`docs/crash_recovery.md`）。
- **并行判定**（`core/prefetch.py`，默认关）：三次互不依赖的判定提前一起发出，用时核对参数，只可能省时间不会改结果。

### 状态存在哪里

| 什么 | 在哪 | 谁写 |
|---|---|---|
| 世界真相：时间、线索、信号、幕、谁在哪、**读人的心智状态**（`minds`） | `data/saves/<存档>.json` | `GodEnv.to_dict()`，每次行动结束的事务提交点 |
| 每个 NPC 对玩家的好感、信任 | 同一个存档文件的 `relationships` | `Relationship`，按评估结果改 |
| 长期记忆：摘要、事实、**反思**、即兴设定 | `data/memories/<存档>/<npc>.jsonl`，共享即兴设定在 `_world_canon.jsonl` | `NPCAgent.remember_session` |
| 本次见面的对话窗口 + 滚动摘要 | 内存里（`core/working_memory.py`） | 每次见面重置，靠长期记忆和存档接上 |
| 事务日志、行动日志 | `data/uow/<存档>/` | `UnitOfWork` / `Journal` |
| 每个事件一行的 trace | `data/traces/` | `Tracer` |

### 记忆是怎么写的

谈话结束时（`NPCAgent.remember_session`）：**摘要** → 事实（必须在原文里有依据才入库；和相近旧事实比一次：新增 / 更新 / 作废 / 不用存）→ **反思**。

反思**不是对 NPC 的评判**，而是 NPC 自己对旅行者的琢磨（“我对旅行者的看法”）：谈话至少 2 轮、且上次反思以来新记忆的重要度累计到 6 才做；
先挑 2~3 个此刻最想弄明白的问题，每个问题去整个记忆库检索，再下结论并写明依据；以 `kind="reflection"` 存进该 NPC 的 jsonl。
同一问题的新结论替换旧的，最多留 3 条，超过 80 个世界回合不再带进上下文。下次见面作为“你自己上次琢磨出来的”放进上下文，并用它带出相关往事。

### 提示词的排布与前缀缓存

NPC 说话那次请求是：`system`（每个角色整局只构建一次，逐字节稳定）+ 历史对话（逐条 user / assistant）+ 最后一条 user（本轮情境 + 对方刚说的话 + 现在）。

- trace 的每个 `llm` 事件记 `cached_tokens`；局末统计行的“前缀缓存”是 `cached_tokens / prompt_tokens`，对**所有**调用求和。
- 这是服务商的**前缀缓存**：只有从第一个 token 起连续一致的部分才能复用，缓存存在服务商的推理服务器上，不在本地。
- 命中率偏低（实测见过约 10%）的原因，**由代码推断、未逐条用 trace 验证**：
  玩家这一句在最末尾，影响很小；真正的原因是——最新一轮永远是新内容；本轮情境块每轮都变且可能很长；历史只留最近 8 轮，窗口滑动后前缀整体错位；
  评估 / 核对 / 记录 / 读人 / 选项 / 反思各有自己的 system，每轮每种只调一次，几乎没有可复用的前缀；通过聚合路由时同一模型可能落到不同后端。
- 本项目走 OpenAI 兼容接口，读的是 `prompt_tokens_details.cached_tokens`，**代码里没有发 `cache_control`**。如果后端是 Anthropic 的 Claude 模型，它要显式标记才缓存，没标记就几乎不会命中；
  Claude 原生接口的字段是 `cache_read_input_tokens` / `cache_creation_input_tokens`，没有现成的命中率，要自己算。

### 容易搞错的几点

| 常见的说法 | 实际 |
|---|---|
| “GOD、TREE、ENV 是三个 agent” | 是三层。每层 = 代码部分（真相）+ 一个模型提议者（导演 / 分支 / 场景）。`GodEnv` 本身是代码，不是 agent |
| “所有的模型调用都是 agent” | 只有 NPC 说话是会自己选工具的循环；其余是一问一答的单一职责调用 |
| “adapter 也是 agent” | adapter 是 agent 的模型一侧（提示词 + 调用 + 校验），不是独立角色；反思、摘要、压缩、声明抽取等都只是 NPC 流程里的一步 |
| “Reflection 是对 NPC 的评判” | 方向相反：是 NPC 对玩家的自省，存进该 NPC 自己的记忆，下次见面当“我上次的想法”用 |
| “最终状态由谁维护” | 世界 + 关系 + 读人状态在 `data/saves/`，记忆在 `data/memories/`，都在事务提交点落盘；模型不能直接写任何一份 |
| “选项生成器必须给三句、每句分好感 / 打听情报之类的类型” | 3 是上限，模型看情况给 0~3 句；曾经的 `intent` / `tone` 标签已去掉，只有“话本身就在做某件事”才由标注员标出来 |
| “选项生成器的标签只是装饰” | 旧的 `intent` 会作为输入标签传给 NPC、也被自动玩家拿来选项；现在普通台词和自己打字一样标 `自由发言`，告辞 / 邀请等由标注员判定 |
| “一轮是 18 个节点” | 是 19 个（旧文档里有的地方写 18，以 `TURN_NODES` 为准） |
| “KV cache 是服务商维护的一张全局表 / 我每次输入不同所以命中低” | 是每个推理实例各自的前缀表，不是全局的；输入在最末尾本身影响小，命中低主要是 prompt 结构和调用类型多 |
| “缓存命中率 Claude 会直接给” | 不会，API 只给读写的 token 数，要自己算；本项目也没发 `cache_control` |

### 角色速查

| 角色 | 职责 | 看到什么 |
|---|---|---|
| NPC | 评估 → 说话（有界 ReAct，只读工具）→ 生成后核对 → 记录 | 自己的人物卡 + 此刻的心思 + 窗口内对话 + 按“自己知道什么”过滤的上下文 |
| Reviewer | 独立核实 NPC 声称的进度 | 只看信号定义和本轮原文 |
| 选项生成器 | 替玩家想 0~3 句台词 | 只有玩家视角，看不到 NPC 的秘密和待达成信号；另带一份“玩家以为这个 NPC 大概知道什么” |
| 读人 ×2（`llm/mind_llm_adapter.py`） | 每轮之后各维护一份心智状态：NPC 以为玩家知道 / 想听什么（她可以故意不说），玩家以为 NPC 知道什么 | NPC 一边能看她自己的底细，玩家一边只看线索本和对话，互相看不到 |
| 导演 / 分支 / 场景 | GOD / TREE / ENV 三层的模型部分：调节奏、把自由说出的方案对应到预设方案、把自由查看对应到物件 | 结构化的全局状态，导演**没有玩家原话** |
| GodEnv（非 LLM） | 世界状态的唯一真相 | — |

详细的设计决策、每条机制为什么这样做，见 `docs/architecture.md`；一轮对话的节点图见 `docs/turn_graph.md`；事务与崩溃恢复见 `docs/crash_recovery.md`。

## 可观测、重放与评测

- **trace**：每个事件一行 JSONL（`data/traces/`），带层级 span 和全局 seq；`llm` 事件含完整 messages、原始输出、usage、缓存命中、耗时；密钥自动脱敏。
  `python3 tools/view_trace.py` 可视化最新一局；`python3 tools/replay.py TRACE --seq 42 --n 3` 把某次模型调用原样重放几次。
- **整局重放**：`python3 tools/replay_game.py <trace>` 用记下的输入和模型回答把整局再跑一遍并逐段核对状态——验证的是**我们的代码**，不是模型；
  `--state-at N` 直接重建第 N 次输入之前的世界。测试里有一局“金标准局”（`tests/golden/`），世界走向变了就红。
- **提示词预览**：`python3 tools/preview_prompt.py <存档> <npc> --say …`，不调模型，打印某个 NPC 此刻会收到的完整提示词；`--find 词` 查某个词有没有进他的提示词。
- **剧情内容校验**：`python3 tools/lint_story.py`，不跑游戏，推演一遍找出走不通的地方（永远拿不到的线索、达成不了的信号、见不到的人）。

## 目录

```
main.py / web/server.py        入口（终端 / 网页）
config/settings.py             环境变量与 .env → Settings（LLM / Harness / Game 三组）
content/                       故事：world.json、npcs/、quests/、lore/、skills/、story.json，纯数据
core/                          世界规则与状态：god_env、physical、dialogue_session、turn_graph、explorer、agent_loop、
                               reviewer、director、branch、scene、claims、judge、guardrails、agendas、actions、
                               uow / crashpoints / journal（事务与续跑）、prefetch、story_lint、eventlog …
models/                        personality / memory / relationship / retrieval（BM25）/ knowledge
llm/                           后端客户端（gemma_client、http_pool、free_first、model_picker、mock_backend）与各个单一职责的小 agent
engine/                        game_engine（组装、存档）、trace、recording、journaled
tools/                         view_trace / replay / replay_game / preview_prompt / lint_story / monitor / pick_model
tests/                         pytest（全离线：假 HTTP 服务、金标准局、崩溃矩阵、续跑矩阵、故事测试）
docs/                          architecture（设计决策）/ code_map（想改某件事去哪）/ turn_graph / crash_recovery / harness_design / reference_projects
```
