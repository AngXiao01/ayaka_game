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

```
GodEnv（剧情真相 + 时钟）     线索、信号、幕、结局；按分钟走的时钟
  └─ PhysicalWorld（空间真相）  谁在哪 = 作息 + 计划；同行队伍；路上的口信
       └─ 认知（每个 agent 一份） 亲眼看到的 + 别人带的话；没有就按作息猜

GameEngine（Harness 外壳：组装、存档 = 事务提交点、预算、trace）
 └─ 地点层循环（Explorer，确定性）：交谈 / 查看 / 移动 / 等待 / 线索本 / 人物 / 日志
     └─ DialogueSession（一次交谈）：一轮 = 18 个节点（TURN_NODES）
          玩家输入 → 输入防护 → 判定 → 结算 → 上下文 → NPC 回复（ReAct 循环）→ 进度审核 → 世界结算
```

| 角色 | 职责 | 看到什么 |
|---|---|---|
| NPC | 评估 → 说话（有界 ReAct，只读工具）→ 生成后核对 → 记录 | 自己的人物卡 + 此刻的心思 + 窗口内对话 + 按“自己知道什么”过滤的上下文 |
| Reviewer | 独立核实 NPC 声称的进度 | 只看信号定义和本轮原文 |
| 选项生成器 | 替玩家想 3 句台词 | 只有玩家视角，看不到 NPC 的秘密和待达成信号 |
| 导演 / 分支 / 场景 | GOD / tree / ENV 三层的模型部分：调节奏、把自由说出的方案对应到预设方案、把自由查看对应到物件 | 结构化的全局状态，导演**没有玩家原话** |
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
