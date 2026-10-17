# ayaka_game 代码地图

## 一句话

这是一个**回合制文字调查游戏**。世界的状态（时间、线索、谁在哪）由**代码规则**管，说话和判断的部分交给**大模型**；
剧情内容（人物、线索、地图）全在 `content/` 的 JSON 里，不在代码里。

三层，从下往上读：

```
content/ 剧情内容（JSON/Markdown）        ← 改剧情只动这里
core/ models/ 规则与状态（纯 Python）     ← 世界怎么运转，唯一的“真相”
llm/ 模型调用（一个个单一职责的小 agent） ← 把规则翻译成提示，再把模型回复整理成结构
engine/ web/ main.py 入口与外壳           ← 把以上串起来、存档、记 trace、网页
```

## 目录

| 目录 | 干什么 | 重点文件 |
|---|---|---|
| `content/` | 全部剧情数据 | `quests/ayaka_main.json`（线索、信号、幕、结局、议程）、`world.json`（地点、物件、格局）、`npcs/*.json`（角色卡）、`skills/*.md`、`lore/*.md`（设定资料）、`_design/story_bible.md`（剧情大纲，只给人看） |
| `core/` | 世界规则与状态 | `god_env.py`（**状态的唯一真相**：时间、线索、信号、幕、结局）、`dialogue_session.py`（**一次谈话的主循环**）、`eventlog.py`（状态差异与重建）、`explorer.py`（地点层菜单）、`physical.py`（谁在哪、传话）、`npc_agent.py`（一个 NPC 的大脑）、`agent_loop.py`（带工具的循环）、`claims.py`、`agendas.py`、`actions.py` |
| `models/` | 数据对象 | `personality.py`（角色卡）、`memory.py`（长期记忆）、`relationship.py`（好感与信任）、`retrieval.py`（检索） |
| `llm/` | 模型调用 | `npc_llm_adapter.py`（一句回复的三步）、`npc_stages.py`（评估、核对、记录）、`player_option_llm_adapter.py`（替玩家想台词）、`gemma_client.py`（HTTP 客户端）、`free_first.py`、`mock_backend.py`（离线假模型，测试用）、`replay.py`（用 trace 里记下的回答重放）、`schemas.py`（所有结构化输出的格式与校验） |
| `engine/` | 外壳 | `game_engine.py`（组装一切、地点层主循环、存档）、`trace.py`（把每一步记成 JSON 行）、`recording.py`（记录玩家输入和状态差异，供重放） |
| `web/` | 网页版 | `server.py`、`index.html` |
| `config/` | 配置 | `settings.py`（所有环境变量） |
| `tests/` `evals/` | 测试与评测 | `tests/`（离线，用假模型）、`evals/claim_cases.py`（真话/谎话用例） |
| `data/` | 运行时产物 | `saves/`（存档）、`memories/`（各 NPC 长期记忆）、`traces/`（每局的 trace） |

## 一个回合是怎么走的

玩家在网页里点一下或打一句话，之后：

1. `web/server.py` → `GameEngine.run()`（`engine/game_engine.py`）：在**地点层**循环，`Explorer.menu()` 列出能做的事（交谈、查看、移动、等待……）。
2. 选了“交谈”→ 建 `DialogueSession`，进入 `_turn()`（`core/dialogue_session.py`，最长的一个函数，按顺序做下面这些）：
   - 玩家输入先过 `guard`（清洗、防注入）。
   - 玩家的话：`judge.turn`（便宜的判断：威胁、突兀、邀请、告别）、`ClaimChecker`（核对“我做过X”）、`branch`（是不是在提处置方案）。
   - 代码结算：硬问掷骰、同行能不能答应、要不要派人去查。
   - `build_ctx()` 把这一轮要给 NPC 的上下文拼好：记忆检索、上次见面交接、身边的人、提到的地方。
3. `NPCAgent._call()`（`core/npc_agent.py`）→ `NPCLLMAdapter.reply()`（`llm/npc_llm_adapter.py`），一句回复分四步：
   **评估**（情绪、好感、打算）→ **说话**（带工具的循环，只写台词）→ **核对**（对照事实，不对就重说）→ **记录**（说了哪件事、有无新设定、谎话、带话）。
4. `SignalReviewer` 审核“这轮是不是真的达成了某个剧情信号”，`GodEnv.apply_signals()` 落地（给线索、推进幕、出结局）。
5. 对话窗口过长就压缩；结束时写长期记忆（`NPCAgent.remember_session`）；NPC 的议程、物理世界（传话、移动）随时间推进。

## 想改 X，看哪里

| 我想… | 去看 |
|---|---|
| 改剧情、线索、结局、谁什么时候说什么 | `content/quests/ayaka_main.json` |
| 改某个人的性格、说话、称呼、一开始知道什么 | `content/npcs/<id>.json` |
| 改地图、物件、格局 | `content/world.json` |
| 改某个人为什么答非所问、编细节 | `llm/npc_llm_adapter.py` 的系统提示和 `build_state_block`；`llm/npc_stages.py` |
| 改玩家选项的风格、称呼 | `llm/player_option_llm_adapter.py` |
| 改“谁什么时候能知道什么” | `core/god_env.py` 的 `signal_status`；`knows_clues`；`content/quests` 的信号条件 |
| 改时间怎么流逝、天色 | `core/god_env.py` 的 `pass_time`、`daylight` |
| 加一个新的判断（是不是 X） | `core/judge.py`（便宜的类型化判断）或 `llm/` 下新加一个单一职责的 agent |
| 改记忆、检索 | `models/memory.py`、`core/working_memory.py`、`core/context_fs.py` |
| 看某一局发生了什么 | `data/traces/*.jsonl`，用 `python3 tools/view_trace.py <文件>` |
| 把一局原样重放 / 看第 N 步时的世界 | `python3 tools/replay_game.py <文件> [--state-at N]`，原理见 `core/eventlog.py` |
| 加测试 | `tests/`，用 `MockBackend`，不花钱 |

## 建议的阅读顺序

1. **先别读代码，读 `content/_design/story_bible.md` 和 `docs/harness_design.md`**，搞清楚故事和设计原则。
2. 看 `content/quests/ayaka_main.json` 里的 `clues` 和 `signals`：这是整个游戏的“剧情图”，每个信号写明谁、在什么条件下说出什么、给什么线索。
3. `core/god_env.py`：只看类的字段和 `signal_status`、`apply_signals`、`npc_task_detail`（它把“此刻 NPC 该知道什么”整理成字典）。
4. `core/dialogue_session.py` 的 `_turn()`：从上往下读一遍，对照上面“一个回合”的列表。
5. `llm/npc_llm_adapter.py` 的 `reply()` 和 `build_state_block()`：看 NPC 最终看到什么提示。
6. 其余按需查。

最长的三个文件 `god_env.py`（948 行）、`dialogue_session.py`（738 行）、`schemas.py`（506 行）不要从头读到尾，按上面的函数名跳着看。

## 怎么跑与怎么看

- 玩：`FREE_FIRST=1 OPENROUTER_MODEL=qwen/qwen3.7-flash python3 web/server.py --new`
- 离线测试：`~/miniforge3/envs/torch_arm/bin/python -m pytest -q`
- 离线跑一局：`python3 main.py --mock --auto cooperative`
- 看 trace：`python3 tools/view_trace.py data/traces/<文件>.jsonl`
- 重放一局：`python3 tools/replay_game.py data/traces/<文件>.jsonl`（只对新版本录的 trace 有效）
