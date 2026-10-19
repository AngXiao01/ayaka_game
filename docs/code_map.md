# 代码地图

## 一句话

一个**回合制文字调查游戏**。世界的状态（时间、线索、谁在哪）由**代码规则**管，说话和判断交给**大模型**；  
剧情内容（人物、线索、地图）全在内容目录的 JSON 里（默认 `content/`），不在代码里。

```
content/            剧情内容（JSON / Markdown）        ← 改剧情只动这里
core/ models/       规则与状态（纯 Python）            ← 世界怎么运转，唯一的“真相”
llm/                模型调用（一个个单一职责的小 agent） ← 把规则翻译成提示，再把模型回复整理成结构
engine/ web/ main.py  入口与外壳                        ← 串起来、存档、记 trace、网页
```

## 目录

| 目录             | 干什么      | 重点文件                                                                                                                                                                                                                                                               |
| -------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `content/`     | 一个完整的故事  | `quests/*.json`（线索、信号、幕、结局、议程）、`world.json`（地点、物件、格局）、`npcs/*.json`（人物卡）、`lore/*.md`（设定资料）、`skills/*.md`、`story.json`（主线 id 和主角 NPC）                                                                                                                               |
| `core/`        | 世界规则与状态  | `god_env.py`（**状态的唯一真相**：时间、线索、信号、幕、结局）、`dialogue_session.py`（**一次谈话**：18 个节点）、`turn_graph.py`、`explorer.py`（地点层菜单）、`physical.py`（谁在哪、传话）、`npc_agent.py`（一个 NPC 的大脑）、`agent_loop.py`（有界 ReAct）、`claims.py`、`agendas.py`、`actions.py`、`story_lint.py`、`eventlog.py` |
| `core/` 里管可靠性的 | 事务、续跑、并行 | `uow.py`（事务）、`crashpoints.py`（崩溃注入点）、`journal.py`（行动日志）、`prefetch.py`（并行判定）                                                                                                                                                                                        |
| `models/`      | 数据对象     | `personality.py`（人物卡）、`memory.py`（长期记忆）、`relationship.py`（好感与信任）、`retrieval.py`（BM25）、`knowledge.py`（设定库）                                                                                                                                                          |
| `llm/`         | 模型调用     | `npc_llm_adapter.py`（NPC 上下文与系统提示）、`npc_stages.py`（评估、核对、记录）、`player_option_llm_adapter.py`、`gemma_client.py`（HTTP 客户端）、`http_pool.py`（连接复用）、`decider.py`（便宜的判定器）、`mock_backend.py`（离线假模型）、`replay.py`、`schemas.py`                                                |
| `engine/`      | 外壳       | `game_engine.py`（组装一切、地点层主循环、存档 = 事务提交点）、`trace.py`、`recording.py`、`journaled.py`                                                                                                                                                                                  |
| `web/`         | 网页版      | `server.py`、`index.html`                                                                                                                                                                                                                                           |
| `config/`      | 配置       | `settings.py`（所有环境变量，样例见 `env.example`）                                                                                                                                                                                                                            |
| `tests/`       | 测试       | 全离线，用假模型                                                                                                                                                                                                                                                           |
| `data/`        | 运行时产物    | `saves/`（存档）、`memories/`（各 NPC 长期记忆）、`traces/`（每局的 trace）、`uow/`（事务日志）                                                                                                                                                                                             |

## 一个回合是怎么走的

玩家在网页里点一下或打一句话，之后：

1. `GameEngine.run()` 在**地点层**循环，`Explorer.menu()` 列出能做的事（交谈、查看、移动、等待……）；每次行动是一个事务，结束时 `save()` 提交。
2. 选了“交谈”→ 建 `DialogueSession`，一轮按 `TURN_NODES` 的顺序走 18 个节点（`docs/turn_graph.md`）：  
   选动作 → 解释动作 → 输入防护 → 判定（语气、告别、邀请、说中的推测）→ 同行 / 威胁 / 掷骰 / 派人 → 拼上下文（记忆检索、节拍、重复提问）→ 声明核对 → 分支识别 → **NPC 回复** → 进度审核与结算 → 世界上的后果 → 判断这次谈话怎么收尾。
3. NPC 回复：`NPCAgent` → `NPCLLMAdapter.reply()`，四步：**评估** → **说话**（有界 ReAct，只读工具）→ **核对**（对照事实，不对带理由重说）→ **记录**。
4. `SignalReviewer` 审核“这轮是不是真的达成了某个信号”，`GodEnv.apply_signals()` 落地（给线索、推进幕、出结局）。
5. 对话窗口过长就压缩；结束时写长期记忆；NPC 的议程、物理世界随时间推进。

## 想改 X，看哪里

| 我想…                                          | 去看                                                                                                                                                       |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 改剧情、线索、结局、谁什么时候说什么                           | `<内容目录>/quests/*.json`                                                                                                                                   |
| 改某个人的性格、说话、称呼、一开始知道什么                        | `<内容目录>/npcs/<id>.json`                                                                                                                                  |
| 改地图、物件、格局                                    | `<内容目录>/world.json`                                                                                                                                      |
| 换一个新故事                                       | 建一个目录（`world.json`、`npcs/`、`quests/`、`lore/`、`skills/`、`story.json`），`CONTENT_DIR=<目录>` 启动；检查：`python3 tools/lint_story.py --content <目录>`；例子 `content/` |
| 让某个物件要等剧情走到某个标记才看得到                          | 物件里写 `requires_flags` + `locked_desc`；标记可以由议程按时间立起来                                                                                                      |
| 让某个人物要等某个标记才出现 / 让议程让某人瞬间到场                  | 人物卡 `present_when_flags`、`absent_when_flags` / 议程 `do` 里写 `instant: true`                                                                                |
| 要求“线索里至少掌握 N 条”                              | 信号 / 提议里写 `requires_clues_any` + `any_min: N`                                                                                                            |
| 让某个抉择要先知道某条线索才出现在菜单里                         | 提议里写 `requires_clues`                                                                                                                                    |
| 给信号加“被问到又不能说”的腔调 / 硬问失败时的口风                  | 信号里写 `lock_style`（direct / hesitate / deflect / bargain）、`partial`                                                                                       |
| 改某个人为什么答非所问、编细节                              | `llm/npc_llm_adapter.py` 的系统提示和 `build_state_block`；`llm/npc_stages.py`                                                                                  |
| 改玩家选项的风格、称呼                                  | `llm/player_option_llm_adapter.py`                                                                                                                       |
| 改“谁什么时候能知道什么”                                | `core/god_env.py` 的 `signal_status`；人物卡的 `knows_clues`；信号条件                                                                                              |
| 改时间怎么流逝、天色                                   | `core/god_env.py` 的 `pass_time`、`daylight`                                                                                                               |
| 加一个新的判断（是不是 X）                               | `core/judge.py`（便宜的类型化判断）或 `llm/` 下新加一个单一职责的 agent                                                                                                       |
| 改记忆、检索                                       | `models/memory.py`、`core/working_memory.py`、`core/context_fs.py`                                                                                         |
| 看 / 改“一轮对话”的流程                               | `core/dialogue_session.py` 的 `TURN_NODES` 和 `_n_*`；执行器 `core/turn_graph.py`；说明 `docs/turn_graph.md`                                                      |
| 开 / 关一句话里几次判定的并行                             | `PARALLEL_JUDGES=1`（默认关；mock 与重放永远不开）                                                                                                                    |
| 崩溃后恢复、验证“崩在哪都不会记忆领先于世界”                      | `core/uow.py`、`core/crashpoints.py`、`tests/crash_harness.py`；说明 `docs/crash_recovery.md`                                                                 |
| 崩溃回滚之后“问过的不重问、选过的不重选”                        | `core/journal.py`、`engine/journaled.py`、`tests/test_resume.py`；`RESUME_AFTER_CRASH=0` 关掉                                                                 |
| 看某一局发生了什么                                    | `data/traces/*.jsonl`，`python3 tools/view_trace.py <文件>`                                                                                                 |
| 看某个 NPC 此刻会收到什么提示词 / 某个词有没有进他的提示词            | `python3 tools/preview_prompt.py <存档> <npc> [--say …] [--find 词1,词2]`，不调模型                                                                               |
| 把一局原样重放 / 看第 N 步时的世界                         | `python3 tools/replay_game.py <文件> [--state-at N]`                                                                                                       |
| 改了规则后确认没改坏别的                                 | `pytest` 里的金标准局会重放；有意改变行为后重录：`python3 tools/replay_game.py --make-golden tests/golden/mock_game.jsonl.gz`                                                |
| 验证“真话 / 谎话 / 半真半假 / 没法核实”被代码逐条核对，与 NPC 回什么无关 | `tests/test_deaf_liar.py`                                                                                                                                |
| 加测试                                          | `tests/`，用 `MockBackend`，不花钱                                                                                                                             |

## 建议的阅读顺序

1. 先读 `docs/architecture.md`（设计原则）和一个故事的设计稿（`content/_design/plot.md`）。
2. 看 `quests/*.json` 里的 `clues` 和 `signals`：这是整个游戏的“剧情图”，每个信号写明谁、在什么条件下说出什么、给什么线索。
3. `core/god_env.py`：只看类的字段和 `signal_status`、`apply_signals`、`npc_task_detail`（它把“此刻 NPC 该知道什么”整理成字典）。
4. `core/dialogue_session.py`：对照 `docs/turn_graph.md` 的节点表，从 `TURN_NODES` 往下读。
5. `llm/npc_llm_adapter.py` 的 `reply()` 和 `build_state_block()`：看 NPC 最终看到什么提示。
6. 其余按需查。`god_env.py`、`dialogue_session.py`、`schemas.py` 都很长，按函数名跳着看。

## 怎么跑与怎么看

- 玩：`python3 web/server.py --new`（网页）或 `python3 main.py`（终端）；模型相关的环境变量见 `env.example`
- 离线测试：`~/miniforge3/envs/torch_arm/bin/python -m pytest -q`
- 离线跑一局：`python3 main.py --mock --auto cooperative`
- 看 trace：`python3 tools/view_trace.py data/traces/<文件>.jsonl`
- 检查剧情内容：`python3 tools/lint_story.py`（有 error 退出码为 1；`--strict` 连疑点也算失败）
