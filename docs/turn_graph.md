# 一轮对话是一张图

## 为什么

`DialogueSession._turn()` 以前是一个 287 行的函数：动作分派、判定、掷骰、构建上下文、世界移动、结算全挤在里面，
控制流靠缩进和 if/elif 表达，没有人看得到全貌。现在它只有几行：按图跑 19 个节点。模型调用那一侧早就拆得很窄（十几个单一职责的小适配器），
真正需要拆的是这个流程控制，而不是再多一个 agent。

## 现在是什么

19 个节点按固定顺序执行（`core/dialogue_session.py` 的 `TURN_NODES`，执行器在 `core/turn_graph.py`，只用标准库）。
每个节点：函数体从共享状态 `TurnState` 取变量、结尾写回，登记了读哪些字段、写哪些字段。唯一的条件边：玩家选了“保存并退出”，在第 1 个节点之后直接结束。

| # | 节点 | 干什么 | 读 | 写 |
|---|---|---|---|---|
| 1 | `pick_action` | 取上下文、列选项、等玩家选一个动作；选了“保存并退出”就在这里结束这一轮。 | turn | ctx, options, action, end, done |
| 2 | `interpret_action` | 把玩家选的动作翻成“说了什么、什么意图、让哪些事此刻能谈、递了什么给对方看”（具体到每种动作见 _act_*）。 | turn, ctx, action | nid, before, forced, shown, press, text, intent, ctx |
| 3 | `guard_input` | 输入防护；打字说出手上线索的内容，等于出示了。 | action, text, nid, before, forced | guard, forced |
| 4 | `prefetch` | 把互不依赖的三次判定（语气 / 声明抽取 / 方案识别）提前一起发出去；结果在原来的节点里取，状态仍按原顺序改。 | action, guard, nid, ctx | pending |
| 5 | `judge_say` | 一句自由发言只判一次（Jev 或大模型）：语气、接不接得上、是不是在邀请、说中了哪几条推测。 | action, guard, nid, ctx, pending | phys, invite, felt |
| 6 | `detect_leaving` | 玩家打字说“我该走了”：走不走是玩家的事。 | action, guard, nid, felt, intent | leaving, intent |
| 7 | `decide_invite` | 同行答不答应由代码按好感和对方有没有空判定。 | phys, action, guard, nid, felt, invite | invite |
| 8 | `social_effects` | 威胁、掷骰信不信、派人去查玩家说出的推测。 | phys, action, guard, felt, nid | dispatch, believe, threatened, odd |
| 9 | `stash_conv` | 把这一轮的结算结果挂到对话状态上，NPC 的情境会读它。 | phys, action, nid, dispatch, believe, press, threatened, odd, forced, invite | — |
| 10 | `build_turn_ctx` | 按这句话检索往事、选这一轮提哪件事（节拍）、识别重复提问。 | turn, text, guard, action, intent, shown, nid | ctx |
| 11 | `check_claims` | 声明核对：玩家说自己做过什么，对着真实足迹核对。 | ctx, guard, action, nid, pending | — |
| 12 | `log_guard` | 记输入防护的结果。 | text, guard | — |
| 13 | `classify_branch` | 自由输入里的处置方案 → 预设方案。 | turn, action, guard, nid, intent, ctx, pending | intent, ctx |
| 14 | `echo_player` | 把玩家这句话（和硬问的结果）显示出来。 | action, guard, press | — |
| 15 | `npc_reply` | NPC 回话：评估 → 说话（ReAct）→ 核对 → 记录。 | ctx, guard, intent, nid | skills, context_lines, reply |
| 16 | `settle` | 进度审核与结算：线索、关系、幕。 | turn, guard, intent, action, shown, skills, options, context_lines, forced, reply | newly |
| 17 | `world_effects` | 对话之后落到世界上的事：派人去查、凭信任相信、同行、请对方先回去。 | phys, nid, reply, newly | dismissed, went, ids, deg |
| 18 | `update_minds` | 两个读人 agent 各更新一份心智状态：NPC 以为玩家知道 / 想听什么（她可以故意不说），玩家以为 NPC 知道什么（core/mind.py）。 | turn, nid, guard, reply, action | — |
| 19 | `decide_end` | 这一轮之后谈话怎么收尾：结局、告辞 / 接着走、对方走了或结束、时间到、停滞，否则继续。 | action, reply, dismissed, leaving, went, ids, deg | end |

每个节点跑完会：记一条 trace 事件 `node`（节点名、耗时 ms）、放一个崩溃点 `turn.after.<节点名>`（崩溃矩阵会自动覆盖）。

- **动作分派表**：`interpret_action` 里按动作种类的分派是 `DialogueSession.ACTION_HANDLERS`：动作种类 → 一个 `_act_*` 方法
  （ask / press / clue / report / propose / confront / leave / invite / dismiss），没登记的（自由发言等）走 `_act_say`。
- **静态检查**（`tests/test_turn_graph.py`）：用 AST 核对“登记的读写”和“节点代码实际读写的字段”完全一致；
  图的 lint 检查每个节点读的字段前面一定有节点写过、没有走不到的节点、没有环；节点顺序被钉住（改顺序要有意识地改测试）。
- **内层 ReAct 循环不改成图**：它的价值在于模型自己决定下一步用哪个工具，把它画成节点只会把“循环”伪装成“流程”。

## 并行判定（默认关）

`prefetch` 节点把三次互不依赖的模型调用——`judge.turn`（语气 / 邀请 / 告别 / 说中的推测）、声明抽取、方案识别的便宜判定——提前一起发出去；
状态仍然在原来的节点里、按原来的顺序改，那几个节点只是把“问模型”换成“取预取的结果”。

- **用时核对**：预取登记的是调用参数；取结果时现在要问的参数必须与之完全相同才用预取的答案，否则当场串行重问。所以预取只可能省时间，不会改变结果。
  参数由同一个函数算出（`_judge_args`、`claims.request`、`branch.request`），不可能写成两份。
- **关掉时同一条代码路径**：`Prefetch.start` 什么都不做，`get` 就是直接调用。mock 和重放后端永远不开。
- **trace 线程安全**：`Tracer` 加了锁（序号、写文件、计数器一起），`span` 每个线程各一份，预取线程沿用提交时的标签并多带一个 `prefetch` 标记。
- **重放容忍乱序**：见 `docs/architecture.md` 的“可复现与可调试”。
- **实测**（真模型，4 轮自由发言）：稳态下每轮只省 0.3~1 秒，但预取线程的**第一次连接**要自己握手——在不稳的网络上有一次多等了 15 秒。
  每一项固定用自己的一条线程（连接按线程保活），所以只在第一轮付一次；但短局里这笔账是亏的，所以默认关，`PARALLEL_JUDGES=1` 打开。
  真正的大头不在这里：每轮约 16 秒里，选项生成约 4 秒、NPC 回话约 7 秒，三次判定加起来只有 1~2 秒。

## 崩溃后从哪接着走

没有做成“已完成的节点跳过”：跳过有副作用的节点要求把一个世界序列化成检查点，并保证它和旁边的记忆文件对得上——那是在重新发明事务。做法是反过来：

1. 状态永远靠事务（`core/uow.py`）：重启时先**回滚**到这次行动开始之前，一个有副作用的节点都不跳；
2. 再用**行动日志**（`core/journal.py`）让重新跑一遍不白花钱、不换说法：模型调用、便宜判定、玩家的选择按“请求内容的哈希”记下来，回滚后同一段代码重新执行，问法一字不差的直接用当时的答案，其余照常现问。

详见 `docs/crash_recovery.md`。
