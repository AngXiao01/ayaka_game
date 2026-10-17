"""一轮对话里各个 agent 共享的只读上下文。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TurnContext:
    player_name: str
    scene: dict                 # World.scene_context()（ENV 层）
    world: dict                 # GodEnv.world_state_summary()
    relationship: dict          # Relationship.summary()
    memories: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)   # 工作记忆窗口内的原文 {"speaker", "text", "intent"?, "reply"?}
    history_summary: str = ""                           # 窗口外较早对话的滚动摘要
    canon: list[str] = field(default_factory=list)      # 之前即兴编出、必须保持一致的设定
    handoff: dict = field(default_factory=dict)         # 上次见面的结构化交接（上下文重置后接续的唯一来源）
    tics: list[str] = field(default_factory=list)       # 代码从最近几句话里量出来的说话套路（句式、开头、称呼）
    avoid: list[str] = field(default_factory=list)      # 刚被代码发现重复的自己的话：这一轮换个说法
    turn: int = 0
    player_view: dict = field(default_factory=dict)     # 玩家视角（Explorer.player_view），只给替玩家想台词的选项 agent
    place: dict = field(default_factory=dict)           # 物理世界里这个 NPC 该知道的：身边的人、同行、提到的人在哪、口信

    @property
    def task(self) -> dict:
        return self.world["npc_task_detail"]
