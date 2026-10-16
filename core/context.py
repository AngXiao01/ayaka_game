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
    turn: int = 0
    player_view: dict = field(default_factory=dict)     # 玩家视角（Explorer.player_view），只给替玩家想台词的选项 agent

    @property
    def task(self) -> dict:
        return self.world["npc_task_detail"]
