"""心智状态维护 agent（两个，各管一边，互相看不到）：每轮对话之后，更新「一方以为对方知道什么 / 对什么感兴趣」。

  npc    站在 NPC 的位置：玩家大概知道什么、对什么感兴趣；NPC 自己愿不愿意主动说（可以故意不说，出于她自己的心思）。
         依据：这次对话原文 + NPC 自己知道的底细。输出只进 NPC 自己的提示词。
  player 站在玩家的位置：这个 NPC 大概知道什么。依据只有玩家自己知道的和对方亲口说过的话——不许用 NPC 的底细。
         输出只进替玩家想台词的选项 agent。

只维护 1~3 个当下相关的主题，不是百科；拿不准就写 maybe。
"""
from __future__ import annotations

from typing import Any

from config.settings import TaskParams
from core import mind as mind_state
from llm.base import LLMBackend, LLMError, LLMRequest
from llm.structured import call_json

SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {
    "type": "object", "properties": {
        "topic": {"type": "string"}, "belief": {"type": "string", "enum": list(mind_state.BELIEFS)},
        "interest": {"type": "integer"}, "share": {"type": "string", "enum": list(mind_state.SHARES)},
        "why": {"type": "string"}}, "required": ["topic", "belief"]}}}, "required": ["items"]}

NPC_SYSTEM = """你是{npc}的「读人」助手。每轮对话之后，你替{npc}更新她对{player}的判断：哪些话题，{player}大概已经知道、大概感兴趣；{npc}自己愿不愿意主动提。

规则：
1. 只记 1~3 个和刚才这几句话直接相关的话题（人、地点、事件、物品，topic 不超过 12 个字），不要写成百科。
2. belief：knows = {player}多半知道（对话里说过、或明摆着该知道）；maybe = 不一定；unknown = 多半不知道。{player}没提过、{npc}也没说过的事，一律不要写成 knows。
3. interest：0 没兴趣 / 1 一般 / 2 很感兴趣，只看{player}在对话里表现出来的。
4. share 是{npc}自己主观的决定，不是客观的：tell = 她愿意主动提；withhold = 她出于自己的心思（顾虑、礼数、没到火候、不想让对方为难）先不说；undecided = 看情形。{npc}自己的底细里写着不能轻易说的事，一律 withhold。
5. why 用一句话（不超过 20 字）说明你为什么这样判断。
6. 上一轮已有的判断，对话里没有新信息就不要重复写。没有新判断就输出 {{"items": []}}。

只输出一个 JSON 对象：{{"items": [{{"topic": "……", "belief": "maybe", "interest": 1, "share": "undecided", "why": "……"}}]}}，不要任何其他文字。"""

PLAYER_SYSTEM = """你是{player}的「读人」助手。每轮对话之后，你替{player}更新他对{npc}的判断：哪些话题，{npc}大概知道。

规则：
1. 只依据{player}自己已经知道的事和{npc}在对话里亲口说过的话。{player}没听说过的人名、地点、案情，不能凭空假设{npc}知道，更不能借此让{player}自己“知道”了。
2. 只记 1~3 个和刚才这几句话直接相关的话题（topic 不超过 12 个字）。
3. belief：knows = {npc}亲口说过或明摆着知道；maybe = 有可能知道；unknown = 多半不知道（比如是外地的事、{npc}从没提过、不在她的圈子里）。
4. why 用一句话（不超过 20 字）说明依据。没有新判断就输出 {{"items": []}}。

只输出一个 JSON 对象：{{"items": [{{"topic": "……", "belief": "maybe", "why": "……"}}]}}，不要任何其他文字。"""


class MindLLMAdapter:
    def __init__(self, backend: LLMBackend, params: TaskParams, tracer: Any = None, repair_attempts: int = 1):
        self.backend, self.params, self.tracer, self.repair_attempts = backend, params, tracer, repair_attempts

    def update(self, side: str, player: str, npc: str, exchange: str, current: list[dict], known: str,
               last_player: str = "") -> list[dict]:
        """返回这一轮的新判断（已规整）。known：这一边自己知道的东西（NPC 一边是她的底细，玩家一边是线索本）。"""
        system = (NPC_SYSTEM if side == "npc" else PLAYER_SYSTEM).format(player=player, npc=npc)
        have = "\n".join(f"- {x['topic']}：{x['belief']}" for x in current) or "（还没有判断）"
        user = (f"# {'你（' + npc + '）自己知道的底细' if side == 'npc' else player + '自己知道的事'}\n{known or '（无）'}\n\n"
                f"# 已有的判断\n{have}\n\n# 刚才的对话\n{exchange}\n\n请更新。只输出 JSON。")
        req = LLMRequest(task="mind_update",
                         messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                         temperature=0.0, max_tokens=min(350, self.params.max_tokens), json_schema=SCHEMA,
                         schema_name="mind_update", meta={"side": side, "last_player": last_player, "npc": npc})
        try:
            return call_json(self.backend, req, mind_state.clean, self.tracer, self.repair_attempts)
        except LLMError:
            return []
