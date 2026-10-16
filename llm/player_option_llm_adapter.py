"""玩家台词选项：语气各异、接得上话、不重复，并给玩家指出可以追问的方向。"""
from __future__ import annotations

from typing import Any

from config.settings import TaskParams
from core.context import TurnContext
from llm.base import LLMBackend, LLMRequest
from llm.schemas import PLAYER_OPTIONS_SCHEMA, PLAYER_TONES, PLAYER_INTENTS, validate_player_options
from llm.structured import call_json

SYSTEM = """你是一款原神同人叙事游戏的「台词选项生成器」。玩家（{player}）正在和{npc}说话，你替玩家想出接下来可以说的 {n} 句话。

要求：
1. {n} 句话的语气各不相同，语气从这些里选：{tones}；意图从这些里选：{intents}。
2. 每句都要接得上{npc}刚才说的话，带着具体的细节（提到对方说过的东西、眼前的景物、自己的经历），不要空泛的客套。
3. 至少一句在推进玩家眼下的目标；如果「可以追问的方向」不为空，至少一句要自然地朝那个方向问。
4. 不要重复玩家之前说过的话，也不要和之前的选项换汤不换药。
5. 以{player}第一人称，像真人聊天一样口语、简短，每句不超过 30 字；大多数不需要括号动作，最多一句在开头带一个简短动作。
6. 选项要有内容，不要只是客套和附和：可以追问对方刚才话里的细节、分享旅行者自己的见闻（蒙德的风车与酒、璃月港的灯火、一路上遇到的人和事）、说出自己的看法、打趣对方；允许有一句俏皮、大胆或带点私人感情的话，让对话有起伏。

只输出一个 JSON 对象，不要任何其他文字，不要代码块：
{{"options": [{{"text": "……", "intent": "打听情报", "tone": "好奇"}}, …]}}"""


def _render_history(ctx: TurnContext, npc_name: str, last_n: int = 8) -> str:
    lines = []
    for h in ctx.history[-last_n:]:
        if h["speaker"] == "player":
            lines.append(f"{ctx.player_name}：「{h['text']}」")
        else:
            act = f"（{h['reply']['action']}）" if h["reply"].get("action") else ""
            lines.append(f"{npc_name}{act}：「{h['text']}」")
    return "\n".join(lines) or "（刚刚见面，还没说话）"


class PlayerOptionLLMAdapter:
    def __init__(self, backend: LLMBackend, params: TaskParams, tracer: Any = None, repair_attempts: int = 2):
        self.backend = backend
        self.params = params
        self.tracer = tracer
        self.repair_attempts = repair_attempts

    def generate(self, ctx: TurnContext, npc_name: str, n: int = 3) -> list[dict]:
        t, w = ctx.task, ctx.world
        said = [h["text"] for h in ctx.history if h["speaker"] == "player"]
        directions = t["info_goals"] or [x.split("（")[0] for x in t.get("locked_info", [])][:2]
        event = f"\n私人时光：「{t['event']['title']}」——{t['event']['premise']}" if t.get("event") else ""
        user = f"""# 场景
{ctx.scene['name']}，{w['time_label']}，局势{w['tension_label']}。距{w['deadline_label']}还有 {w['deadline_left']} 个时段。{event}

# 玩家眼下的目标
{w['quest_title']}·第 {w['stage_no']}/{w['stage_total']} 幕：{t['objective'] or t['stage_title']}

# 可以追问的方向（{npc_name}知道、但还没说的事）
{chr(10).join('- ' + d for d in directions) or '（暂无）'}

# {npc_name}对玩家的态度
{ctx.relationship['tier']}（好感 {ctx.relationship['affinity']}，信任 {ctx.relationship['trust']}）

# 最近的对话
{_render_history(ctx, npc_name)}

# 玩家之前说过的话（不要重复）
{chr(10).join('- ' + s for s in said[-6:]) or '（无）'}

请给出 {n} 句。"""
        req = LLMRequest(
            task="player_options",
            messages=[{"role": "system", "content": SYSTEM.format(player=ctx.player_name, npc=npc_name, n=n,
                                                                  tones="、".join(PLAYER_TONES),
                                                                  intents="、".join(PLAYER_INTENTS))},
                      {"role": "user", "content": user}],
            temperature=self.params.temperature, max_tokens=self.params.max_tokens,
            json_schema=PLAYER_OPTIONS_SCHEMA, schema_name="player_options",
            meta={"objective": t["objective"] or t["stage_title"], "turn": ctx.turn},
        )
        return call_json(self.backend, req, validate_player_options, self.tracer, self.repair_attempts)
