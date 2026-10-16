"""玩家台词选项：语气各异、接得上话、不重复，并给玩家指出可以追问的方向。

信息隔离：这个 agent 替玩家说话，只能看玩家视角（ctx.player_view：线索本、可以问的方向、对话原文）。
NPC 的待达成信号、锁着的秘密里写着答案，一律不给——给了，选项就会替玩家「先知道」小吉、辰巳印坊这些还没查到的事。
"""
from __future__ import annotations

from typing import Any

from config.settings import TaskParams
from core.context import TurnContext
from llm.base import LLMBackend, LLMRequest
from llm.schemas import PLAYER_OPTIONS_SCHEMA, PLAYER_TONES, PLAYER_INTENTS, validate_player_options
from llm.structured import call_json

OPTION_INTENTS = [i for i in PLAYER_INTENTS if i != "冒犯"]   # 想冒犯的话玩家可以自己打字，不往嘴里塞

SYSTEM = """你替{player}想台词。{player}是游历过蒙德、璃月等地的旅行者，独自来到稻妻；为人礼貌、好奇、有分寸，说话像普通人。{player}正在和{npc}说话，你替{player}想几句接下来可以说的话，给玩家参考——玩家也可以自己打字。

要求：
1. 给 0~{n} 句，看情况：接下来明显有几个不同的方向时才给满 {n} 句；对方只是在闲聊、或者明显该{player}自己回答一个具体问题时，给一两句就够，觉得不需要提示时可以一句不给（options 为空）。不要凑数，不要每次都是「一句关心、一句查案、一句玩笑」的固定套路。语气从这些里选：{tones}；意图从这些里选：{intents}。
2. 每句都要接得上{npc}刚才说的话，带着具体的细节（对方说过的东西、眼前的景物、自己的旅途见闻），不要空泛的客套。
3. 如果「可以问的方向」不为空，至少一句自然地朝那个方向问；方向为空时顺着对话聊，最多一句提到眼下的目标。私人时光里不谈公务。
4. 不要重复{player}之前说过的话，也不要和之前的选项换汤不换药。
   视角别串：选项是{player}说的话，不要复述{npc}刚说的话，也不要说只有{npc}才会说的话（{npc}对别人的叮嘱、{npc}的身份口吻、{npc}家里的事由{npc}来安排）。
5. 以{player}第一人称，像真人聊天一样口语、简短，每句不超过 30 字；大多数不需要括号动作，最多一句在开头带一个简短动作。
6. 选项要有内容：可以追问对方刚才话里的细节、分享旅途见闻（蒙德的风车与酒、璃月港的灯火、一路上遇到的人和事）、说出自己的看法、打趣对方。
7. 只用{player}已经知道的事：「{player}已经知道的事」里列的线索，和对话里对方亲口说过的话。{player}还没听说过的人名、地点、案情细节一律不提，也不要替{player}猜出答案——{player}只能问，不能先知道。
   不要编造{player}做过、看到过的事：「去过」「看见」「刚从……回来」只能用足迹里有的；线索本里的事如果是听人说的，就只能说「听说」「据说」，不能说成自己亲眼见到。没去过的地方，可以说想去、问在哪，不能说去过。提到案子时，只能引用线索本里的内容，或者提出疑问。
8. 不出戏：每句都是{player}在稻妻当面说出口的话。不提「游戏」「玩家」「剧情」「任务」「选项」「系统」这类说法；不提这个故事里没有出场的同伴（比如派蒙）；不搬出与眼前无关的原作剧情。
10. 「现在可以做的事」里的事，选项可以顺带做：一句话本身就是在做其中一件（告辞、邀对方同行、问那个关键问题、汇报、提出方案），就在这句上写 "act": 对应的编号，这句话会直接起作用；只是聊天就不写 act。每件事最多一句，不要硬凑。告别的话一定要带上告辞的编号。
9. 有分寸：对初次见面或还不熟的人礼貌克制；亲昵、调情或带私人感情的话，只在对方是神里绫华、且关系到了「友好」以上时才用，最多一句。

只输出一个 JSON 对象，不要任何其他文字，不要代码块：
{{"options": [{{"text": "……", "intent": "打听情报", "tone": "好奇"}}, {{"text": "……", "intent": "告别", "tone": "温和", "act": "leave"}}, …]}}"""


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
        t, w, pv = ctx.task, ctx.world, ctx.player_view
        said = [h["text"] for h in ctx.history if h["speaker"] == "player"]
        player = ctx.player_name
        event = f"\n私人时光：「{t['event']['title']}」——{t['event']['premise']}" if t.get("event") else ""
        user = f"""# 场景
{ctx.scene['name']}，{w['time_label']}，局势{w['tension_label']}。距{w['deadline_label']}还有 {w['deadline_left']} 个时段。{event}

# {player}眼下的目标
{t['objective'] or t['stage_title']}

# {player}已经知道的事（线索本）
{chr(10).join('- ' + c for c in pv.get('clues', [])) or '（还没有线索）'}

# {player}的足迹（只有这些是{player}亲身经历的）
{pv.get('footprint') or '（无）'}

# 可以问的方向（{player}这边的想法，不是答案）
{chr(10).join('- ' + d for d in pv.get('directions', [])) or '（暂无，顺着对话聊就好）'}

# {npc_name}对{player}的态度
{ctx.relationship['tier']}（好感 {ctx.relationship['affinity']}，信任 {ctx.relationship['trust']}）

# 现在可以做的事（编号: 说明）
{chr(10).join(f"- {a['id']}: {a['desc']}" for a in pv.get('actions', [])) or '（无）'}

# 最近的对话
{_render_history(ctx, npc_name)}

# {player}之前说过的话（不要重复）
{chr(10).join('- ' + s for s in said[-6:]) or '（无）'}

请给出 0~{n} 句。"""
        req = LLMRequest(
            task="player_options",
            messages=[{"role": "system", "content": SYSTEM.format(player=player, npc=npc_name, n=n,
                                                                  tones="、".join(PLAYER_TONES),
                                                                  intents="、".join(OPTION_INTENTS))},
                      {"role": "user", "content": user}],
            temperature=self.params.temperature, max_tokens=self.params.max_tokens,
            json_schema=PLAYER_OPTIONS_SCHEMA, schema_name="player_options",
            meta={"objective": t["objective"] or t["stage_title"], "turn": ctx.turn},
        )
        return call_json(self.backend, req, validate_player_options, self.tracer, self.repair_attempts)
