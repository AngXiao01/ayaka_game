"""分支判定的几层问题（交给 llm/decider.py 的 Jev 来答）。

每个函数都是「先问 Jev，答得有把握就用，否则返回 None 让调用方走原来的关键词规则」。
Jev 只回答类型化的问题（是 / 否、几选一、打档位）和概率，不写台词、不改状态——用不用、怎么用由代码决定。
state 里只放判断需要的东西（玩家原话、此刻在场的人、可选项），不放案情真相。
"""
from __future__ import annotations

from typing import Any, Optional

NONE = "none"
SURE = 0.7            # 「是 X」要有这么大把握才采纳
SURE_NOT = 0.8        # 「什么都不是」要有这么大把握才跳过后面的模型调用


def _decider(god: Any):
    d = getattr(god, "decider", None)
    return d if d is not None and d.name != "off" else None


def proposal(god: Any, npc_name: str, text: str, props: list[dict]) -> Optional[tuple[str, float]]:
    """玩家这句话提出了哪个预设处置方案。返回 (flag 或 "none", 把握)；None = 没判。"""
    d = _decider(god)
    if not d or not props:
        return None
    crit = {p["flag"]: f"{p['label'].lstrip('⚖️ ').replace('提议：', '')}。例：{p['text']}" for p in props}
    crit[NONE] = "没有明确主张任何一种处置办法：只是在提问、闲聊、犹豫，或者同时说了好几种"
    out = d.decide("proposal", {"listener": npc_name, "player_said": text},
                   {"proposal": {"type": "choice", "instructions": "玩家这句话明确主张了哪一种处置办法？", "criteria": crit}})
    if not out or "proposal" not in out:
        return None
    return out["proposal"].answer, out["proposal"].p


def action(god: Any, text: str, verbs: dict, people: dict[str, str]) -> Optional[tuple[str, Optional[str]]]:
    """地点层的自由输入是哪个行动、对谁。people = {npc_id: 名字（附带身份）}。返回 (verb 或 "none", npc 或 None)。"""
    d = _decider(god)
    if not d:
        return None
    crit = {k: f"{v['label']}（例：{'、'.join(v['keywords'][:3])}）" for k, v in verbs.items()}
    crit[NONE] = "不是这些行动：只是想看看、描述环境、自言自语，或者别的事"
    qs = {"verb": {"type": "choice", "instructions": "玩家想做的是哪一种行动？", "criteria": crit}}
    if people:
        who = {k: v for k, v in people.items()}
        who[NONE] = "没有针对哪个人"
        qs["target"] = {"type": "choice", "instructions": "这个行动针对的是谁？", "criteria": who}
    out = d.decide("action", {"player_wants": text, "people_here": list(people.values())}, qs)
    if not out or "verb" not in out:
        return None
    v = out["verb"]
    if v.answer == NONE:
        return (NONE, None) if v.p >= SURE_NOT else None
    if v.p < SURE:
        return None
    t = out.get("target")
    return v.answer, (t.answer if t and t.answer != NONE and t.p >= SURE else None)


def turn(god: Any, npc_name: str, text: str, hyps: list[dict], last_line: str = "",
         situation: str = "") -> Optional[dict]:
    """对话里每一句自由发言只问一次：语气 + 说中了哪几条推测，合在一个请求里（省一次往返）。
    返回 {"threat": bool, "specific": 0~2, "hypotheses": [...]}；None = 没判。"""
    d = _decider(god)
    if not d or len(text.strip()) < 4:
        return None
    qs = {
        "threat": {"type": "noul", "instructions": "玩家是不是在威胁、恐吓或者逼迫对方？",
                   "criteria": {"true": "玩家用后果、报官、动手之类的话施压，或者语气凶狠地逼问",
                                "false": "玩家语气平和，或者只是认真地追问"}},
        "specific": {"type": "score", "instructions": "玩家说出的推断有多具体、有多少依据？",
                     "criteria": ["没有推断，或者只是泛泛地猜", "说出了具体的人或事", "具体到人、做了什么、怎么知道的"]},
        "abrupt": {"type": "noul", "instructions": "玩家这句话是不是来得很突兀：和对方刚说的话、眼下的处境接不上，换了谁听了都会愣一下？",
                   "criteria": {"true": "话题或打算突然跳开，跟刚才聊的、眼下要做的事对不上（比如正查着案子忽然说要出远门）",
                                "false": "顺着刚才的话说，或者是自然的转换话题、告别、寒暄"}},
        "invite": {"type": "noul", "instructions": "玩家是不是在邀请对方现在和自己一起去某处、一起走走？",
                   "criteria": {"true": "玩家请对方此刻同行、陪自己去哪儿（哪怕说得委婉）",
                                "false": "没有邀请；或者只是约以后、问对方要不要去、提到别人一起"}}}
    for h in hyps:
        qs[h["id"]] = {"type": "noul", "instructions": f"玩家是不是在明确断言：{h['desc']}？",
                       "criteria": {"true": f"玩家把「{h['desc']}」当成判断说了出来（哪怕换了说法）",
                                    "false": "玩家没这么说，只是在提问、猜测别的事，或者只是提到了相关的人和东西"}}
    out = d.decide("turn", {"listener": npc_name, "listener_just_said": last_line, "situation": situation,
                            "player_said": text}, qs)
    if not out:
        return None
    th, sp, ab = out.get("threat"), out.get("specific"), out.get("abrupt")
    return {"threat": bool(th and th.yes(SURE)), "specific": sp.answer if sp else 0,
            "abrupt": bool(ab and ab.yes(SURE)),
            **({"invite": out["invite"].yes(SURE)} if "invite" in out else {}),
            "hypotheses": [h for h in hyps if h["id"] in out and out[h["id"]].yes(SURE)]}


def unbacked(god: Any, footprint: str, texts: list[str]) -> Optional[set[int]]:
    """选项里哪几句替玩家编了没有的经历（说自己去过没去过的地方、亲眼见过没见过的东西）。返回要丢掉的下标。"""
    d = _decider(god)
    if not d or not texts or not footprint:
        return None
    qs = {f"o{i}": {"type": "noul", "instructions": f"这句话「{t}」是不是声称说话的人亲自去过、见过或做过某件事，而足迹里并没有？",
                    "criteria": {"true": "说了「刚去了」「我看见」「我查过」之类亲身经历，足迹里却没有这件事",
                                 "false": "没有声称亲身经历，或者经历和足迹对得上；「听说」「据说」「想去」都不算"}}
          for i, t in enumerate(texts)}
    out = d.decide("unbacked", {"speaker_footprint": footprint}, qs)
    if not out:
        return None
    return {i for i in range(len(texts)) if f"o{i}" in out and out[f"o{i}"].yes(SURE)}


def conceded(god: Any, npc_name: str, speech: str, player_text: str, signals: list) -> Optional[list[str]]:
    """NPC 这句话是不是已经亲口说出 / 承认了某件事（关键词没认出来的说法）。说出口就算，免得剧情和对话对不上。"""
    d = _decider(god)
    if not d or not signals or not speech.strip():
        return None
    qs = {s.id: {"type": "noul", "instructions": f"{npc_name}这句话是不是已经承认或说出了：{s.desc}？",
                 "criteria": {"true": f"{npc_name}亲口承认 / 说出了这件事（换了说法也算），听的人会觉得这事他认了",
                              "false": "没有承认，只是在反问、嘴硬、含糊其辞，或者说的是别的事"}}
          for s in signals}
    out = d.decide("conceded", {"speaker": npc_name, "player_said": player_text, "speaker_said": speech}, qs)
    if not out:
        return None
    return [k for k, v in out.items() if v.yes(SURE)]


def persuasion(god: Any, npc_name: str, point: str, lines: list[str]) -> Optional[int]:
    """玩家前面几句的理由，对「{point}」有多有说服力：0 没讲理由 / 1 有点道理 / 2 很有说服力。"""
    d = _decider(god)
    if not d or not lines:
        return None
    out = d.decide("persuasion", {"listener": npc_name, "recent_dialogue": lines[-6:]}, {
        "strength": {"type": "score", "instructions": f"玩家想让{npc_name}认下「{point}」。对话里玩家给出的理由有多有说服力？",
                     "criteria": ["没讲什么理由，只是一口咬定", "有一点道理，但还说不太通", "理由很充分，换了谁都难以反驳"]}})
    return out["strength"].answer if out and "strength" in out else None
