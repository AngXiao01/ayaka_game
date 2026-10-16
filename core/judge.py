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


def turn(god: Any, npc_name: str, text: str, hyps: list[dict]) -> Optional[dict]:
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
                     "criteria": ["没有推断，或者只是泛泛地猜", "说出了具体的人或事", "具体到人、做了什么、怎么知道的"]}}
    for h in hyps:
        qs[h["id"]] = {"type": "noul", "instructions": f"玩家是不是在明确断言：{h['desc']}？",
                       "criteria": {"true": f"玩家把「{h['desc']}」当成判断说了出来（哪怕换了说法）",
                                    "false": "玩家没这么说，只是在提问、猜测别的事，或者只是提到了相关的人和东西"}}
    out = d.decide("turn", {"listener": npc_name, "player_said": text}, qs)
    if not out:
        return None
    th, sp = out.get("threat"), out.get("specific")
    return {"threat": bool(th and th.yes(SURE)), "specific": sp.answer if sp else 0,
            "hypotheses": [h for h in hyps if h["id"] in out and out[h["id"]].yes(SURE)]}
