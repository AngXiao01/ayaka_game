"""心智状态：对话里的一方「以为对方知道什么、对什么感兴趣」，以及（NPC 一方）自己愿不愿意说。

为什么要有：
  · NPC 说话前要有“这个人大概知道什么、想听什么”的判断，才不会把对方已经知道的事当新闻讲，也不会对着不相干的人把什么都倒出来；
    说不说是她自己的主观决定（可以故意不说），所以判断里带一个 share：tell 愿意提 / withhold 先不说 / undecided 看情形。
  · 替玩家想台词的选项 agent 要有“玩家以为这个 NPC 大概知道什么”的判断，不然会写出「杜瓦尔女士是枫丹来的画家」这种
    默认对方一定知道、而且玩家自己也还不知道的话。

两份各自独立、各由一个小模型维护（llm/mind_llm_adapter.py），互相看不到对方的内容——
  side="npc"     NPC 对玩家的判断（NPC 心里的，玩家和选项 agent 都看不到）
  side="player"  玩家对这个 NPC 的判断（只依据玩家自己知道的和对方亲口说过的）

这个文件只管状态怎么合并、怎么写进提示词；不依赖游戏的任何东西。
"""
from __future__ import annotations

from typing import Any

MAX_ITEMS = 8
BELIEFS = ("knows", "maybe", "unknown")
SHARES = ("tell", "withhold", "undecided")
_BELIEF_TEXT = {"knows": "多半知道", "maybe": "不一定知道", "unknown": "多半不知道"}
_SHARE_TEXT = {"tell": "愿意提", "withhold": "先不说", "undecided": "看情形"}


def clean(obj: Any) -> list[dict]:
    """模型给的条目 → 规整过的条目；认不得的字段一律回到保守值，主题太长或为空的丢掉。"""
    items = obj.get("items") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        return []
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        topic = str(it.get("topic") or "").strip()
        if not 1 <= len(topic) <= 16:
            continue
        belief = it.get("belief") if it.get("belief") in BELIEFS else "maybe"
        share = it.get("share") if it.get("share") in SHARES else "undecided"
        try:
            interest = max(0, min(2, int(it.get("interest", 1))))
        except (TypeError, ValueError):
            interest = 1
        out.append({"topic": topic, "belief": belief, "interest": interest, "share": share,
                    "why": str(it.get("why") or "").strip()[:40]})
    return out[:4]


def merge(items: list[dict], updates: list[dict], turn: int) -> list[dict]:
    """同一主题后来的判断覆盖前面的；超过上限时丢掉最旧又最没兴趣的。"""
    by_topic = {x["topic"]: dict(x) for x in items}
    for u in updates:
        by_topic[u["topic"]] = {**u, "turn": turn}
    merged = list(by_topic.values())
    if len(merged) > MAX_ITEMS:
        merged.sort(key=lambda x: (x.get("interest", 1), x.get("turn", 0)), reverse=True)
        merged = merged[:MAX_ITEMS]
    return merged


def render_npc(items: list[dict]) -> str:
    """写进 NPC 提示词的一节。没有判断时返回空串。"""
    if not items:
        return ""
    lines = []
    for x in items:
        interest = "很感兴趣" if x.get("interest", 1) >= 2 else "兴趣不大" if x.get("interest", 1) == 0 else "有点兴趣"
        lines.append(f"- {x['topic']}：对方{_BELIEF_TEXT[x['belief']]}，{interest}；你{_SHARE_TEXT[x['share']]}"
                     + (f"（{x['why']}）" if x.get("why") else ""))
    return "\n".join(lines)


def render_player(items: list[dict], npc: str) -> str:
    """写进选项 agent 提示词的一节：玩家以为{npc}大概知道什么。永远带一条保底的规矩，没有判断时也一样。"""
    lines = [f"- {x['topic']}：{npc}{_BELIEF_TEXT[x['belief']]}" + (f"（{x['why']}）" if x.get("why") else "")
             for x in items]
    base = (f"{npc}知道什么，玩家本来是不清楚的：{npc}没提过的人和事、玩家自己也没听说过的东西，"
            f"不要当成{npc}一定知道的共同话题。想提时用询问或试探的口气，不要用陈述句替两个人都‘知道’了。")
    return base + ("\n" + "\n".join(lines) if lines else "")
