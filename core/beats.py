"""Beat 选择（storylet 的另一半）：信号图决定“什么能说”，这里由代码决定“这一轮顺势提哪一件”。

成熟的互动叙事把“条件满足的事件”做成候选池，再由另一个模块挑一件织进下一句（Dramamancer 的 interpretation / instantiation 分工）。
以前我们把所有此刻能说的事一股脑列给 NPC，让他“一次一件”——小模型要么全抖出来，要么哪件都不提，节奏全凭运气。
现在代码按这几样给候选打分，只把最合适的一件递给 NPC，其余这一轮不提：

  相关度   玩家刚说的话和这件事（描述 / 关键问题 / 引导）有多贴
  意图     玩家是在推进任务 / 打听情报 / 汇报，还是只是闲聊
  耐心     连着几轮没有进展，就更该找机会提一件
  先后     信号在任务文件里的先后（同分时前面的先）

闲聊、又没有贴合的话头、也还没卡住的时候，一件都不提：不硬塞剧情。
"""
from __future__ import annotations

from typing import Any, Optional

from models.retrieval import bigram_jaccard

TASKY = ("推进任务", "打听情报", "汇报", "出示线索", "表达关心", "试探")
OFFER_AT = 0.5


def pick_beat(god: Any, pending: list[dict], player_text: str, intent: Optional[str],
              stuck_turns: int = 0) -> tuple[Optional[dict], list[dict]]:
    """返回 (这一轮要织进去的那一件 | None, 各候选的打分——给 trace)。"""
    achieved = set(god.achieved)
    order = {s.id: i for i, s in enumerate(god.quest.signals)}
    scored = []
    for p in pending:
        if any(r not in achieved for r in p.get("requires", [])):
            continue
        sd = god.quest.signal(p["id"])
        label = ((sd.ask or {}).get("label", "") if sd else "") + " " + (sd.lead if sd and sd.lead else "")
        rel = bigram_jaccard(player_text or "", p["desc"] + " " + label) * 4.0
        score = rel + (0.6 if intent in TASKY else 0.0) + 0.15 * min(stuck_turns, 6) - order.get(p["id"], 0) * 0.001
        scored.append({"id": p["id"], "desc": p["desc"], "score": round(score, 3)})
    scored.sort(key=lambda x: -x["score"])
    best = scored[0] if scored and scored[0]["score"] >= OFFER_AT else None
    return best, scored
