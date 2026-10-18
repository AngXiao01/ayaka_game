"""NPC 回复的确定性护栏（不调模型、不花钱），在“核对员”之前先过一遍。

成熟的游戏 NPC 管线都在生成之后加一道“知识边界 / 设定违规”检查：角色不该知道的东西，说出口就是违规，
违规就带着理由重说，而不是靠提示词祈祷。这里只做能用代码判的那一类：

  knowledge_boundary  NPC 这句话里出现了他此刻“还不该知道”的案情词（见 quest 的 secrets：真相里的关键事实，
                      谁本来就知道、被出示什么线索后才知道、哪个信号达成后才知道）。
                      对方刚说过的词不算（NPC 复述对方的话、说“裂纹？我没听说过”都是正常的）。
  slip_missing        这一轮规定要漏一句口风（硬问失败，见信号的 partial），NPC 这句话里却没有那个意思。
                      小模型常被人物卡里写死的“推说不知道”压过去，所以由代码查：没带上就带着理由重说一次。
"""
from __future__ import annotations

from typing import Iterable

from models.retrieval import coverage

SLIP_COVERAGE = 0.25          # 口风里的词至少这个比例出现在回复里才算漏了（实测：漏了的 ≥0.32，没漏的 ≤0.11）


def knowledge_boundary(speech: str, action: str, blocked: dict[str, str], said_by_player: Iterable[str]) -> list[str]:
    said = "".join(said_by_player)
    text = (speech or "") + (action or "")
    hit: dict[str, str] = {}
    for term, sid in blocked.items():
        if term in text and term not in said:
            hit.setdefault(sid, term)
    return [f"你说到了「{term}」，可这件事你并不知道（没人告诉过你，你也没亲眼见过），不能说，也不能暗示你知道。"
            for term in hit.values()][:3]


def slip_missing(speech: str, slip: str) -> list[str]:
    if not slip or coverage(slip, speech or "") >= SLIP_COVERAGE:
        return []
    return [f"这一轮你被逼问，要在这句话里漏出一点口风，大意是「{slip}」，可你这句话里没有这个意思。"
            "用你自己的话把它说出来（一句就够，不展开细节，不把事情说出来），其余照常回避，别说「没有」「不知道」把它否掉。"]
