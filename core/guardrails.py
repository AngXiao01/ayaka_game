"""NPC 回复的确定性护栏（不调模型、不花钱），在“核对员”之前先过一遍。

成熟的游戏 NPC 管线都在生成之后加一道“知识边界 / 设定违规”检查：角色不该知道的东西，说出口就是违规，
违规就带着理由重说，而不是靠提示词祈祷。这里只做能用代码判的那一类：

  knowledge_boundary  NPC 这句话里出现了他此刻“还不该知道”的案情词（见 quest 的 secrets：真相里的关键事实，
                      谁本来就知道、被出示什么线索后才知道、哪个信号达成后才知道）。
                      对方刚说过的词不算（NPC 复述对方的话、说“裂纹？我没听说过”都是正常的）。
"""
from __future__ import annotations

from typing import Iterable


def knowledge_boundary(speech: str, action: str, blocked: dict[str, str], said_by_player: Iterable[str]) -> list[str]:
    said = "".join(said_by_player)
    text = (speech or "") + (action or "")
    hit: dict[str, str] = {}
    for term, sid in blocked.items():
        if term in text and term not in said:
            hit.setdefault(sid, term)
    return [f"你说到了「{term}」，可这件事你并不知道（没人告诉过你，你也没亲眼见过），不能说，也不能暗示你知道。"
            for term in hit.values()][:3]
