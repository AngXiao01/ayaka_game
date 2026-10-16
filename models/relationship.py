"""NPC 对玩家的关系：好感（喜不喜欢）与信任（敢不敢托付），各 0~100。"""
from __future__ import annotations

from dataclasses import asdict, dataclass

# (好感下限, 名称, 给模型的行为指引)
TIERS = [
    (0, "疏远", "礼数周全但明显保持距离，不谈私事，回答简短克制。"),
    (25, "初识", "客气有礼，愿意交谈，但仍在观察对方的品性，私事点到为止。"),
    (45, "友好", "语气放松了些，会主动分享稻妻见闻，偶尔流露少女的一面。"),
    (65, "信赖", "把对方当作可托付之人，愿意谈及家族的难处与自己的心事。"),
    (85, "知己", "在对方面前不必是「白鹭公主」，会坦率表达心意、开些小玩笑。"),
]


def _clamp(x: int) -> int:
    return max(0, min(100, x))


@dataclass
class Relationship:
    affinity: int = 30
    trust: int = 20

    def apply(self, affinity_delta: int, trust_delta: int) -> None:
        self.affinity = _clamp(self.affinity + affinity_delta)
        self.trust = _clamp(self.trust + trust_delta)

    def tier(self) -> tuple[str, str]:
        name, hint = TIERS[0][1], TIERS[0][2]
        for lo, n, h in TIERS:
            if self.affinity >= lo:
                name, hint = n, h
        return name, hint

    def summary(self) -> dict:
        name, hint = self.tier()
        return {"affinity": self.affinity, "trust": self.trust, "tier": name, "behavior_hint": hint}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Relationship":
        return cls(affinity=int(d.get("affinity", 30)), trust=int(d.get("trust", 20)))
