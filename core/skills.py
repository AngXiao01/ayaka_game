"""NPC 技能（渐进式披露）。

content/skills/*.md：front matter 写 name / description / triggers，正文是具体做法。
- system prompt 里只放「技能索引」（每个一行），常驻但很短
- 命中触发条件（玩家意图、关键词、当前幕、注入标记）时由路由器把正文注入本轮情境
- 其余技能模型可以自己调用 load_skill 按需加载
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    body: str
    intents: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)
    on_injection: bool = False
    only_when_locked: bool = False      # 只有对方心里有暂时不肯说的事时才适用
    npcs: list[str] = field(default_factory=list)   # 只适用于这些角色（空 = 谁都能用）


def _parse(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path.name} 缺少 front matter")
    _, fm, body = text.split("---", 2)
    meta: dict[str, str] = {}
    for line in fm.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()

    def lst(key: str) -> list[str]:
        return [x.strip() for x in meta.get(key, "").split(",") if x.strip()]

    return Skill(name=meta["name"], description=meta.get("description", ""), body=body.strip(),
                 intents=lst("intents"), keywords=lst("keywords"), stages=lst("stages"), npcs=lst("npcs"),
                 on_injection=meta.get("on_injection", "").lower() == "true",
                 only_when_locked=meta.get("only_when_locked", "").lower() == "true")


class SkillLibrary:
    def __init__(self, skills: list[Skill]):
        self.skills = {s.name: s for s in skills}

    @classmethod
    def load(cls, skills_dir: Path) -> "SkillLibrary":
        return cls([_parse(p) for p in sorted(skills_dir.glob("*.md"))] if skills_dir.is_dir() else [])

    def index(self) -> str:
        return "\n".join(f"- {s.name}：{s.description}" for s in self.skills.values()) or "（无）"

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def route(self, intent: str | None, text: str, stage_id: str, injection: bool, limit: int = 1,
              has_locked: bool = True, npc: str | None = None) -> list[Skill]:
        """规则路由：确定性、零 token。"""
        hits: list[tuple[int, Skill]] = []
        for s in self.skills.values():
            if s.only_when_locked and not has_locked:
                continue
            if s.npcs and npc not in s.npcs:
                continue
            score = 0
            if injection and s.on_injection:
                score += 10
            if intent and intent in s.intents:
                score += 3
            if stage_id in s.stages:
                score += 2
            score += sum(1 for k in s.keywords if k and k in text)
            if score:
                hits.append((score, s))
        hits.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in hits[:limit]]
