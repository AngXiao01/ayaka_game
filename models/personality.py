"""角色卡：从 content/npcs/<id>.json 加载，渲染成 system prompt 片段。

两种渲染（PERSONA_MODE）：
  full   全部内容进 system（整局不变，前缀缓存友好）
  index  渐进式披露：system 只放「核心」（身份、性格、说话方式、语气示例、禁忌）和一份细节索引；
         喜好、人际、看法、往事这些细节按本轮话题由代码挑出来附在情境里，模型也可以用 load_persona 自己查
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DETAIL_SECTIONS = ("喜好", "人际", "看法", "往事")
CHAT_INTENTS = ("闲聊", "表达关心", "礼貌问候", "赞美", "自由发言")


def _cues(key: str) -> list[str]:
    """一个话题/人名的触发词：本身，加上按标点拆开后不短于两个字的片段。"""
    parts = [x for x in re.split(r"[「」、，,/（）()\s]|这个|这些", key) if len(x) >= 2]
    return list(dict.fromkeys([key] + parts))


@dataclass
class Personality:
    id: str
    name: str
    title: str
    identity: str
    core_traits: list[str]
    speech_style: list[str]
    likes: list[str] = field(default_factory=list)
    dislikes: list[str] = field(default_factory=list)
    relations: dict[str, str] = field(default_factory=dict)
    taboos: list[str] = field(default_factory=list)
    style_examples: list[str] = field(default_factory=list)
    fallback_lines: dict[str, list[str]] = field(default_factory=dict)
    farewell_lines: list[str] = field(default_factory=list)
    role: str = "minor"                                        # companion | minor
    brief: str = ""                                            # 给玩家看的一句话介绍
    initial: dict = field(default_factory=lambda: {"affinity": 30, "trust": 20})
    schedule: list[dict] = field(default_factory=list)         # [{location, phases, acts?}]，按顺序取第一条匹配
    stances: dict[str, str] = field(default_factory=dict)      # 每一幕的立场，key 为 act id 或 default
    secrets: list[dict] = field(default_factory=list)          # [{min_trust, text}]
    fallback_options: list[dict] = field(default_factory=list)
    opinions: dict[str, str] = field(default_factory=dict)     # 话题 -> 她自己的看法（让聊天有态度）
    anecdotes: list[str] = field(default_factory=list)         # 可以随口讲的小事、往事（让聊天有细节）
    days: dict[str, str] = field(default_factory=dict)         # 第几天 -> 今天的近况与心情（让聊天有「当下」）
    curiosities: list[str] = field(default_factory=list)       # 想问对方的事（让聊天是双向的）
    knows_clues: list[str] = field(default_factory=list)       # 这个人一开始就知道的线索（别人拿这些给他看，他不会意外；也不会出现在「出示线索」菜单里）
    addressed_as: list[str] = field(default_factory=list)      # 玩家（一个有礼貌的外来旅行者）该怎么称呼他 / 她，选项台词只许用这些
    aliases: list[str] = field(default_factory=list)           # 别人怎么称呼他（「大小姐」「白鹭公主」），用来认出话里提到的人
    outing: dict = field(default_factory=dict)                 # 能不能和玩家一起出门：min_affinity / phases / tell / secret / note …
    confidants: list[str] = field(default_factory=list)        # 见过玩家之后会去跟谁说（托人带话）
    absent_when_flags: list[str] = field(default_factory=list) # 剧情走到这些标记之后就不在地图上了（比如被送官）
    present_when_flags: list[str] = field(default_factory=list) # 这些标记都立起来之后才出现在地图上（比如茶会开始后才到场的客人）

    @classmethod
    def load(cls, path: Path) -> "Personality":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def load_all(cls, npc_dir: Path) -> dict[str, "Personality"]:
        out = {}
        for p in sorted(npc_dir.glob("*.json")):
            persona = cls.load(p)
            out[persona.id] = persona
        return out

    def location_at(self, phase: str, act: str) -> str | None:
        for slot in self.schedule:
            if phase in slot.get("phases", []) and (not slot.get("acts") or act in slot["acts"]):
                return slot["location"]
        return None

    def stance(self, act: str) -> str:
        return self.stances.get(act) or self.stances.get("default", "")

    def to_prompt(self, player_name: str, mode: str = "full") -> str:
        if mode == "index":
            return self.to_prompt_index(player_name)

        def bullets(items: list[str]) -> str:
            return "\n".join(f"- {x.format(player_name=player_name)}" for x in items)

        parts = [
            f"姓名：{self.name}（{self.title}）",
            f"身份：{self.identity}",
            "性格：\n" + bullets(self.core_traits),
            "说话方式：\n" + bullets(self.speech_style),
        ]
        if self.likes:
            parts.append("喜欢：" + "、".join(self.likes))
        if self.dislikes:
            parts.append("反感：" + "、".join(self.dislikes))
        if self.relations:
            parts.append("人际：\n" + "\n".join(f"- {k}：{v}" for k, v in self.relations.items()))
        if self.opinions:
            parts.append("你的看法（被问到或聊到时，说出你自己的态度，不要和稀泥）：\n"
                         + "\n".join(f"- {k}：{v}" for k, v in self.opinions.items()))
        if self.anecdotes:
            parts.append("你的往事与小事（聊天时可以自然地讲起，一次只讲一件，讲过的不要再讲）：\n" + bullets(self.anecdotes))
        if self.style_examples:
            parts.append("语气示例（只参考语气，不要照抄）：\n" + bullets(self.style_examples))
        if self.taboos:
            parts.append("禁止：\n" + bullets(self.taboos))
        return "\n\n".join(parts)

    # ------------------------------------------------------------ 渐进式披露

    def _core_parts(self, player_name: str) -> list[str]:
        def bullets(items: list[str]) -> str:
            return "\n".join(f"- {x.format(player_name=player_name)}" for x in items)

        parts = [f"姓名：{self.name}（{self.title}）", f"身份：{self.identity}",
                 "性格：\n" + bullets(self.core_traits), "说话方式：\n" + bullets(self.speech_style)]
        if self.style_examples:
            parts.append("语气示例（只参考语气，不要照抄）：\n" + bullets(self.style_examples))
        if self.taboos:
            parts.append("禁止：\n" + bullets(self.taboos))
        return parts

    def to_prompt_index(self, player_name: str) -> str:
        idx = []
        if self.likes or self.dislikes:
            idx.append(f"- 喜好：喜欢的 {len(self.likes)} 样、反感的 {len(self.dislikes)} 样")
        if self.relations:
            idx.append("- 人际：" + "、".join(self.relations))
        if self.opinions:
            idx.append("- 看法：" + "、".join(self.opinions))
        if self.anecdotes:
            idx.append(f"- 往事：{len(self.anecdotes)} 件可以随口讲的小事")
        parts = self._core_parts(player_name)
        if idx:
            parts.append("人物细节索引（和话题相关时会附在「当前情境」里；需要时也可以用 load_persona 查某一类的全文）：\n"
                         + "\n".join(idx))
        return "\n\n".join(parts)

    def detail(self, section: str, player_name: str = "") -> str:
        fmt = lambda x: x.format(player_name=player_name)    # noqa: E731
        if section == "喜好":
            return "喜欢：" + "、".join(self.likes) + "\n反感：" + "、".join(self.dislikes)
        if section == "人际":
            return "\n".join(f"- {k}：{v}" for k, v in self.relations.items())
        if section == "看法":
            return "\n".join(f"- {k}：{v}" for k, v in self.opinions.items())
        if section == "往事":
            return "\n".join(f"- {fmt(x)}" for x in self.anecdotes)
        return ""

    def relevant_details(self, text: str, intent: str | None, turn: int, player_name: str = "") -> list[tuple[str, str]]:
        """规则路由：按这句话提到的人和话题、以及对话的性质，挑出本轮该附上的细节（确定性、零 token）。"""
        text = text or ""
        out: list[tuple[str, str]] = []
        rel = [f"- {k}：{v}" for k, v in self.relations.items()
               if any(c in text for c in _cues(k) + ([k[-2:]] if len(k) >= 3 else []))]   # 「绫人」也算提到神里绫人
        if rel:
            out.append(("人际", "\n".join(rel)))
        ops = [f"- {k}：{v}" for k, v in self.opinions.items() if any(c in text for c in _cues(k))]
        if ops:
            out.append(("你的看法（说出你自己的态度，不要和稀泥）", "\n".join(ops)))
        taste = [x for x in self.likes + self.dislikes if any(c in text for c in _cues(x))]
        if taste or intent in ("闲聊", "赞美"):
            out.append(("喜好", self.detail("喜好")))
        if self.anecdotes and intent in CHAT_INTENTS:
            out.append(("可以随口讲的一件小事（讲过的不要再讲）",
                        "- " + self.anecdotes[turn % len(self.anecdotes)].format(player_name=player_name)))
        return out
