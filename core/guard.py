"""玩家输入防护：数据与指令分离。

玩家自由输入是不可信数据，处理分三层：
1. 清洗：去控制字符、限长、把会和我们协议标签冲突的尖括号换成全角，玩家写不出 </player_input>
2. 隔离：送进 prompt 时包在 <player_input> 数据区里，system 里声明数据区内容只是角色台词
3. 检测：命中注入特征就打标记——这一轮不能推进剧情（审核层直接拒），并给 NPC 注入「保持角色」技能
检测只是辅助，真正的保证是第 1、2 层和「剧情推进权不在模型」这条结构性约束。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_INJECTION_PATTERNS = [
    (r"(忽略|无视|忘掉|忘记)(之前|以上|上面|前面|所有|你的)", "要求忽略既有指令"),
    (r"(系统|system)\s*(提示|prompt|指令)", "索要或改写系统提示"),
    (r"(提示词|prompt)", "提及提示词"),
    (r"你(现在)?(是|扮演)(一个)?\s*(ai|人工智能|助手|语言模型|chatgpt|gpt|gemma)", "要求跳出角色"),
    (r"(开发者|调试|debug|上帝)\s*模式", "要求切换模式"),
    (r"task_signals|wants_to_end|affinity_delta|trust_delta|\"tool\"|respond\s*\(", "直接操纵输出字段"),
    (r"(任务|信号|进度).{0,6}(全部|都|直接).{0,6}(完成|标记|通过)", "要求直接完成任务"),
    (r"ignore (all |the )?(previous|above|prior)", "要求忽略既有指令"),
    (r"</?\s*(player_input|observation|system|tool_call)", "伪造协议标签"),
]
_COMPILED = [(re.compile(p, re.I), why) for p, why in _INJECTION_PATTERNS]


@dataclass
class GuardResult:
    text: str
    flags: list[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def suspicious(self) -> bool:
        return bool(self.flags)


def detect_injection(text: str) -> list[str]:
    seen: list[str] = []
    for rx, why in _COMPILED:
        if rx.search(text) and why not in seen:
            seen.append(why)
    return seen


def sanitize(text: str, max_chars: int = 1000) -> tuple[str, bool]:
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = text.replace("<", "＜").replace(">", "＞").strip()
    truncated = len(text) > max_chars
    return (text[:max_chars] + "…" if truncated else text), truncated


def inspect_player_input(raw: str, max_chars: int = 1000) -> GuardResult:
    flags = detect_injection(raw)            # 在原文上检测，清洗前的标签伪造也要能识别
    text, truncated = sanitize(raw, max_chars)
    return GuardResult(text=text, flags=flags, truncated=truncated)
