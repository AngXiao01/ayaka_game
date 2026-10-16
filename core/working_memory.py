"""短期工作记忆：本次见面的对话窗口 + 滚动摘要。

窗口内保留原文；超出 window_turns 时，把最老的一批对话压成「前情提要」（LLM 摘要，失败退回抽取式），
既控制上下文长度，又不像直接截断那样丢掉早先的承诺和事实。
每次见面开始时清空（Context Reset），跨见面的连续性靠长期记忆和外化的世界状态，不靠上下文。
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from llm.base import LLMError
from llm.messages import estimate_tokens


def render_lines(entries: list[dict], player_name: str, npc_name: str) -> list[str]:
    out = []
    for e in entries:
        if e["speaker"] == "player":
            out.append(f"{player_name}：「{e['text']}」")
        else:
            act = f"（{e['reply'].get('action')}）" if e.get("reply", {}).get("action") else ""
            out.append(f"{npc_name}{act}：「{e['text']}」")
    return out


class WorkingMemory:
    def __init__(self, window_turns: int, player_name: str, npc_name: str,
                 compressor: Optional[Callable[[str, list[str]], str]] = None, mode: str = "llm",
                 tracer: Any = None):
        self.window = max(2, window_turns)
        self.player_name = player_name
        self.npc_name = npc_name
        self.compressor = compressor
        self.mode = mode
        self.tracer = tracer
        self.entries: list[dict] = []
        self.summary = ""

    def add(self, entry: dict) -> None:
        self.entries.append(entry)

    def lines(self, entries: Optional[list[dict]] = None) -> list[str]:
        return render_lines(self.entries if entries is None else entries, self.player_name, self.npc_name)

    def player_turns(self) -> int:
        return sum(1 for e in self.entries if e["speaker"] == "player")

    def maybe_compress(self) -> bool:
        """玩家发言轮数超窗就压缩最老的一半。返回是否发生了压缩。"""
        if self.player_turns() <= self.window:
            return False
        keep = self.window // 2
        cut, seen = len(self.entries), 0
        for i in range(len(self.entries) - 1, -1, -1):        # 从后往前数，保留最近 keep 个玩家回合
            if self.entries[i]["speaker"] == "player":
                seen += 1
                if seen == keep:
                    cut = i
                    break
        dropped, self.entries = self.entries[:cut], self.entries[cut:]
        dropped_lines = self.lines(dropped)
        before = estimate_tokens("\n".join(dropped_lines) + self.summary)
        used = self.mode
        if self.mode == "llm" and self.compressor:
            try:
                self.summary = self.compressor(self.summary, dropped_lines)
            except LLMError as e:
                used = "extractive"
                if self.tracer:
                    self.tracer.log("fallback", where="compress", error=str(e))
        if used != "llm" or not self.compressor:
            if self.mode != "off":
                used = "extractive"
                brief = "；".join(l[:40] for l in dropped_lines)
                self.summary = (self.summary + "；" + brief).strip("；")[-400:]
        if self.tracer:
            self.tracer.log("compress", mode=used, dropped_entries=len(dropped), kept_entries=len(self.entries),
                            tokens_before=before, tokens_after=estimate_tokens(self.summary), summary=self.summary)
        return True
