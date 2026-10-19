"""玩家：真人（终端输入）、自动（无人值守跑 Gemma / 评测用）、脚本（测试用）。

choose(options, ctx) 同时用于地点菜单（ctx 为 None）和对话（ctx 为 TurnContext）。
返回 PlayerAction：kind 与所选选项一致；对话里自由输入时 kind="say"、intent=自由发言。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union

from core.context import TurnContext
from core.player_option_generator import PlayerOption
from llm.schemas import FREE_INTENT

QUIT_WORDS = {"q", "quit", "exit", "退出"}


@dataclass
class PlayerAction:
    kind: str                   # 与 PlayerOption.kind 一致；quit 表示退出游戏；confront 为对质（含 tid / clue）
    text: str = ""
    intent: str = ""
    meta: dict = field(default_factory=dict)

    @classmethod
    def of(cls, o: PlayerOption) -> "PlayerAction":
        return cls(o.kind, o.text, o.intent, dict(o.meta))


def render_option(i: int, o: PlayerOption) -> str:
    if o.kind == "say":
        return f"  {i}. 「{o.text}」"
    if o.meta.get("line"):
        return f"  {i}. 「{o.text}」  〔→ {o.meta.get('act_label', '')}〕"
    return f"  {i}. {o.text}"


class HumanPlayer:
    def __init__(self, input_fn: Callable[[str], str] = input, out: Callable[[str], None] = print):
        self.input_fn = input_fn
        self.out = out

    def choose(self, options: list[PlayerOption], ctx: Optional[TurnContext]) -> PlayerAction:
        free = ctx is not None
        for i, o in enumerate(options, 1):
            self.out(render_option(i, o))
        self.out("  （输入编号；对话中也可以直接打字说你想说的话；q 保存并退出）" if free
                 else "  （输入编号；也可以直接写想查看什么，比如「看看廊下的茶具」；q 保存并退出）")
        while True:
            try:
                s = self.input_fn("> ").strip()
            except (EOFError, KeyboardInterrupt):
                return PlayerAction("quit")
            if not s:
                continue
            if s.lower() in QUIT_WORDS:
                return PlayerAction("quit")
            if s.isdigit() and 1 <= int(s) <= len(options):
                return PlayerAction.of(options[int(s) - 1])
            if free:
                return PlayerAction("say", s, FREE_INTENT)
            return PlayerAction("look", s, "查看")


class AutoPlayer:
    """cooperative：像认真的玩家一样查案；chaotic：随机乱选（含冒犯），用来压测稳定性。"""
    stops_at_ending = True          # 无人值守：到结局就停，不进后日谈

    def __init__(self, style: str = "cooperative", seed: Optional[int] = None,
                 out: Callable[[str], None] = print, farewell_after: int = 6):
        self.style = style
        self.rng = random.Random(seed)
        self.out = out
        self.farewell_after = farewell_after

    def _say(self, o: PlayerOption) -> PlayerAction:
        self.out(f"  ▸ {o.text}")
        return PlayerAction.of(o)

    def choose(self, options: list[PlayerOption], ctx: Optional[TurnContext]) -> PlayerAction:
        if self.style == "chaotic":
            pool = [o for o in options if o.kind not in ("quit", "view")]
            return self._say(self.rng.choice(pool))
        return self._say(self._converse(options, ctx) if ctx is not None else self._explore(options))

    def _explore(self, options: list[PlayerOption]) -> PlayerOption:
        by = lambda k: [o for o in options if o.kind == k]   # noqa: E731
        if by("event") and self.rng.random() < 0.8:
            return by("event")[0]
        talk = sorted((o for o in by("talk") if o.meta.get("pending")), key=lambda o: -o.meta["pending"])
        if talk:
            return talk[0]
        clue = [o for o in by("examine") if o.meta.get("clue")]
        if clue:
            return clue[0]
        moves = [o for o in by("move") if o.meta.get("useful")]
        if moves:
            return self.rng.choice(moves)
        unseen = [o for o in by("examine") if not o.meta.get("seen")]
        if unseen and self.rng.random() < 0.3:
            return unseen[0]
        companion = [o for o in by("talk") if o.meta.get("companion")]
        if companion and self.rng.random() < 0.3:
            return companion[0]
        return by("wait")[0]

    def _converse(self, options: list[PlayerOption], ctx: TurnContext) -> PlayerOption:
        by = lambda k: [o for o in options if o.kind == k]   # noqa: E731
        if by("report"):
            return by("report")[0]
        if by("ask") and (ctx.turn > 1 or self.rng.random() < 0.5):   # 关键问题：问出口就有进展（偶尔先寒暄一句）
            return by("ask")[0]
        for k in ("testimony", "evidence"):                     # 对质的二级菜单：挑有用的，没有就算了
            if by(k):
                good = [o for o in by(k) if o.meta.get("useful")]
                return good[0] if good else by("cancel")[0]
        if by("confront_menu") and by("confront_menu")[0].meta.get("useful"):
            return by("confront_menu")[0]
        props = [o for o in by("propose") if not o.meta.get("chosen")]
        if props and not any(o.meta.get("chosen") for o in by("propose")):
            mercy = [o for o in props if "mercy" in o.meta["flag"]]
            return (mercy or props)[0] if self.rng.random() < 0.8 else self.rng.choice(props)
        useful = [o for o in by("clue") if o.meta.get("useful")]
        if useful:
            return useful[0]
        if by("clue") or by("cancel"):
            return (by("cancel") or by("clue"))[0]
        if by("show_menu") and by("show_menu")[0].meta.get("useful"):
            return by("show_menu")[0]
        pending = bool(ctx.task["pending_signals"])
        says = by("say")
        persuadable = bool(ctx.task.get("locked_info")) and ctx.turn < self.farewell_after
        if pending or ctx.task.get("event") or persuadable:
            task = [o for o in says if o.intent in ("推进任务", "打听情报", "表达关心", "礼貌问候")]
            if task:
                return self.rng.choice(task)
            if says:
                return self.rng.choice(says)
        if ctx.turn >= self.farewell_after or (not pending and ctx.turn >= 2):
            return by("leave")[0]
        safe = [o for o in says if o.intent not in ("冒犯", "试探")]
        return self.rng.choice(safe or says or by("leave"))


Step = Union[int, str, tuple, dict]


class ScriptedPlayer:
    """按脚本出招：
    int    选第几个选项（从 1 开始）
    "q"    退出
    tuple  (text, intent) 自由发言
    dict   选第一个 kind 与 meta 都匹配的选项，如 {"kind": "talk", "npc": "ayaka"}
    """

    def __init__(self, script: Sequence[Step], out: Callable[[str], None] = lambda s: None):
        self.script = list(script)
        self.out = out
        self.seen_options: list[list[PlayerOption]] = []

    def choose(self, options: list[PlayerOption], ctx: Optional[TurnContext]) -> PlayerAction:
        self.seen_options.append(options)
        step = self.script.pop(0) if self.script else "q"
        if step == "q":
            return PlayerAction("quit")
        if isinstance(step, tuple):
            return PlayerAction("say", step[0], step[1])
        if isinstance(step, dict):
            want = dict(step)
            kind = want.pop("kind", None)
            for o in options:
                if (kind is None or o.kind == kind) and all(o.meta.get(k) == v for k, v in want.items()):
                    return PlayerAction.of(o)
            raise ValueError(f"脚本步骤 {step} 没有匹配的选项：{[(o.kind, o.meta) for o in options]}")
        return PlayerAction.of(options[min(int(step), len(options)) - 1])
