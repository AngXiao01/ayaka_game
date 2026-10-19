"""自由行动：玩家在地点层打出「我想做什么」，代码把它归到一个动词，按规则和概率判定结算，写回世界状态。

模型（或关键词）只负责「这句话是哪个动词、对谁」；能不能做、成不成、得到什么，都由这里决定，
所以玩家可以跟踪、蹲守、搜查、告发、通风报信，剧情跟着世界状态走，而不是跟着剧本走。
动词表和「做了会发现什么」写在 quests/*.json 的 actions 里。没对上任何动词的话，交回场景描写（SceneAgent）。
"""
from __future__ import annotations

from typing import Any, Optional

from core import judge
from core.dice import roll, tag


class ActionResolver:
    def __init__(self, god: Any, world: Any, personas: dict, explorer: Any):
        self.god, self.world, self.personas, self.ex = god, world, personas, explorer
        self.spec = god.quest.actions or {}

    def parse(self, text: str) -> Optional[tuple[str, Optional[str]]]:
        """返回 (动词, 对象 NPC)；不是自由行动就返回 None。"""
        verbs = self.spec.get("verbs", {})
        for verb, v in verbs.items():
            if any(k in text for k in v["keywords"]):
                return verb, self._target(text)
        # 关键词没对上：问便宜的 Jev（说法绕一点也能认出来），拿不准就交回场景描写
        people = {n: f"{self.personas[n].name}（{self.personas[n].title}）" for n in self._present()}
        j = judge.action(self.god, text, verbs, people)
        if not j or j[0] == judge.NONE:
            return None
        return j[0], j[1] or self._target(text)

    def _present(self) -> list[str]:
        return [n for n in self.ex.npcs_here() if n not in self.ex.party]

    def _target(self, text: str) -> Optional[str]:
        phys = self.god.physical
        named = [n for n in (phys.mentioned(text) if phys is not None else []) if n in self.personas]
        if named:
            return named[0]
        here = self._present()
        return here[0] if len(here) == 1 else None

    def resolve(self, verb: str, target: Optional[str], text: str) -> tuple[str, list[str]]:
        """执行一个自由行动，返回 (给玩家看的文字, 新得到的线索)。"""
        v = self.spec["verbs"][verb]
        g = self.god
        if v.get("needs_target") and not target:
            return f"（你想{v['label']}谁？这里没有合适的人。）", []
        if v.get("needs_target") and target not in self._present():
            return f"（{self.personas[target].name}不在这里。）", []
        out = getattr(self, f"_{verb}")(v, target, text)
        if g.tracer_log:
            g.tracer_log("free_action", verb=verb, target=target, text=text, flags=g.flags[-3:])
        return out

    def _set(self, *flags: str) -> None:
        for f in flags:
            if f not in self.god.flags:
                self.god.flags.append(f)

    def _gain(self, clues: list[str], lines: list[str]) -> list[str]:
        got = []
        for c in clues:
            if self.god.destroyed(c):
                lines.append(f"（{self.god.destroyed(c)}）")
            elif self.god.give_clue(c):
                got.append(c)
                lines.append(f"    📜 获得线索「{self.god.quest.clue_name(c)}」")
        return got

    # ------------------------------------------------------------ 各个动词

    def _follow(self, v: dict, target: str, text: str) -> tuple[str, list[str]]:
        g, name = self.god, self.personas[target].name
        parts = [("", v["p"])]
        if f"alerted:{target}" in g.flags:
            parts.append(("对方正提防着", -0.25))
        if g.phase in ("黄昏", "夜晚"):
            parts.append(("天色掩护", 0.15))
        if self.ex.party:
            parts.append((f"带着{len(self.ex.party)}个人不好藏", -0.1 * len(self.ex.party)))
        ok, pct = roll(g, f"follow:{target}", sum(p for _, p in parts))
        line = tag(pct, ok, [f"{k}{p:+.0%}" for k, p in parts if k])
        g.pass_time(v["minutes"])
        if not ok:
            self._set(f"alerted:{target}")
            g.journal.append(f"{g.time_label()}　想跟踪{name}，被发现了")
            return f"{line}\n{name}走到街角，忽然回头看了你一眼，脚步快了起来。你只好装作路过——对方大概起了疑心。", []
        d = next((x for x in self.spec.get("discoveries", []) if x["verb"] == "follow" and x["target"] == target), None)
        if not d:
            return f"{line}\n你远远跟了{name}一阵。对方只是办些日常琐事，没什么可疑的。", []
        lines = [line, d["text"]]
        self._set(*d.get("set_flags", []))
        got = self._gain(d.get("clues", []), lines)
        g.journal.append(f"{g.time_label()}　跟踪{name}：{d['text']}")
        return "\n".join(lines), got

    def _stakeout(self, v: dict, target: Optional[str], text: str) -> tuple[str, list[str]]:
        g = self.god
        here = self.world.name(g.location)
        g.advance(1)                                  # 守到下一个时段：期间发生在这里的事（agenda 的 witness）都会看到
        who = [self.personas[n].name for n in self._present()]
        seen = "、".join(who) if who else "没什么人"
        return f"你在{here}找了个不起眼的位置守着，一直到{g.clock_label()}。这会儿这里有：{seen}。", []

    def _search(self, v: dict, target: Optional[str], text: str) -> tuple[str, list[str]]:
        lines, got = [], []
        for o in self.world.objects(self.god.location):
            desc, gained = self.ex.examine(o["id"])
            if gained or "🎲" in desc:
                lines.append(f"【{o['name']}】{desc}")
                got += gained
        for c in got:
            lines.append(f"    📜 获得线索「{self.god.quest.clue_name(c)}」")
        return ("\n".join(lines) or "你把这里里外外翻了一遍，没找到什么新东西。"), got

    def _report_authority(self, v: dict, target: Optional[str], text: str) -> tuple[str, list[str]]:
        g = self.god
        flag = self.spec.get("report_flag", "reported")
        if flag in g.flags:
            return self.spec.get("report_already", "你已经报过了，对方说会处理。"), []
        g.pass_time(v["minutes"])
        needs = set(self.spec.get("report_needs_any", []))
        if needs and not needs & set(g.clues):
            return self.spec["report_refused"], []
        self._set(flag)
        if g.quest.companion in self.ex.party:
            self._set(f"{flag}_{g.quest.companion}")
        g.journal.append(f"{g.time_label()}　{self.spec.get('report_journal', '去报了案')}")
        return self.spec["report_text"], []

    def _warn(self, v: dict, target: str, text: str) -> tuple[str, list[str]]:
        g = self.god
        g.pass_time(v["minutes"])
        self._set(f"warned_{target}", f"alerted:{target}")
        g.journal.append(f"{g.time_label()}　私下给{self.personas[target].name}通风报信")
        return self.spec.get("warn_text", f"你私下提醒了{self.personas[target].name}。"), []
