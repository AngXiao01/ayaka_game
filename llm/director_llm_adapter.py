"""导演 agent 的模型调用。输入只有 Director.view() 给的结构化视图，没有玩家原话。"""
from __future__ import annotations

import json
from typing import Any

from config.settings import TaskParams
from llm.base import LLMBackend, LLMRequest
from llm.schemas import director_schema, validate_director
from llm.structured import call_json

SYSTEM = """你是一个叙事调查游戏的「导演」。每到一个新的时段，你看一眼全局状态，决定要不要调一调节奏。

你只能提议下面几种动作（每次最多 {n} 个，什么都不做就输出空数组）：
- nudge：让某个角色下次见面时找机会主动提起一件事。npc 填角色 id，signal 必须是这个角色 signals 里 status=available 的 id。
- rumor：玩家卡住时放一条传闻，指个方向。npc 或 location 二选一，必须出自 rumor_targets；连续 2 个时段以上没有进展才会生效。
- move_npc：让某个角色这个时段出现在另一个地方（只能是他 schedule_locations 里的地方），方便玩家碰到他。
- tension：把气氛调松或调紧一点，delta 取 -0.1 ~ 0.1。

原则：
1. 玩家进展顺利时少干预，空数组就是最好的选择；卡住了再帮一把，而且只指方向，不给答案。
2. 你不能直接给线索、不能让任何事情「已经发生」、不能改结局；这些只能由玩家在对话和调查里自己达成。
3. 不认识的 id 不要写；字段用不到就填空字符串，delta 用不到填 0。
4. 截止时间临近（slots_left 很小）而主线还没推进时，优先让能推进主线的人出现在玩家身边。

只输出一个 JSON 对象，不要任何其他文字，不要代码块：
{{"note": "一句话说明你的判断", "actions": [{{"kind": "nudge", "npc": "...", "signal": "...", "location": "", "delta": 0, "reason": "..."}}]}}"""


class DirectorLLMAdapter:
    def __init__(self, backend: LLMBackend, params: TaskParams, tracer: Any = None, repair_attempts: int = 1):
        self.backend = backend
        self.params = params
        self.tracer = tracer
        self.repair_attempts = repair_attempts

    def propose(self, view: dict, max_actions: int) -> dict:
        npc_ids = [n["id"] for n in view["npcs"]]
        signal_ids = sorted({s["id"] for n in view["npcs"] for s in n["signals"]})
        location_ids = list(view["locations"])
        req = LLMRequest(
            task="director",
            messages=[{"role": "system", "content": SYSTEM.format(n=max_actions)},
                      {"role": "user", "content": "# 当前全局状态\n" + json.dumps(view, ensure_ascii=False, indent=1)
                                                  + "\n\n请给出这个时段的导演动作。"}],
            temperature=self.params.temperature, max_tokens=self.params.max_tokens,
            json_schema=director_schema(npc_ids, signal_ids, location_ids), schema_name="director",
            meta={"view": view},
        )
        return call_json(self.backend, req, lambda o: validate_director(o, max_actions), self.tracer,
                         self.repair_attempts)
