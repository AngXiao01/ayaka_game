"""ReAct 循环的四道闸与收紧工具集。"""
import itertools
import json

from core.agent_loop import LoopLimits, ReActLoop, StopReason
from core.tools import Tool, ToolRegistry
from llm.base import LLMError, LLMResponse
from llm.messages import check_conversation
from llm.schemas import npc_reply_schema, validate_npc_reply

REPLY = {"action": "", "speech": "好的。", "emotion": "平静", "intent": "", "affinity_delta": 0, "trust_delta": 0,
         "tension_delta": 0, "task_signals": [], "wants_to_end": False}


class Scripted:
    """按脚本输出每一步；记录每次请求，便于检查约束。"""
    name = "scripted"

    def __init__(self, steps):
        self.steps = list(steps)
        self.reqs = []

    def complete(self, req):
        check_conversation(req.messages)          # 每一次请求都必须是合法编排
        self.reqs.append(req)
        s = self.steps.pop(0) if self.steps else {"tool": "respond", "args": REPLY}
        if isinstance(s, Exception):
            raise s
        text = s if isinstance(s, str) else json.dumps({"thought": "", **s}, ensure_ascii=False)
        return LLMResponse(text=text, usage={"prompt_tokens": 100, "completion_tokens": 20})


def registry(calls=None, fail=False):
    def lore(args):
        if fail:
            raise RuntimeError("boom")
        (calls if calls is not None else []).append(args)
        return "社奉行" * 400
    return ToolRegistry([Tool("lookup_lore", "查设定", {"topic": "主题"}, lore, cacheable=True),
                         Tool("respond", "回复", {}, None, terminal=True)])


def run(steps, limits=None, reg=None, clock=None, **kw):
    b = Scripted(steps)
    loop = ReActLoop(b, reg or registry(), limits or LoopLimits(), 0.7, 300, **({"clock": clock} if clock else {}))
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    return loop.run(msgs, lambda a: validate_npc_reply(a, []), **kw), b


def test_direct_respond():
    r, b = run([{"tool": "respond", "args": REPLY}])
    assert r.stop is StopReason.FINISHED and r.final["speech"] == "好的。" and r.forced_by is None
    assert len(b.reqs) == 1


def test_tool_then_respond_with_truncated_observation():
    calls = []
    r, b = run([{"tool": "lookup_lore", "args": {"topic": "社奉行"}}, {"tool": "respond", "args": REPLY}],
               limits=LoopLimits(obs_max_chars=50), reg=registry(calls))
    assert r.stop is StopReason.FINISHED and calls == [{"topic": "社奉行"}]
    obs = b.reqs[1].messages[-1]["content"]
    assert "<observation" in obs and "已截断" in obs and len(obs) < 300


def test_unknown_tool_is_rejected_not_guessed():
    r, b = run([{"tool": "search_web", "args": {"q": "x"}}, {"tool": "respond", "args": REPLY}])
    assert r.stop is StopReason.FINISHED
    assert r.steps[0].error == "unknown_tool"
    assert "不存在" in b.reqs[1].messages[-1]["content"]


def test_schema_enum_lists_registered_tools():
    _, b = run([{"tool": "respond", "args": REPLY}])
    assert b.reqs[0].json_schema["properties"]["tool"]["enum"] == ["lookup_lore", "respond"]


def test_repeat_detection_forces_finish():
    same = {"tool": "lookup_lore", "args": {"topic": "社奉行"}}
    r, b = run([same, same, same, {"tool": "respond", "args": REPLY}], limits=LoopLimits(max_steps=6, max_repeat=1))
    assert r.stop is StopReason.FINISHED and r.forced_by is StopReason.REPEAT
    # 被强制后工具集收紧为只剩 respond（解码层 enum 约束）
    assert b.reqs[-1].json_schema["properties"]["tool"]["enum"] == ["respond"]
    assert "检测到重复调用" in b.reqs[-1].messages[-1]["content"]


def test_max_steps_forces_respond_on_last_step():
    steps = [{"tool": "lookup_lore", "args": {"topic": f"t{i}"}} for i in range(2)] + [{"tool": "respond", "args": REPLY}]
    r, b = run(steps, limits=LoopLimits(max_steps=3), final_args_schema=npc_reply_schema([]))
    assert r.forced_by is StopReason.MAX_STEPS and len(b.reqs) == 3
    assert "required" not in b.reqs[0].json_schema["properties"]["args"]
    assert b.reqs[2].json_schema["properties"]["args"]["required"]   # 最后一步 args 也被 schema 约束


def test_never_responds_hits_hard_stop():
    steps = [{"tool": "lookup_lore", "args": {"topic": f"t{i}"}} for i in range(10)]
    r, _ = run(steps, limits=LoopLimits(max_steps=3))
    assert r.stop is StopReason.MAX_STEPS and r.final is None


def test_token_budget_forces_then_stops():
    steps = [{"tool": "lookup_lore", "args": {"topic": f"t{i}"}} for i in range(10)]
    r, b = run(steps, limits=LoopLimits(max_steps=10, token_budget=200))
    assert r.stop is StopReason.BUDGET
    assert b.reqs[-1].json_schema["properties"]["tool"]["enum"] == ["respond"]


def test_timeout():
    t = itertools.chain([0, 0], itertools.repeat(100))
    r, _ = run([{"tool": "lookup_lore", "args": {"topic": "a"}}], limits=LoopLimits(timeout_s=10),
               clock=lambda: next(t))
    assert r.stop is StopReason.TIMEOUT


def test_tool_exception_becomes_observation():
    r, b = run([{"tool": "lookup_lore", "args": {"topic": "a"}}, {"tool": "respond", "args": REPLY}],
               reg=registry(fail=True))
    assert r.stop is StopReason.FINISHED and r.steps[0].error == "tool_exception"
    assert "RuntimeError" in b.reqs[1].messages[-1]["content"] and "boom" not in b.reqs[1].messages[-1]["content"]


def test_invalid_final_args_get_feedback():
    r, b = run([{"tool": "respond", "args": {"speech": ""}}, {"tool": "respond", "args": REPLY}])
    assert r.stop is StopReason.FINISHED and r.steps[0].error == "invalid_final"


def test_llm_error_stops_loop():
    r, _ = run([LLMError("down")])
    assert r.stop is StopReason.LLM_ERROR


def test_tool_cache():
    calls = []
    reg = registry(calls)
    run([{"tool": "lookup_lore", "args": {"topic": "a"}}, {"tool": "respond", "args": REPLY}], reg=reg)
    r, _ = run([{"tool": "lookup_lore", "args": {"topic": "a"}}, {"tool": "respond", "args": REPLY}], reg=reg)
    assert len(calls) == 1 and r.steps[0].cached


def test_allow_tools_false_restricts_to_respond():
    _, b = run([{"tool": "respond", "args": REPLY}], allow_tools=False)
    assert b.reqs[0].json_schema["properties"]["tool"]["enum"] == ["respond"]


def test_bare_reply_tolerated_as_respond():
    r, _ = run([json.dumps(REPLY, ensure_ascii=False)])
    assert r.stop is StopReason.FINISHED
