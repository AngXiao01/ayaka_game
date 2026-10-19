"""图执行器本身，以及“一轮对话”这张图：顺序、数据流、登记的读写和代码实际读写一致。"""
import ast
import json
from pathlib import Path

import pytest

from core import crashpoints
from core.dialogue_session import TURN_NODES, TurnState, turn_graph
from core.turn_graph import Graph
from engine.game_engine import GameEngine
from llm.mock_backend import MockBackend
from core.player_agent import ScriptedPlayer
from tests.test_physical import settings


# ------------------------------------------------------------ 执行器

class S:
    def __init__(self):
        self.log, self.stop = [], False


def toy():
    g = Graph("toy", "a", prefix="toy")
    for n in "abc":
        g.node(n, lambda s, n=n: s.log.append(n), writes=[n])
    g.edge("a", None, when=lambda s: s.stop)
    g.chain(list("abc"))
    return g


def test_nodes_run_in_order_with_a_trace_callback():
    seen = []
    s = toy().run(S(), on_node=lambda name, dt: seen.append(name))
    assert s.log == ["a", "b", "c"] and seen == ["a", "b", "c"]


def test_a_conditional_edge_ends_the_run_early():
    s = S()
    s.stop = True
    assert toy().run(s).log == ["a"]


def test_every_node_boundary_is_a_crash_point():
    crashpoints.arm("toy.after.b")
    s = S()
    with pytest.raises(crashpoints.SimulatedCrash):
        toy().run(s)
    crashpoints.disarm()
    assert s.log == ["a", "b"]                                    # 崩在 b 跑完之后、c 开始之前


def test_lint_finds_unreachable_nodes_unwritten_reads_and_cycles():
    g = Graph("bad", "a").node("a", None, writes=["x"]).node("b", None, reads=["x", "z"]).node("lonely", None)
    g.chain(["a", "b"])
    out = g.lint()
    assert any("lonely" in p and "走不到" in p for p in out) and any("读了 z" in p for p in out)
    assert not any("读了 x" in p for p in out)
    loop = Graph("loop", "a").node("a", None).node("b", None).edge("a", "b").edge("b", "a")
    assert any("有环" in p for p in loop.lint())
    with pytest.raises(RuntimeError, match="环"):
        Graph("loop", "a").node("a", lambda s: None).edge("a", "a").run(S())


# ------------------------------------------------------------ 一轮对话这张图

PIPELINE = ["pick_action", "interpret_action", "guard_input", "prefetch", "judge_say", "detect_leaving", "decide_invite", "social_effects",
            "stash_conv", "build_turn_ctx", "check_claims", "log_guard", "classify_branch", "echo_player", "npc_reply", "settle",
            "world_effects", "update_minds", "decide_end"]


def test_the_turn_pipeline_order_is_pinned():
    """顺序不是随便的：模型调用的先后（判定 → 声明核对 → 分支 → NPC 回话）会影响重放和金标准局。要改顺序请有意识地改这里。"""
    assert [n for n, _, _ in TURN_NODES] == PIPELINE
    assert turn_graph().lint(initial=["turn"]) == []              # 每个节点读的字段，前面一定有节点写过


def _actual_fields():
    """每个 _n_<节点> 实际读写了 st 的哪些字段。节点交给的处理器（_act_*）算在 interpret_action 头上。"""
    tree = ast.parse(Path("core/dialogue_session.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DialogueSession")
    shared = {"_judge_args": ("prefetch", "judge_say")}               # 预取和实际调用共用的参数函数：两个节点都算读了它读的字段
    out = {}
    for fn in cls.body:
        if isinstance(fn, ast.FunctionDef) and (fn.name.startswith(("_n_", "_act_")) or fn.name in shared):
            owners = shared.get(fn.name) or ("interpret_action" if fn.name.startswith("_act_") else fn.name[3:],)
            for n in ast.walk(fn):
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "st":
                    for owner in owners:
                        reads, writes = out.setdefault(owner, (set(), set()))
                        (writes if isinstance(n.ctx, ast.Store) else reads).add(n.attr)
    return out


def test_the_declared_reads_and_writes_match_what_each_node_really_does():
    actual = _actual_fields()
    wrong = []
    for name, reads, writes in TURN_NODES:
        r, w = actual[name]
        # 写：必须一致。读：登记的都要真的读了；多读的只能是这个节点自己先写过的（处理器读节点刚填好的 nid、forced 之类）
        if set(writes) != w or not set(reads) <= r or (r - set(reads)) - w:
            wrong.append(f"{name}: 登记 读{sorted(reads)} 写{sorted(writes)} / 实际 读{sorted(r)} 写{sorted(w)}")
    assert not wrong, "\n".join(wrong)
    assert set(actual) == set(PIPELINE)
    assert {f for _, r, w in TURN_NODES for f in (*r, *w)} <= set(TurnState.__dataclass_fields__)


def test_every_action_kind_has_a_handler_and_unknown_ones_fall_back_to_say():
    from core.dialogue_session import DialogueSession
    assert set(DialogueSession.ACTION_HANDLERS) == {"ask", "press", "clue", "report", "propose", "confront", "leave", "invite", "dismiss"}
    assert all(callable(getattr(DialogueSession, h)) for h in DialogueSession.ACTION_HANDLERS.values())
    assert callable(DialogueSession._act_say)                       # 自由发言等没登记的动作


def test_a_real_game_leaves_one_node_event_per_step_in_order(tmp_path):
    f = tmp_path / "data/saves/c.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"world": {"act_id": "act2", "phase_idx": 1, "clues": ["c_case", "c_shops"]}}))
    eng = GameEngine(settings(tmp_path), MockBackend(seed=0), ScriptedPlayer(
        [{"kind": "talk", "npc": "ayaka"}, ("你好，聊聊茶会吧。", "闲聊"), {"kind": "leave"}, "q"]), out=lambda s: None, save_name="c")
    eng.run()
    names = [json.loads(l)["name"] for l in eng.tracer.path.read_text().splitlines() if '"event": "node"' in l]
    assert names[:len(PIPELINE)] == PIPELINE                       # 第一轮完整走了一遍
    assert all(n in PIPELINE for n in names) and len(names) >= 2 * len(PIPELINE)
