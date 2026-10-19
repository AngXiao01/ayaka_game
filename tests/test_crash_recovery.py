"""崩溃恢复：一次行动是一个事务；崩在任何一处，重启后都回到“最后一次提交”，不会出现“记忆领先于世界”。"""
import json

import pytest

from config.settings import load_settings
from core import crashpoints
from core.player_agent import ScriptedPlayer
from core.uow import UnitOfWork
from engine.game_engine import GameEngine
from llm.mock_backend import MockBackend
from tests import crash_harness as ch

SCRIPT = [{"kind": "talk", "npc": "ayaka"}, ("你好，聊聊茶会吧。", "闲聊"), ("再聊聊。", "闲聊"), {"kind": "leave"},
          {"kind": "talk", "npc": "thoma"}, ("那天夜里有谁来敲过后门吗？", "打听情报"), {"kind": "leave"}, "q"]
STATIC_POINTS = {"react.after_respond": 7, "react.after_tool": 2,                       # ReAct 循环里：工具（RAG 检索）刚拿到观察 / 模型刚决定怎么答
                 "journal.after_append": 55,                                                       # 行动日志：每记下一条模型答案 / 判定 / 玩家选择之后
                 "uow.after_backup": 4, "memory.after_append": 4, "save.before_state": 3, "uow.after_state": 3, "memory.after_rewrite": 1}
def make_engine(data_dir, new_game):
    s = load_settings(env_file=data_dir / "none.env")
    s.game.data_dir, s.game.seed, s.harness.game_token_budget = data_dir, 3, 0
    return GameEngine(s, MockBackend(seed=1), ScriptedPlayer(list(SCRIPT)), out=lambda x: None, save_name="c", new_game=new_game)


def consistent(eng):
    """跨文件的不变量：记忆里写着“见过面”，世界里就必须有见面次数和会话数。"""
    for nid, a in eng.npcs.items():
        sessions = [m.extra.get("session", 0) for m in a.memory.items if m.kind == "dialogue_summary"]
        if sessions and (eng.god.met.get(nid, 0) < 1 or max(sessions) > eng.god.session_count):
            return False, f"{nid} 的记忆里有第 {max(sessions)} 次会话，世界里 met={eng.god.met.get(nid, 0)} session_count={eng.god.session_count}"
    return True, ""


def _baseline_once():
    import tempfile
    from pathlib import Path
    return ch.baseline(make_engine, Path(tempfile.mkdtemp()))


_BASE = _baseline_once()                         # 矩阵的行是从基线里现场枚举出来的（节点崩溃点随图自动出现）
ALL = [(name, n) for name, count in sorted(_BASE["points"].items()) for n in range(1, count + 1)]


@pytest.fixture(scope="module")
def base():
    return _BASE


def test_the_matrix_covers_every_crash_point(base, tmp_path):
    points = base["points"]
    assert {k: v for k, v in points.items() if not k.startswith("turn.after.")} == STATIC_POINTS, \
        "副作用点的集合变了：新加的点要放进 STATIC_POINTS，矩阵才会覆盖它"
    from core.dialogue_session import TURN_NODES
    nodes = [n for n, _, _ in TURN_NODES]
    missing = [n for n in nodes if f"turn.after.{n}" not in points]
    assert not missing, f"这些节点在场景里一次都没跑到，它们的崩溃点没被覆盖：{missing}"
    assert sorted(base["by_tx"]) == [0, 1, 2, 3]


@pytest.mark.parametrize("point,nth", ALL, ids=[f"{p}#{n}" for p, n in ALL])
def test_a_crash_anywhere_restarts_at_the_last_commit(base, tmp_path, point, nth):
    eng = ch.crash_and_restart(make_engine, tmp_path, point, nth)
    ok, why = ch.recovered_to_last_commit(base, tmp_path)
    assert ok, why                                                       # 磁盘回到最后一次提交时的样子
    assert consistent(eng) == (True, "")                                  # 记忆和世界没有错位
    assert eng.recovery["action"] in ("none", "cleaned", "rolled_back")
    assert not (tmp_path / "uow" / "c" / "marker.json").exists()          # 日志清干净了
    eng.player = ScriptedPlayer([{"kind": "talk", "npc": "ayaka"}, ("崩溃之后接着聊。", "闲聊"), {"kind": "leave"}, "q"])
    eng.run()                                                             # 还能接着玩，而且接着玩之后仍然一致
    assert consistent(eng) == (True, "")


def test_crashing_in_the_middle_of_a_save_is_recognised_as_committed(base, tmp_path):
    """崩在“存档已经写完、备份还没清”：这次行动其实已经提交了，恢复只能清理，不能把记忆回滚掉。"""
    eng = ch.crash_and_restart(make_engine, tmp_path, "uow.after_state", 1)
    assert eng.recovery["action"] == "cleaned" and ch.committed_tx(tmp_path) == 1
    assert len(eng.npcs["ayaka"].memory.items) > 0                        # 绫华的记忆还在：存档和记忆是一起提交的


def test_the_matrix_can_see_the_original_bug(base, tmp_path, monkeypatch):
    """反向对照：把恢复关掉 = 回到原来的状态。同样的检查必须失败，否则上面那些测试证明不了任何事。"""
    monkeypatch.setattr(UnitOfWork, "recover", lambda self: {"action": "none"})
    eng = ch.crash_and_restart(make_engine, tmp_path, "save.before_state", 1)
    ok, why = ch.recovered_to_last_commit(base, tmp_path)
    assert not ok and "mem/ayaka.jsonl" in why
    assert consistent(eng)[0] is False                                    # 世界说没见过面，绫华的记忆里却有这次谈话


def test_a_crash_inside_the_react_loop_leaves_no_trace_even_though_retrieval_marks_memories_as_used(base, tmp_path):
    """RAG 检索会把取到的记忆标成“刚被用到”（accessed）——读也有内存里的副作用。它只在提交时随存档一起落盘，
    所以崩在循环中间（第 2 次工具刚拿到观察，托马的一条记忆刚被标成用过），磁盘上一点变化都没有。"""
    crashpoints.arm("react.after_tool", 2)
    crashed = make_engine(tmp_path, True)
    with pytest.raises(crashpoints.SimulatedCrash):
        crashed.run()
    crashpoints.disarm()
    assert any(m.accessed > 0 for m in crashed.npcs["thoma"].memory.items) and crashed.npcs["thoma"].memory._dirty   # 崩之前：内存里确实标了
    eng = make_engine(tmp_path, False)                                                                       # 重启
    ok, why = ch.recovered_to_last_commit(base, tmp_path)
    assert ok, why
    assert all(m.accessed == 0 for a in eng.npcs.values() for m in a.memory.items)                           # 之后：没有任何残留
