"""崩溃之后接着走：先回滚（状态永远靠事务），再带着日志重新跑一遍——问过的不重问、选过的不重选。

日志不是检查点：没有“跳过有副作用的节点”这回事。有副作用的节点照常重新执行、照常走事务；
日志只让它们拿到和上次一样的输入（模型的答案、判定、玩家的选择）。
"""
import json
from pathlib import Path

import pytest

from config.settings import load_settings
from core import crashpoints
from core.journal import Journal, content_key
from core.player_agent import ScriptedPlayer
from core.uow import UnitOfWork
from engine.game_engine import GameEngine
from llm.mock_backend import MockBackend
from tests import crash_harness as ch
from tests.test_crash_recovery import ALL, SCRIPT, _BASE, consistent

STARTS = [0, 4, 7, 8]                                 # 三次行动各自在脚本里的起点（talk 绫华…leave / talk 托马…leave / q；最后一个 = 全部做完）


class CountingMock(MockBackend):
    """记下真正发出去的请求。包装在日志的里面，所以命中日志的调用不会出现在这里。"""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.sent: list[str] = []

    def complete(self, req):
        self.sent.append(req.task)
        return super().complete(req)


def make(data_dir: Path, new_game: bool, script, resume=True):
    s = load_settings(env_file=data_dir / "none.env")
    s.game.data_dir, s.game.seed, s.harness.game_token_budget = data_dir, 3, 0
    s.harness.resume_after_crash = resume
    be = CountingMock(seed=1)
    return GameEngine(s, be, ScriptedPlayer(list(script)), out=lambda x: None, save_name="c", new_game=new_game), be


def crash(data_dir: Path, point: str, nth: int, resume=True):
    crashpoints.arm(point, nth)
    try:
        make(data_dir, True, SCRIPT, resume)[0].run()
        raise AssertionError(f"{point}#{nth} 没有触发崩溃")
    except crashpoints.SimulatedCrash:
        pass
    finally:
        crashpoints.disarm()


def journal_inputs(data_dir: Path) -> int:
    f = data_dir / "uow" / "c" / "journal.jsonl"
    if not f.exists():
        return 0
    n, live = 0, False
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            break
        if row.get("kind") == "begin":
            live = row["tx"] == ch.committed_tx(data_dir) + 1               # 属于已经提交过的事务的日志是过期的
        n += live and row.get("kind") == "input"
    return n


def restart(data_dir: Path, resume=True):
    """重启（恢复 + 读日志）：玩家接着要做的是脚本里剩下的。返回 (引擎, 后端, 重启时日志里的记录数)，还没开始跑。"""
    tx = ch.committed_tx(data_dir)
    k = journal_inputs(data_dir) if resume else 0
    eng, be = make(data_dir, False, SCRIPT[STARTS[tx] + k:], resume)
    return eng, be, (len(eng.journal._entries) if eng.journal else 0)


def restart_and_finish(data_dir: Path, resume=True):
    eng, be, entries = restart(data_dir, resume)
    eng.run()
    return eng, be, entries


@pytest.mark.parametrize("point,nth", ALL, ids=[f"{p}#{n}" for p, n in ALL])
def test_after_a_crash_everything_the_journal_holds_is_used_and_nothing_else_changes(tmp_path, point, nth):
    crash(tmp_path, point, nth)
    eng, be, entries = restart(tmp_path)
    ok, why = ch.recovered_to_last_commit(_BASE, tmp_path)
    assert ok, why                                                           # 先回滚：磁盘等于最后一次提交
    assert consistent(eng) == (True, "")
    eng.run()
    ev = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    used = sum(sum(e["served"].values()) for e in ev if e["event"] == "resume_done")
    assert used == entries, f"日志里有 {entries} 条记录，只用上了 {used} 条（其余是请求变了，说明重跑和上次不一样）"
    assert consistent(eng) == (True, "") and ch.committed_tx(tmp_path) >= 3  # 一路做到最后，一致
    assert not (tmp_path / "uow" / "c" / "journal.jsonl").exists()           # 提交之后日志清掉
    assert not (tmp_path / "uow" / "c" / "marker.json").exists()


def test_a_late_crash_reuses_the_words_that_were_already_said(tmp_path):
    """崩在第一次谈话的最后一轮回话之后：前面每一轮模型说过的话，重启后一字不差地用上，不再重问。"""
    crash(tmp_path, "turn.after.npc_reply", 3)
    first = [json.loads(l) for l in next((tmp_path / "traces").glob("c-*.jsonl")).read_text().splitlines()]
    said = [e["output"] for e in first if e["event"] == "llm" and e["task"] == "npc_step"]
    eng, be, entries = restart_and_finish(tmp_path)
    ev = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    again = [e["output"] for e in ev if e["event"] == "llm" and e["task"] == "npc_step"]
    assert said and all(s in again for s in said)                            # 说过的话原样在
    done = next(e for e in ev if e["event"] == "resume_done")
    assert done["served"]["input"] == 4 and done["served"]["llm"] > 10        # 玩家的 4 次选择、十几次模型调用都没重来
    assert be.sent.count("npc_step") < len(said) + 6                          # 现问的 NPC 回话只剩崩溃之后才发生的那几次


def test_without_the_journal_the_same_crash_asks_everything_again(tmp_path):
    """反向对照：关掉日志，同一处崩溃，重启后一条都用不上，玩家得把整步重做。"""
    crash(tmp_path, "turn.after.npc_reply", 3, resume=False)
    assert journal_inputs(tmp_path) == 0
    eng, be, entries = restart_and_finish(tmp_path, resume=False)
    assert entries == 0 and not any(e["event"] == "resume_done" for e in
                                    (json.loads(l) for l in eng.tracer.path.read_text().splitlines()))
    with_journal = CountingMock(seed=1)
    crash(tmp_path / "j", "turn.after.npc_reply", 3)
    eng2, be2, _ = restart_and_finish(tmp_path / "j")
    assert len(be2.sent) < len(be.sent)                                      # 有日志少发了请求


def test_a_run_whose_requests_changed_falls_back_to_asking_live(tmp_path):
    """日志里的请求内容和现在不一样（代码 / 世界变了）：一条都不命中，全部现问，仍然一致地做完——最坏就是没有日志。"""
    crash(tmp_path, "turn.after.npc_reply", 3)
    f = tmp_path / "uow" / "c" / "journal.jsonl"
    rows = [json.loads(l) for l in f.read_text().splitlines()]
    for r in rows[1:]:
        if r["kind"] in ("llm", "decide"):
            r["key"] = "0" * 20
    f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    eng = make(tmp_path, False, SCRIPT[STARTS[0] + journal_inputs(tmp_path):])[0]
    eng.run()
    ev = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    done = next(e for e in ev if e["event"] == "resume_done")
    assert "llm" not in done["served"] and done["served"]["input"] == 4
    assert consistent(eng) == (True, "") and ch.committed_tx(tmp_path) == 3


def test_a_changed_menu_stops_replaying_the_players_old_choices(tmp_path):
    """玩家的选择按顺序用，每次连同当时的选项一起校验：选项变了，这一次和之后的都不再用日志里的，交还给玩家。"""
    crash(tmp_path, "turn.after.npc_reply", 3)
    f = tmp_path / "uow" / "c" / "journal.jsonl"
    rows = [json.loads(l) for l in f.read_text().splitlines()]
    inputs = [r for r in rows if r["kind"] == "input"]
    inputs[1]["key"] = "x" * 20                                              # 第 2 次选择当时的选项和现在对不上
    f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    eng = make(tmp_path, False, SCRIPT[STARTS[0] + 1:])[0]                   # 只有第 1 次选择会被日志代替，其余玩家自己来
    eng.run()
    ev = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    assert next(e for e in ev if e["event"] == "resume_done")["served"]["input"] == 1
    assert consistent(eng) == (True, "")


def test_the_npc_random_numbers_are_restored_to_where_the_action_began(tmp_path):
    """NPC 的随机数（比如挑哪句告别语）重启后是从种子起算的；日志记着这次行动开始时的状态，接上时放回去。"""
    crash(tmp_path, "turn.after.npc_reply", 3)
    f = tmp_path / "uow" / "c" / "journal.jsonl"
    saved = json.loads(f.read_text().splitlines()[0])["meta"]["npc_rng"]
    eng = make(tmp_path, False, SCRIPT[STARTS[0] + journal_inputs(tmp_path):])[0]
    assert eng.npcs["ayaka"].rng.getstate() != (3, (), None)                 # 重启后是种子状态
    eng.run()
    assert saved["ayaka"][1] and "ayaka" in saved                            # 日志里确实存了


# ---------------------------------------------------------------- 日志本身

class FakeUow:
    def __init__(self, tx=1):
        self.tx, self.on_commit = tx, []


def test_journal_ignores_a_torn_last_line_and_a_stale_transaction(tmp_path):
    f = tmp_path / "j.jsonl"
    uow = FakeUow(2)
    j = Journal(f, uow)
    assert j.begin({"a": 1}) is True
    j.record("llm", "k1", {"text": "你好"})
    j.record("llm", "k2", {"text": "再见"})
    with f.open("a", encoding="utf-8") as fh:
        fh.write('{"kind": "llm", "key": "k3", "pay')                      # 写到一半崩了
    again = Journal(f, uow)                                                # 重启
    assert again.started and again.meta == {"a": 1} and len(again._entries) == 2
    assert again.begin() is False                                          # 接着上次的，不是新的一次
    assert again.serve("llm", "k1") == (True, {"text": "你好"}) and again.serve("llm", "k1") == (False, None)   # 每条只用一次
    assert again.serve("llm", "nope") == (False, None)
    uow.tx = 3                                                             # 事务已经提交过了：日志过期
    assert Journal(f, uow).started is False and not f.exists()


def test_duplicate_identical_requests_are_served_in_order(tmp_path):
    uow = FakeUow()
    j = Journal(tmp_path / "j.jsonl", uow)
    j.begin()
    j.record("llm", "same", {"n": 1})
    j.record("llm", "same", {"n": 2})
    j2 = Journal(tmp_path / "j.jsonl", uow)
    assert [j2.serve("llm", "same")[1], j2.serve("llm", "same")[1]] == [{"n": 1}, {"n": 2}]


def test_a_commit_discards_the_journal(tmp_path):
    uow = UnitOfWork(tmp_path / "u", 0)
    j = Journal(tmp_path / "u" / "journal.jsonl", uow)
    j.begin()
    j.record("input", "k", {"kind": "quit"})
    assert j.path.exists()
    uow.commit(lambda tx: None)
    assert not j.path.exists() and not j.started and j.begin() is True        # 下一次行动是新的


def test_content_key_depends_on_the_whole_request():
    assert content_key({"a": [1, 2]}) == content_key({"a": [1, 2]}) != content_key({"a": [1, 3]})
