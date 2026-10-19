"""并行判定：只许省时间，不许改结果。"""
import json
import threading
import time

import pytest

from core.player_agent import ScriptedPlayer
from core.prefetch import Prefetch
from engine.game_engine import GameEngine
from engine.trace import Tracer
from llm.decider import Decision, ScriptedDecider
from llm.mock_backend import MockBackend
from tests.test_physical import settings


# ---------------------------------------------------------------- Prefetch 本身

def test_disabled_prefetch_is_a_plain_call():
    pf, seen = Prefetch(enabled=False), []
    pf.start("a", lambda x: seen.append(("thread", threading.current_thread().name)), (1,))      # 什么都不做
    assert not seen
    assert pf.get("a", lambda x: x + 1, (1,)) == 2 and pf.hits == 0


def test_prefetched_answer_is_used_only_when_the_arguments_match():
    pf, calls = Prefetch(enabled=True), []

    def fn(x):
        calls.append(x)
        return x * 10
    pf.start("a", fn, (1,))
    assert pf.get("a", fn, (1,)) == 10 and pf.hits == 1 and calls == [1]            # 对得上：用预取的，没有多问
    pf.start("b", fn, (2,))
    pf._started["b"][1].result()                                                    # 等预取那次跑完，再看计数
    assert pf.get("b", fn, (3,)) == 30 and pf.misses == 1 and sorted(calls) == [1, 2, 3]   # 参数变了：作废，当场重问
    assert pf.get("c", fn, (4,)) == 40                                              # 根本没预取：当场问
    pf.start("d", fn, None)                                                         # None = 这一轮不问
    assert "d" not in pf._started


def test_errors_surface_where_the_answer_is_used_not_where_it_was_started():
    pf = Prefetch(enabled=True)

    def boom(x):
        raise ValueError("模型挂了")
    pf.start("a", boom, (1,))                                                       # 提交时不抛
    with pytest.raises(ValueError):
        pf.get("a", boom, (1,))


def test_prefetched_calls_really_overlap():
    pf = Prefetch(enabled=True)

    def slow(x):
        time.sleep(0.25)
        return x
    t0 = time.monotonic()
    for k in "abc":
        pf.start(k, slow, (k,))
    assert [pf.get(k, slow, (k,)) for k in "abc"] == ["a", "b", "c"]
    assert time.monotonic() - t0 < 0.55                                             # 串行要 0.75


def test_worker_threads_keep_the_callers_trace_labels(tmp_path):
    t = Tracer(tmp_path / "t.jsonl", "r")
    pf = Prefetch(enabled=True, tracer=t)
    with t.span(agent="npc", turn=3):
        pf.start("a", lambda: t.log("hello"), ())
    pf.get("a", lambda: None, ())
    rec = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines() if '"hello"' in l][0]
    assert rec["span"]["agent"] == "npc" and rec["span"]["turn"] == 3 and rec["span"]["prefetch"] == "a"
    assert t.span_ctx == {}                                                         # 调用方这边没被污染


def test_tracer_counts_survive_concurrent_logging(tmp_path):
    t = Tracer(tmp_path / "t.jsonl", "r")
    ths = [threading.Thread(target=lambda: [t.log("fallback", where="x", error="e") for _ in range(200)]) for _ in range(8)]
    [x.start() for x in ths]
    [x.join() for x in ths]
    lines = (tmp_path / "t.jsonl").read_text().splitlines()
    assert t.stats["fallbacks"] == 1600 and len(lines) == 1600
    assert sorted(json.loads(l)["seq"] for l in lines) == list(range(1, 1601))      # 序号不重不漏，每行是完整的 JSON


# ---------------------------------------------------------------- 整局：串行与并行结果一样

class SlowMock(MockBackend):
    name = "slowmock"                                           # 不是 mock：引擎才会开并行

    def complete(self, req):
        time.sleep(0.12)
        return super().complete(req)


class SlowDecider(ScriptedDecider):
    def decide(self, tag, state, questions):
        time.sleep(0.12)
        return super().decide(tag, state, questions)


SAY = "我昨天去过茶室，还看了那只茶盏，我觉得这事一定另有隐情，我们该把它交给奉行所处置。"


def play(tmp_path, parallel):
    tmp_path.mkdir(parents=True, exist_ok=True)
    s = settings(tmp_path)
    s.harness.parallel_judges = parallel
    save = tmp_path / "data/saves/p.json"
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps({"world": {"act_id": "act2", "phase_idx": 1, "location": "kamisato_estate",
                                          "clues": ["c_case", "c_shops"]}, "relationships": {"ayaka": {"affinity": 70, "trust": 50}}}))
    d = SlowDecider({"turn": {"threat": (False, 0.9), "specific": (1, 1.0), "abrupt": (False, 0.9)},
                     "proposal": {"proposal": ("none", 0.9)}})
    out = []
    eng = GameEngine(s, SlowMock(seed=0), ScriptedPlayer([{"kind": "talk", "npc": "ayaka"}, (SAY, "自由发言"), "q"]),
                     out=out.append, save_name="p", decider=d)
    t0 = time.monotonic()
    eng.run()
    dt = time.monotonic() - t0
    ev = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    return eng, "\n".join(out), ev, dt


def shape(ev):
    """去掉时间、序号、耗时这些每次都不同的东西后，每种事件的内容。"""
    drop = {"seq", "ts", "run", "span", "latency_s", "latency", "ms", "usage", "cost", "elapsed_s"}

    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items() if k not in drop}
        return [clean(v) for v in x] if isinstance(x, list) else x
    return sorted(json.dumps(clean(e), ensure_ascii=False, sort_keys=True, default=str)
                  for e in ev if e.get("event") not in ("node", "game_start", "state_init", "session_end", "game_end"))


def test_parallel_judgments_change_nothing_but_the_wait(tmp_path):
    a, text_a, ev_a, t_serial = play(tmp_path / "serial", False)
    b, text_b, ev_b, t_par = play(tmp_path / "par", True)
    assert b.god.to_dict() == a.god.to_dict()                                       # 世界状态一样
    assert text_b == text_a                                                         # 玩家看到的一样
    assert shape(ev_b) == shape(ev_a)                                               # 发出的每一次请求、做出的每一个判定都一样
    assert any(e.get("span", {}).get("prefetch") for e in ev_b)                     # 并行确实发生了
    assert not any(e.get("span", {}).get("prefetch") for e in ev_a)
    assert t_par < t_serial                                                         # 而且更快


def test_mock_and_replay_backends_stay_serial(tmp_path):
    s = settings(tmp_path)
    s.harness.parallel_judges = True
    eng = GameEngine(s, MockBackend(seed=0), ScriptedPlayer([{"kind": "talk", "npc": "ayaka"}, ("你好，今天天气不错呀。", "自由发言"), "q"]),
                     out=lambda x: None, save_name="p")
    eng.run()
    ev = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    assert ev and not any(e.get("span", {}).get("prefetch") for e in ev)
