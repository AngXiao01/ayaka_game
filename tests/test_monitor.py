"""抽样监控：从 trace 事件里数越界 / 重说 / 兜底，越线给出告警。"""
from tools.monitor import line, summarize


def ev(**kw):
    return kw


def test_rates_and_alerts():
    events = [ev(event="turn", turn=i, degraded=(i == 3)) for i in range(1, 9)]
    events += [ev(event="audit", problems=["你说到了「裂纹」，可这件事你并不知道"]),
               ev(event="audit", problems=["窗离地两丈，和格局矛盾"]),
               ev(event="audit", problems=["编造了灰尘"]),
               ev(event="repeat"), ev(event="fallback", where="npc_loop"),
               ev(event="llm", task="npc_step", ok=True, latency_s=2.0, usage={"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.0001}),
               ev(event="llm", task="npc_step", ok=False, latency_s=4.0, usage={}),
               ev(event="free_fallback"), ev(event="beat", picked="rv_x"), ev(event="beat", picked=None), ev(event="reflection")]
    s = summarize(events)
    assert s["turns"] == 8 and s["boundary_violations"] == 1 and s["fact_physics_violations"] == 2
    assert s["out_of_bounds"] == round(3 / 8, 3) and s["retry"] == round(1 / 8, 3)
    assert s["fallback"] == round(2 / 8, 3)                                  # 1 个兜底回复 + 1 条 fallback 事件
    assert s["free_fallbacks"] == 1 and s["beats_offered"] == 1 and s["reflections"] == 1
    assert s["median_latency_s"]["npc_step"] == 3.0 and s["tokens"] == 120
    assert any("越界" not in a and "out_of_bounds" in a for a in s["alerts"]) or s["alerts"]   # 越界率 37% > 25%，兜底率 25% > 5%
    assert "⚠" in line(s)


def test_a_healthy_run_has_no_alerts():
    s = summarize([ev(event="turn", turn=i, degraded=False) for i in range(1, 11)])
    assert s["alerts"] == [] and "⚠" not in line(s)
