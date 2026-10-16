"""进度审核：规则、独立审核、证据落地、依赖 DAG、fail-closed。"""
from core.reviewer import SignalReviewer
from llm.base import LLMError
from tests.helpers import make_god


class StubAdapter:
    def __init__(self, verdicts=None, fail=False):
        self.verdicts, self.fail, self.calls = verdicts or {}, fail, 0

    def review(self, npc, player, signals, ctx, pl, nl):
        self.calls += 1
        if self.fail:
            raise LLMError("down")
        return self.verdicts


def rv(adapter=None, mode="llm", policy="reject", turns=2):
    g = make_god()
    g.begin_conversation("ayaka")
    g.conv["turns"] = turns
    return g, SignalReviewer(g, adapter, "神里绫华", "旅行者", mode, policy)


def review(r, claimed, **kw):
    args = dict(opening=False, injection_flags=[], player_text="请问您找我来有什么事？我愿意帮忙。",
                npc_speech="其实是有一桩事想拜托您。", context_lines=[])
    args.update(kw)
    return r.review(claimed, **args)


def test_opening_and_injection_rejected_by_rules():
    ad = StubAdapter()
    _, r = rv(ad)
    assert review(r, ["asked_purpose"], opening=True).accepted == []
    assert review(r, ["asked_purpose"], injection_flags=["x"]).accepted == []
    assert review(r, ["rv_aen_messenger"]).rejected[0].layer == "rule"     # 别人的信号
    assert ad.calls == 0                       # 规则层拦下的不浪费审核调用


def test_min_turns_makes_signal_not_yet_pending():
    _, r = rv(mode="rules", turns=1)
    res = review(r, ["asked_purpose", "accepted_commission"])
    assert res.accepted == ["asked_purpose"]
    assert next(v for v in res.verdicts if v.id == "accepted_commission").layer == "rule"


def test_reviewer_accepts_with_grounded_evidence():
    _, r = rv(StubAdapter({"asked_purpose": {"accept": True, "evidence": "有什么事", "reason": "问了来意"}}))
    assert review(r, ["asked_purpose"]).accepted == ["asked_purpose"]


def test_hallucinated_evidence_rejected():
    _, r = rv(StubAdapter({"asked_purpose": {"accept": True, "evidence": "我愿意全力以赴", "reason": "x"}}))
    res = review(r, ["asked_purpose"])
    assert res.accepted == [] and res.rejected[0].layer == "grounding"


def test_reviewer_rejects():
    _, r = rv(StubAdapter({"asked_purpose": {"accept": False, "evidence": "", "reason": "没问"}}))
    assert review(r, ["asked_purpose"]).rejected[0].layer == "reviewer"


def test_dag_prereq_same_turn_and_missing():
    _, r = rv(mode="rules")
    assert review(r, ["accepted_commission", "asked_purpose"]).accepted == ["asked_purpose", "accepted_commission"]
    _, r = rv(mode="rules")
    res = review(r, ["accepted_commission"])
    assert res.accepted == [] and res.rejected[0].layer == "dag"


def test_fail_closed_and_fail_open():
    _, r = rv(StubAdapter(fail=True), policy="reject")
    assert review(r, ["asked_purpose"]).rejected[0].layer == "policy"
    _, r = rv(StubAdapter(fail=True), policy="accept")
    assert review(r, ["asked_purpose"]).accepted == ["asked_purpose"]


def test_degraded_skips_llm():
    ad = StubAdapter({"asked_purpose": {"accept": False, "evidence": "", "reason": ""}})
    _, r = rv(ad)
    assert review(r, ["asked_purpose"], degraded=True).accepted == ["asked_purpose"] and ad.calls == 0


def test_ellipsis_spliced_evidence_checked_segment_by_segment():
    from core.reviewer import grounded, _norm
    text = _norm("伪信事件：有人冒用社奉行印鉴。收信的商铺：花见坂的木漏茶室与志村屋都收到过伪信。")
    assert grounded("伪信事件：有人冒用社奉行印鉴……收信的商铺：花见坂的木漏茶室", text)
    assert not grounded("伪信事件：有人冒用社奉行印鉴……送信人是个戴斗笠的人", text)
    assert not grounded("", text)
