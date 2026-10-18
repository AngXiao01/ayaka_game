"""锁着的事，NPC 条件没满足就说出口，等于绕过了门槛（游戏规则是“NPC 说出口就算数”）——生成时必须挡住。"""
import json

from core.guardrails import locked_reveal
from core.player_agent import ScriptedPlayer
from engine.game_engine import GameEngine
from llm.mock_backend import MockBackend
from tests.test_physical import settings

EV = [{"id": "rv_soji_fire", "groups": [["青白", "灯火", "火"], ["人影", "蹲", "拾"]], "why": "对方还没让你知道不是你一个人看见"}]


def test_a_line_that_completes_the_evidence_is_stopped_with_the_reason():
    leak = "等等！……真看见了青白的火，还有个人影蹲在那儿。"
    out = locked_reveal(leak, "", EV, ["失窃那晚你看见什么了"])
    assert out and "不能说" in out[0] and "对方还没让你知道不是你一个人看见" in out[0]
    assert locked_reveal("……那晚海边有件事，我一直没敢说。", "", EV, []) == []               # 只漏口风、没凑齐证据：放行
    assert locked_reveal(leak, "", EV, ["你是不是看见了青白的火和人影蹲着"]) == []              # 是玩家先说的词，复述不算泄露
    assert locked_reveal(leak, "", [], []) == []


class Leaky(MockBackend):
    """每次说话都把秘密说出口。"""
    def _reply(self, meta):
        r = super()._reply(meta)
        r["speech"] = "那晚海滩上有一团青白的火，还有个人影蹲在那儿拾东西。"
        return r


def test_the_engine_sends_a_premature_reveal_back(tmp_path):
    f = tmp_path / "data/saves/c.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"world": {"act_id": "act2", "phase_idx": 1, "location": "ritou", "clues": ["c_case", "c_shops"]}}))
    eng = GameEngine(settings(tmp_path), Leaky(seed=0), ScriptedPlayer([{"kind": "talk", "npc": "soji"}, ("今天海边风大吗？", "闲聊"), {"kind": "leave"}, "q"]),
                     out=lambda s: None, save_name="c")
    eng.run()
    ev = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    blocked = [e for e in ev if e["event"] == "audit" and any("现在不能说" in p for p in e["problems"])]
    assert blocked and "青白" in blocked[0]["speech"]                                    # 第一版就被代码拦下，记了理由
    told = [m["content"] for e in ev if e["event"] == "llm" and e["task"] == "npc_step" for m in e["messages"] if "你上一版回答有问题" in m["content"]]
    assert told                                                                          # 重说的那一版看得到理由
