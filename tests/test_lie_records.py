"""写进记忆的“谎话”也要过“不该知道的词”检查：模型写的“真相”里带着本人不可能知道的东西，就是编的。"""
import json
from types import SimpleNamespace

from core.player_agent import ScriptedPlayer
from engine.game_engine import GameEngine
from llm.mock_backend import MockBackend
from tests.test_physical import settings


def ayaka(tmp_path):
    f = tmp_path / "data/saves/c.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"world": {"act_id": "act2", "phase_idx": 1, "location": "kamisato_estate", "clues": ["c_case", "c_shops"]}}))
    eng = GameEngine(settings(tmp_path), MockBackend(seed=0), ScriptedPlayer(["q"]), out=lambda s: None, save_name="c")
    return eng, eng.npcs["ayaka"]


def test_a_lie_whose_truth_mentions_what_she_cannot_know_is_dropped(tmp_path):
    eng, a = ayaka(tmp_path)
    said = "茶盏怎么会是我藏的？"
    reply = SimpleNamespace(speech=said, lies=[{"said": said, "truth": "茶盏是狐狸为修补旧裂纹带走的，现在在神龛上", "lie_id": "improvised"}])
    out = a.record_lies(reply, eng.god, now=3)
    assert not out["kept"] and "不该知道" in out["dropped"][0]["reason"]
    assert not eng.god.testimonies and not [m for m in a.memory.items if m.kind == "lie"]


def test_an_ordinary_lie_is_still_recorded(tmp_path):
    eng, a = ayaka(tmp_path)
    said = "兄长大人知道的，是他让我请您帮忙"
    reply = SimpleNamespace(speech=said, lies=[{"said": said, "truth": "兄长并不知道，你不想让他再为这件事分心", "lie_id": "ayaka_brother"}])
    out = a.record_lies(reply, eng.god, now=3)
    assert out["kept"] and eng.god.testimonies and [m for m in a.memory.items if m.kind == "lie"]


def test_improvised_world_facts_may_not_describe_places(tmp_path):
    """真实 trace：绫华随口一句“花见坂的清晨最热闹”，被存成了全世界共享的设定，可作者写的花见坂清晨是“街上还不算拥挤”。"""
    eng, a = ayaka(tmp_path)
    reply = SimpleNamespace(speech="花见坂的清晨最热闹……不过，还得劳烦您多留意周围。", action="",
                            new_facts=[{"fact": "花见坂的清晨最热闹", "about": "world"}])
    out = a.remember_improvisations(reply, now=3, provenance={})
    assert not out["kept"] and "地点" in out["dropped"][0]["reason"]
    mine = SimpleNamespace(speech="小时候我在花见坂的团子摊前走丢过，是托马把我找回来的。", action="",
                           new_facts=[{"fact": "小时候在花见坂的团子摊前走丢过，被托马找了回来", "about": "self"}])
    assert a.remember_improvisations(mine, now=4, provenance={})["kept"]               # 关于她自己的往事不受影响
    other = SimpleNamespace(speech="托马会弹三味线，只是弹得不太好。", action="",
                            new_facts=[{"fact": "托马会弹三味线，但弹得不太好", "about": "world"}])
    assert a.remember_improvisations(other, now=5, provenance={})["kept"]               # 不涉及地点的照常
