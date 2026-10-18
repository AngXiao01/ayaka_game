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
    said = "茶碗怎么会是我藏的？"
    reply = SimpleNamespace(speech=said, lies=[{"said": said, "truth": "茶碗是狐狸为修补旧裂纹带走的，现在在神龛上", "lie_id": "improvised"}])
    out = a.record_lies(reply, eng.god, now=3)
    assert not out["kept"] and "不该知道" in out["dropped"][0]["reason"]
    assert not eng.god.testimonies and not [m for m in a.memory.items if m.kind == "lie"]


def test_an_ordinary_lie_is_still_recorded(tmp_path):
    eng, a = ayaka(tmp_path)
    said = "兄长大人知道的，是他让我请您帮忙"
    reply = SimpleNamespace(speech=said, lies=[{"said": said, "truth": "兄长并不知道，你不想让他再为这件事分心", "lie_id": "ayaka_brother"}])
    out = a.record_lies(reply, eng.god, now=3)
    assert out["kept"] and eng.god.testimonies and [m for m in a.memory.items if m.kind == "lie"]
