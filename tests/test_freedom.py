"""放宽判定（来自真实 trace 的场景）：打字说出线索等于出示、NPC 说出口就算、私人时光不花时间。"""
import json

from config.settings import load_settings
from core.player_agent import ScriptedPlayer
from engine.game_engine import GameEngine
from llm.mock_backend import MockBackend
from tests.helpers import WORLD, make_god


def engine(tmp_path, script, state):
    s = load_settings(env_file=tmp_path / "none.env")
    s.game.data_dir, s.game.seed, s.harness.game_token_budget = tmp_path / "data", 0, 0
    save = tmp_path / "data/saves/f.json"
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps({"world": state}))
    eng = GameEngine(s, MockBackend(seed=0), ScriptedPlayer(script), out=lambda x: None, save_name="f")
    eng.run()
    return eng


def test_saying_who_sent_you_counts_as_showing_the_commission(tmp_path):
    # trace：玩家打字「是神里小姐托我来的」，与平软化了，却还要再点一次「出示线索」
    eng = engine(tmp_path, [{"kind": "talk", "npc": "yohei"}, ("是神里小姐托我来的，想看看丢东西的货架。", "自由发言"), "q"],
                 {"day": 1, "phase_idx": 1, "location": "shimura_shop", "act_id": "act2", "clues": ["c_case", "c_shops"]})
    assert "c_letter" in eng.god.clues and "c_case" in eng.god.known["yohei"]


def test_only_clues_the_player_has_can_be_told():
    g = make_god({"act_id": "act2", "clues": ["c_case"]})
    g.begin_conversation("soji")
    assert g.mentioned_clues("soji", "那晚小吉抱着布包来找过你吧？") == []          # 还没拿到「深夜的布包」
    g.clues.append("c_messenger")
    assert g.mentioned_clues("soji", "那晚小吉抱着布包来找过你吧？") == ["c_messenger"]


def test_npc_saying_it_counts_even_without_the_button():
    # trace：阿缘自己说出「小吉抱着布包往离岛去了」，却没给线索
    g = make_god({"act_id": "act2", "clues": ["c_case", "c_shops"]})
    g.begin_conversation("aen")
    heard = g.spoken_signals("那晚夜深了，我看见印坊的小吉抱着个布包，慌慌张张往离岛去了。", "（拿出线索「失窃的委托」给对方看）")
    assert heard == ["rv_aen_messenger"]
    assert g.apply_signals(heard, heard=True)[0]["clues"] == ["深夜的布包"]


def test_npc_leak_while_locked_still_counts():
    # trace：给托马看了「货架上的短毛」（不是他要的线索），他还是说出了小吉那天在茶室里等过
    g = make_god({"act_id": "act2", "clues": ["c_case", "c_shops", "c_letter"]})
    g.begin_conversation("thoma")
    g.mark_known("thoma", ["c_letter"])
    assert "rv_thoma_workshop" not in g.available_ids(check_turns=False)
    speech = "短毛？说起来……那天傍晚印坊的小吉来送请柬样张，我让他在茶室里等了一会儿。不会跟他有关吧？"
    assert g.apply_signals(g.spoken_signals(speech, "志村屋丢纸的货架上留着几根棕色的短毛"), heard=True)
    assert "c_workshop" in g.clues


def test_echoing_the_player_is_not_a_confession():
    g = make_god({"act_id": "act2", "clues": ["c_case", "c_messenger"], "location": "ritou", "phase_idx": 1})
    g.begin_conversation("soji")
    assert g.spoken_signals("我、我不知道什么金缮，也没补过什么！", "是你带小吉去找金缮匠补茶碗的吧？") == []
    assert g.spoken_signals("……他抱着一只缺了口的碗，我带他去找了船上的金缮匠。", "那晚小吉找你做什么？") == ["rv_soji_confess"]


def test_private_time_does_not_spend_time(tmp_path):
    state = {"day": 1, "phase_idx": 3, "location": "kamisato_estate", "act_id": "act2", "clues": ["c_case", "c_shops"]}
    save = tmp_path / "data/saves/f.json"
    save.parent.mkdir(parents=True)
    save.write_text(json.dumps({"world": state, "relationships": {"ayaka": {"affinity": 45, "trust": 30}}}))
    s = load_settings(env_file=tmp_path / "none.env")
    s.game.data_dir, s.game.seed, s.harness.game_token_budget = tmp_path / "data", 0, 0
    lines = [{"kind": "event", "event": "ev_sword"}] + [("月色真好。", "闲聊")] * (WORLD.turns_per_phase + 2) + ["q"]
    eng = GameEngine(s, MockBackend(seed=0), ScriptedPlayer(lines), out=lambda x: None, save_name="f")
    eng.run()
    assert eng.god.time_label() == "第1天·夜晚"


def test_deadline_warning_in_next_steps():
    from core.explorer import Explorer
    from models.personality import Personality
    from models.relationship import Relationship
    from tests.helpers import ROOT
    personas = Personality.load_all(ROOT / "content/npcs")
    g = make_god({"act_id": "act2", "day": 3, "phase_idx": 1, "clues": ["c_case", "c_shops"]})
    ex = Explorer(g, WORLD, personas, lambda n: Relationship(30, 20), out=lambda s: None)
    assert ex.leads()[0].startswith("⏳ 离雪见茶会只剩 2 个时段")
