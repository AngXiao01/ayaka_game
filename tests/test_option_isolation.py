"""替玩家想台词的选项 agent 只能看玩家视角：不给 NPC 的秘密，剧透和出戏的选项由代码拦下。"""
import json

from config.settings import load_settings
from core.context import TurnContext
from core.explorer import Explorer
from core.player_agent import ScriptedPlayer
from core.player_option_generator import PlayerOptionGenerator
from engine.game_engine import GameEngine
from llm.mock_backend import MockBackend
from models.personality import Personality
from models.relationship import Relationship
from tests.helpers import ROOT, WORLD, make_god

PERSONAS = Personality.load_all(ROOT / "content/npcs")
SECRETS = ["小吉", "辰巳", "母亲病", "药钱", "印样"]


def explorer(g):
    rels = {n: Relationship(30, 20) for n in PERSONAS}
    return Explorer(g, WORLD, PERSONAS, lambda n: rels[n], out=lambda s: None)


def test_option_request_carries_no_npc_secrets(tmp_path):
    s = load_settings(env_file=tmp_path / "none.env")
    s.game.data_dir, s.game.seed, s.harness.game_token_budget = tmp_path / "data", 0, 0
    save = tmp_path / "data/saves/iso.json"
    save.parent.mkdir(parents=True)
    save.write_text(json.dumps({"world": {"day": 1, "phase_idx": 1, "location": "ritou", "act_id": "act2",
                                          "clues": ["c_case", "c_shops"]}}))
    eng = GameEngine(s, MockBackend(seed=0), ScriptedPlayer([{"kind": "talk", "npc": "soji"}, "q"]),
                     out=lambda x: None, save_name="iso")
    eng.run()
    reqs = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    opts = [e for e in reqs if e["event"] == "llm" and e["task"] == "player_options"]
    assert opts
    text = "".join(m["content"] for e in opts for m in e["messages"])
    assert not [t for t in SECRETS if t in text]                 # 宗次心里的「是小吉托我送的」没给出去
    npc = next(e for e in reqs if e["event"] == "llm" and e["task"] == "npc_step")
    assert "小吉" in npc["messages"][-1]["content"]               # NPC 自己当然知道


def test_player_view_grows_with_what_the_player_learns():
    g = make_god({"act_id": "act2", "clues": ["c_case", "c_shops"], "location": "ritou", "phase_idx": 1})
    g.begin_conversation("soji")
    pv = explorer(g).player_view("soji")
    assert "宗次" in pv["known_text"]                            # 人就站在面前
    assert "小吉" not in pv["known_text"] and "小吉" in pv["spoiler_terms"]
    assert pv["directions"] == []                                 # 手上没有能让他开口的东西
    g.clues.append("c_messenger")
    pv = explorer(g).player_view("soji")
    assert pv["directions"] == ["把「送信人」的事说给对方听，看对方什么反应"]
    g.clues.append("c_soji")
    assert "小吉" in explorer(g).player_view("soji")["known_text"]


class Stub:
    def __init__(self, options):
        self.options = options

    def generate(self, ctx, npc, n):
        return [{"text": t, "intent": "打听情报", "tone": "好奇"} for t in self.options]


def test_spoiler_and_ooc_options_are_dropped():
    ctx = TurnContext(player_name="旅行者", scene={}, world={}, relationship={},
                      history=[{"speaker": "npc", "text": "我、我就是个扛货的。", "reply": {"action": ""}}],
                      player_view={"known_text": "宗次 离岛的搬运工", "spoiler_terms": ["小吉", "宗次", "辰巳"]})
    gen = PlayerOptionGenerator(Stub(["是小吉托你送的信吧？", "这个剧情选项好怪", "宗次，你在码头干了多久？",
                                      "扛货累不累？要不要吃个饭团？"]))
    opts, _ = gen.generate(ctx, "宗次", [])
    assert [o.text for o in opts] == ["宗次，你在码头干了多久？", "扛货累不累？要不要吃个饭团？"]
    ctx.history.append({"speaker": "npc", "text": "是小吉……小吉托我送的。", "reply": {"action": ""}})
    opts, _ = gen.generate(ctx, "宗次", [])
    assert "是小吉托你送的信吧？" in [o.text for o in opts]       # 对方亲口说了，玩家就知道了


def test_npcs_only_know_the_commission_after_being_told():
    g = make_god({"act_id": "act2", "clues": ["c_case", "c_shops"]})
    g.begin_conversation("yohei")
    assert "暗中查访" not in g.world_state_summary(PERSONAS["yohei"], 20)["premise"]
    g.mark_known("yohei", ["c_case"])
    assert "暗中查访" in g.world_state_summary(PERSONAS["yohei"], 20)["premise"]
    assert "暗中查访" in g.world_state_summary(PERSONAS["ayaka"], 20)["premise"]
