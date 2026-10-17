"""上下文：按需翻看的文件系统（权限即视角）、不对劲的话会被注意到、选项不串视角。"""
import json

from core.context import TurnContext
from core.context_fs import ContextFS
from core.player_agent import ScriptedPlayer
from core.player_option_generator import PlayerOptionGenerator
from engine.game_engine import GameEngine
from llm.decider import ScriptedDecider
from llm.mock_backend import MockBackend
from models.knowledge import LoreBase
from tests.helpers import ROOT, WORLD
from tests.test_living_world import living
from tests.test_physical import PERSONAS, settings


def fs(state, npc="ayaka"):
    g, phys, ex, _ = living(state)
    return g, ContextFS(npc, g, WORLD, PERSONAS, LoreBase.load(ROOT / "content/lore"), lambda k: ["上次聊了茶泡饭"])


def test_files_are_scoped_to_what_the_npc_knows():
    g, f = fs({"act_id": "act2", "clues": ["c_case", "c_messenger", "c_seal_rubbing"]})
    assert "/地点/璃月" in f.ls("/地点") and "七八天" in f.read("/地点/璃月")
    assert "走路约" in f.read("/地点/离岛")
    g.known["ayaka"] = ["c_case"]
    case = f.read("/案子/我知道的")
    assert "失窃的委托" in case and "碎瓷片" not in case            # 玩家手上有，但没给她看过
    assert "托马" in f.read("/人物/托马") and "茶泡饭" in f.read("/我/记忆")


def test_ledger_of_what_the_npc_already_said_is_a_file():
    g, f = fs({"act_id": "act2", "clues": ["c_case"], "achieved": ["yohei_shelf"]}, npc="yohei")
    sid = next(s.id for s in g.quest.signals if s.npc == "yohei" and "爪印" in s.desc)
    g.achieved.append(sid)
    f.canon = lambda: ["我进货的纸来自美浓"]
    text = f.read("/我/说过的事")
    assert "泥爪印" in text and "美浓" in text                      # 说出口的事实 + 随口编的设定都在
    assert "/我/说过的事" in f.ls("/我")
    other = ContextFS("ayaka", g, WORLD, PERSONAS, f.lore, lambda k: [])
    assert "泥爪印" not in other.read("/我/说过的事")                # 每个人只有自己说过的


def test_echo_of_own_recent_line_is_detected():
    from core.npc_agent import NPCAgent
    ctx = type("C", (), {"history": [{"speaker": "npc", "text": "这位客官，买东西请看货架，不买的话……恕不奉陪。"},
                                     {"speaker": "player", "text": "是狸猫吗"}]})()
    agent = NPCAgent.__new__(NPCAgent)
    assert agent._echoes(ctx, "这位客官，买东西请看货架，不买的话恕不奉陪。")
    assert not agent._echoes(ctx, "狸猫的爪尖不外露，这串印子分趾，是狗爪。")


def npc_prompts(eng):
    ev = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    return [e["messages"][-1]["content"] for e in ev if e["event"] == "llm" and e["task"] == "npc_step"]


def run(tmp_path, script, decider=None):
    save = tmp_path / "data/saves/c.json"
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps({"world": {"act_id": "act2", "phase_idx": 1, "clues": ["c_case", "c_shops"]}}))
    eng = GameEngine(settings(tmp_path), MockBackend(seed=0), ScriptedPlayer(script), out=lambda s: None,
                     save_name="c", decider=decider)
    eng.run()
    return eng


def test_npc_judges_fit_before_speaking(tmp_path):
    # 接不接得上由 NPC 模型自己判（fit 放在回复的第一个字段，先判后说），不再用代码匹配地名
    from llm.schemas import npc_reply_schema, validate_npc_reply
    assert list(npc_reply_schema([])["properties"])[0] == "fit"
    assert validate_npc_reply({"speech": "嗯？", "fit": "毫不相干"}, [])["fit"] == "毫不相干"
    assert validate_npc_reply({"speech": "嗯。"}, [])["fit"] == "接得上"
    eng = run(tmp_path, [{"kind": "talk", "npc": "ayaka"}, ("好的，那我先去璃月了", "自由发言"), "q"])
    ev = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    system = next(e["messages"][0]["content"] for e in ev if e["event"] == "llm" and e["task"] == "npc_step")
    assert "毫不相干" in system and '"fit"' in system and "read" in system


def test_abrupt_topic_jump_judged_by_jev(tmp_path):
    d = ScriptedDecider({"turn": {"abrupt": (True, 0.9), "threat": (False, 0.9), "specific": (0, 1.0)}})
    eng = run(tmp_path, [{"kind": "talk", "npc": "ayaka"}, ("对了，我其实很会做烤鱼。", "自由发言"), "q"], d)
    assert "接不上" in npc_prompts(eng)[-1]
    state = next(c[1] for c in reversed(d.calls) if c[0] == "turn")
    assert state["listener_just_said"] and state["situation"]            # Jev 看得到上一句和眼下的处境


def test_options_never_parrot_the_npc():
    class Stub:
        def generate(self, ctx, npc, n):
            return [{"text": "璃月……托马常提起那里的海与商船。祝您一路顺风。", "intent": "闲聊", "tone": "温和"},
                    {"text": "等等，我说错了，我是想去离岛码头。", "intent": "闲聊", "tone": "直率"}]
    ctx = TurnContext(player_name="旅行者", scene={}, world={}, relationship={}, player_view={"known_text": ""},
                      history=[{"speaker": "npc", "text": "璃月……托马常提起那里的海与商船。祝您一路顺风，愿那里的风景能为您带来新的灵感。",
                                "reply": {"action": ""}}])
    opts, _ = PlayerOptionGenerator(Stub()).generate(ctx, "绫华", [])
    assert [o.text for o in opts] == ["等等，我说错了，我是想去离岛码头。"]


def test_footprint_separates_hearsay_from_experience():
    g, phys, ex, _ = living({"act_id": "act2", "clues": ["c_case", "c_shops"], "location": "kamisato_estate"})
    fp = ex.footprint()
    assert "去过：神里屋敷" in fp and "木漏茶室" in fp.split("还没去过：")[1].split("\n")[0]
    ex.move("hanamizaka")
    assert "花见坂" in ex.footprint().split("\n")[0]


def test_options_claiming_unbacked_visits_are_dropped(tmp_path):
    class Stub:
        def generate(self, ctx, npc, n):
            assert "还没去过" in ctx.player_view["footprint"]               # 选项 agent 看得到足迹
            return [{"text": "刚去了木漏茶室，听说那晚连油豆腐和茶筅都丢了。", "intent": "打听情报", "tone": "直率"},
                    {"text": "听说木漏茶室也丢了东西？", "intent": "打听情报", "tone": "好奇"}]
    d = ScriptedDecider({"unbacked": {"o0": (True, 0.9), "o1": (False, 0.9)}})
    save = tmp_path / "data/saves/c.json"
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps({"world": {"act_id": "act2", "phase_idx": 1, "clues": ["c_case", "c_shops"]}}))
    seen = []

    class Spy(ScriptedPlayer):
        def choose(self, opts, c):
            if c is not None:
                seen.append([o.text for o in opts if o.kind == "say"])
            return super().choose(opts, c)

    eng = GameEngine(settings(tmp_path), MockBackend(seed=0), Spy([{"kind": "talk", "npc": "ayaka"}, "q"]),
                     out=lambda s: None, save_name="c", decider=d)
    eng.option_gen.adapter = Stub()
    eng.run()
    assert seen[0] == ["听说木漏茶室也丢了东西？"]
    assert "木漏茶室" in d.calls[-1][1]["speaker_footprint"]


def test_elapsed_since_is_computed_not_guessed():
    g, _ = fs({"act_id": "act1"})
    now = g.minute
    assert "不到五分钟" in g.elapsed_since(now - 2)
    assert "不是昨天" in g.elapsed_since(now - 30) and "30 分钟" in g.elapsed_since(now - 30)
    assert g.elapsed_since(now - 1440) == "昨天" and g.elapsed_since(now - 2 * 1440) == "2 天前"
