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
    state = d.calls[-1][1]
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
