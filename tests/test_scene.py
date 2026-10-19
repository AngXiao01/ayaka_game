"""场景 agent：自由描述要查看的东西；对得上物件就走原规则，对不上就即兴描写但不能给线索、不能碰案情。"""
import json

from config.settings import load_settings
from core.explorer import Explorer
from core.player_agent import ScriptedPlayer
from core.scene import NOTHING, SceneAgent
from engine.game_engine import GameEngine
from llm.mock_backend import MockBackend
from models.memory import MemoryStore
from models.personality import Personality
from models.relationship import Relationship
from tests.helpers import QUEST, ROOT, WORLD, make_god

PERSONAS = Personality.load_all(ROOT / "tests/fixtures/yuebai/npcs")


class FakeAdapter:
    def __init__(self, out):
        self.out = out
        self.calls = []

    def describe(self, text, scene, objects, known, guard_terms):
        self.calls.append({"text": text, "known": known})
        return self.out


def make_scene(tmp_path, adapter=None, mode="llm", state=None):
    g = make_god(state)
    rels = {nid: Relationship.from_dict(p.initial) for nid, p in PERSONAS.items()}
    ex = Explorer(g, WORLD, PERSONAS, lambda nid: rels[nid], out=lambda s: None)
    canon = MemoryStore(tmp_path / "canon.jsonl")
    return SceneAgent(g, WORLD, ex, canon, QUEST.canon_guard_terms, adapter, mode=mode), g, canon


def test_matching_object_goes_through_normal_rules(tmp_path):
    sc, g, _ = make_scene(tmp_path, FakeAdapter({"object": "o_shrine", "description": "", "new_fact": ""}),
                          state={"location": "chinju_forest"})
    desc, gained = sc.look("看看稻荷神龛")
    assert gained == [] and "o_shrine" in g.examined          # 没有前置线索，照规则不给
    g.clues.append("c_round_prints")
    g.misses["o_shrine"] = 2                                  # 判定必成，这里只测规则
    desc, gained = sc.look("再看看神龛后面")
    assert gained == ["c_den"]


def test_improvised_detail_is_remembered_and_fed_back(tmp_path):
    ad = FakeAdapter({"object": "none", "description": "石灯笼的底座上长着一圈青苔，像是很久没人打理。",
                      "new_fact": "石灯笼底座长着一圈青苔"})
    sc, g, canon = make_scene(tmp_path, ad)
    desc, gained = sc.look("看看石灯笼")
    assert "青苔" in desc and gained == [] and not g.clues
    assert any("青苔" in m.text and m.extra["location"] == "kamisato_estate" for m in canon.items)
    sc.look("石灯笼旁边有什么")
    assert any("青苔" in k for k in ad.calls[-1]["known"])          # 下一次描写会带上已确定的细节


def test_case_terms_and_ungrounded_facts_are_dropped(tmp_path):
    sc, g, canon = make_scene(tmp_path, FakeAdapter({"object": "none", "description": "墙角落着几根白色的狐狸毛，细细软软的。",
                                                      "new_fact": "墙角有几根狐狸毛"}))
    desc, _ = sc.look("看看墙角")
    assert desc == NOTHING and not canon.items
    sc2, _, canon2 = make_scene(tmp_path / "b", FakeAdapter({"object": "none", "description": "廊下很安静。",
                                                              "new_fact": "托马昨晚在廊下弹三味线"}))
    sc2.look("看看廊下")
    assert not canon2.items


def test_rules_and_off_modes(tmp_path):
    sc, g, _ = make_scene(tmp_path, mode="rules")
    assert sc.look("托马送来的邀请函")[0] == WORLD.examine("kamisato_estate", "o_invitation", [])[0]
    assert sc.look("看看天上的云")[0] == NOTHING
    off, _, _ = make_scene(tmp_path / "c", mode="off")
    assert "菜单" in off.look("看看天上的云")[0]


def test_free_look_from_the_map_costs_no_time(tmp_path):
    s = load_settings(env_file=tmp_path / "none.env")
    s.game.data_dir = tmp_path / "data"
    s.harness.game_token_budget = 0
    eng = GameEngine(s, MockBackend(seed=0), ScriptedPlayer([("仔细看看院子里的樱树", "查看"), "q"]), out=lambda x: None)
    t0 = eng.god.time_label()
    eng.run()
    ev = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    look = next(e for e in ev if e["event"] == "scene")
    assert look["object"] == "none" and look["description"] and eng.god.time_label() == t0
    assert any(e["event"] == "llm" and e["task"] == "scene" for e in ev)
