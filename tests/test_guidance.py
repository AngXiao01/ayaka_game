"""新手引导：关键问题按钮、动作直达的进度、「下一步」、背景速览与名词速查。"""
import json

from config.settings import load_settings
from core.explorer import Explorer
from core.player_agent import ScriptedPlayer
from engine.game_engine import GameEngine
from llm.mock_backend import MockBackend
from models.personality import Personality
from models.relationship import Relationship
from tests.helpers import ROOT, WORLD, make_god

PERSONAS = Personality.load_all(ROOT / "tests/fixtures/yuebai/npcs")
TALK = lambda npc: {"kind": "talk", "npc": npc}   # noqa: E731


class StrictMock(MockBackend):
    """审核员一律驳回：用来证明动作直达的进度不依赖模型上报和审核。"""

    def _review(self, meta):
        return {"verdicts": [{"id": s, "accept": False, "evidence": "", "reason": "不认"}
                             for s in meta.get("signals", [])]}


def settings(tmp_path):
    s = load_settings(env_file=tmp_path / "none.env")
    s.game.data_dir = tmp_path / "data"
    s.game.seed = 0
    s.harness.game_token_budget = 0
    return s


def engine(tmp_path, script, state=None, backend=None, name="g"):
    if state:
        save = tmp_path / f"data/saves/{name}.json"
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(json.dumps({"world": state}))
    out = []
    eng = GameEngine(settings(tmp_path), backend or StrictMock(seed=0), ScriptedPlayer(script), out=out.append,
                     save_name=name)
    eng.run()
    return eng, out


def events(eng):
    return [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]


def test_ask_buttons_progress_act1_without_reviewer(tmp_path):
    eng, out = engine(tmp_path, [TALK("ayaka"), {"kind": "ask", "signal": "asked_purpose"},
                                 {"kind": "ask", "signal": "accepted_commission"}, "q"])
    g = eng.god
    assert g.act_id == "act2" and {"c_case", "c_shops"} <= set(g.clues)
    assert eng.tracer.stats["signals_by_action"] == 2 and eng.tracer.stats["signals_accepted"] == 0
    npc_reqs = [e for e in events(eng) if e["event"] == "llm" and e["task"] == "npc_step"]
    assert any("这一轮你要说出口的事" in e["messages"][-1]["content"] and "悄悄查清" in e["messages"][-1]["content"]
               for e in npc_reqs)
    # 先显示对方说的话，再显示获得的线索
    accept = next(i for i, line in enumerate(out) if "这件事就交给我吧" in line)
    clue = next(i for i, line in enumerate(out) if "获得线索「失窃的委托」" in line)
    assert any(line.startswith("\n神里绫华") for line in out[accept:clue])


def test_accept_button_only_after_asking_purpose(tmp_path):
    g = make_god()
    g.begin_conversation("ayaka")
    assert [s.id for s in g.askable()] == ["asked_purpose"]       # 还没问来意，不能先接委托
    g.apply_signals(["asked_purpose"], forced=True)
    assert [s.id for s in g.askable()] == ["accepted_commission"]  # 不用等聊满两轮


def test_showing_evidence_forces_confession_and_no_lie(tmp_path):
    state = {"day": 1, "phase_idx": 1, "location": "ritou", "act_id": "act2",
             "clues": ["c_case", "c_shops", "c_bundle"]}
    eng, _ = engine(tmp_path, [TALK("soji"), {"kind": "show_menu"}, {"kind": "clue", "clue": "c_bundle"}, "q"],
                    state)
    assert "c_foxfire" in eng.god.clues
    assert not [t for t in eng.god.testimonies if t["kind"] == "lie"]
    turn = next(e for e in events(eng) if e["event"] == "turn" and e["player"] and e["player"]["kind"] == "clue")
    assert turn["new_signals"] == ["rv_soji_fire"]


def test_yohei_shows_letter_once_he_sees_the_commission(tmp_path):
    state = {"day": 1, "phase_idx": 1, "location": "shimura_shop", "act_id": "act2", "clues": ["c_case", "c_shops"]}
    eng, _ = engine(tmp_path, [TALK("yohei"), {"kind": "show_menu"}, {"kind": "clue", "clue": "c_case"}, "q"], state)
    assert "c_tanuki_hair" in eng.god.clues


def test_propose_decides_without_reviewer(tmp_path):
    state = {"day": 2, "phase_idx": 0, "location": "kamisato_estate", "act_id": "act3",
             "clues": ["c_case", "c_shops", "c_bundle", "c_tanuki_hair", "c_foxfire", "c_double"],
             "achieved": ["asked_purpose", "accepted_commission", "reported_findings"]}
    eng, out = engine(tmp_path, [TALK("ayaka"), {"kind": "propose", "flag": "proposed_mercy"}], state)
    assert eng.god.resolved and eng.god.ending == "mercy"
    assert any("温情的裁决" in line for line in out)


def explorer(g, aff=30):
    rels = {n: Relationship(aff, 20) for n in PERSONAS}
    return Explorer(g, WORLD, PERSONAS, lambda n: rels[n], out=lambda s: None)


def test_leads_follow_the_investigation():
    g = make_god()
    ex = explorer(g)
    assert ex.leads() == ["和绫华交谈，问问她特意请你来是为了什么事（就在这里）"]
    g.act_id, g.clues, g.achieved = "act2", ["c_case", "c_shops"], ["asked_purpose", "accepted_commission"]
    leads = ex.leads()
    assert any("阿缘" in x and "木漏茶室·清晨、午后、黄昏" in x for x in leads)
    assert any("与平" in x and "出示「失窃的委托」" in x for x in leads)
    assert ex.checklist_line().count("✘") == 3
    g.clues += ["c_bundle", "c_tanuki_hair"]
    leads = ex.leads()
    assert any("离岛码头问问宗次" in x and "出示「深夜的布包」" in x and "离岛·午后、黄昏、夜晚" in x for x in leads)
    assert ex.checklist_line().count("✔") == 2
    g.clues.append("c_bowl_on_altar")
    leads = ex.leads()
    assert any("向绫华汇报" in x for x in leads)


def test_event_leads_hint_at_closeness_without_showing_numbers():
    g = make_god({"act_id": "act2"})
    assert any("关系更近时也许会有意外的相遇" in x for x in explorer(g, 30).leads())
    assert not any("好感" in x for x in explorer(g, 30).leads())
    assert any("夜晚去神里屋敷走走" in x for x in explorer(g, 41).leads())


def test_clues_that_point_at_the_apprentice_unlock_workshop():
    assert WORLD.unlocked("print_workshop", ["c_gate_visitor"])    # 托马说夜里折回来的是「小吉」，能找到印坊去
    assert WORLD.unlocked("print_workshop", ["c_bundle"])          # 阿缘看见的是「印坊的学徒小吉」，也能找过去
    assert not WORLD.unlocked("print_workshop", ["c_tanuki_hair", "c_round_prints", "c_foxfire"])   # 狸猫那条线、海边的火都指不到印坊
    assert not WORLD.unlocked("print_workshop", [])


def test_primer_and_glossary(tmp_path):
    eng, out = engine(tmp_path, [{"kind": "view", "view": "glossary"}, {"kind": "view", "view": "leads"}, "q"])
    text = "\n".join(out)
    assert "给第一次来稻妻的你" in text and "怎么玩" in text
    assert "· 社奉行：" in text and "【下一步】" in text
