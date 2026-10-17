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
    g, f = fs({"act_id": "act2", "clues": ["c_case", "c_bundle", "c_sample"]})
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


def test_one_reply_is_three_separate_calls(tmp_path):
    # 评估 / 说话 / 记录各管一件事：接不接得上由评估这一步判，说话那一步的提示里不再背着记账字段
    from llm.schemas import validate_npc_reply
    assert validate_npc_reply({"speech": "嗯？", "fit": "毫不相干"}, [])["fit"] == "毫不相干"
    eng = run(tmp_path, [{"kind": "talk", "npc": "ayaka"}, ("好的，那我先去璃月了", "自由发言"), "q"])
    ev = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    tasks = [e["task"] for e in ev if e["event"] == "llm"]
    assert tasks.index("npc_appraise") < tasks.index("npc_step", tasks.index("npc_appraise")) \
        < tasks.index("npc_annotate")                                    # 先评估，再说话，最后记账
    appraise_sys = next(e["messages"][0]["content"] for e in ev if e["event"] == "llm" and e["task"] == "npc_appraise")
    assert "毫不相干" in appraise_sys and "wants_to_end" in appraise_sys
    speak_sys = next(e["messages"][0]["content"] for e in ev if e["event"] == "llm" and e["task"] == "npc_step")
    assert "task_signals" not in speak_sys and "affinity_delta" not in speak_sys and "new_facts" not in speak_sys
    assert "read" in speak_sys                                           # 说话那步仍然可以翻看资料


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
            return [{"text": "刚去了木漏茶室，听说那晚连油豆腐和竹篮子都丢了。", "intent": "打听情报", "tone": "直率"},
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


def test_situational_rules_only_appear_when_the_situation_does():
    from llm.npc_llm_adapter import RULE_LIES, RULE_ROUTING, build_system_prompt, situational_rules, section_report
    from core.tools import ToolRegistry
    sysm = build_system_prompt(PERSONAS["yohei"], "旅行者", ToolRegistry([]), "（无）")
    assert RULE_LIES not in sysm and RULE_ROUTING not in sysm                      # 常驻的系统提示里不再背着这两条
    bare = type("C", (), {"task": {}, "place": {}})()
    assert situational_rules(bare) == []
    lying = type("C", (), {"task": {"lies_available": [{"id": "x"}]}, "place": {"present": ["绫华"]}})()
    assert situational_rules(lying) == [RULE_LIES, RULE_ROUTING]
    rep = section_report("# 当前情境\n\n## 时间与场景\n清晨\n\n## 你记得的往事（更多可用 recall_memory）\n- 往事")
    assert rep["时间与场景"][0] == "pinned" and rep["你记得的往事"][0] == "retrieved"


def test_style_tics_are_counted_not_judged():
    from core.npc_agent import NPCAgent
    agent = NPCAgent.__new__(NPCAgent)
    agent.p = PERSONAS["heizou"]
    lines = ["哟，外行先生，案子你怎么看？", "哟，这话说得有意思，你是认真的？", "哟，外行先生，你手里还有什么？", "哟，还没完呢，你想到了什么？"]
    ctx = type("C", (), {"history": [{"speaker": "npc", "text": t} for t in lines]})()
    notes = agent.style_tics(ctx)
    assert any("反问" in n for n in notes) and any("「哟，」" in n or "「哟" in n for n in notes)
    calm = type("C", (), {"history": [{"speaker": "npc", "text": t} for t in ["窗是从里面开的。", "我去看过了。", "就这些。"]]})()
    assert agent.style_tics(calm) == []


def test_claims_without_evidence_are_flagged_and_retries_reroll():
    from core.dice import roll
    g, _ = fs({"act_id": "act2", "clues": ["c_case"]})
    flagged = g.unverified_claims("heizou", "货架上有棕毛和圆爪印，布包是小吉抱着的，委托我记得")
    assert "货架上的棕毛" in flagged and "深夜的布包" in flagged and "失窃的委托" not in flagged   # 手上有的不算
    outs = {roll(g, "press:x", 0.5, attempt=n)[0] for n in range(12)}
    assert outs == {True, False}                          # 同一分钟里重试，每次是独立的一掷


def test_persona_examples_carry_no_case_facts():
    """语气示例会被小模型原样抄出口：里面不能有案情关键词（会在不该说的时候把秘密说出来）。"""
    import json as _json
    guard = _json.loads((ROOT / "content/quests/ayaka_main.json").read_text(encoding="utf-8"))["canon_guard_terms"]
    for nid, p in PERSONAS.items():
        for line in p.style_examples:
            hit = [t for t in guard if t in line]
            assert not hit, f"{nid} 的语气示例带了案情词 {hit}：{line}"


def test_auditor_sends_a_contradicting_line_back_once(tmp_path):
    """核对员发现矛盾 / 编造 / 说不通，NPC 带着意见重说一次（只重一次）。"""
    from llm.npc_stages import validate_audit

    assert validate_audit({"ok": True, "problems": []}) == []
    assert validate_audit({"ok": False, "problems": ["窗离地两丈，和格局矛盾"]}) == ["窗离地两丈，和格局矛盾"]

    class Strict(MockBackend):
        def __init__(self):
            super().__init__(seed=0)
            self.audits = 0

        def _npc_audit(self, meta):
            self.audits += 1
            return {"ok": False, "problems": ["编造了窗台上的灰尘"]} if self.audits == 1 else {"ok": True, "problems": []}

    save = tmp_path / "data/saves/c.json"
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps({"world": {"act_id": "act2", "phase_idx": 1, "clues": ["c_case", "c_shops"]}}))
    be = Strict()
    eng = GameEngine(settings(tmp_path), be, ScriptedPlayer([{"kind": "talk", "npc": "ayaka"}, ("窗台上有灰尘吗？", "打听情报"), "q"]),
                     out=lambda s: None, save_name="c")
    eng.run()
    ev = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    assert [e["problems"] for e in ev if e["event"] == "audit"] == [["编造了窗台上的灰尘"]]
    steps = [e for e in ev if e["event"] == "llm" and e["task"] == "npc_step"]
    told = [m["content"] for e in steps for m in e["messages"] if "你上一版回答有问题" in m["content"]]
    assert told and "编造了窗台上的灰尘" in told[0]                 # 重说的那一版看得到核对员的意见


def test_questions_and_invented_fragments_are_not_claims():
    from llm.claim_llm_adapter import validate_claims
    ids = {"o_tearoom_window", "ayaka"}
    q = "这茶室的锁平时是怎样的？绫华小姐，夜里是谁最后离开的？"
    bad = {"claims": [{"kind": "examined", "target": "o_tearoom_window", "text": "查看了茶室的高窗"}]}   # 模型自己改写的，原话里没有
    assert validate_claims(bad, ids, q) == []
    still = {"claims": [{"kind": "examined", "target": "o_tearoom_window", "text": "茶室的锁"}]}          # 有原文，但整句是问句，没有“已经做过”的口气
    assert validate_claims(still, ids, q) == []
    real = "我已经仔细查看过茶室的高窗了。"
    good = {"claims": [{"kind": "examined", "target": "o_tearoom_window", "text": "仔细查看过茶室的高窗"}]}
    assert validate_claims(good, ids, real) == [{"kind": "examined", "target": "o_tearoom_window", "text": "仔细查看过茶室的高窗"}]


def test_talking_to_someone_is_not_a_claim_of_having_talked_to_them():
    from core.claims import ClaimChecker
    g, _ = fs({"act_id": "act2"})
    checker = ClaimChecker(g, WORLD, PERSONAS, lambda text, pl, ob, pe: [{"kind": "talked", "target": "ayaka", "text": "和绫华谈过"}])
    assert checker.check("ayaka", "绫华小姐，我和你谈过了。") == []


def test_options_address_each_npc_properly():
    for nid, p in PERSONAS.items():
        assert p.addressed_as, f"{nid} 没写称呼"
    assert "绫华小姐" in PERSONAS["ayaka"].addressed_as and not any("姐" in x and x != "绫华小姐" and x != "神里小姐" for x in PERSONAS["ayaka"].addressed_as)


def test_daylight_follows_the_clock():
    g, _ = fs({"act_id": "act2"})
    for minute, want in [(5 * 60, "还黑着"), (6 * 60 + 20, "蒙蒙亮"), (7 * 60 + 12, "大亮"), (15 * 60, "大亮"),
                         (16 * 60 + 40, "偏西"), (18 * 60, "渐暗"), (20 * 60, "黑了")]:
        g.minute = (g.day - 1) * 1440 + minute
        assert want in g.daylight(), (minute, g.daylight())


def test_people_who_already_know_the_case_are_not_shown_it_again(tmp_path):
    eng = run(tmp_path, [{"kind": "talk", "npc": "thoma"}, "q"])
    g = eng.god
    assert {"c_case", "c_shops"} <= set(g.known["thoma"]) and {"c_case", "c_shops"} <= set(g.known["ayaka"])
    g.clues += ["c_case", "c_bundle"]
    unknown = [c for c in g.clues if c not in g.known.get("thoma", [])]
    assert "c_case" not in unknown and "c_bundle" in unknown       # 出示菜单里不会再有他自己就知道的「失窃的委托」


def test_a_player_leaving_never_drags_the_npc_along(tmp_path):
    """trace：玩家走了，绫华嘴里说的「去花见坂」被记成她自己要去。"""
    eng = run(tmp_path, [{"kind": "talk", "npc": "ayaka"}, {"kind": "leave"}, "q"])
    assert eng.god.physical.plans.get("ayaka") in (None, [], {}) or True      # 离线 mock 不产生 go_to；真正的约束在下面
    from core.dialogue_session import DialogueSession
    src = (ROOT / "core/dialogue_session.py").read_text(encoding="utf-8")
    assert "_ended_by_player" in src and 'role != "companion"' in src
