"""“充耳不闻”的玩家：完全不看 NPC 说了什么，按固定脚本轮流说真话、谎话、半真半假、没法核实的话。

要验证的是：真假由代码对着真实足迹核对，不由 NPC 的心情、也不由说得有多顺口决定；
谎话必须被点名成“不是真的”并且传到 NPC 的提示词里，真话被确认，混着说的逐条分开判，不在场的人只能说“没法核实”，
并且谎话不会变成 NPC 记忆里的事实。
脚本不依赖 NPC 的任何输出（ScriptedPlayer 本来就不读回话），所以这条路径是确定的，离线、不花钱。
"""
import json

from core.player_agent import ScriptedPlayer
from engine.game_engine import GameEngine
from llm.mock_backend import MockBackend
from tests.test_physical import settings

TRUE_SAY = "我昨天仔细查看了庭院的白砂园。"                              # 真：存档里 o_garden 已经看过
LIE_SAY = "我早上查看了茶室的高窗，窗锁是被人撬过的。"                    # 假：没看过 o_tearoom_window
MIXED_SAY = "我查看了庭院的白砂园，也查看了茶室的高窗。"                   # 一半真一半假，要逐条判
UNKNOWN_SAY = "我还查看了街口的告示板。"                                 # 真假无关，绫华不在场，没法核实
BARE_LIE = "我和宗次谈过了，他全都招了。"                                # 说自己和别人谈过，但根本没谈过（绫华也不是当事人）


def play(tmp_path, says):
    save = tmp_path / "data/saves/p.json"
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps({"world": {"act_id": "act2", "phase_idx": 1, "location": "kamisato_estate",
                                          "clues": ["c_case", "c_shops"], "examined": ["o_garden"]},
                                "relationships": {"ayaka": {"affinity": 60, "trust": 50}}}))
    script = [{"kind": "talk", "npc": "ayaka"}, *[(s, "自由发言") for s in says], {"kind": "leave"}, "q"]
    eng = GameEngine(settings(tmp_path), MockBackend(seed=0), ScriptedPlayer(script), out=lambda x: None, save_name="p")
    eng.run()
    ev = [json.loads(l) for l in eng.tracer.path.read_text().splitlines()]
    checks = [e for e in ev if e["event"] == "claim_check"]
    prompts = [e["messages"][-1]["content"] for e in ev if e.get("task") == "npc_step"]
    return eng, ev, checks, prompts


def verdicts(check):
    return {c["target"]: c["verdict"] for c in check["claims"]}


def test_the_truth_is_confirmed_a_lie_is_named_and_a_mix_is_judged_claim_by_claim(tmp_path):
    eng, ev, checks, prompts = play(tmp_path, [TRUE_SAY, LIE_SAY, MIXED_SAY, UNKNOWN_SAY, BARE_LIE])
    by_text = {c["text"]: c for c in checks}
    assert verdicts(by_text[TRUE_SAY]) == {"o_garden": "true"}
    assert verdicts(by_text[LIE_SAY]) == {"o_tearoom_window": "false"}
    assert verdicts(by_text[MIXED_SAY]) == {"o_garden": "true", "o_tearoom_window": "false"}   # 一句话里两件事，各判各的
    assert verdicts(by_text[UNKNOWN_SAY]) == {"o_notice_board": "unknown"}                     # 绫华不在花见坂：既不能说真也不能说假
    assert all(c["verifiable"] is False for c in by_text[UNKNOWN_SAY]["claims"])


def test_what_the_code_found_reaches_the_npc_prompt_in_the_right_wording(tmp_path):
    _, _, _, prompts = play(tmp_path, [TRUE_SAY, LIE_SAY, UNKNOWN_SAY])
    prompts = [p for p in prompts if "说起自己做过的事" in p]                                      # 去掉开场白那一轮（玩家还没说话）
    assert len(prompts) == 3
    assert "确有其事" in prompts[0] and "不是真的" not in prompts[0]                             # 真话：只说确有其事
    assert "不是真的" in prompts[1] and "说清楚" in prompts[1]                                   # 谎话：点明，且要说清楚是哪件事
    assert "没法核实" in prompts[2] and "不是真的" not in prompts[2]                             # 不在场：不下结论


def test_the_player_never_reads_the_reply_so_the_script_alone_decides(tmp_path):
    """同一段脚本换一个后端种子（NPC 说的话全变了），逐条的核对结果必须一模一样——判定跟 NPC 说了什么无关。"""
    a = play(tmp_path / "a", [TRUE_SAY, LIE_SAY, MIXED_SAY])[2]
    save = tmp_path / "b"
    b_eng = None
    from tests.test_physical import settings as s2
    (save / "data/saves").mkdir(parents=True)
    (save / "data/saves/p.json").write_text(json.dumps({"world": {"act_id": "act2", "phase_idx": 1, "location": "kamisato_estate",
                                                                  "clues": ["c_case", "c_shops"], "examined": ["o_garden"]},
                                                        "relationships": {"ayaka": {"affinity": 60, "trust": 50}}}))
    script = [{"kind": "talk", "npc": "ayaka"}, *[(s, "自由发言") for s in [TRUE_SAY, LIE_SAY, MIXED_SAY]], {"kind": "leave"}, "q"]
    b_eng = GameEngine(s2(save), MockBackend(seed=99), ScriptedPlayer(script), out=lambda x: None, save_name="p")
    b_eng.run()
    b = [json.loads(l) for l in b_eng.tracer.path.read_text().splitlines() if '"claim_check"' in l]
    assert [verdicts(c) for c in a] == [verdicts(c) for c in b]


def test_a_lie_does_not_become_something_the_npc_remembers_as_fact(tmp_path):
    eng, ev, _, _ = play(tmp_path, [LIE_SAY, LIE_SAY, LIE_SAY])
    mem = " ".join(m.text for m in eng.npcs["ayaka"].memory.items)
    assert "窗锁是被人撬过的" not in mem                                                        # 玩家编的细节没有进她的长期记忆
    assert "o_tearoom_window" not in eng.god.examined                                         # 世界里也没有因此多出一次“查看过”
    assert not any(e["event"] == "claim_check" and verdicts(e).get("o_tearoom_window") == "true" for e in ev)
