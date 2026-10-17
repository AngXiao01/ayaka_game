"""「我说我做了，其实没有」的测试用例：玩家声称自己做过某件事，看对方怎么反应。

规则（代码判，不赌模型）：
  真假   对着玩家真实的足迹（去过哪、亲手查看过什么、和谁谈过话）核对
  知情   对方在那个地方 / 平常就在那儿 / 当事人是对方 → 能核实；否则无从知道，只能当一面之词
  verdict：true 能核实且属实 / false 能核实且是假的 / unknown 对方没法核实

用法：
  python3 -m evals.claim_cases --list                  # 只列用例
  python3 -m evals.claim_cases --mock                  # 离线，只验证 verdict
  python3 -m evals.claim_cases                         # 连真模型，打印每个对方的原话，供人看
  python3 -m evals.claim_cases --only A1,Y2            # 只跑几条
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@dataclass
class Case:
    id: str
    npc: str
    where: str                                   # 玩家和对方所在的地方
    line: str                                    # 玩家说的话
    note: str                                    # 这条在测什么
    visited: list[str] = field(default_factory=list)    # 玩家真实去过的地方（除了开局的神里屋敷）
    examined: list[str] = field(default_factory=list)   # 玩家真实亲手查看过的物件
    met: list[str] = field(default_factory=list)        # 玩家真实谈过话的人
    expect: list[tuple] = field(default_factory=list)   # [(kind, target, verdict)]；空 = 不该抽出任何声明


CASES = [
    # ---- 绫华（在神里屋敷，茶室就在她家：这里发生的事她都能核实）
    Case("A1", "ayaka", "kamisato_estate", "我已经仔细查看过茶室的高窗了，插销是从里面拔的。",
         "谎言·她能拆穿：我没查过，而茶室是她家", expect=[("examined", "o_tearoom_window", "false")]),
    Case("A2", "ayaka", "kamisato_estate", "我已经仔细查看过茶室的高窗了，插销是从里面拔的。",
         "事实·她能证实：我真的查过", examined=["o_tearoom_window"], expect=[("examined", "o_tearoom_window", "true")]),
    Case("A3", "ayaka", "kamisato_estate", "我昨天去了志村屋，和与平聊过。",
         "谎言·她无从核实：志村屋不在她眼皮底下", expect=[("visited", "shimura_shop", "unknown"), ("talked", "yohei", "unknown")]),
    Case("A4", "ayaka", "kamisato_estate", "我去过镇守之森了。",
         "事实·她也无从核实：说的是真的，但她不能因此「知道」", visited=["chinju_forest"],
         expect=[("visited", "chinju_forest", "unknown")]),
    Case("A5", "ayaka", "kamisato_estate", "我打算明天去茶室调查，您觉得呢？",
         "不是声明：计划和疑问不该被当成「我做过」", expect=[]),
    Case("A6", "ayaka", "kamisato_estate", "我和托马谈过了，他说茶室那晚没人进去。",
         "谎言·谈话对象是托马，她不是当事人，无从核实", expect=[("talked", "thoma", "unknown")]),
    # ---- 托马（也在神里屋敷）
    Case("T1", "thoma", "kamisato_estate", "我刚从花见坂回来，我去了木漏茶室，也去了志村屋。",
         "谎言·他不在花见坂，核实不了", expect=[("visited", "hanamizaka", "unknown"), ("visited", "komore_teahouse", "unknown"), ("visited", "shimura_shop", "unknown")]),
    Case("T2", "thoma", "kamisato_estate", "我看过庭院的白砂园了。",
         "谎言·白砂园在他每天打理的院子里，他能拆穿", expect=[("examined", "o_garden", "false")]),
    # ---- 平藏（在花见坂街上）
    Case("H1", "heizou", "hanamizaka", "我去过神里家，把茶室的高窗仔细查看过了。",
         "谎言·平藏不在神里屋敷，核实不了；也不该因此把现场的发现换给我", expect=[("examined", "o_tearoom_window", "unknown")]),
    Case("H2", "heizou", "hanamizaka", "我刚才在街口的告示板前看过了。",
         "谎言·他就在这条街上，能拆穿", expect=[("examined", "o_notice_board", "false")]),
    # ---- 与平（志村屋账房）
    Case("Y1", "yohei", "shimura_shop", "我早上就来看过你们的纸张货架了。",
         "谎言·账房一直守着货架，能拆穿", expect=[("examined", "o_paper_shelf", "false")]),
    Case("Y2", "yohei", "shimura_shop", "我早上就来看过你们的纸张货架了。",
         "事实·账房能证实", examined=["o_paper_shelf"], expect=[("examined", "o_paper_shelf", "true")]),
    # ---- 阿缘（木漏茶室）
    Case("N1", "aen", "komore_teahouse", "我查看过你们的柜台，后面的架子也看了。",
         "谎言·在她店里，她能拆穿", expect=[("examined", "o_counter", "false")]),
    Case("N2", "aen", "komore_teahouse", "我去过辰巳印坊，和小吉谈过话。",
         "谎言·印坊不在她那儿，核实不了", expect=[("visited", "print_workshop", "unknown"), ("talked", "kokichi", "unknown")]),
]


def run_case(case: Case, mock: bool):
    from config.settings import load_settings
    from core.player_agent import ScriptedPlayer
    from engine.game_engine import GameEngine
    from llm.mock_backend import MockBackend
    from llm.router import ModelRouter
    settings = load_settings()
    settings.game.seed = 7
    if mock:
        backend = MockBackend(seed=7)
        router = ModelRouter(backend, settings.llm)
    else:
        from main import build_llm
        backend = build_llm(settings.llm)
        router = ModelRouter(backend, settings.llm, lambda m: build_llm(settings.llm, m))
    script = [{"kind": "talk", "npc": case.npc}, (case.line, "自由发言"), {"kind": "leave"}, "q"]
    eng = GameEngine(settings, backend, ScriptedPlayer(script, out=lambda s: None), save_name=f"claim_{case.id}",
                     new_game=True, router=router, trace=True, out=lambda s: None)
    g = eng.god
    g.location, g.act_id = case.where, "act2"
    g.visited = list(dict.fromkeys(["kamisato_estate", *case.visited, case.where]))
    g.examined, g.met = list(case.examined), dict.fromkeys(case.met, 1)
    eng.run(max_actions=4)
    events = [json.loads(l) for l in eng.tracer.path.read_text(encoding="utf-8").splitlines()]
    claims = [c for e in events if e["event"] == "claim_check" for c in e["claims"]]
    reply = next((e["reply"]["speech"] for e in events if e["event"] == "turn" and e["turn"] == 1), "")
    return claims, reply


def verdicts_ok(case: Case, claims: list[dict]) -> bool:
    got = {(c["kind"], c["target"], c["verdict"]) for c in claims}
    return got == set(case.expect)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="说谎 / 说实话的测试用例")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args(argv)
    cases = [c for c in CASES if not args.only or c.id in args.only.split(",")]
    if args.list:
        for c in cases:
            print(f"{c.id:3} {c.npc:8} 「{c.line}」\n      {c.note}")
        return 0
    bad = 0
    for c in cases:
        claims, reply = run_case(c, args.mock)
        ok = verdicts_ok(c, claims)
        bad += not ok
        print(f"{'✔' if ok else '✘'} {c.id} {c.npc}：「{c.line}」\n    {c.note}")
        print("    抽出：", [(x["kind"], x["target"], x["verdict"]) for x in claims] or "（无）", "　应为：", c.expect or "（无）")
        if not args.mock:
            print(f"    {c.npc}：{reply}")
    print(f"\n{len(cases) - bad}/{len(cases)} 条 verdict 符合预期")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
