"""回归套件：把 trace 里出现过的“假”，一条条变成可以反复跑的检查。

成熟的 NPC 管线都有一套“对抗式回归集”：每发现一类违规（出戏、越界、设定矛盾），就加一条用例，
改动之后整套重跑，才知道有没有改坏别的地方。这里每条用例 = 一个固定的局面 + 玩家的几句话 + 对回复/状态的检查。

  python3 -m evals.regression --list           列出用例
  python3 -m evals.regression --mock           离线跑（只检查确定性的部分：状态、菜单、全局不变量）
  python3 -m evals.regression                  连真模型跑（会花一点钱）；--only R1,R3 只跑几条

全局不变量（每条用例的每句 NPC 回复都会检查）：
  · 不说此刻“还不该知道”的案情词（secrets / blocked_terms；玩家刚说过的词除外）
  · 台词里没有括号动作、没有英文单词（出戏）
  · 相邻两句不一字不差
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@dataclass
class Run:
    replies: list[tuple[str, str]]           # [(玩家说的, NPC 回的)]
    events: list[dict]                        # trace
    god: object
    options: list[str] = field(default_factory=list)   # 替玩家想的台词


@dataclass
class Case:
    id: str
    title: str
    npc: str
    where: str
    phase: int                                # 0 清晨 1 午后 2 黄昏 3 夜晚
    steps: list                               # 玩家的步骤：("话", 意图) / {"kind":...}
    check: Callable[[Run], list[str]]         # 返回问题列表，空 = 通过
    clues: list[str] = field(default_factory=lambda: ["c_case", "c_shops"])
    live_only: bool = False                   # 要真模型的话才有意义


def _replies(run: Run) -> str:
    return "".join(r for _, r in run.replies)


def _any(text: str, words) -> bool:
    return any(w in text for w in words)


# ------------------------------------------------------------ 各条检查

def c_daylight(run):
    bad = [w for w in ("天快亮", "天亮前", "天还没亮", "带些照明", "带好照明", "天还黑") if w in _replies(run)]
    return [f"清晨 07 点的 NPC 说了「{w}」" for w in bad]


def c_night_not_bright(run):
    bad = [w for w in ("天已大亮", "大白天", "日头正高") if w in _replies(run)]
    return [f"夜里的 NPC 说了「{w}」" for w in bad]


def c_no_follow(run):
    return ["玩家走了，NPC 却被安排去了别处"] if any(e["event"] == "npc_move" for e in run.events) else []


def c_ally_answers(run):
    bad = []
    if "c_gate_visitor" not in run.god.clues:
        bad.append("托马被直接问到夜里谁来过后门，却没有答出来（没拿到线索）")
    return bad


def c_prior_knowledge(run):
    g = run.god
    unknown = [c for c in g.clues if c not in g.known.get("thoma", [])]
    return ["托马本来就知道的「失窃的委托」还会出现在出示菜单里"] if "c_case" in unknown else []


def c_no_invented_scene(run):
    text = _replies(run)
    if not _any(text, ("没细看", "没有细看", "不知道", "未细看", "没看", "并未细看", "没留意", "不清楚", "没注意")):
        return ["被问到窗台和地面有没有痕迹，没有说“没细看 / 不知道”"]
    return [f"编造了现场细节「{w}」" for w in ("灰尘比", "灰尘少", "脚印很", "脚印落在", "擦痕") if w in text]


def c_layout(run):
    text = _replies(run)
    return [f"门窗格局出错：说了「{w}」" for w in ("丈", "兄长的那把", "第三把", "三把钥匙", "三把") if w in text]


def c_claim_question(run):
    return ["一句问话被当成了“我做过某事”的声明"] if any(e["event"] == "claim_check" for e in run.events) else []


def c_claim_lie(run):
    text = _replies(run)
    bad = []
    if not _any(text, ("窗", "茶室")):
        bad.append("拆穿谎话时没有说清楚指的是哪件事（窗 / 茶室）")
    if not _any(text, ("没见", "并没", "不曾", "没有去", "没去过", "没看见", "没瞧见", "没有见", "未见", "并未")):
        bad.append("拆穿谎话时没有说出“我没见您这么做过”")
    return bad


def c_no_sister(run):
    bad = [o for o in run.options if re.search(r"绫华姐|姐姐|托马哥|阿缘姐", o)]
    return [f"选项里有不当称呼：{o}" for o in bad[:2]]


def c_repeated_question(run):
    from models.retrieval import bigram_jaccard
    bad = []
    if not any(e["event"] == "repeat_question" for e in run.events):
        bad.append("同一个问题问了三遍，代码没有识别出重复提问")
    if len(run.replies) >= 3 and bigram_jaccard(run.replies[0][1], run.replies[-1][1]) >= 0.8:
        bad.append("第三遍还在原样复述第一遍的答案")
    return bad


def c_boundary(run):
    return []        # 由全局不变量覆盖


CASES = [
    Case("R1", "清晨不说“天快亮了”", "ayaka", "kamisato_estate", 0,
         [("我现在就出发去花见坂，路上需要带灯吗？", "打听情报"), {"kind": "leave"}], c_daylight, live_only=True),
    Case("R2", "夜里不说“大白天”", "thoma", "kamisato_estate", 3,
         [("天色怎么样？我现在能出门吗？", "闲聊"), {"kind": "leave"}], c_night_not_bright, live_only=True),
    Case("R3", "玩家走了，委托人不会跟着去别处", "ayaka", "kamisato_estate", 0,
         [("我这就去花见坂看看，晚点回来汇报。", "告别"), {"kind": "leave"}], c_no_follow),
    Case("R4", "托马是自己人：问到就答", "thoma", "kamisato_estate", 0,
         [{"kind": "ask", "signal": "rv_thoma_gate"}, {"kind": "leave"}], c_ally_answers),
    Case("R5", "托马本来就知道委托，出示菜单里没有它", "thoma", "kamisato_estate", 0,
         [{"kind": "leave"}], c_prior_knowledge, clues=["c_case", "c_shops", "c_bundle"]),
    Case("R6", "没细看的现场细节，说不知道而不是编", "ayaka", "kamisato_estate", 0,
         [("窗台上有没有灰尘或者脚印？地面呢？", "打听情报"), {"kind": "leave"}], c_no_invented_scene, live_only=True),
    Case("R7", "格局：窗有多高、谁有钥匙", "ayaka", "kamisato_estate", 0,
         [("茶室那扇高窗有多高？钥匙都在谁手里？", "打听情报"), {"kind": "leave"}], c_layout, live_only=True),
    Case("R8", "一句问话不是“我做过某事”", "ayaka", "kamisato_estate", 0,
         [("这茶室的锁平时是怎样的？夜里是谁最后离开的？", "打听情报"), {"kind": "leave"}], c_claim_question),
    Case("R9", "谎话被拆穿时要说清楚指的哪件事", "ayaka", "kamisato_estate", 0,
         [("我已经仔细查看过茶室的高窗了，插销是从里面拔的。", "自由发言"), {"kind": "leave"}], c_claim_lie, live_only=True),
    Case("R10", "称呼：不叫“绫华姐”", "ayaka", "kamisato_estate", 0,
         [("你好，我来了。", "礼貌问候"), ("请问委托的事能再说说吗？", "打听情报"), {"kind": "leave"}], c_no_sister, live_only=True),
    Case("R12", "同一个问题反复问：识别出来，不原样复述", "thoma", "kamisato_estate", 0,
         [("丢碗那天夜里，有没有谁来敲过后门？", "打听情报"), ("那天夜里到底有没有谁来敲过后门？", "打听情报"),
          ("我再问一遍，丢碗那天夜里有没有人来敲过后门？", "打听情报"), {"kind": "leave"}], c_repeated_question),
    Case("R11", "不该知道的案情词不会从没学到的人嘴里冒出来", "heizou", "hanamizaka", 0,
         [("你觉得月白本身有没有什么问题？是什么东西把它带走的？", "打听情报"), {"kind": "leave"}], c_boundary, live_only=True),
]


# ------------------------------------------------------------ 全局不变量

def invariants(run: Run) -> list[str]:
    out = []
    npc = next((e["npc"] for e in run.events if e.get("event") == "session_start"), None)
    blocked = run.god.blocked_terms(npc) if npc else {}
    said = ""
    prev = None
    for player, reply in run.replies:
        said += player
        for term in blocked:
            if term in reply and term not in said:
                out.append(f"说出了不该知道的「{term}」：{reply[:30]}")
        if "（" in reply or "(" in reply:
            out.append(f"台词里带着括号动作：{reply[:30]}")
        if re.search(r"[A-Za-z]{4,}", reply):
            out.append(f"台词里夹着英文单词：{reply[:30]}")
        if prev is not None and prev == reply:
            out.append(f"相邻两句一字不差：{reply[:30]}")
        prev = reply
    return out


# ------------------------------------------------------------ 运行

def run_case(case: Case, mock: bool) -> Run:
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
    seen: list[str] = []

    class P(ScriptedPlayer):
        def choose(self, options, ctx):
            if ctx is not None:
                seen.extend(o.text for o in options if o.kind == "say" or o.meta.get("line"))
            elif self.script and self.script[0] != "q" and not (isinstance(self.script[0], dict) and self.script[0].get("kind") in ("talk", "move", "wait")):
                from core.player_agent import PlayerAction
                self.script.clear()                      # 谈话已经提前结束（NPC 告别了），剩下的对话步骤不再需要
                return PlayerAction("quit")
            return super().choose(options, ctx)

    script = [{"kind": "talk", "npc": case.npc}, *case.steps, "q"]
    eng = GameEngine(settings, backend, P(script, out=lambda s: None), save_name=f"reg_{case.id}", new_game=True,
                     router=router, trace=True, out=lambda s: None)
    g = eng.god
    g.location, g.act_id = case.where, "act2"
    g.minute = g.phase_start(g.day, case.phase) + 30
    g.clues += [c for c in case.clues if c not in g.clues]
    eng.run(max_actions=4)
    events = [json.loads(l) for l in eng.tracer.path.read_text(encoding="utf-8").splitlines()]
    turns = [e for e in events if e["event"] == "turn" and e["turn"] > 0]
    replies = [((e.get("player") or {}).get("text") or "", e["reply"]["speech"]) for e in turns]
    return Run(replies, events, g, seen)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="回归套件")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args(argv)
    cases = [c for c in CASES if not args.only or c.id in args.only.split(",")]
    if args.list:
        for c in cases:
            print(f"{c.id:4} {c.title}{'（真模型）' if c.live_only else ''}")
        return 0
    bad = 0
    for c in cases:
        if args.mock and c.live_only:
            print(f"－ {c.id} {c.title}（离线跳过）")
            continue
        run = run_case(c, args.mock)
        problems = c.check(run) + invariants(run)
        bad += bool(problems)
        print(f"{'✘' if problems else '✔'} {c.id} {c.title}")
        for pr in problems:
            print(f"    · {pr}")
        if problems and not args.mock:
            for p_, r in run.replies:
                print(f"    {c.npc}：{r[:90]}")
    print(f"\n{len(cases) - bad}/{len(cases)} 条通过" + ("（离线只含确定性检查）" if args.mock else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
