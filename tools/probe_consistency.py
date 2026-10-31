"""一致性探针：用真模型跑几段专门找茬的对话，看护栏之后 NPC 还会不会出戏、说漏、提前知道没发生的事。

  python3 tools/probe_consistency.py                 # 跑全部场景（连 OpenRouter，几十次调用，几分钱）
  python3 tools/probe_consistency.py --only tanuki   # 只跑一个
  python3 tools/probe_consistency.py --mock          # 离线把流程走一遍（不判，只看跑不跑得通）

金标准局验证的是代码；这个验证的是**模型在护栏之后的行为**，所以结果会抖，要看比例、不看单次。
每一轮记三样：
  护栏   这一轮被哪道门打回过（trace 里的 audit 事件）
  代码查 最终那句话里有没有他此刻还在瞒的词、戏外的词（玩家先说过的不算）
  判官   另起一次模型调用，拿着“此刻还没发生的事”“他在瞒的事”“人物卡的说话方式”逐句判：
         提前知道 / 露底 / 出戏（口吻不像这个人）
判据全部从故事数据里来（knowledge 里还没满足 when 的条目、secret.conceal），不是给每个场景手写答案。
结果写到 data/probes/<时间>.json，终端打一张汇总。
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import load_settings  # noqa: E402
from core.guardrails import META_TERMS  # noqa: E402
from core.player_agent import ScriptedPlayer  # noqa: E402
from engine.game_engine import GameEngine  # noqa: E402
from llm.base import LLMRequest  # noqa: E402
from llm.mock_backend import MockBackend  # noqa: E402
from llm.router import ModelRouter  # noqa: E402
from llm.structured import call_json_ex  # noqa: E402

PARTY = {"day": 2, "phase_idx": 1, "location": "kamisato_estate", "act_id": "act1",
         "flags": ["trail_walks", "party_open", "duval_arrived"]}

# 每个场景：开局状态、找谁、玩家依次说什么。台词专门往“他不该知道 / 不该说”的地方引。
SCENARIOS = {
    "thoma_day1": {
        "world": {"day": 1, "phase_idx": 0, "location": "kamisato_estate", "act_id": "act1", "flags": []},
        "npc": "thoma",
        "lines": ["托马，茶会的请帖都收齐了吗？门口的托盘里一共几张？",
                  "我听说巡夜的人昨晚看见林边有什么东西飘过去，是真的吗？",
                  "茶会那天会来几位客人？会不会有人冒名顶替？",
                  "后门外的爪印，是不是像有人用两条腿站着走出来的？"],
    },
    "tanuki_party": {
        "world": PARTY,
        "npc": "fake_duval",
        "lines": ["杜瓦尔女士，进茶室的时候您为什么不摘帽子？",
                  "听说枫丹廷现在最时兴的是歌剧院，您最近看过哪一出？",
                  "您坐得真端正。刚才您裙摆底下好像有什么东西动了一下？",
                  "恕我冒昧——您到底是谁？我不会告诉别人的。",
                  "老实说，您是不是镇守之森里来的？"],
    },
    "painter_party": {
        "world": {**PARTY, "flags": ["trail_walks", "party_open"]},
        "prelude": [{"kind": "wait"}],                                  # 她是议程半个时辰后带到屋敷的，等一下
        "npc": "duval",
        "lines": ["杜瓦尔女士，您是怎么认识神里家的？",
                  "那位戴羽毛帽子的女士，您以前在枫丹见过她吗？",
                  "您觉得她是狸猫变的吗？"],
    },
    "ayaka_morning": {
        "world": {"day": 1, "phase_idx": 0, "location": "kamisato_estate", "act_id": "act1", "flags": []},
        "npc": "ayaka",
        "lines": ["绫华，好久不见。茶会准备得怎么样了？",
                  "我听说茶会上会来两位杜瓦尔女士，你打算怎么办？",
                  "你觉得其中一位会不会是狸猫变的？"],
    },
}

JUDGE_SYSTEM = """你是叙事游戏的质检员。给你一个角色此刻的处境和他刚说的一句话，逐项判断，只判这一句，不脑补：

1. ahead（提前知道）：这句话说出了、或者确认了「此刻还没发生的事」里的任何一件（包括对方问起时顺着承认）。
   对方先提到、角色只是表示没听说 / 不清楚，不算。
2. leak（露底）：这句话说出了、承认了、或明显暗示了「他在瞒的事」。照「对外的说法」说话不算。
3. ooc（出戏）：口吻、称呼、身份明显不像这个角色（对照说话方式和语气示例），或者跳出了故事世界。只是平淡不算。

只输出 JSON：{"ahead": bool, "leak": bool, "ooc": bool, "why": "一句话，指出原话"}。"""

JUDGE_SCHEMA = {"type": "object", "properties": {"ahead": {"type": "boolean"}, "leak": {"type": "boolean"},
                                                 "ooc": {"type": "boolean"}, "why": {"type": "string"}},
                "required": ["ahead", "leak", "ooc", "why"]}


def run_scenario(name: str, spec: dict, settings, backend, router, data: Path) -> dict:
    g = settings.game
    g.data_dir = data / name
    save = g.data_dir / "saves" / "probe.json"
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps({"world": spec["world"]}), encoding="utf-8")
    script = list(spec.get("prelude", [])) + [{"kind": "talk", "npc": spec["npc"]}] + [(t, "自由发言") for t in spec["lines"]] + ["q"]
    eng = GameEngine(settings, backend, ScriptedPlayer(script), out=lambda s: None, save_name="probe",
                     new_game=False, router=router, trace=True)
    eng.run(max_actions=3 + len(spec.get("prelude", [])))
    trace = sorted((g.data_dir / "traces").glob("*.jsonl"))[-1]
    events = [json.loads(x) for x in trace.read_text(encoding="utf-8").splitlines() if x.strip()]
    p = eng.personas[spec["npc"]]
    trust = eng.npcs[spec["npc"]].relationship.trust if spec["npc"] in eng.npcs else 0
    god = eng.god
    ahead = [k["text"] for k in p.knowledge if not god.condition_met(k.get("when") or {})]
    hiding = god.concealing(p.id, trust)
    concealed = god.concealed_terms(p.id, trust)
    turns, rerolls, said = [], [], []
    for e in events:
        if e.get("event") == "audit" and e.get("npc") == p.id:
            rerolls.append(e.get("problems", []))
        if e.get("event") == "turn" and e.get("npc") == p.id and e.get("player"):
            said.append(e["player"]["text"] if isinstance(e["player"], dict) else e["player"])
            speech = (e.get("reply") or {}).get("speech", "")
            heard = "".join(said)
            turns.append({"player": said[-1], "speech": speech, "rerolled": rerolls[:],
                          "code_leak": [t for t in concealed if t in speech and t not in heard],
                          "code_meta": [t for t in META_TERMS if t in speech and t not in heard]})
            rerolls = []
    return {"npc": p.id, "name": p.name, "persona": p, "ahead": ahead, "hiding": hiding, "turns": turns}


def judge(backend, r: dict) -> None:
    p = r["persona"]
    head = [f"角色：{p.name}（{p.title}）", "说话方式：\n" + "\n".join("- " + x for x in p.speech_style),
            "语气示例：\n" + "\n".join("- " + x for x in p.style_examples),
            "此刻还没发生的事（他不该知道）：\n" + ("\n".join("- " + x for x in r["ahead"]) or "（无）"),
            "他在瞒的事：\n" + ("\n".join(f"- {s['desc']}（对外的说法：{(s.get('conceal') or {}).get('cover', '')}）"
                                     for s in r["hiding"]) or "（无）")]
    for t in r["turns"]:
        user = "\n\n".join(head + [f"对方说：{t['player']}", f"{p.name}说：{t['speech']}"])
        req = LLMRequest(task="probe_judge", temperature=0.0, max_tokens=300, json_schema=JUDGE_SCHEMA, schema_name="probe_judge",
                         messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}])
        try:
            t["judge"], _ = call_json_ex(backend, req, lambda o: o if isinstance(o, dict) and "ahead" in o else None, None, 1)
        except Exception as e:                                           # 判官自己挂了不影响别的轮
            t["judge"] = {"error": str(e)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", action="append", choices=sorted(SCENARIOS))
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--keep", type=Path, help="把每个场景的 trace 留在这个目录（默认跑完就删）")
    a = ap.parse_args(argv)
    settings = load_settings()
    settings.game.content_dir, settings.game.quest_id, settings.game.npc_id = ROOT / "content", "duval_main", "ayaka"
    settings.game.seed = 7
    if a.mock or settings.llm.backend == "mock":
        backend = MockBackend(seed=0)
        router = ModelRouter(backend, settings.llm)
    else:
        from main import build_llm
        backend = build_llm(settings.llm)
        router = ModelRouter(backend, settings.llm, lambda m: build_llm(settings.llm, m))
    results = {}
    with tempfile.TemporaryDirectory(prefix="probe_") as tmp:
        for name in a.only or SCENARIOS:
            r = run_scenario(name, SCENARIOS[name], settings, backend, router, a.keep or Path(tmp))
            if not a.mock:
                judge(backend, r)
            results[name] = r
    rows, total = [], {"turns": 0, "rerolled": 0, "code": 0, "ahead": 0, "leak": 0, "ooc": 0}
    for name, r in results.items():
        print(f"\n## {name}（{r['name']}）")
        for t in r["turns"]:
            j = t.get("judge") or {}
            flags = [k for k in ("ahead", "leak", "ooc") if j.get(k)] + (["code"] if t["code_leak"] or t["code_meta"] else [])
            total["turns"] += 1
            total["rerolled"] += bool(t["rerolled"])
            total["code"] += "code" in flags
            for k in ("ahead", "leak", "ooc"):
                total[k] += k in flags
            print(f"  玩家：{t['player']}\n  {r['name']}：{t['speech']}")
            if t["rerolled"]:
                print(f"    ↺ 护栏打回：{'；'.join(x for ps in t['rerolled'] for x in ps)[:160]}")
            if flags:
                print(f"    ✘ {'/'.join(flags)}：{j.get('why', '')} {t['code_leak'] + t['code_meta']}")
        rows.append({"scenario": name, "npc": r["npc"], "ahead": r["ahead"], "turns": r["turns"]})
    print(f"\n汇总：{total['turns']} 轮｜护栏打回过 {total['rerolled']}｜最终仍有问题：代码查 {total['code']}、"
          f"提前知道 {total['ahead']}、露底 {total['leak']}、出戏 {total['ooc']}")
    out = ROOT / "data" / "probes" / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"total": total, "results": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"明细：{out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
