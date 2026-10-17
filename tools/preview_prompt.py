"""提示词预览：不调模型，打印某个 NPC 此刻会收到的完整提示词。

查「谁不该知道什么」「这句话到底带了哪些上下文」时，以前要真跑一轮再翻 trace；这里直接看。

  python3 tools/preview_prompt.py web ayaka                              开场那一轮
  python3 tools/preview_prompt.py web thoma --say "那晚有谁敲过后门？" --intent 打听情报
  python3 tools/preview_prompt.py web heizou --say "你觉得月白出了什么问题？" --find 守夜狐,裂纹
  python3 tools/preview_prompt.py web aen --stage all --no-system        这一轮的每一次请求（评估 / 说话 / 核对 / 记录）

做法：把存档和记忆复制到临时目录，用一个「只记录请求、回答用 mock」的后端把真实的对话流程跑一轮，
所以打印出来的就是真实代码拼出来的提示词（同一份 build_system_prompt / build_state_block / fit_state）。
不会碰你的存档、记忆和 trace，也不会联网、不花钱。

两点要知道：
  · 玩家会被临时挪到 NPC 此刻所在的地点（NPC 在路上就没法预览）。
  · 「你这一轮的心态」这一节来自评估步骤的输出，历史里的开场白也是模型说的——这里都是 mock 给的，真实模型会不同；
    开场那一轮没有这一节。其余小节（时间、局势、知道的线索、往事、检索、规矩……）都由代码决定，与真局一致。
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STAGES = {"speak": "npc_step", "appraise": "npc_appraise", "audit": "npc_audit", "annotate": "npc_annotate"}
STAGE_NAMES = {"npc_step": "说话", "npc_appraise": "评估", "npc_audit": "核对", "npc_annotate": "记录"}


class PreviewError(Exception):
    """用户能看懂、能自己处理的问题（存档不存在、NPC 不在场……）。"""


@dataclass
class Preview:
    save: str
    npc: str
    npc_name: str
    say: Optional[str]
    intent: Optional[str]
    place: str
    moved_from: Optional[str]
    time_label: str
    requests: list = field(default_factory=list)       # 这一轮发给 NPC 的 LLMRequest，按发出顺序
    context: dict = field(default_factory=dict)        # trace 里这一轮的 context 事件：各小节字数、被预算丢掉的小节


def _capture_backend():
    from llm.mock_backend import MockBackend

    class CaptureBackend(MockBackend):
        """回答用 mock，同时把每个请求记下来。"""
        name = "mock"

        def __init__(self):
            super().__init__(seed=0)
            self.captured = []

        def complete(self, req):
            self.captured.append(req)
            return super().complete(req)

    return CaptureBackend()


def capture(save: str, npc: str, *, say: Optional[str] = None, intent: str = "自由发言",
            data_dir: Optional[Path] = None, settings=None) -> Preview:
    from core.player_agent import ScriptedPlayer
    from engine.game_engine import GameEngine
    from llm.decider import Decider
    from llm.router import ModelRouter

    if settings is None:
        from config.settings import load_settings
        settings = load_settings()               # 沿用 .env 里的开关（人物细节模式、预算……），提示词才和真局一致
    src = Path(data_dir or settings.game.data_dir)
    save_file = src / "saves" / f"{save}.json"
    if not save_file.is_file():
        have = sorted(p.stem for p in (src / "saves").glob("*.json")) if (src / "saves").is_dir() else []
        raise PreviewError(f"没有存档「{save}」。现有存档：{'、'.join(have) or '（无）'}")
    tmp = Path(tempfile.mkdtemp(prefix="ayaka_preview_"))
    try:
        (tmp / "saves").mkdir()
        shutil.copy(save_file, tmp / "saves" / f"{save}.json")
        if (src / "memories" / save).is_dir():
            shutil.copytree(src / "memories" / save, tmp / "memories" / save)
        settings = copy.deepcopy(settings)
        settings.game.data_dir = tmp
        settings.harness.game_token_budget = 0                       # 预算降级会改提示词，预览按“没降级”看
        backend = _capture_backend()
        script = [{"kind": "talk", "npc": npc}] + ([(say, intent)] if say else []) + [{"kind": "leave"}, "q"]
        eng = GameEngine(settings, backend, ScriptedPlayer(script), out=lambda s: None, save_name=save, new_game=False,
                         router=ModelRouter(backend, settings.llm), decider=Decider(), trace=True)
        if npc not in eng.personas:
            raise PreviewError(f"没有这个 NPC「{npc}」。可选：{'、'.join(sorted(eng.personas))}")
        g = eng.god
        name = eng.personas[npc].name
        where = g.physical.where(npc)
        if where is None:
            plan = g.physical.plans.get(npc)
            if plan and plan.get("location") and plan.get("arrive", 0) > g.minute:
                raise PreviewError(f"{name}正在去{eng.world.name(plan['location'])}的路上，还要约 {plan['arrive'] - g.minute} 分钟才到，"
                                   f"现在没有可以交谈的位置（游戏里菜单上也找不到她）。")
            raise PreviewError(f"{name}这个时段不在场（不在任何能去的地方），没有可以交谈的位置。")
        moved_from = g.location if g.location != where else None
        g.location = where
        try:
            eng.run(max_actions=6)
        except ValueError:
            here = [o.meta["npc"] for o in eng.explorer.menu() if o.kind == "talk"]
            raise PreviewError(f"{name}在{eng.world.name(where)}，但此刻没法和他交谈（菜单里没有这个选项）。"
                               f"这里能交谈的：{'、'.join(here) or '（无）'}") from None
        opening = say is None
        reqs = [r for r in backend.captured if r.task in STAGE_NAMES and bool(r.meta.get("opening")) == opening
                and (opening or r.meta.get("player_text") == say)]
        if not reqs:
            raise PreviewError("这一轮没有发出任何请求（谈话可能在开场就结束了）。")
        events = [json.loads(line) for line in eng.tracer.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ctx = next((e for e in events if e.get("event") == "context" and e.get("npc") == npc
                    and e.get("turn") == (0 if opening else 1)), {})
        return Preview(save, npc, name, say, intent if say else None, eng.world.name(where),
                       eng.world.name(moved_from) if moved_from else None, g.time_label(), reqs, ctx)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------ 渲染

def _sections(content: str) -> list[tuple[str, str]]:
    """把情境块按「## 标题」拆开：[(标题, 正文)]。第一段（没有标题的）叫“开头”。"""
    out = []
    for i, sec in enumerate(re.split(r"\n(?=## )", content)):
        m = re.match(r"## ([^\n：（]*)", sec)
        out.append((m.group(1).strip() if m else ("开头" if i == 0 else ""), sec))
    return out


def locate(messages: list[dict], term: str) -> list[str]:
    """term 出现在哪些位置：system、某条历史消息，或情境块的哪几个小节。"""
    hits = []
    for i, m in enumerate(messages):
        content = m.get("content") or ""
        if term not in content:
            continue
        if m["role"] == "system":
            hits.append("system（人设与规矩）")
        elif i == len(messages) - 1:
            hits += [f"情境块「{t}」" for t, body in _sections(content) if term in body]
        else:
            hits.append(f"第 {i} 条{m['role']}消息（历史）")
    return hits


def find_report(pv: Preview, terms: list[str], stage: str = "speak") -> list[str]:
    reqs = _pick(pv, stage)
    lines = []
    for t in terms:
        seen: list[str] = []
        for r in reqs:
            for h in locate(r.messages, t):
                tag = f"[{STAGE_NAMES[r.task]}] {h}" if stage == "all" else h
                if tag not in seen:
                    seen.append(tag)
        lines.append(f"  「{t}」：" + ("；".join(seen) if seen else "不在提示词里"))
    return lines


def _pick(pv: Preview, stage: str) -> list:
    if stage == "all":
        return list(pv.requests)
    task = STAGES[stage]
    got = [r for r in pv.requests if r.task == task]
    return got[:1] if stage == "speak" else got


def render(pv: Preview, stage: str = "speak", show_system: bool = True, find: Optional[list[str]] = None) -> str:
    from llm.messages import estimate_tokens
    reqs = _pick(pv, stage)
    out = []
    what = "开场" if pv.say is None else f"玩家说「{pv.say}」（意图：{pv.intent}）"
    out.append(f"══ 提示词预览 · {pv.npc_name}（{pv.npc}）· {'全部请求' if stage == 'all' else STAGE_NAMES[STAGES[stage]]} ══")
    out.append(f"存档 {pv.save}：{pv.time_label}，在{pv.place}" + (f"（玩家被临时从{pv.moved_from}挪到这里）" if pv.moved_from else ""))
    out.append(f"这一轮：{what}")
    out.append("没有调用真实模型。" + ("开场没有评估这一步。" if pv.say is None else
                                       "「你这一轮的心态」来自 mock 的评估、历史里的开场白也是 mock 说的，真实模型会不同。"))
    if not reqs:
        out.append(f"\n（这一轮没有「{STAGE_NAMES[STAGES[stage]]}」这一步）")
        return "\n".join(out)
    c = pv.context
    if c:
        out.append(f"\n── 情境块概览：系统消息 {c.get('system', '?')} 字 · 历史 {c.get('history', '?')} 字 · "
                   f"被预算丢掉的小节：{'、'.join(c['dropped']) if c.get('dropped') else '无'}")
        for title, (tier, n) in (c.get("sections") or {}).items():
            out.append(f"   {tier:<10} {n:>5} 字  {title}")
    if find:
        out.append("\n── 关键词检查（只看提示词里有没有，不看模型会不会说）")
        out += find_report(pv, find, stage)
    for r in reqs:
        toks = estimate_tokens("".join(m["content"] for m in r.messages))
        extra = f" · 可用工具：{'、'.join(r.meta['allowed_tools'])}" if r.meta.get("allowed_tools") else ""
        out.append(f"\n╔═ 请求：{STAGE_NAMES[r.task]}（{r.task}{'，第 %d 步' % (r.meta['step'] + 1) if 'step' in r.meta else ''}）"
                   f" · {len(r.messages)} 条消息 · 约 {toks} token · 输出格式 {r.schema_name}{extra}")
        for i, m in enumerate(r.messages):
            if m["role"] == "system" and not show_system:
                out.append(f"┌─ [system] {len(m['content'])} 字（已隐藏，去掉 --no-system 可看）")
                continue
            out.append(f"┌─ [{m['role']}] {len(m['content'])} 字")
            out.append(m["content"])
    return "\n".join(out)


def to_json(pv: Preview, stage: str = "speak") -> str:
    return json.dumps({"save": pv.save, "npc": pv.npc, "say": pv.say, "intent": pv.intent, "place": pv.place,
                       "time": pv.time_label, "context": pv.context,
                       "requests": [{"task": r.task, "schema": r.schema_name, "messages": r.messages,
                                     "allowed_tools": r.meta.get("allowed_tools"), "step": r.meta.get("step")}
                                    for r in _pick(pv, stage)]}, ensure_ascii=False, indent=2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="不调模型，打印某个 NPC 此刻会收到的完整提示词")
    ap.add_argument("save", help="存档名（data/saves/<名字>.json），比如 web")
    ap.add_argument("npc", help="NPC 的 id，比如 ayaka、thoma、heizou")
    ap.add_argument("--say", help="玩家这一轮说的话；不给就看开场那一轮")
    ap.add_argument("--intent", default="自由发言", help="玩家话的意图（打听情报 / 闲聊 / 出示线索 / 自由发言……）")
    ap.add_argument("--stage", choices=[*STAGES, "all"], default="speak", help="看哪一步的请求，默认说话")
    ap.add_argument("--find", help="逗号分隔的词：检查它们是否出现在提示词里、出现在哪一节")
    ap.add_argument("--no-system", action="store_true", help="不打印很长的系统消息（人设与规矩）")
    ap.add_argument("--json", action="store_true", help="输出 JSON，方便别的工具读")
    ap.add_argument("--data-dir", help="数据目录，默认用设置里的 data/")
    args = ap.parse_args(argv)
    try:
        pv = capture(args.save, args.npc, say=args.say, intent=args.intent,
                     data_dir=Path(args.data_dir) if args.data_dir else None)
    except PreviewError as e:
        print(f"✘ {e}", file=sys.stderr)
        return 2
    if args.json:
        print(to_json(pv, args.stage))
    else:
        print(render(pv, args.stage, not args.no_system, [t.strip() for t in args.find.split(",") if t.strip()] if args.find else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
