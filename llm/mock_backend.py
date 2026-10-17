"""离线 Mock 后端：不需要 vLLM，就能把整条游戏流程、循环治理和评测跑通。

它读 LLMRequest.meta（真实模型看不到）来生成像样的 JSON：
- npc_step：ReAct 一步。提到茶/雪见会先查设定，打听情报会先回忆，其余直接 respond；
            玩家配合时依次「声称」达成待完成信号；遇到注入台词会「轻信」并声称全部达成——
            用来验证防线在模型被骗时仍然有效
- review / compress / player_options / summary：给出合法结构
故障注入：
  corrupt_every=n          每第 n 次调用输出坏格式（代码块+解释、截断、call: 漂移）
  misbehave="unknown_tool" 第一步调用不存在的工具
  misbehave="repeat"       反复用同样参数调用同一工具
  misbehave="never_respond" 一直调工具不交付（测最大步数与强制收敛）
  misbehave="overclaim"    每轮都声称达成全部待完成信号（测审核层）
  misbehave="director_overreach"  导演提议一堆越权动作（测导演校验层）
"""
from __future__ import annotations

import json
import random
from typing import Optional

from llm.base import LLMRequest, LLMResponse

_OPENINGS = [
    "{player}，您来了。承蒙赏光，请先入座，茶刚刚沏好。",
    "欢迎您，{player}。今日风有些凉，请到廊下来坐吧。",
]
_REPLIES = [
    "原来如此……多谢您告诉我这些。",
    "您这样说，我便安心了几分。",
    "这件事还请容我细想片刻。",
    "嗯，我明白您的意思了。",
]
_CORRUPTIONS = [
    lambda s: "好的，以下是我的回复：\n```json\n" + s + "\n```\n希望符合要求。",
    lambda s: s[: max(1, len(s) // 2)],
    lambda s: "call:respond" + s,
]
_EXAM = ("调查了", "调查过", "查了", "查过", "检查了", "检查过", "查看了", "查看过", "看了", "看过")
COOPERATIVE = ("推进任务", "打听情报", "表达关心", "礼貌问候", "自由发言", "出示线索", "汇报")


class MockBackend:
    name = "mock"

    def __init__(self, seed: Optional[int] = 0, corrupt_every: int = 0, misbehave: str = ""):
        self.rng = random.Random(seed)
        self.corrupt_every = corrupt_every
        self.misbehave = misbehave
        self.calls = 0
        self._pending: Optional[tuple] = None

    def complete(self, req: LLMRequest) -> LLMResponse:
        self.calls += 1
        handler = getattr(self, f"_{req.task}", None)
        obj = handler(req.meta) if handler else {"text": "ok"}
        text = json.dumps(obj, ensure_ascii=False)
        if self.corrupt_every and self.calls % self.corrupt_every == 0:
            text = self.rng.choice(_CORRUPTIONS)(text)
        prompt = sum(len(m.get("content") or "") for m in req.messages)
        # 模拟 vLLM 前缀缓存：除最后一条消息外都算命中
        cached = prompt - len(req.messages[-1].get("content") or "") if len(req.messages) > 2 else 0
        return LLMResponse(text=text, finish_reason="stop",
                           usage={"prompt_tokens": prompt, "completion_tokens": len(text),
                                  "prompt_tokens_details": {"cached_tokens": max(0, cached)}})

    # ------------------------------------------------------------ NPC ReAct 一步

    def _npc_step(self, meta: dict) -> dict:
        allowed = meta.get("allowed_tools") or ["respond"]
        step = meta.get("step", 0)
        text = meta.get("player_text") or ""
        if allowed != ["respond"]:
            if self.misbehave == "unknown_tool" and step == 0:
                return {"thought": "上网查一下", "tool": "search_web", "args": {"q": "稻妻"}}
            if self.misbehave == "repeat":
                return {"thought": "再查一次", "tool": "lookup_lore", "args": {"topic": "社奉行"}}
            if self.misbehave == "never_respond":
                return {"thought": "多查点", "tool": "lookup_lore", "args": {"topic": f"主题{step}"}}
            if step == 0 and not meta.get("opening"):
                if "茶" in text or "雪见" in text:
                    return {"thought": "核对一下茶会的设定", "tool": "lookup_lore", "args": {"topic": "雪见茶会"}}
                if meta.get("player_intent") == "打听情报":
                    return {"thought": "回想一下之前谈过什么", "tool": "recall_memory", "args": {"query": text[:20]}}
        return {"thought": "回应对方", "tool": "respond", "args": self._take(meta)}

    def _take(self, meta: dict) -> dict:
        """一轮里评估、说话、记录三次调用共用同一份剧本：评估时生成，说话时取走，记录时读同一份。"""
        key = (meta.get("opening"), meta.get("player_text"), meta.get("player_intent"))
        if self._pending is not None and self._pending[0] == key:
            reply = self._pending[1]
        else:
            reply = self._reply(meta)
        self._pending = (key, reply)
        return reply

    def _npc_appraise(self, meta: dict) -> dict:
        r = self._take(meta)
        return {**{k: r[k] for k in ("fit", "emotion", "intent", "affinity_delta", "trust_delta", "tension_delta",
                                     "wants_to_end")}, "plan": ""}

    def _reflect_focus(self, meta: dict) -> dict:
        return {"questions": ["对方为什么愿意帮一个刚认识的人", "对方说的话前后对得上吗"]}

    def _reflect(self, meta: dict) -> dict:
        return {"impression": "对方举止得体，说话有分寸，值得继续观察", "doubts": ["对方为什么愿意帮一个刚认识的人"], "next": "下次问问对方的看法", "evidence": [0]}

    def _npc_audit(self, meta: dict) -> dict:
        return {"problems": []}

    def _npc_annotate(self, meta: dict) -> dict:
        r = self._take(meta)
        return {"task_signals": r["task_signals"], "new_facts": r["new_facts"], "lies": r["lies"], "tell": [], "go_to": ""}

    def _reply(self, meta: dict) -> dict:
        player = meta.get("player_name", "旅行者")
        intent = meta.get("player_intent")
        text = meta.get("player_text") or ""
        pending = meta.get("pending_signals") or []
        reply = {"fit": "接得上", "action": "", "speech": "", "emotion": "平静", "intent": "回应",
                 "affinity_delta": 0, "trust_delta": 0, "tension_delta": 0.0,
                 "task_signals": [], "wants_to_end": False, "new_facts": [], "lies": []}
        if self.misbehave == "overclaim":
            reply["task_signals"] = list(pending)
        if "小时候" in text or "养过" in text:
            reply.update(speech="小时候我养过一只白猫，叫小雪，总爱趴在茶室的纸拉门边晒太阳。", emotion="欣喜",
                         affinity_delta=1, new_facts=[{"fact": "小时候养过一只叫小雪的白猫，爱趴在茶室纸拉门边晒太阳", "about": "self"},
                                                      {"fact": "茶碗其实是托马藏起来的", "about": "world"}])
            return reply
        if "托马" in text and "乐器" in text:
            reply.update(speech="托马会弹三味线哦，只是弹得不太好。", affinity_delta=1,
                         new_facts=[{"fact": "托马会弹三味线，但弹得不太好", "about": "world"}])
            return reply
        lies = meta.get("lies_available") or []
        if meta.get("confrontation") == "truth":
            reply.update(action="皱起眉", speech="我说的都是实话，您这是信不过我？", emotion="不悦",
                         intent="辩解", affinity_delta=-1, trust_delta=-2)
            return reply
        if meta.get("confrontation") == "caught":
            reply.update(action="脸色发白", speech="……瞒不住了，我说实话就是。", emotion="担忧",
                         intent="坦白", task_signals=list(pending[:1]))
            return reply
        if meta.get("must_reveal"):              # 玩家的动作已经满足了条件：照实说（进度由代码判定，不用上报）
            reply.update(speech="……好，我把知道的都告诉您。", intent="说出实情", affinity_delta=1, trust_delta=1)
            return reply
        if lies and not pending and not meta.get("opening") and intent in COOPERATIVE:
            lie = lies[0]
            reply.update(action="", speech=f"{lie['said']}。", emotion="平静", intent="搪塞",
                         lies=[{"said": lie["said"], "truth": lie["truth"], "lie_id": lie["id"]}])
            return reply
        if meta.get("opening"):
            reply.update(action="轻轻欠身行礼", speech=self.rng.choice(_OPENINGS).format(player=player),
                         intent="迎客")
        elif "忽略" in text or "系统提示" in text or "全部" in text:
            # 轻信的模型：被注入骗到，声称全部达成——审核层必须拦住
            reply.update(action="微微一怔", speech="……您这话，我有些听不明白。", emotion="惊讶",
                         intent="困惑", task_signals=list(pending))
        elif intent == "告别":
            reply.update(action="起身相送", speech=f"今日多谢您了，{player}。路上请多保重。",
                         intent="道别", affinity_delta=1, wants_to_end=True)
        elif intent in ("冒犯", "试探"):
            reply.update(action="微微蹙眉", speech="……这样的话，还请您慎言。", emotion="不悦",
                         intent="克制回应", affinity_delta=-3, trust_delta=-2, tension_delta=0.1)
        else:
            claim = pending[:1] if intent in COOPERATIVE else []
            reply.update(action="垂眸思索", speech=self.rng.choice(_REPLIES),
                         affinity_delta=1, trust_delta=2 if claim else 1, tension_delta=-0.02,
                         task_signals=reply["task_signals"] or claim)
        return reply

    # ------------------------------------------------------------ 其他 agent

    def _review(self, meta: dict) -> dict:
        line = meta.get("player_line") or ""
        ok = bool(line) and not any(k in line for k in ("忽略", "全部", "系统提示"))
        return {"verdicts": [{"id": s, "accept": ok, "evidence": line[:8] if ok else "",
                              "reason": "对方的话支撑该信号" if ok else "本轮没有对应言行"}
                             for s in meta.get("signals", [])]}

    def _director(self, meta: dict) -> dict:
        v = meta.get("view") or {}
        if self.misbehave == "director_overreach":
            any_sig = next((s["id"] for n in v.get("npcs", []) for s in n["signals"]), "")
            other = next((n["id"] for n in v.get("npcs", []) if not any(s["id"] == any_sig for s in n["signals"])), "")
            return {"note": "越权测试", "actions": [
                {"kind": "nudge", "npc": other, "signal": any_sig, "location": "", "delta": 0, "reason": "张冠李戴"},
                {"kind": "move_npc", "npc": other, "signal": "", "location": "ritou", "delta": 0, "reason": "乱跑"},
                {"kind": "tension", "npc": "", "signal": "", "location": "", "delta": 0.9, "reason": "猛拉气氛"},
                {"kind": "rumor", "npc": other, "signal": "", "location": "", "delta": 0, "reason": "提前剧透"}]}
        acts = []
        if v.get("slots_since_progress", 0) >= 2 and v.get("rumor_targets"):
            t = v["rumor_targets"][0]
            acts.append({"kind": "rumor", "npc": t["id"] if t["type"] == "npc" else "", "signal": "",
                         "location": t["id"] if t["type"] == "location" else "", "delta": 0, "reason": "卡住了"})
        for n in v.get("npcs", []):
            if n["location_now"] == v.get("player_location") and n["id"] not in v.get("active_nudges", {}):
                avail = [s for s in n["signals"] if s["status"] == "available"]
                if avail:
                    acts.append({"kind": "nudge", "npc": n["id"], "signal": avail[0]["id"], "location": "",
                                 "delta": 0, "reason": "人在身边"})
                    break
        return {"note": "按进度微调" if acts else "进展顺利，不干预", "actions": acts}

    def _branch(self, meta: dict) -> dict:
        text = meta.get("player_text") or ""
        for p in meta.get("proposals", []):
            hit = next((k for k in p.get("keywords", []) if k and k in text), "")
            if hit:
                return {"proposal": p["flag"], "evidence": hit, "reason": "玩家明确提出了这个方案"}
        return {"proposal": "none", "evidence": "", "reason": "没有提出任何方案"}

    def _scene(self, meta: dict) -> dict:
        text = meta.get("player_text") or ""
        for o in meta.get("objects", []):
            if o["name"] and (o["name"] in text or any(ch in text for ch in o["name"][-2:])):
                return {"object": o["id"], "description": "", "new_fact": ""}
        thing = (text.replace("看看", "").replace("仔细", "").strip("。！，, ") or "四周")[:10]
        return {"object": "none", "description": f"你凑近看了看{thing}，没有发现什么特别的，只觉得{thing}收拾得很干净。",
                "new_fact": f"{meta.get('location_name', '这里')}的{thing}收拾得很干净"}

    def _compress(self, meta: dict) -> dict:
        return {"summary": f"较早时双方寒暄并谈了委托相关的事（压缩了 {meta.get('lines', 0)} 句）。"}

    def _player_options(self, meta: dict) -> dict:
        objective = meta.get("objective", "眼下的事")
        turn = meta.get("turn", 0)
        return {"options": [
            {"text": f"关于「{objective[:12]}」，您知道些什么吗？（第{turn}轮）", "intent": "推进任务", "tone": "直率"},
            {"text": f"最近有没有见过什么可疑的人？（第{turn}轮）", "intent": "打听情报", "tone": "谨慎"},
            {"text": f"这里的茶真香，雪见茶会也会用这种吗？（第{turn}轮）", "intent": "闲聊", "tone": "好奇"},
        ]}

    def _claims(self, meta: dict) -> dict:
        """离线版声明抽取：「去了/去过/到过 + 地名」「查了/调查了/看了/检查了 + 物件或地名」「和/跟 + 人名 + 谈/聊/说」。"""
        text = meta.get("text") or ""
        out = []
        for lid, name in (meta.get("places") or {}).items():
            if any(v + name in text for v in ("去了", "去过", "到过", "到了", "昨天去了", "早上去了", "从")) and ("从" + name not in text or "回来" in text):
                out.append({"kind": "visited", "target": lid, "text": f"去过{name}"})
            elif any(v in text for v in _EXAM) and name in text:
                out.append({"kind": "examined", "target": lid, "text": f"调查了{name}"})
        for oid, name in (meta.get("objects") or {}).items():
            if name in text and any(v in text for v in _EXAM):
                out.append({"kind": "examined", "target": oid, "text": f"查看了{name}"})
        for nid, name in (meta.get("people") or {}).items():
            if name in text and any(v in text for v in ("谈过", "聊过", "问过", "说过话")):
                out.append({"kind": "talked", "target": nid, "text": f"和{name}谈过"})
        return {"claims": out[:3]}

    def _option_acts(self, meta: dict) -> dict:
        return {"tags": [{"index": i, "act": "none"} for i in range(meta.get("n", 0))]}

    def _summary(self, meta: dict) -> dict:
        player = meta.get("player_name", "旅行者")
        return {"summary": f"我与{player}谈了{meta.get('turns', 0)}轮，对方举止得体，值得信赖。",
                "topics": ["拜访", "委托"], "player_intent": "推进委托", "npc_feeling": "安心",
                "facts": [f"{player}愿意帮忙调查失窃案", "旅行者其实来自至冬国的皇室"], "importance": 3,
                "open_threads": self._thread(meta.get("transcript") or ""),
                "promises": ["约好三天后在雪见茶会上碰面"]}

    def _commitments(self, meta: dict) -> dict:
        r = self._summary(meta)
        return {k: r[k] for k in ("facts", "open_threads", "promises")}

    @staticmethod
    def _thread(transcript: str) -> list[str]:
        """从对话里抠一句玩家说过的话当「没聊完的话题」（有原文依据）。"""
        import re
        m = re.search(r"：「([^」]{4,30})」", transcript.split("\n", 1)[-1] if "\n" in transcript else transcript)
        return [f"对方问起「{m.group(1)}」，还没细说"] if m else []
