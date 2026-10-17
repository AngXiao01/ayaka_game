"""三类结构化输出的 JSON Schema 与校验。

Schema 只用 type / enum / required 这类各家结构化解码都支持的关键字；
数值范围不写进 schema（部分 grammar 后端会 400），统一在校验时 clamp。
校验宁宽勿严：能纠正的就纠正，只有缺了核心字段才判失败、触发修复重试。
"""
from __future__ import annotations

import re
from typing import Any, Optional

EMOTIONS = ["平静", "欣喜", "羞涩", "担忧", "严肃", "不悦", "感动", "好奇", "惊讶"]
PLAYER_INTENTS = ["礼貌问候", "推进任务", "打听情报", "表达关心", "闲聊", "赞美", "试探", "冒犯", "告别"]
PLAYER_TONES = ["温和", "俏皮", "直率", "深情", "谨慎", "好奇", "调侃"]
FREE_INTENT = "自由发言"
MAX_SPEECH_CHARS = 140
MAX_ACTION_CHARS = 24


class ValidationError(ValueError):
    pass


def _int(v: Any, lo: int, hi: int, default: int = 0) -> int:
    try:
        x = int(round(float(v)))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, x))


def _float(v: Any, lo: float, hi: float, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, x))


def _bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "是")
    return bool(v)


def _str(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _strip_quotes(s: str) -> str:
    s = s.strip()
    for left, right in (("「", "」"), ("“", "”"), ('"', '"'), ("『", "』")):
        if len(s) >= 2 and s.startswith(left) and s.endswith(right) and s.count(left) == 1:
            return s[1:-1].strip()
    return s


def _str_list(v: Any) -> list[str]:
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        return []
    return [_str(x) for x in v if _str(x)]


# ---------------------------------------------------------------- NPC 回复

FITS = ["接得上", "换了话题", "有点跳", "毫不相干"]   # 对方这句话和眼下的谈话、处境接不接得上（NPC 自己判）

def npc_reply_schema(signal_ids: list[str], lie_ids: Optional[list[str]] = None) -> dict:
    item = {"type": "string", "enum": signal_ids} if signal_ids else {"type": "string"}
    return {
        "type": "object",
        "properties": {
            # 放在最前面：先判断对方这句话接不接得上，再决定怎么说（模型按顺序生成，先判后说）
            "fit": {"type": "string", "enum": FITS},
            "action": {"type": "string"},
            "speech": {"type": "string"},
            "emotion": {"type": "string", "enum": EMOTIONS},
            "intent": {"type": "string"},
            "affinity_delta": {"type": "integer"},
            "trust_delta": {"type": "integer"},
            "tension_delta": {"type": "number"},
            "task_signals": {"type": "array", "items": item},
            "wants_to_end": {"type": "boolean"},
            "new_facts": {"type": "array", "items": {
                "type": "object",
                "properties": {"fact": {"type": "string"}, "about": {"type": "string", "enum": ["self", "world"]}},
                "required": ["fact", "about"]}},
            "lies": {"type": "array", "items": {
                "type": "object",
                "properties": {"said": {"type": "string"}, "truth": {"type": "string"},
                               "lie_id": {"type": "string", "enum": list(lie_ids or []) + ["improvised"]}},
                "required": ["said", "truth", "lie_id"]}},
            # 可选：托人带话、说完要去别处（不写就是没有）
            "tell": {"type": "array", "items": {
                "type": "object", "properties": {"to": {"type": "string"}, "text": {"type": "string"}},
                "required": ["to", "text"]}},
            "go_to": {"type": "string"},
        },
        "required": ["fit", "action", "speech", "emotion", "intent", "affinity_delta", "trust_delta",
                     "tension_delta", "task_signals", "wants_to_end", "new_facts", "lies"],
    }


# ---------------------------------------------------------------- NPC 回复的三段式：评估 / 说话 / 记录
# 原来一次调用要同时「判断对方这句话、决定心态、写台词、给自己记账（信号/新设定/谎/带话）」，
# 台词质量被记账字段挤掉。现在拆成三次调用，各管一件事，最后由 validate_npc_reply 合成同一个 NPCReply。

def appraise_schema() -> dict:
    return {"type": "object", "properties": {
        "fit": {"type": "string", "enum": FITS}, "emotion": {"type": "string", "enum": EMOTIONS},
        "intent": {"type": "string"}, "plan": {"type": "string"},
        "affinity_delta": {"type": "integer"}, "trust_delta": {"type": "integer"}, "tension_delta": {"type": "number"},
        "wants_to_end": {"type": "boolean"}},
        "required": ["fit", "emotion", "intent", "plan", "affinity_delta", "trust_delta", "tension_delta", "wants_to_end"]}


def validate_appraisal(obj: Any) -> dict:
    if not isinstance(obj, dict):
        raise ValidationError("顶层必须是 JSON 对象")
    fit, emotion = _str(obj.get("fit")), _str(obj.get("emotion"))
    return {"fit": fit if fit in FITS else FITS[0], "emotion": emotion if emotion in EMOTIONS else "平静",
            "intent": _str(obj.get("intent"))[:40], "plan": _str(obj.get("plan"))[:80],
            "affinity_delta": _int(obj.get("affinity_delta"), -5, 5), "trust_delta": _int(obj.get("trust_delta"), -5, 5),
            "tension_delta": round(_float(obj.get("tension_delta"), -0.2, 0.2), 3),
            "wants_to_end": _bool(obj.get("wants_to_end", False))}


SPEAK_SCHEMA = {"type": "object", "properties": {"action": {"type": "string"}, "speech": {"type": "string"}},
                "required": ["action", "speech"]}


def validate_speak(obj: Any) -> dict:
    """说话这一步只交台词和动作；其余字段即使模型多给了也不收。"""
    if not isinstance(obj, dict):
        raise ValidationError("顶层必须是 JSON 对象")
    speech = _strip_quotes(_str(obj.get("speech") or obj.get("dialogue") or obj.get("text")))
    if not speech:
        raise ValidationError("speech 字段为空，必须写出角色说的话")
    if len(speech) > MAX_SPEECH_CHARS:
        raise ValidationError(f"speech 太长了（{len(speech)} 字），像真人聊天那样压缩到一两句、不超过 {MAX_SPEECH_CHARS} 字")
    speech, action = split_stage_directions(speech, _strip_quotes(_str(obj.get("action"))))
    return {"action": action if len(action) <= MAX_ACTION_CHARS else "", "speech": speech}


def annotate_schema(signal_ids: list[str], lie_ids: Optional[list[str]] = None) -> dict:
    full = npc_reply_schema(signal_ids, lie_ids)["properties"]
    keys = ("task_signals", "new_facts", "lies", "tell", "go_to")
    return {"type": "object", "properties": {k: full[k] for k in keys}, "required": ["task_signals", "new_facts", "lies"]}


def validate_annotation(obj: Any, signal_ids: list[str], lie_ids: Optional[list[str]] = None) -> dict:
    if not isinstance(obj, dict):
        raise ValidationError("顶层必须是 JSON 对象")
    return {"task_signals": [x for x in dict.fromkeys(_str_list(obj.get("task_signals"))) if x in signal_ids],
            "new_facts": _new_facts(obj.get("new_facts")), "lies": _lies(obj.get("lies"), lie_ids or []),
            "tell": _tell(obj.get("tell")), "go_to": _str(obj.get("go_to"))[:20]}


_STAGE_DIR = re.compile(r"[（(]([^）)\n]{1,%d})[）)]\s*" % MAX_ACTION_CHARS)


def split_stage_directions(speech: str, action: str) -> tuple[str, str]:
    """模型常把「（推了推眼镜）」直接写进台词里，action 字段却留空。括号里的动作是结构化信息，不属于台词：
    挪进 action（action 已有就丢掉），台词只留说出口的话。纯机械的整理，不评判动作该不该有。"""
    found = _STAGE_DIR.findall(speech)
    clean = _STAGE_DIR.sub("", speech).strip()
    if not clean:                              # 整句都是括号：当作没说话，原样保留交给上游校验
        return speech, action
    return clean, action or (found[0] if found else "")


def validate_npc_reply(obj: Any, signal_ids: list[str], lie_ids: Optional[list[str]] = None) -> dict:
    if not isinstance(obj, dict):
        raise ValidationError("顶层必须是 JSON 对象")
    speech = _strip_quotes(_str(obj.get("speech") or obj.get("dialogue") or obj.get("text")))
    if not speech:
        raise ValidationError("speech 字段为空，必须写出角色说的话")
    if len(speech) > MAX_SPEECH_CHARS:
        raise ValidationError(f"speech 太长了（{len(speech)} 字），像真人聊天那样压缩到一两句、不超过 {MAX_SPEECH_CHARS} 字")
    action = _strip_quotes(_str(obj.get("action")))
    speech, action = split_stage_directions(speech, action)
    if len(action) > MAX_ACTION_CHARS:      # 冗长的神态描写直接丢掉，不值得为它重写一次
        action = ""
    emotion = _str(obj.get("emotion"))
    signals: list[str] = []
    for s in _str_list(obj.get("task_signals")):
        if s in signal_ids and s not in signals:
            signals.append(s)
    fit = _str(obj.get("fit"))
    return {
        "fit": fit if fit in FITS else FITS[0],
        "action": action,
        "speech": speech,
        "emotion": emotion if emotion in EMOTIONS else "平静",
        "intent": _str(obj.get("intent")),
        "affinity_delta": _int(obj.get("affinity_delta"), -5, 5),
        "trust_delta": _int(obj.get("trust_delta"), -5, 5),
        "tension_delta": round(_float(obj.get("tension_delta"), -0.2, 0.2), 3),
        "task_signals": signals,
        "wants_to_end": _bool(obj.get("wants_to_end", False)),
        "new_facts": _new_facts(obj.get("new_facts")),
        "lies": _lies(obj.get("lies"), lie_ids or []),
        "tell": _tell(obj.get("tell")),
        "go_to": _str(obj.get("go_to"))[:20],
    }


def _tell(v: Any, limit: int = 1) -> list[dict]:
    """托人带话：[{to, text}]，宽容解析，每轮最多一条、每条不超过 60 字。"""
    if isinstance(v, dict):
        v = [v]
    out = []
    for it in v if isinstance(v, list) else []:
        if isinstance(it, dict) and _str(it.get("to")) and _str(it.get("text")):
            out.append({"to": _str(it.get("to"))[:12], "text": _str(it.get("text"))[:60]})
    return out[:limit]


def _lies(v: Any, lie_ids: list[str], limit: int = 2) -> list[dict]:
    """这一轮说的谎：[{said, truth, lie_id}]。lie_id 不认识的一律当临时编的。"""
    if isinstance(v, dict):
        v = [v]
    out = []
    for it in v if isinstance(v, list) else []:
        if not isinstance(it, dict):
            continue
        said, truth = _strip_quotes(_str(it.get("said")))[:80], _str(it.get("truth"))[:80]
        if said and truth:
            lid = _str(it.get("lie_id"))
            out.append({"said": said, "truth": truth, "lie_id": lid if lid in lie_ids else "improvised"})
    return out[:limit]


def _new_facts(v: Any, limit: int = 3) -> list[dict]:
    """即兴编出的新设定：[{fact, about: self|world}]，宽容解析，最多 3 条、每条不超过 60 字。"""
    if isinstance(v, (str, dict)):
        v = [v]
    out = []
    for it in v if isinstance(v, list) else []:
        if isinstance(it, str):
            it = {"fact": it, "about": "self"}
        if not isinstance(it, dict):
            continue
        fact = _str(it.get("fact"))[:60]
        about = _str(it.get("about")) if _str(it.get("about")) in ("self", "world") else "self"
        if fact and fact not in [x["fact"] for x in out]:
            out.append({"fact": fact, "about": about})
    return out[:limit]


# ---------------------------------------------------------------- 玩家选项

PLAYER_OPTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "options": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "intent": {"type": "string", "enum": PLAYER_INTENTS},
                    "tone": {"type": "string", "enum": PLAYER_TONES},
                },
                "required": ["text", "intent", "tone"],
            },
        },
    },
    "required": ["options"],
}


def validate_player_options(obj: Any, max_len: int = 60) -> list[dict]:
    items = obj.get("options") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        raise ValidationError("缺少 options 数组")
    out, seen = [], set()
    for it in items:
        if isinstance(it, str):
            it = {"text": it}
        if not isinstance(it, dict):
            continue
        text = _strip_quotes(_str(it.get("text")))[:max_len]
        if not text or text in seen:
            continue
        seen.add(text)
        intent, tone = _str(it.get("intent")), _str(it.get("tone"))
        out.append({"text": text, "intent": intent if intent in PLAYER_INTENTS else "闲聊",
                    "tone": tone if tone in PLAYER_TONES else "", "act": _str(it.get("act"))})
    return out                  # 0~3 句都可以：不是每一轮都需要替玩家想台词


def act_tag_schema(act_ids: list[str]) -> dict:
    return {"type": "object", "properties": {"tags": {"type": "array", "items": {
        "type": "object", "properties": {"index": {"type": "integer"},
                                          "act": {"type": "string", "enum": list(act_ids) + ["none"]}},
        "required": ["index", "act"]}}}, "required": ["tags"]}


def validate_act_tags(obj: Any, n: int, act_ids: list[str]) -> dict[int, str]:
    """{第几句: 动作编号}。每句最多一件事、每件事最多一句；认不得的编号一律当没有。"""
    items = obj.get("tags") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        raise ValidationError("缺少 tags 数组")
    out: dict[int, str] = {}
    used: set[str] = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        i, act = _int(it.get("index"), 0, max(0, n - 1), -1), _str(it.get("act"))
        if 0 <= i < n and act in act_ids and act not in used and i not in out:
            out[i] = act
            used.add(act)
    return out


# ---------------------------------------------------------------- 对话摘要

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "player_intent": {"type": "string"},
        "npc_feeling": {"type": "string"},
        "importance": {"type": "integer"},
    },
    "required": ["summary", "topics", "player_intent", "npc_feeling", "importance"],
}


def validate_summary(obj: Any) -> dict:
    if not isinstance(obj, dict):
        raise ValidationError("顶层必须是 JSON 对象")
    summary = _str(obj.get("summary"))
    if not summary:
        raise ValidationError("summary 字段为空")
    return {
        "summary": summary,
        "topics": _str_list(obj.get("topics"))[:8],
        "player_intent": _str(obj.get("player_intent")),
        "npc_feeling": _str(obj.get("npc_feeling")),
        "facts": _str_list(obj.get("facts"))[:8],
        "importance": _int(obj.get("importance"), 1, 5, default=3),
        "open_threads": _str_list(obj.get("open_threads"))[:4],     # 交接字段宽容解析：缺了就是 []
        "promises": _str_list(obj.get("promises"))[:4],
    }


def validate_commitments(obj: Any) -> dict:
    if not isinstance(obj, dict):
        raise ValidationError("顶层必须是 JSON 对象")
    return {"facts": _str_list(obj.get("facts"))[:8], "open_threads": _str_list(obj.get("open_threads"))[:4],
            "promises": _str_list(obj.get("promises"))[:4]}


# ---------------------------------------------------------------- 进度审核（Reviewer）

def review_schema(signal_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": signal_ids},
                        "accept": {"type": "boolean"},
                        "evidence": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "accept", "evidence", "reason"],
                },
            },
        },
        "required": ["verdicts"],
    }


def validate_review(obj: Any, signal_ids: list[str]) -> dict[str, dict]:
    items = obj.get("verdicts") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        raise ValidationError("缺少 verdicts 数组")
    out: dict[str, dict] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        sid = _str(it.get("id"))
        if sid in signal_ids and sid not in out:
            out[sid] = {"accept": _bool(it.get("accept")), "evidence": _str(it.get("evidence"))[:80],
                        "reason": _str(it.get("reason"))[:120]}
    if not out:
        raise ValidationError("verdicts 里没有任何一个待审核信号的结论")
    return out


# ---------------------------------------------------------------- 工作记忆压缩

COMPRESS_SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}


def validate_compress(obj: Any) -> str:
    s = _str(obj.get("summary")) if isinstance(obj, dict) else ""
    if not s:
        raise ValidationError("summary 为空")
    return s[:400]


# ---------------------------------------------------------------- 导演（GOD 层的模型部分）

DIRECTOR_KINDS = ["nudge", "rumor", "move_npc", "tension"]


def director_schema(npc_ids: list[str], signal_ids: list[str], location_ids: list[str]) -> dict:
    """枚举都来自代码：模型只能在现有的人、信号、地点里挑，写不出不存在的东西。"""
    return {
        "type": "object",
        "properties": {
            "note": {"type": "string"},
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": DIRECTOR_KINDS},
                        "npc": {"type": "string", "enum": npc_ids + [""]},
                        "signal": {"type": "string", "enum": signal_ids + [""]},
                        "location": {"type": "string", "enum": location_ids + [""]},
                        "delta": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["kind", "npc", "signal", "location", "delta", "reason"],
                },
            },
        },
        "required": ["note", "actions"],
    }


def validate_director(obj: Any, max_actions: int = 2) -> dict:
    """只做格式层面的整理；能不能执行由 core/director.py 按游戏规则判。"""
    if not isinstance(obj, dict):
        raise ValidationError("顶层必须是 JSON 对象")
    items = obj.get("actions")
    if not isinstance(items, list):
        raise ValidationError("缺少 actions 数组（什么都不做就写 []）")
    out = []
    for it in items:
        if not isinstance(it, dict) or _str(it.get("kind")) not in DIRECTOR_KINDS:
            continue
        out.append({"kind": _str(it.get("kind")), "npc": _str(it.get("npc")), "signal": _str(it.get("signal")),
                    "location": _str(it.get("location")), "delta": _float(it.get("delta"), -1.0, 1.0),
                    "reason": _str(it.get("reason"))[:80]})
    return {"note": _str(obj.get("note"))[:120], "actions": out[: max(0, max_actions) + 2]}


# ---------------------------------------------------------------- 分支（tree 层的模型部分）

def branch_schema(flags: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "proposal": {"type": "string", "enum": flags + ["none"]},
            "evidence": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["proposal", "evidence", "reason"],
    }


def validate_branch(obj: Any, flags: list[str]) -> dict:
    if not isinstance(obj, dict):
        raise ValidationError("顶层必须是 JSON 对象")
    p = _str(obj.get("proposal"))
    if p not in flags + ["none"]:
        raise ValidationError(f"proposal 只能是 {flags + ['none']} 之一")
    return {"proposal": p, "evidence": _str(obj.get("evidence"))[:60], "reason": _str(obj.get("reason"))[:100]}


# ---------------------------------------------------------------- 场景（ENV 层的模型部分）

def scene_schema(object_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "object": {"type": "string", "enum": object_ids + ["none"]},
            "description": {"type": "string"},
            "new_fact": {"type": "string"},
        },
        "required": ["object", "description", "new_fact"],
    }


def validate_scene(obj: Any, object_ids: list[str]) -> dict:
    if not isinstance(obj, dict):
        raise ValidationError("顶层必须是 JSON 对象")
    o = _str(obj.get("object")) or "none"
    if o not in object_ids + ["none"]:
        raise ValidationError(f"object 只能是 {object_ids + ['none']} 之一")
    desc = _strip_quotes(_str(obj.get("description")))[:120]
    if o == "none" and not desc:
        raise ValidationError("没有对应物件时 description 不能为空")
    return {"object": o, "description": desc, "new_fact": _str(obj.get("new_fact"))[:60]}
