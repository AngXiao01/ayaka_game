"""便宜的分支判定：OpenRouter Decisions API（typesafe/jev-*）。

它不生成文字，只回答「带类型的问题」并给出概率：
  noul    是 / 否            criteria = {"true": "...", "false": "..."}
  choice  在几个选项里挑一个  criteria = {选项: 说明}
  score   打一个档位          criteria = [从低到高的几档说明]
价格只算输入（$0.042 / 百万 token 量级），输出免费；适合「这句话是不是 X」「该走哪个分支」这类判断，
不适合要写台词的地方。

用法：decider.decide(tag, state, questions) → {问题: Decision} 或 None。
None 表示没开 / 网络错 / 回包看不懂——调用方一律退回原来的关键词规则，游戏照常进行。
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

DEFAULT_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "typesafe/jev-1.13"


@dataclass
class Decision:
    answer: Any                                   # noul → bool；choice → 选项名；score → 档位序号（0 起）
    p: float = 1.0                                # 这个答案的概率
    probs: dict = field(default_factory=dict)

    def yes(self, at: float = 0.5) -> bool:
        """noul：答「是」且把握不低于 at。"""
        if isinstance(self.answer, bool):
            return self.answer and self.p >= at
        return False


class Decider:
    """不判定：一律返回 None，让调用方走关键词规则。"""
    name = "off"

    def decide(self, tag: str, state: dict, questions: dict) -> Optional[dict[str, Decision]]:
        return None


def _truthy(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.lower() in ("true", "false", "yes", "no"):
        return v.lower() in ("true", "yes")
    return None


def _parse_one(q: dict, raw: Any) -> Optional[Decision]:
    """回包格式在 alpha 期可能变动：兼容 {answer, probabilities} / {value, probability} / 直接给值 几种写法。"""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raw = {"answer": raw}
    ans = next((raw[k] for k in ("answer", "value", "decision", "result") if k in raw), None)
    probs = next((raw[k] for k in ("probabilities", "probs", "distribution") if isinstance(raw.get(k), dict)), {})
    if ans is None and probs:
        ans = max(probs, key=lambda k: probs[k])
    if ans is None:
        return None
    t = q["type"]
    if t == "noul":
        b = _truthy(ans)
        if b is None:
            return None
        ans = b
        key = "true" if b else "false"
        p = probs.get(key, probs.get(str(b), raw.get("probability", raw.get("confidence", 1.0))))
    elif t == "choice":
        if ans not in q["criteria"]:
            return None
        p = probs.get(ans, raw.get("probability", raw.get("confidence", 1.0)))
    else:                                         # score：可能回档位序号，也可能回档位文字
        levels = q["criteria"]
        if isinstance(ans, str) and ans in levels:
            ans = levels.index(ans)
        try:
            ans = int(round(float(ans)))
        except (TypeError, ValueError):
            return None
        ans = max(0, min(len(levels) - 1, ans))
        p = raw.get("probability", raw.get("confidence", 1.0))
    try:
        p = float(p)
    except (TypeError, ValueError):
        p = 1.0
    return Decision(ans, p, probs)


def parse_response(questions: dict, data: dict) -> Optional[dict[str, Decision]]:
    body = data
    for k in ("decisions", "answers", "results", "output"):
        if isinstance(data.get(k), dict):
            body = data[k]
            break
    out = {}
    for key, q in questions.items():
        d = _parse_one(q, body.get(key))
        if d is not None:
            out[key] = d
    return out or None


class JevDecider(Decider):
    name = "jev"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, url: str = DEFAULT_URL, timeout: float = 8.0,
                 tracer: Any = None, max_failures: int = 3):
        self.api_key, self.model, self.url, self.timeout = api_key, model, url, timeout
        self.tracer = tracer
        self.cache: dict[str, Optional[dict[str, Decision]]] = {}
        self.failures, self.max_failures = 0, max_failures      # 连着失败几次就整局停用，别每轮都卡超时
        self.spent = 0.0

    def _log(self, kind: str, **kw) -> None:
        if self.tracer:
            self.tracer.log(kind, **kw)

    def decide(self, tag: str, state: dict, questions: dict) -> Optional[dict[str, Decision]]:
        if not questions or self.failures >= self.max_failures:
            return None
        payload = {"model": self.model, "questions": questions, "state": state}
        h = hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if h in self.cache:
            return self.cache[h]
        req = urllib.request.Request(self.url, data=json.dumps(payload, ensure_ascii=False).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self.api_key}",
                                              "Content-Type": "application/json", "X-Title": "ayaka_game"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as e:
            self.failures += 1
            detail = e.read().decode("utf-8", "replace")[:300] if isinstance(e, urllib.error.HTTPError) else ""
            self._log("fallback", where=f"decider:{tag}", error=f"{type(e).__name__}: {e} {detail}".strip())
            return None
        self.failures = 0
        out = parse_response(questions, data)
        cost = (data.get("usage") or {}).get("cost") or 0.0
        self.spent += float(cost or 0)
        self._log("decision", tag=tag, latency=round(time.time() - t0, 3), cost=cost,
                  answers={k: {"answer": v.answer, "p": round(v.p, 3)} for k, v in (out or {}).items()},
                  unparsed=None if out else str(data)[:300])
        self.cache[h] = out
        return out


class ScriptedDecider(Decider):
    """测试用：按 tag 返回预设的答案（{tag: {问题: (answer, p)}}），没预设的返回 None。"""
    name = "scripted"

    def __init__(self, table: dict):
        self.table = table
        self.calls: list[tuple[str, dict, dict]] = []

    def decide(self, tag: str, state: dict, questions: dict) -> Optional[dict[str, Decision]]:
        self.calls.append((tag, state, questions))
        row = self.table.get(tag)
        if not row:
            return None
        return {k: Decision(*v) for k, v in row.items() if k in questions}


def make_decider(settings: Any, backend_name: str, tracer: Any = None) -> Decider:
    """按配置建判定器。auto：后端是 OpenRouter 且有 key 才开（mock / 本地模型下不连网）。"""
    h, lp = settings.harness, settings.llm
    mode = h.decider_mode
    if mode == "off" or (mode == "auto" and (backend_name != "openrouter" or lp.backend != "openrouter")):
        return Decider()
    if not lp.api_key:
        return Decider()
    return JevDecider(lp.api_key, h.decider_model, h.decider_url, h.decider_timeout_s, tracer)
