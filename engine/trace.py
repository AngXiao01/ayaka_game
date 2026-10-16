"""结构化 trace：每个事件一行 JSONL，带层级 span（session / turn / agent / step）。

出问题时按 span 定位到具体某一次见面、某一轮、某个 agent 的某一步，
再用 tools/replay.py 把那次 LLM 请求原样重放，而不是从一大坨对话文本里人肉翻。

事件：game_start / session_start / llm / tool_call / loop_end / guard / review / compress / director / branch / scene /
      fallback / budget / turn / session_end
"""
from __future__ import annotations

import json
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from llm.base import LLMRequest, LLMResponse

_SECRET_RES = [re.compile(r"Bearer\s+[A-Za-z0-9._\-]+"), re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}")]


class Tracer:
    def __init__(self, path: Optional[Path] = None, secrets: tuple[str, ...] = ()):
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex[:8]
        self.secrets = tuple(s for s in secrets if s and len(s) >= 6)
        self.span_ctx: dict[str, Any] = {}
        self.seq = 0
        self.stats: dict[str, Any] = {
            "llm_calls": 0, "llm_errors": 0, "json_repairs": 0, "fallbacks": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "latency_s": 0.0, "cost_usd": 0.0,
            "tool_calls": 0, "tool_errors": 0, "tool_cache_hits": 0,
            "loop_stops": {}, "forced_finishes": 0,
            "signals_claimed": 0, "signals_accepted": 0, "signals_rejected": 0,
            "injection_flags": 0, "compressions": 0, "canon_kept": 0, "canon_dropped": 0, "llm_latencies": [],
            "director_runs": 0, "director_applied": 0, "director_rejected": 0, "branch_proposals": 0,
            "scene_looks": 0,
        }

    # ------------------------------------------------------------ span

    @contextmanager
    def span(self, **kv) -> Iterator[None]:
        old = dict(self.span_ctx)
        self.span_ctx.update(kv)
        try:
            yield
        finally:
            self.span_ctx = old

    # ------------------------------------------------------------ 写入

    def _redact(self, obj: Any) -> Any:
        if isinstance(obj, str):
            for rx in _SECRET_RES:
                obj = rx.sub("***", obj)
            for s in self.secrets:
                obj = obj.replace(s, "***")
            return obj
        if isinstance(obj, dict):
            return {k: self._redact(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._redact(v) for v in obj]
        return obj

    def _write(self, rec: dict) -> None:
        self.seq += 1
        if self.path:
            rec = {"seq": self.seq, "ts": round(time.time(), 3), "run": self.run_id,
                   "span": dict(self.span_ctx), **rec}
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(self._redact(rec), ensure_ascii=False) + "\n")

    def log(self, event: str, /, **data) -> None:
        st = self.stats
        if event == "fallback":
            st["fallbacks"] += 1
        elif event == "tool_call" and not data.get("terminal"):
            st["tool_calls"] += 1
            st["tool_errors"] += 0 if data.get("ok") else 1
            st["tool_cache_hits"] += 1 if data.get("cached") else 0
        elif event == "loop_end":
            st["loop_stops"][data["stop"]] = st["loop_stops"].get(data["stop"], 0) + 1
            st["forced_finishes"] += 1 if data.get("forced_by") else 0
        elif event == "review":
            st["signals_claimed"] += len(data.get("claimed", []))
            st["signals_accepted"] += len(data.get("accepted", []))
            st["signals_rejected"] += len(data.get("rejected", []))
        elif event == "guard" and data.get("flags"):
            st["injection_flags"] += 1
        elif event == "compress":
            st["compressions"] += 1
        elif event == "director":
            st["director_runs"] += 1
            st["director_applied"] += len(data.get("applied", []))
            st["director_rejected"] += len(data.get("rejected", []))
        elif event == "branch" and data.get("applied"):
            st["branch_proposals"] += 1
        elif event == "scene":
            st["scene_looks"] += 1
        elif event == "canon_write":
            st["canon_kept"] += len(data.get("kept", []))
            st["canon_dropped"] += len(data.get("dropped", []))
        self._write({**data, "event": event})

    def log_llm(self, req: LLMRequest, resp: Optional[LLMResponse], attempt: int, ok: bool,
                error: Optional[str] = None) -> None:
        st = self.stats
        st["llm_calls"] += 1
        if attempt > 0:
            st["json_repairs"] += 1
        cached = 0
        if resp is None:
            st["llm_errors"] += 1
        else:
            u = resp.usage or {}
            cached = int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
            st["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
            st["completion_tokens"] += int(u.get("completion_tokens") or 0)
            st["cached_tokens"] += cached
            st["cost_usd"] += float(u.get("cost") or 0)          # OpenRouter 每次响应都带实际扣费；vLLM / mock 没有
            st["latency_s"] = round(st["latency_s"] + resp.latency_s, 3)
            st["llm_latencies"].append(resp.latency_s)
        self._write({
            "event": "llm", "task": req.task, "attempt": attempt, "ok": ok, "error": error,
            "temperature": req.temperature, "max_tokens": req.max_tokens,
            "schema_name": req.schema_name if req.json_schema else None,
            "json_schema": req.json_schema,
            "messages": req.messages,
            "output": resp.text if resp else None,
            "reasoning": (resp.reasoning or None) if resp else None,
            "finish_reason": resp.finish_reason if resp else None,
            "usage": resp.usage if resp else None,
            "cached_tokens": cached,
            "latency_s": resp.latency_s if resp else None,
        })

    # ------------------------------------------------------------ 汇总

    def summary(self) -> dict:
        st = dict(self.stats)
        lat = sorted(st.pop("llm_latencies"))
        pick = (lambda q: lat[min(len(lat) - 1, int(q * len(lat)))]) if lat else (lambda q: 0.0)
        st["latency_p50"] = round(pick(0.5), 3)
        st["latency_p95"] = round(pick(0.95), 3)
        st["cache_hit_ratio"] = round(st["cached_tokens"] / st["prompt_tokens"], 3) if st["prompt_tokens"] else 0.0
        st["cost_usd"] = round(st["cost_usd"], 4)
        return st
