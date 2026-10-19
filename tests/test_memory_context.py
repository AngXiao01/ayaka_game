from pathlib import Path

from core.working_memory import WorkingMemory
from llm.base import LLMError
from models.knowledge import LoreBase
from models.memory import MemoryItem, MemoryStore
from models.retrieval import BM25, coverage

ROOT = Path(__file__).resolve().parent.parent


def test_bm25_ranks_relevant_first():
    bm = BM25(["社奉行掌管祭祀与文化", "花见坂是热闹的商业街", "托马是家政官"])
    assert bm.top("花见坂的商铺", 1)[0][0] == 1
    assert bm.top("完全无关的词汇xyz", 3) == []


def test_lore_lookup():
    lb = LoreBase.load(ROOT / "tests/fixtures/yuebai/lore")
    assert lb.search("眼狩令现在还有吗", 1)[0].title.startswith("眼狩令")


def test_memory_dedupe_and_atomic_rewrite(tmp_path):
    s = MemoryStore(tmp_path / "m.jsonl")
    s.add(MemoryItem(text="旅行者答应帮忙调查伪信", importance=2))
    item, merged = s.add(MemoryItem(text="旅行者答应帮忙调查伪信。", importance=4))
    assert merged and item.hits == 2 and item.importance == 4 and len(s.items) == 1
    assert len(MemoryStore(tmp_path / "m.jsonl").items) == 1


def test_fact_grounding():
    transcript = "旅行者：「我愿意帮忙调查伪信」"
    assert coverage("旅行者愿意帮忙调查伪信", transcript) >= 0.5
    assert coverage("旅行者来自至冬国皇室", transcript) < 0.5


def entries(n):
    out = []
    for i in range(n):
        out += [{"speaker": "player", "text": f"第{i}句", "intent": "闲聊"},
                {"speaker": "npc", "text": f"回应{i}", "reply": {"action": ""}}]
    return out


def test_working_memory_compresses_oldest():
    calls = []
    wm = WorkingMemory(4, "旅行者", "绫华", lambda prev, lines: calls.append(lines) or "提要", "llm")
    for e in entries(5):
        wm.add(e)
    assert wm.maybe_compress()
    assert wm.summary == "提要" and wm.player_turns() == 2 and wm.entries[0]["speaker"] == "player"
    assert calls and "第0句" in calls[0][0]
    assert not wm.maybe_compress()


def test_working_memory_extractive_fallback():
    def boom(prev, lines):
        raise LLMError("down")
    wm = WorkingMemory(2, "旅行者", "绫华", boom, "llm")
    for e in entries(3):
        wm.add(e)
    wm.maybe_compress()
    assert "第0句" in wm.summary
