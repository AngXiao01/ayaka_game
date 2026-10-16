from models.memory import MemoryItem, MemoryStore
from models.relationship import Relationship


def test_memory_persist_and_retrieve(tmp_path):
    p = tmp_path / "m.jsonl"
    s = MemoryStore(p)
    s.add(MemoryItem(text="旅行者喜欢吃三彩团子", world_time=1, importance=2))
    s.add(MemoryItem(text="我们谈到了伪信与志村屋", world_time=5, importance=4, topics=["伪信"]))
    p.open("a").write("坏掉的一行\n")
    s2 = MemoryStore(p)
    assert len(s2.items) == 2
    top = s2.retrieve("志村屋的伪信", k=1)
    assert top[0].text.startswith("我们谈到了伪信")


def test_relationship_clamp_and_tier():
    r = Relationship(affinity=95, trust=3)
    r.apply(10, -10)
    assert (r.affinity, r.trust) == (100, 0)
    assert r.tier()[0] == "知己"
    assert Relationship(affinity=10).tier()[0] == "疏远"


def test_dotenv_inline_comments(tmp_path, monkeypatch):
    from config.settings import load_dotenv
    for k in ("A_INT", "A_Q", "A_JSON"):
        monkeypatch.delenv(k, raising=False)
    p = tmp_path / ".env"
    p.write_text('A_INT=2      # 注释\nA_Q="a # b"  # 注释\nA_JSON={"enable_thinking": false}\n', encoding="utf-8")
    load_dotenv(p)
    import os
    assert os.environ["A_INT"] == "2" and os.environ["A_Q"] == "a # b"
    assert os.environ["A_JSON"] == '{"enable_thinking": false}'
