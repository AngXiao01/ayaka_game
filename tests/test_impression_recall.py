"""对话前用“整体印象”当检索焦点：翻出和印象有关、但玩家这句话没提到的往事。"""
import json

from tests.test_physical import settings
from tools import preview_prompt as pp

REFLECTION = "我对旅行者的看法：他对茶室的高窗格外上心，问得很细；我的疑问：他是不是已经看出了什么？；下次我打算：多留意他看茶室的神情。"


def seed(tmp_path, with_reflection=True):
    s = settings(tmp_path)
    (tmp_path / "data/saves").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/saves/c.json").write_text(json.dumps({"world": {"act_id": "act2", "phase_idx": 1, "location": "kamisato_estate",
                                                                      "clues": ["c_case", "c_shops"]}}))
    mem = tmp_path / "data/memories/c"
    mem.mkdir(parents=True, exist_ok=True)
    fillers = ["托马今天早上烤焦了鱼", "庭院的白砂园每天都要耙一遍", "风铃在廊下响了一下午", "老梅树的枝头落了一层雪",
               "石灯笼夜里会一盏盏点亮", "茶会的采买清单还没有对完"]
    items = [{"text": "旅行者蹲在茶室的高窗下看了很久，还问窗台有多高", "kind": "fact", "importance": 2, "world_time": 1}] + [
        {"text": t, "kind": "fact", "importance": 5, "world_time": 2 + i} for i, t in enumerate(fillers)]       # 更近更重要：普通检索会选它们
    if with_reflection:
        items.append({"text": REFLECTION, "kind": "reflection", "importance": 4, "world_time": 9, "extra": {"focus": ["他看出了什么"]}})
    (mem / "ayaka.jsonl").write_text("".join(json.dumps({"topics": [], "extra": {}, "hits": 1, **it}, ensure_ascii=False) + "\n" for it in items))
    return s


def state(tmp_path, s, say):
    return pp.capture("c", "ayaka", say=say, intent="闲聊", settings=s).requests[0].messages[-1]["content"]


def test_the_overall_impression_pulls_in_a_memory_the_line_never_mentions(tmp_path):
    s = seed(tmp_path)
    block = state(tmp_path, s, "今天的茶点看起来不错。")                                      # 这句话和高窗毫无关系
    section = block[block.index("## 你记得的往事"):]
    assert "蹲在茶室的高窗下看了很久" in section                                              # 印象里说他对高窗上心 → 翻出了这条
    assert sum(t in section.split("\n\n")[0] for t in ("烤焦了鱼", "白砂园", "风铃", "老梅树", "石灯笼", "采买清单")) <= 3   # 普通检索那 3 条之外，只多这一条


def test_without_an_impression_nothing_extra_is_pulled(tmp_path):
    s = seed(tmp_path, with_reflection=False)
    block = state(tmp_path, s, "今天的茶点看起来不错。")
    assert "蹲在茶室的高窗下看了很久" not in block


def test_the_reflection_itself_is_not_returned_as_a_memory(tmp_path):
    from core.player_agent import ScriptedPlayer
    from engine.game_engine import GameEngine
    from llm.mock_backend import MockBackend
    s = seed(tmp_path)
    a = GameEngine(s, MockBackend(seed=0), ScriptedPlayer(["q"]), out=lambda x: None, save_name="c").npcs["ayaka"]
    got = a.recall_by_impression(now=5)
    assert got and all(not g.startswith("我对旅行者的看法") for g in got) and len(got) <= 2


def test_excluded_items_do_not_distort_the_scores(tmp_path):
    """被排除的条目（这里是反思，里面就含着查询的字）不能参与归一化，否则真正相关的往事被压成零头。"""
    from models.memory import MemoryStore
    store = MemoryStore(tmp_path / "data/memories/c/ayaka.jsonl")
    seed(tmp_path)
    store = MemoryStore(tmp_path / "data/memories/c/ayaka.jsonl")
    q = "他对茶室的高窗格外上心，问得很细；我的疑问：他是不是已经看出了什么？"
    assert "蹲在茶室的高窗下看了很久" in store.retrieve(q, k=2, now=0, touch=False, exclude_kinds=("reflection",))[0].text
