"""锁着的事被问到时的几种躲法：同样是“现在不说”，换着腔调；底线不变。"""
import copy
import json
from pathlib import Path

from core.god_env import LOCK_STYLES
from core.story_lint import lint
from models.personality import Personality
from tests.helpers import QUEST, make_god
from tests.test_physical import settings
from tools import lint_story
from tools import preview_prompt as pp


def locked_line(npc, style=None, clues=("c_case", "c_shops")):
    quest = copy.deepcopy(QUEST)
    if style is not None:
        for s in quest.signals:
            if s.npc == npc:
                s.lock_style = style
    g = make_god({"act_id": "act2", "clues": list(clues)})
    g.quest = quest
    g.begin_conversation(npc)
    persona = Personality.load_all(Path("tests/fixtures/yuebai/npcs"))[npc]
    return g.npc_task_detail(persona, 30)["locked_info"]


def test_each_style_changes_the_wording_but_keeps_the_floor():
    texts = {st: locked_line("soji", st) for st in LOCK_STYLES}
    assert len({tuple(v) for v in texts.values()}) == len(LOCK_STYLES)            # 四种说法互不相同
    for st, lines in texts.items():
        assert lines and all("现在不说，因为" in l for l in lines)
        assert all("不要把内容漏出来" in l or "不要透露内容" in l or "不要岔开话题" in l for l in lines), st
        assert all("装作不知道" in l or "答非所问" in l or "不要透露内容本身" in l for l in lines), st


def test_unset_style_is_the_old_direct_wording():
    assert locked_line("aen") == locked_line("aen", "direct")
    assert LOCK_STYLES["direct"] in locked_line("soji", "direct")[0]


def test_real_content_gives_different_characters_different_dodges():
    styles = {s.npc: s.lock_style for s in QUEST.signals if s.lock_style}
    assert styles == {"yohei": "bargain", "soji": "hesitate", "kokichi": "deflect", "heizou": "bargain", "fox": "deflect"}
    assert "欲言又止" in locked_line("soji")[0]                                      # 宗次怕被笑话：支吾
    assert "开口的条件" in locked_line("yohei")[0]                                  # 与平要凭据才肯带路：谈条件


def test_the_style_reaches_the_prompt_the_npc_sees(tmp_path):
    s = settings(tmp_path)
    f = tmp_path / "data/saves/c.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"world": {"act_id": "act2", "phase_idx": 1, "location": "ritou", "clues": ["c_case", "c_shops"]}}))
    pv = pp.capture("c", "soji", say="那晚海边有没有什么怪事？", intent="打听情报", settings=s)
    state = pv.requests[0].messages[-1]["content"]
    assert "欲言又止" in state


def test_unknown_style_is_a_lint_error():
    q, w, p = (copy.deepcopy(x) for x in lint_story.load())
    q.signals[0].lock_style = "shout"
    assert any(f.code == "bad_ref" and "shout" in f.msg for f in lint(q, w, p))
