"""台词选项可以「顺带做事」：告辞的话说了就走、邀请的话说了就邀——不再一边是台词、一边是另一个按钮。"""
import json

from core.context import TurnContext
from core.player_agent import ScriptedPlayer
from core.player_option_generator import PlayerOptionGenerator
from engine.game_engine import GameEngine
from llm.mock_backend import MockBackend
from tests.test_physical import settings

ACTS = [{"id": "leave", "desc": "结束这次谈话，离开", "kind": "leave", "meta": {}, "tag": "🚪 告辞"},
        {"id": "invite", "desc": "邀对方一起出去走走", "kind": "invite", "meta": {}, "tag": "🚶 邀请同行"}]


class Stub:
    def __init__(self, options):
        self.options = options

    def generate(self, ctx, npc, n):
        return [dict(o) for o in self.options]


def ctx():
    return TurnContext(player_name="旅行者", scene={}, world={}, relationship={}, history=[],
                       player_view={"known_text": "", "actions": ACTS})


def test_line_carries_its_action():
    opts, _ = PlayerOptionGenerator(Stub([
        {"text": "承蒙挂心，那我先去花见坂了。", "act": "leave"},
        {"text": "难得天晴，要不要一起去走走？", "act": "invite"},
        {"text": "这茶真香。", "act": "ask:nope"}])).generate(ctx(), "绫华", [])
    assert [(o.kind, o.meta.get("act")) for o in opts] == [("leave", "leave"), ("invite", "invite"), ("say", None)]
    assert opts[0].meta["line"] and opts[0].meta["act_label"] == "🚪 告辞"


def test_untagged_line_stays_a_line_and_each_act_once():     # 谁在做哪件事由标注员说了算，台词本身不再带意图
    opts, _ = PlayerOptionGenerator(Stub([
        {"text": "那我先告辞了。"},
        {"text": "改天再来叨扰。", "act": "leave"}])).generate(ctx(), "绫华", [])
    assert [o.kind for o in opts] == ["say", "leave"]


def run(tmp_path, options, script, state=None, rels=None):
    save = tmp_path / "data/saves/o.json"
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps({"world": state or {"act_id": "act2", "phase_idx": 1, "clues": ["c_case", "c_shops"]},
                                "relationships": rels or {}}))
    shown = []

    class Spy(ScriptedPlayer):
        def choose(self, opts, c):
            if c is not None:
                shown.append([(o.text, o.kind, bool(o.meta.get("line"))) for o in opts])
            return super().choose(opts, c)

    out = []
    eng = GameEngine(settings(tmp_path), MockBackend(seed=0), Spy(script), out=out.append, save_name="o")
    eng.option_gen.adapter = Stub(options)
    eng.run()
    return eng, shown, out


def test_spoken_farewell_ends_talk_and_replaces_the_button(tmp_path):
    line = {"text": "承蒙挂心，那我先去花见坂看看。", "act": "leave"}
    eng, shown, out = run(tmp_path, [line], [{"kind": "talk", "npc": "ayaka"}, 1, "q"])
    first = shown[0]
    assert first[0] == (line["text"], "leave", True)
    assert not any(t == "🚪 离开" for t, _, _ in first)                 # 台词已经带上了告辞，不再另给按钮
    assert any(line["text"] in x for x in out) and len(shown) == 1        # 说完就走，谈话结束


def test_declined_invite_button_disappears(tmp_path):
    eng, shown, _ = run(tmp_path, [], [{"kind": "talk", "npc": "ayaka"}, {"kind": "invite"}, ("今天天气不错", "自由发言"), "q"],
                        rels={"ayaka": {"affinity": 10, "trust": 10}})
    assert any("邀" in t for t, k, _ in shown[0] if k == "invite")
    assert not any(k == "invite" for _, k, _ in shown[1])                 # 刚被婉拒，这次谈话里不再摆出邀请按钮
