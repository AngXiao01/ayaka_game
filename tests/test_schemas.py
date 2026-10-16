import pytest

from llm.schemas import ValidationError, validate_npc_reply, validate_player_options, validate_summary


def test_npc_reply_clamps_and_filters():
    r = validate_npc_reply({"speech": "「您好」", "emotion": "狂喜", "affinity_delta": 99, "trust_delta": "-2",
                            "tension_delta": 3, "task_signals": ["a", "zzz", "a"], "wants_to_end": "false"},
                           ["a", "b"])
    assert r["speech"] == "您好"
    assert r["emotion"] == "平静"
    assert (r["affinity_delta"], r["trust_delta"], r["tension_delta"]) == (5, -2, 0.2)
    assert r["task_signals"] == ["a"]
    assert r["wants_to_end"] is False


def test_npc_reply_requires_speech():
    with pytest.raises(ValidationError):
        validate_npc_reply({"action": "沉默"}, [])


def test_options_dedupe_and_bare_list():
    out = validate_player_options([{"text": "A", "intent": "推进任务"}, {"text": "A"}, "B"])
    assert [o["text"] for o in out] == ["A", "B"]
    assert out[1]["intent"] == "闲聊"
    assert len(validate_player_options({"options": [{"text": "只有一个"}]})) == 1   # 给几句由模型看情况
    assert validate_player_options({"options": []}) == []                            # 不需要提示时可以一句不给
    with pytest.raises(ValidationError):
        validate_player_options({"answer": "没有 options 字段"})


def test_summary_defaults():
    s = validate_summary({"summary": "记得了", "importance": 9, "facts": "一条事实"})
    assert s["importance"] == 5 and s["facts"] == ["一条事实"]


def test_npc_reply_length_guards():
    from llm.schemas import MAX_SPEECH_CHARS
    with pytest.raises(ValidationError, match="太长"):
        validate_npc_reply({"speech": "啊" * (MAX_SPEECH_CHARS + 1)}, [])
    r = validate_npc_reply({"speech": "嗯。", "action": "微微侧过头，目光落在院中静谧的樱树上，随后又带着一丝试探的意味"}, [])
    assert r["action"] == "" and r["speech"] == "嗯。"
    assert validate_npc_reply({"speech": "嗯。", "action": "递过茶盏"}, [])["action"] == "递过茶盏"
