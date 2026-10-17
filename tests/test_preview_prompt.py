"""提示词预览工具：看到的是真实代码拼出的提示词，且不碰存档、不联网。"""
import json

import pytest

from tests.test_physical import settings
from tools import preview_prompt as pp

AT_ESTATE = {"act_id": "act2", "phase_idx": 0, "location": "kamisato_estate", "clues": ["c_case", "c_shops"]}


def make_save(tmp_path, world=None, name="c"):
    s = settings(tmp_path)
    f = tmp_path / f"data/saves/{name}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"world": world or AT_ESTATE}))
    return s


def tree(root):
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def test_opening_prompt_is_the_real_one_and_nothing_is_written(tmp_path):
    s = make_save(tmp_path)
    before = tree(tmp_path / "data")
    pv = pp.capture("c", "thoma", settings=s)
    assert tree(tmp_path / "data") == before                                      # 存档、记忆、trace 目录都没动
    first = pv.requests[0]
    assert first.task == "npc_step" and not any(r.task == "npc_appraise" for r in pv.requests)    # 开场没有评估这一步
    assert "托马" in first.messages[0]["content"]                                  # 系统消息是托马的人设
    assert "## 时间与场景" in first.messages[-1]["content"] and "## 你此刻的心思" in first.messages[-1]["content"]
    assert pv.context["sections"] and pv.npc_name == "托马"


def test_a_player_line_adds_appraisal_and_shows_up_in_the_prompt(tmp_path):
    s = make_save(tmp_path)
    q = "那天夜里有没有谁来敲过后门？"
    pv = pp.capture("c", "thoma", say=q, intent="打听情报", settings=s)
    tasks = [r.task for r in pv.requests]
    assert "npc_appraise" in tasks and tasks.index("npc_appraise") < tasks.index("npc_step")
    speak = pp._pick(pv, "speak")
    assert len(speak) == 1 and f'<player_input intent="打听情报">{q}</player_input>' in speak[0].messages[-1]["content"]
    assert "你这一轮的心态" in speak[0].messages[-1]["content"]
    text = pp.render(pv, "speak", show_system=False)
    assert "mock" in text and "══ 提示词预览 · 托马" in text and "（已隐藏" in text


def test_find_reports_where_a_term_sits_and_what_is_absent(tmp_path):
    s = make_save(tmp_path)
    pv = pp.capture("c", "thoma", say="那天夜里有没有谁来敲过后门？", intent="打听情报", settings=s)
    got = dict(l.strip().split("：", 1) for l in pp.find_report(pv, ["守夜狐", "托马"]))
    assert got["「守夜狐」"] == "不在提示词里"                                      # 还没查到狐狸，托马的提示词里就不该有
    assert "system" in got["「托马」"]
    msgs = [{"role": "system", "content": "甲乙丙"}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"},
            {"role": "user", "content": "# 头\n\n## 局势\n有乙在\n\n## 别的\n没有"}]
    assert pp.locate(msgs, "乙") == ["system（人设与规矩）", "情境块「局势」"] and pp.locate(msgs, "丁") == []


def test_other_stages_and_json(tmp_path):
    s = make_save(tmp_path)
    pv = pp.capture("c", "thoma", say="你好，聊聊茶会吧。", intent="闲聊", settings=s)
    assert [r.task for r in pp._pick(pv, "appraise")] == ["npc_appraise"]
    assert {r.task for r in pp._pick(pv, "all")} >= {"npc_appraise", "npc_step", "npc_annotate"}
    data = json.loads(pp.to_json(pv, "all"))
    assert data["npc"] == "thoma" and data["requests"][0]["messages"][0]["role"] == "system"


def test_problems_are_explained(tmp_path):
    s = make_save(tmp_path)
    with pytest.raises(pp.PreviewError, match="没有存档"):
        pp.capture("nope", "thoma", settings=s)
    with pytest.raises(pp.PreviewError, match="没有这个 NPC"):
        pp.capture("c", "nobody", settings=s)
    moving = dict(AT_ESTATE, physical={"plans": {"ayaka": {"kind": "go", "location": "hanamizaka", "arrive": 99999,
                                                          "until": 99999, "reason": "花见坂"}}})
    s2 = make_save(tmp_path, moving, name="m")
    with pytest.raises(pp.PreviewError, match="正在去.*的路上"):
        pp.capture("m", "ayaka", settings=s2)


def test_cli(tmp_path, monkeypatch, capsys):
    s = make_save(tmp_path)
    monkeypatch.setattr("config.settings.load_settings", lambda *a, **k: s)
    assert pp.main(["c", "thoma", "--no-system", "--find", "守夜狐"]) == 0
    out = capsys.readouterr().out
    assert "提示词预览" in out and "关键词检查" in out and "不在提示词里" in out
    assert pp.main(["c", "thoma", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["requests"]
    assert pp.main(["nope", "thoma"]) == 2 and "没有存档" in capsys.readouterr().err
