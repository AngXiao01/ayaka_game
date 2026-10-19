"""tools/prune_data.py：只留网页上说够话的对局。"""
import json

from tools import prune_data


def make_game(data, save, says, location_inputs=0, stamps=("20260101-000000",)):
    (data / "saves").mkdir(parents=True, exist_ok=True)
    (data / "saves" / f"{save}.json").write_text("{}")
    for sub in ("memories", "uow"):
        (data / sub / save).mkdir(parents=True, exist_ok=True)
        (data / sub / save / "x").write_text("x")
    (data / "traces").mkdir(parents=True, exist_ok=True)
    per = [says // len(stamps) + (i < says % len(stamps)) for i in range(len(stamps))]
    for stamp, n in zip(stamps, per):
        rows = [{"event": "input", "at": "dialogue", "kind": "say"}] * n
        rows += [{"event": "input", "at": "location", "kind": "talk"}] * location_inputs
        (data / "traces" / f"{save}-{stamp}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n{半截")


def test_keeps_only_web_games_with_enough_dialogue(tmp_path):
    make_game(tmp_path, "p_abc", 10)
    make_game(tmp_path, "p_short", 9, location_inputs=30)         # 地点层的操作不算说话
    make_game(tmp_path, "web", 12, stamps=("20260101-000000", "20260102-000000"))   # 几个 trace 累加
    make_game(tmp_path, "default", 50)                            # 终端对局不留
    keep, drop = prune_data.prune(tmp_path, apply=True)
    assert keep == ["p_abc", "web"] and drop == ["default", "p_short"]
    assert sorted(p.name for p in (tmp_path / "saves").iterdir()) == ["p_abc.json", "web.json"]
    assert sorted(p.name for p in (tmp_path / "memories").iterdir()) == ["p_abc", "web"]
    assert sorted(p.name for p in (tmp_path / "uow").iterdir()) == ["p_abc", "web"]
    assert not list((tmp_path / "traces").glob("p_short*")) and not list((tmp_path / "traces").glob("default*"))
    assert len(list((tmp_path / "traces").glob("web-*"))) == 2


def test_dry_run_touches_nothing(tmp_path, capsys):
    make_game(tmp_path, "p_short", 3)
    assert prune_data.main(["--data", str(tmp_path)]) == 0
    assert (tmp_path / "saves" / "p_short.json").exists() and "将删" in capsys.readouterr().out
