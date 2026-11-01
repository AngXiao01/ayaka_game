"""分享模式：一人一局、人满拒绝、邀请码校验。"""
import argparse
import sys
import time

from tests.helpers import ROOT

sys.path.insert(0, str(ROOT / "web"))
import server  # noqa: E402


def args(tmp_path, **kw):
    a = argparse.Namespace(save="web", new=False, mock=True, incognito=False, share=True, code="雪见,初雪",
                           max_games=2, budget=1000, host="127.0.0.1", port=0)
    vars(a).update(kw)
    return a


def test_one_game_per_player_and_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("GAME_DATA_DIR", str(tmp_path))
    sh = server.Share(args(tmp_path))
    try:
        a, b = sh.game("a" * 32), sh.game("b" * 32)
        assert a is not b and a.save != b.save
        assert sh.game("a" * 32) is a                       # 同一个人回来还是那一局
        assert sh.game("c" * 32) is None                    # 人满了
        assert sh.check_code("初雪") and not sh.check_code("别的")
        assert sh.check_code("雪见")
        tok = sh.token()
        assert len(tok) == 64 and sh.token() == tok         # 同一次启动 token 稳定
    finally:
        sh.stop_all()


def test_share_idle_reaper_releases_slot(tmp_path, monkeypatch):
    monkeypatch.setenv("GAME_DATA_DIR", str(tmp_path))
    sh = server.Share(args(tmp_path, max_games=1))
    try:
        g = sh.game("a" * 32)
        assert g is not None and sh.game("b" * 32) is None
        g.last_seen = 0                                     # 假装闲置超时
        server.Share.IDLE = 0
        # 直接走 reaper 逻辑：停掉闲置局并腾出名额
        with sh.lock:
            idle = [k for k, x in sh.games.items() if time.time() - x.last_seen > 0]
            gone = [sh.games.pop(k) for k in idle]
        for x in gone:
            x.stop()
        assert sh.game("b" * 32) is not None
    finally:
        server.Share.IDLE = 30 * 60
        sh.stop_all()
