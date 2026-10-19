"""崩溃矩阵：枚举所有崩溃点，每个点崩一次，恢复后磁盘状态必须等于“最后一次提交时的样子”。

换到别的 agent 游戏复用，只要给三样东西：
  make_engine(data_dir, new_game)   用确定性的模型替身和固定脚本建一个引擎（同一个目录再建一次 = 重启 = 触发恢复）
  fingerprint(data_dir)             权威状态 + 旁边的副作用文件的指纹（去掉墙钟这类每次都不同的字段）
  consistent(engine)                跨文件的不变量（比如：有“见过面”的记忆，世界里就该有见面次数）
副作用代码里放 core.crashpoints.hit("名字")，再让 UnitOfWork 包住“权威状态的提交”，其余都是这里的通用逻辑：
  1. 基线：不崩跑一遍，每次提交后记下指纹；同时记下每个崩溃点被经过几次；
  2. 对每个（点，第 n 次）：在那里崩 → 重启 → 指纹必须等于基线里“存档记着的那次提交”的指纹 → 还能接着玩；
  3. 反向对照：把恢复关掉，同样的检查必须失败——否则这个矩阵是摆设。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from core import crashpoints


def fingerprint(data_dir: Path, save: str = "c") -> dict:
    """存档 + 各 NPC 的记忆文件的指纹。记忆里的 ts 是墙钟，每次都不同，去掉。"""
    out = {}
    f = data_dir / "saves" / f"{save}.json"
    if f.exists():
        out["save"] = hashlib.sha1(json.dumps(json.loads(f.read_text(encoding="utf-8")), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    mem = data_dir / "memories" / save
    for p in sorted(mem.glob("*.jsonl")) if mem.is_dir() else []:
        rows = [{k: v for k, v in json.loads(l).items() if k != "ts"} for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        out[f"mem/{p.name}"] = hashlib.sha1(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return out


def committed_tx(data_dir: Path, save: str = "c") -> int:
    f = data_dir / "saves" / f"{save}.json"
    return json.loads(f.read_text(encoding="utf-8")).get("tx", 0) if f.exists() else 0


def baseline(make_engine: Callable, data_dir: Path) -> dict:
    """不崩跑一遍：{"by_tx": {提交号: 指纹}, "points": {崩溃点: 经过次数}}。提交号 0 = 什么都还没有。"""
    crashpoints.disarm()
    eng = make_engine(data_dir, True)
    by_tx = {0: {}}
    original = eng.uow.commit

    def commit(persist):
        tx = original(persist)
        by_tx[tx] = fingerprint(data_dir)
        return tx
    eng.uow.commit = commit
    eng.run()
    points = crashpoints.seen()
    crashpoints.disarm()
    return {"by_tx": by_tx, "points": points}


def crash_and_restart(make_engine: Callable, data_dir: Path, point: str, nth: int):
    """在 point 第 nth 次经过时崩，然后在同一个目录重建引擎（= 重启，恢复在构造函数里发生）。"""
    crashpoints.arm(point, nth)
    try:
        make_engine(data_dir, True).run()
        raise AssertionError(f"{point}#{nth} 没有触发崩溃——基线里它被经过的次数和这里对不上")
    except crashpoints.SimulatedCrash:
        pass
    finally:
        crashpoints.disarm()
    return make_engine(data_dir, False)


def recovered_to_last_commit(base: dict, data_dir: Path) -> tuple[bool, str]:
    tx = committed_tx(data_dir)
    want, got = base["by_tx"].get(tx), fingerprint(data_dir)
    if got == want:
        return True, ""
    return False, f"存档记着提交号 {tx}，磁盘指纹和基线里那次提交对不上：多/少/变了的文件 = " + \
        str(sorted(k for k in set(got) | set(want or {}) if got.get(k) != (want or {}).get(k)))
