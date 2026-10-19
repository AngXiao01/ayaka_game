"""一次“行动”的事务：权威状态（存档）是唯一的提交点，期间写过的副作用文件可以回滚。

要解决的问题：agent 的一次行动会碰好几处持久化的东西——
  · 权威状态：世界存档（时间、线索、谁在哪）；
  · 旁边的文件：NPC 的长期记忆、共享设定。
旁边的文件是“即时追加”的，权威状态却要等行动结束才存。进程在中间被杀，重启后就是：
世界说“没见过面”，NPC 的记忆里却写着“我们谈了三轮”，还会被塞进下次的提示词。

做法（写前日志 / undo log，思路和数据库回滚一样）：
  · 行动里第一次要写某个文件之前，先把它原样备份一份（文件当时不存在也记下来），并把“正在进行的事务号”落盘；
  · 行动结束时调用 commit：先把权威状态原子地写下去（里面带着事务号），再清掉备份；
  · 重启时 recover：权威状态里记的“最后提交号”如果小于日志里正在进行的事务号 → 那次行动没提交 →
    把备份还原回去，等于它从没发生；如果已经提交（崩在“写完存档、清备份之前”）→ 只清理。
为什么回滚而不是续跑：这类行动的内容来自模型，没法让它重新说出同一句话；而重放同一段输入是安全的
（权威状态没动，回到行动开始之前），玩家把这一步再做一遍即可。真正对外、不可撤销的副作用（发消息、扣款）
不能靠回滚——那种要用幂等键 + 先问下游“落地了吗”，见 docs/architecture.md。

这个文件不依赖游戏的任何东西。
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Callable, Optional

from core import crashpoints


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


class UnitOfWork:
    def __init__(self, root: Path, committed_tx: int = 0):
        self.root = Path(root)
        self.committed_tx = committed_tx            # 权威状态里记着的最后提交号
        self.marker = self.root / "marker.json"
        self.bak = self.root / "bak"
        self._files: dict[str, Optional[str]] = {}   # 本事务里碰过的文件 → 备份文件名（None = 当时不存在）

    @property
    def tx(self) -> int:
        return self.committed_tx + 1                # 正在进行的事务号

    # ------------------------------------------------------------ 写之前

    def touch(self, path: Path) -> None:
        """要写 path 之前调用：本事务里第一次碰它就备份。必须在真正写之前，写前日志才算数。"""
        key = str(Path(path).resolve())
        if key in self._files:
            return
        self.bak.mkdir(parents=True, exist_ok=True)
        if Path(path).exists():
            name = f"{len(self._files)}.bak"
            shutil.copy2(path, self.bak / name)
            self._files[key] = name
        else:
            self._files[key] = None
        _atomic_write(self.marker, json.dumps({"tx": self.tx, "files": self._files}, ensure_ascii=False))
        crashpoints.hit("uow.after_backup")

    # ------------------------------------------------------------ 提交

    def commit(self, persist_state: Callable[[int], None]) -> int:
        """persist_state(tx) 把权威状态原子地写下去（要把 tx 一起存进去）。写完才清备份。"""
        tx = self.tx
        persist_state(tx)
        crashpoints.hit("uow.after_state")          # 存档写完了，备份还没清：重启时要认出“其实已经提交了”
        self._clear()
        self.committed_tx = tx
        return tx

    def _clear(self) -> None:
        self.marker.unlink(missing_ok=True)
        shutil.rmtree(self.bak, ignore_errors=True)
        self._files = {}

    # ------------------------------------------------------------ 重启

    def recover(self) -> dict:
        """启动时先于任何读取调用。返回 {"action": "none" | "cleaned" | "rolled_back", "tx": …, "files": n}。"""
        if not self.marker.exists():
            shutil.rmtree(self.bak, ignore_errors=True)
            return {"action": "none"}
        info = json.loads(self.marker.read_text(encoding="utf-8"))
        if info["tx"] <= self.committed_tx:           # 存档已经带着这个事务号了：只是没来得及清备份
            self._clear()
            return {"action": "cleaned", "tx": info["tx"], "files": 0}
        for path, name in info["files"].items():       # 没提交：还原。每一步都可以重复做，恢复过程中再崩也没关系
            p = Path(path)
            if name is None:
                p.unlink(missing_ok=True)
            else:
                _atomic_write(p, (self.bak / name).read_text(encoding="utf-8"))
        self._clear()
        return {"action": "rolled_back", "tx": info["tx"], "files": len(info["files"])}
