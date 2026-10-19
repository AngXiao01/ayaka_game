"""一个很小的“节点 + 条件边”执行器（只用标准库，与游戏无关）。

为什么不用 LangGraph：我们只要“显式的步骤、显式的条件、每步可观测”，不要它的持久化和人工介入——
那些我们已经有（core/uow.py 的事务、tools/replay_game.py 的重放）。多一个依赖不值得。

一个图 = 若干节点 + 若干边：
  · 节点：函数 fn(state)，只通过共享的 state 读写；登记时写明 reads / writes（哪些字段），用来做静态检查和以后判断哪些节点可以并行；
  · 边：从一个节点到下一个节点，可以带条件 when(state)；按登记顺序取第一条成立的；dst=None 表示结束；
  · 每个节点跑完会：记一条 trace（节点名、耗时）、放一个崩溃点（core/crashpoints，名字 `<prefix>.after.<节点名>`）。

lint 检查：有没有走不到的节点、有没有“读了却没有任何前面的节点写过”的字段、有没有环（目前只支持无环图；
以后要加“重说”这类回边，再给每个节点加最大访问次数）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from core import crashpoints


@dataclass
class Node:
    name: str
    fn: Callable[[Any], None]
    reads: frozenset = field(default_factory=frozenset)
    writes: frozenset = field(default_factory=frozenset)


class Graph:
    def __init__(self, name: str, start: str, prefix: str = "graph"):
        self.name, self.start, self.prefix = name, start, prefix
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, list[tuple[Optional[Callable[[Any], bool]], Optional[str]]]] = {}

    # ------------------------------------------------------------ 搭图

    def node(self, name: str, fn: Callable, reads: Iterable[str] = (), writes: Iterable[str] = ()) -> "Graph":
        self.nodes[name] = Node(name, fn, frozenset(reads), frozenset(writes))
        return self

    def edge(self, src: str, dst: Optional[str], when: Optional[Callable[[Any], bool]] = None) -> "Graph":
        self.edges.setdefault(src, []).append((when, dst))
        return self

    def chain(self, names: list[str]) -> "Graph":
        """顺序连起来：a→b→c，最后一个→结束。"""
        for a, b in zip(names, names[1:] + [None]):
            self.edge(a, b)
        return self

    # ------------------------------------------------------------ 跑

    def run(self, state: Any, on_node: Optional[Callable[[str, float], None]] = None) -> Any:
        name: Optional[str] = self.start
        visited = 0
        while name is not None:
            visited += 1
            if visited > len(self.nodes) + 1:
                raise RuntimeError(f"图 {self.name} 里出现了环：目前只支持无环图")
            node = self.nodes[name]
            t0 = time.monotonic()
            node.fn(state)
            if on_node is not None:
                on_node(name, time.monotonic() - t0)
            crashpoints.hit(f"{self.prefix}.after.{name}")
            name = self._next(name, state)
        return state

    def _next(self, name: str, state: Any) -> Optional[str]:
        for when, dst in self.edges.get(name, []):
            if when is None or when(state):
                return dst
        return None

    # ------------------------------------------------------------ 检查与描述

    def lint(self, initial: Iterable[str] = ()) -> list[str]:
        problems: list[str] = []
        for src, outs in self.edges.items():
            for _, dst in outs:
                if src not in self.nodes or (dst is not None and dst not in self.nodes):
                    problems.append(f"边 {src}→{dst} 指向不存在的节点")
        if self.start not in self.nodes:
            return problems + [f"起点 {self.start} 不存在"]
        succ = {n: [d for _, d in self.edges.get(n, []) if d is not None and d in self.nodes] for n in self.nodes}
        reach, todo = {self.start}, [self.start]                      # 从起点走得到的节点
        while todo:
            for d in succ[todo.pop()]:
                if d not in reach:
                    reach.add(d)
                    todo.append(d)
        problems += [f"节点 {n} 从起点走不到" for n in self.nodes if n not in reach]
        pred = {n: [m for m in reach if n in succ[m]] for n in reach}
        waiting = {n: len(pred[n]) for n in reach}                      # 拓扑排序，顺便查环
        topo, queue = [], [n for n in reach if waiting[n] == 0]
        while queue:
            n = queue.pop(0)
            topo.append(n)
            for d in succ[n]:
                waiting[d] -= 1
                if waiting[d] == 0:
                    queue.append(d)
        if len(topo) != len(reach):
            return problems + [f"图 {self.name} 有环"]
        out: dict[str, set[str]] = {}                                   # 离开该节点时一定已经写过的字段
        for n in topo:
            before = set(initial) if n == self.start else set.intersection(*[out[m] for m in pred[n]])
            problems += [f"节点 {n} 读了 {f}，但在它之前没有任何节点一定写过它" for f in sorted(self.nodes[n].reads - before)]
            out[n] = before | self.nodes[n].writes
        return problems

    def describe(self) -> str:
        lines = [f"图 {self.name}（起点 {self.start}）"]
        for n, node in self.nodes.items():
            outs = ", ".join(("" if w is None else "[条件] ") + (d or "结束") for w, d in self.edges.get(n, [])) or "结束"
            lines.append(f"  {n:18} 读 {sorted(node.reads)} 写 {sorted(node.writes)} → {outs}")
        return "\n".join(lines)
