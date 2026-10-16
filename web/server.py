"""网页版：本机起一个小 HTTP 服务，在后台线程里跑同一套 GameEngine。

  python3 web/server.py                 连 LLM_BACKEND 指定的后端（默认 OpenRouter），继续上次的存档 web
  python3 web/server.py --new           清档重来
  python3 web/server.py --mock          离线 Mock
  python3 web/server.py --incognito     无痕模式：存档、记忆都放临时目录，不写 trace，退出即删

浏览器打开 http://127.0.0.1:8770 。页面轮询 /api/state 拿新输出和当前选项，点按钮或打字 POST /api/choose。
引擎是同步的：WebPlayer.choose() 阻塞等网页的回答——引擎、harness、trace 一行不改。
"""
from __future__ import annotations

import argparse
import atexit
import json
import shutil
import signal
import sys
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import load_settings  # noqa: E402
from core.player_agent import PlayerAction  # noqa: E402
from engine.game_engine import GameEngine  # noqa: E402
from llm.base import LLMError  # noqa: E402
from llm.mock_backend import MockBackend  # noqa: E402
from llm.router import ModelRouter  # noqa: E402
from llm.schemas import FREE_INTENT  # noqa: E402

PAGE = Path(__file__).resolve().parent / "index.html"


class WebIO:
    """引擎线程与 HTTP 线程之间的共享状态。"""

    def __init__(self):
        self.cond = threading.Condition()
        self.lines: list[dict] = []
        self.options: list[dict] = []
        self.allow_free = False
        self.in_dialogue = False
        self.waiting = False
        self.answer: Optional[dict] = None
        self.stop = False
        self.finished = False
        self.error = ""

    def write(self, text: str, kind: str = "out") -> None:
        with self.cond:
            for part in str(text).split("\n"):
                self.lines.append({"id": len(self.lines), "text": part, "kind": kind})
            self.cond.notify_all()


class WebPlayer:
    def __init__(self, io: WebIO, player_name: str):
        self.io = io
        self.player_name = player_name

    def choose(self, options, ctx) -> PlayerAction:
        io = self.io
        with io.cond:
            io.options = [{"i": i, "text": o.text, "kind": o.kind, "intent": o.intent, "tone": o.tone}
                          for i, o in enumerate(options)]
            io.allow_free = True              # 对话里自由发言；地点层自由写想查看什么
            io.in_dialogue = ctx is not None
            io.answer, io.waiting = None, True
            io.cond.notify_all()
            while io.answer is None and not io.stop:
                io.cond.wait(timeout=1.0)
            ans, io.answer, io.waiting, io.options = io.answer, None, False, []
        if io.stop or ans is None:
            return PlayerAction("quit")
        if "text" in ans:
            if ctx is None:
                io.write(f"▸ 查看：{ans['text']}", "action")
                return PlayerAction("look", ans["text"], "查看")
            io.write(f"{self.player_name}：「{ans['text']}」", "player")
            return PlayerAction("say", ans["text"], FREE_INTENT)
        o = options[ans["index"]]
        if o.kind == "say":
            io.write(f"{self.player_name}：「{o.text}」", "player")
        elif o.kind not in ("view",):
            io.write(f"▸ {o.text}", "action")
        return PlayerAction.of(o)


class Game:
    def __init__(self, args):
        self.args = args
        self.io = WebIO()
        self.engine: Optional[GameEngine] = None
        self.thread: Optional[threading.Thread] = None
        self.tmpdir: Optional[str] = None

    def cleanup(self) -> None:
        """无痕模式：删掉这一局的临时目录（存档、长期记忆、即兴设定全在里面）。"""
        if self.tmpdir:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            self.tmpdir = None

    def start(self, new_game: bool) -> None:
        self.io = io = WebIO()
        settings = load_settings()
        settings.game.debug = True            # 调试行照常输出，页面上默认隐藏，可以勾选显示
        if self.args.incognito:
            self.cleanup()
            self.tmpdir = tempfile.mkdtemp(prefix="ayaka_incognito_")
            settings.game.data_dir = Path(self.tmpdir)
            new_game = True
        if self.args.mock:
            backend = MockBackend(seed=0)
            router = ModelRouter(backend, settings.llm)
        else:
            from llm.diagnostics import start_hint
            from main import build_llm
            try:
                backend = build_llm(settings.llm)
                backend.list_models()
            except LLMError as e:
                io.error = f"连不上 {settings.llm.backend}（{settings.llm.base_url}）：{e}。{start_hint(settings.llm.backend)}"
                io.finished = True
                return
            router = ModelRouter(backend, settings.llm, lambda m: build_llm(settings.llm, m))
        self.engine = GameEngine(settings, backend, WebPlayer(io, settings.game.player_name), out=io.write,
                                 save_name=self.args.save, new_game=new_game, router=router,
                                 trace=not self.args.incognito)

        def run():
            try:
                self.engine.run(max_actions=1000)
                io.write(f"\n[{backend.name}] {self.engine.stats_line()}", "meta")
            except Exception as e:  # noqa: BLE001
                io.error = f"{type(e).__name__}: {e}"
                io.write(traceback.format_exc(), "meta")
            finally:
                with io.cond:
                    io.finished, io.waiting = True, False
                    io.cond.notify_all()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def restart(self, new_game: bool) -> None:
        with self.io.cond:
            self.io.stop = True
            self.io.cond.notify_all()
        if self.thread:
            self.thread.join(timeout=60)
        self.start(new_game)

    def side_panel(self) -> dict[str, Any]:
        e = self.engine
        if e is None:
            return {}
        g, q = e.god, e.quest
        people = []
        for nid, p in e.personas.items():
            if not g.met.get(nid) and p.role != "companion":
                continue
            r = e.npcs[nid].relationship
            people.append({"name": p.name, "title": p.title, "brief": p.brief, "tier": r.tier()[0],
                           "affinity": r.affinity, "trust": r.trust,
                           "where": e.explorer.player_whereabouts(nid)})     # 玩家知道的去向，不是真相
        hints = e.explorer.hints
        return {
            "status": g.status_line(), "time": f"{g.time_label()} {g.clock_label()}", "location": e.world.name(g.location),
            "act": f"{g.act.title}：{g.act.objective}" + ("（已告一段落）" if g.resolved else ""),
            "hint": "主线告一段落，可以随意走走、拜访认识的人，或者约人同行。" if g.resolved else g.act.hint,
            "party": [e.personas[n].name for n in e.explorer.party],
            "leads": e.explorer.leads() if hints else [],
            "checklist": [{"label": label, "ok": ok} for label, ok in g.checklist()] if hints else [],
            "glossary": [{"term": k, "desc": v} for k, v in q.glossary.items()],
            "how_to_play": [x for x in q.how_to_play if not x.startswith("【")],     # 去掉标题行
            "clues": [{"name": q.clue_name(c), "desc": q.clues[c]["desc"]} for c in g.clues],
            "clue_total": len(q.clues),
            "testimonies": [{"npc": e.personas[t["npc"]].name, "said": t["said"], "time": t["time"],
                             "caught": t["caught"]} for t in g.testimonies],
            "people": people, "journal": g.journal[-12:], "ending": g.ending,
            "incognito": bool(self.args.incognito),
        }


GAME: Optional[Game] = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index.html"):
            data = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path.startswith("/api/state"):
            since = 0
            if "since=" in self.path:
                try:
                    since = int(self.path.split("since=")[1].split("&")[0])
                except ValueError:
                    since = 0
            io = GAME.io
            with io.cond:
                payload = {"lines": io.lines[since:], "next": len(io.lines), "options": io.options,
                           "allow_free": io.allow_free, "in_dialogue": io.in_dialogue,
                           "waiting": io.waiting, "finished": io.finished,
                           "error": io.error}
            payload["panel"] = GAME.side_panel()
            self._json(payload)
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path == "/api/choose":
            io = GAME.io
            with io.cond:
                if not io.waiting:
                    return self._json({"ok": False, "error": "现在还不能操作"}, 409)
                if "index" in body and 0 <= int(body["index"]) < len(io.options):
                    io.answer = {"index": int(body["index"])}
                elif body.get("text", "").strip() and io.allow_free:
                    io.answer = {"text": body["text"].strip()[:300]}
                else:
                    return self._json({"ok": False, "error": "无效的选择"}, 400)
                io.cond.notify_all()
            return self._json({"ok": True})
        if self.path == "/api/restart":
            GAME.restart(bool(body.get("new")))
            return self._json({"ok": True})
        self._json({"error": "not found"}, 404)


def main(argv=None) -> int:
    global GAME
    ap = argparse.ArgumentParser(description="神里绫华 · 网页版")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--save", default="web")
    ap.add_argument("--new", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--incognito", action="store_true", help="无痕：不留存档、记忆和 trace")
    args = ap.parse_args(argv)
    GAME = Game(args)
    atexit.register(GAME.cleanup)
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))      # 被外部停掉时也走 atexit 清理
    GAME.start(args.new)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"神里绫华 · 网页版：http://127.0.0.1:{args.port}  "
          + ("（无痕模式：不留存档和记录，退出即删）" if args.incognito else "（Ctrl-C 停止，进度会保存）"), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        GAME.io.stop = True
        if GAME.engine and not args.incognito:
            GAME.engine.save()
        GAME.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
