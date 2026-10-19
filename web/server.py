"""网页版：本机起一个小 HTTP 服务，在后台线程里跑同一套 GameEngine。

  python3 web/server.py                 连 LLM_BACKEND 指定的后端（默认 OpenRouter），继续上次的存档 web
  python3 web/server.py --new           清档重来
  python3 web/server.py --mock          离线 Mock
  python3 web/server.py --incognito     无痕模式：存档、记忆都放临时目录，不写 trace，退出即删

浏览器打开 http://127.0.0.1:8770 。页面轮询 /api/state 拿新输出和当前选项，点按钮或打字 POST /api/choose。
引擎是同步的：WebPlayer.choose() 阻塞等网页的回答——引擎、harness、trace 一行不改。

分享给朋友玩（多人各玩各的）：
  python3 web/server.py --share --code 一个邀请码 [--max-games 5] [--budget 600000]
  · 每个浏览器（cookie）一局，存档按人分开（saves/p_<id>），关掉页面再来接着玩
  · 没有邀请码只能看到登录页，接口一律 401——别人拿到网址也花不了你的 API 额度
  · 同时最多 --max-games 局；闲置 30 分钟的局自动存档并释放，回来时从存档继续
  · 每人一局的 token 上限 --budget（用完会提示，不再调用模型）
  · 服务只听本机 127.0.0.1，公网访问走隧道（cloudflared / Tailscale Funnel），不用开路由器端口
"""
from __future__ import annotations

import argparse
import atexit
import os
import hashlib
import hmac
import secrets
import time
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
        self.status = ""                    # 引擎此刻在忙什么："" = 对方在想，options = 在准备你可说的话

    def set_status(self, s: str) -> None:
        with self.cond:
            self.status = s

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
            io.options = [{"i": i, "text": o.text, "kind": o.kind, "intent": o.intent,
                           "line": bool(o.meta.get("line")), "act": o.meta.get("act_label", "")}
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
        elif o.meta.get("line"):
            pass                                  # 带动作的台词：对话里会作为玩家的话显示出来
        elif o.kind not in ("view",):
            io.write(f"▸ {o.text}", "action")
        return PlayerAction.of(o)


class Game:
    def _monitor_line(self) -> str:
        try:
            from tools.monitor import line, load, summarize
            return "\n" + line(summarize(load([self.engine.tracer.path])))
        except Exception:  # noqa: BLE001 —— 监控统计出错不能影响游戏收尾
            return ""

    def __init__(self, args, save: Optional[str] = None):
        self.args = args
        self.save = save or args.save or "web"
        self.last_seen = time.time()
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
        if getattr(self.args, "budget", 0):
            settings.harness.game_token_budget = self.args.budget      # 分享模式：每人一局的 token 上限
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
            from main import build_llm, with_free_first
            try:
                backend = build_llm(settings.llm)
                backend.list_models()
                backend = with_free_first(settings.llm, backend)
            except LLMError as e:
                io.error = f"连不上 {settings.llm.backend}（{settings.llm.base_url}）：{e}。{start_hint(settings.llm.backend)}"
                io.finished = True
                return
            router = ModelRouter(backend, settings.llm, lambda m: build_llm(settings.llm, m))
        self._free_first = backend if hasattr(backend, "free") else None
        self.engine = GameEngine(settings, backend, WebPlayer(io, settings.game.player_name), out=io.write,
                                 save_name=self.save, new_game=new_game, router=router,
                                 trace=not self.args.incognito)
        self.engine.status = io.set_status
        if self._free_first is not None:
            self._free_first.tracer = self.engine.tracer       # 回落的时候写进 trace

        def run():
            try:
                self.engine.run(max_actions=1000)
                io.write(f"\n[{backend.name}] {self.engine.stats_line()}"
                         + (f"\n免费优先：{self._free_first.summary()}" if self._free_first is not None else "")
                         + (self._monitor_line() if self.engine.tracer.path else ""), "meta")
            except Exception as e:  # noqa: BLE001
                io.error = f"{type(e).__name__}: {e}"
                io.write(traceback.format_exc(), "meta")
            finally:
                with io.cond:
                    io.finished, io.waiting = True, False
                    io.cond.notify_all()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """停下这一局（引擎会在下一次等输入时退出并存档）。"""
        with self.io.cond:
            self.io.stop = True
            self.io.cond.notify_all()
        if self.thread:
            self.thread.join(timeout=30)

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


GAME: Optional[Game] = None          # 单机模式：只有一局
SHARE: Optional["Share"] = None      # 分享模式：一人一局

LOGIN_PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>雪见茶会</title><style>body{font-family:system-ui,sans-serif;background:#f3f1ec;display:grid;place-items:center;
min-height:100vh;margin:0}form{background:#fff;padding:28px;border-radius:14px;box-shadow:0 2px 12px #0001;width:min(320px,90vw)}
input,button{width:100%;box-sizing:border-box;padding:10px;margin-top:10px;font-size:16px;border-radius:8px;border:1px solid #ccc}
button{background:#5b7fa8;color:#fff;border:0}p{color:#c33}</style>
<form method="get" action="/"><b>雪见茶会与消失的「初雪」</b><div>请输入邀请码</div>
<input name="code" autofocus autocomplete="off"><button>进入</button>{MSG}</form>"""


class Share:
    """分享模式：邀请码 + 一个浏览器一局。"""
    IDLE = 30 * 60

    def __init__(self, args):
        self.args = args
        self.secret = secrets.token_hex(16)              # 每次启动换一个：重启后要重新输一次邀请码
        self.codes = [c.strip() for c in args.code.split(",") if c.strip()]
        self.games: dict[str, Game] = {}
        self.lock = threading.Lock()
        threading.Thread(target=self._reaper, daemon=True).start()

    def token(self) -> str:
        return hmac.new(self.secret.encode(), b"ok", hashlib.sha256).hexdigest()

    def check_code(self, code: str) -> bool:
        return any(hmac.compare_digest(code.encode(), c.encode()) for c in self.codes)

    def game(self, sid: str) -> Optional[Game]:
        """这个玩家的那一局；没有就开一局（存档在就接着玩）。人满时返回 None。"""
        with self.lock:
            g = self.games.get(sid)
            if g is None:
                if len(self.games) >= self.args.max_games:
                    return None
                g = Game(self.args, save=f"p_{sid[:12]}")
                g.start(new_game=False)
                self.games[sid] = g
            g.last_seen = time.time()
            return g

    def _reaper(self) -> None:
        while True:
            time.sleep(60)
            now = time.time()
            with self.lock:
                idle = [k for k, g in self.games.items() if now - g.last_seen > self.IDLE]
                gone = [self.games.pop(k) for k in idle]
            for g in gone:                                # 闲置太久：存档、释放名额；回来时从存档继续
                g.stop()

    def stop_all(self) -> None:
        for g in list(self.games.values()):
            g.stop()


def _cookies(header: str) -> dict:
    out = {}
    for part in (header or "").split(";"):
        if "=" in part:
            k, v = part.strip().split("=", 1)
            out[k] = v
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _authed(self) -> bool:
        return SHARE is None or hmac.compare_digest(_cookies(self.headers.get("Cookie")).get("ayaka_auth", ""),
                                                    SHARE.token())

    def _game(self) -> Optional[Game]:
        if SHARE is None:
            return GAME
        sid = _cookies(self.headers.get("Cookie")).get("ayaka_sid", "")
        if not sid or len(sid) != 32:
            return None
        return SHARE.game(sid)

    def _login(self, msg: str = "") -> None:
        data = LOGIN_PAGE.replace("{MSG}", f"<p>{msg}</p>" if msg else "").encode()
        self.send_response(200 if not msg else 401)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if SHARE is not None and "code=" in self.path and self.path.split("?")[0] in ("/", "/index.html"):
            from urllib.parse import parse_qs, urlparse
            code = (parse_qs(urlparse(self.path).query).get("code") or [""])[0].strip()
            if not SHARE.check_code(code):
                return self._login("邀请码不对")
            sid = _cookies(self.headers.get("Cookie")).get("ayaka_sid") or secrets.token_hex(16)
            self.send_response(303)
            self.send_header("Location", "/")
            for k, v in (("ayaka_auth", SHARE.token()), ("ayaka_sid", sid)):
                self.send_header("Set-Cookie", f"{k}={v}; Path=/; Max-Age=2592000; HttpOnly; SameSite=Lax")
            self.end_headers()
            return
        if not self._authed():
            if self.path.startswith("/api/"):
                return self._json({"error": "需要邀请码"}, 401)
            return self._login()
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
            game = self._game()
            if game is None:
                return self._json({"error": "现在人满了，过一会儿再来"}, 503)
            io = game.io
            with io.cond:
                payload = {"lines": io.lines[since:], "next": len(io.lines), "options": io.options,
                           "allow_free": io.allow_free, "in_dialogue": io.in_dialogue,
                           "waiting": io.waiting, "finished": io.finished, "status": io.status,
                           "error": io.error}
            payload["panel"] = game.side_panel()
            self._json(payload)
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 4096:
            return self._json({"error": "请求太大"}, 413)
        body = json.loads(self.rfile.read(n) or b"{}")
        if not self._authed():
            return self._json({"error": "需要邀请码"}, 401)
        game = self._game()
        if game is None:
            return self._json({"error": "现在人满了，过一会儿再来"}, 503)
        if self.path == "/api/choose":
            io = game.io
            with io.cond:
                if not io.waiting:
                    return self._json({"ok": False, "error": "现在还不能操作"}, 409)
                if "index" in body and 0 <= int(body["index"]) < len(io.options):
                    io.answer = {"index": int(body["index"])}
                elif body.get("text", "").strip() and io.allow_free:
                    io.answer = {"text": body["text"].strip()[:1000]}
                else:
                    return self._json({"ok": False, "error": "无效的选择"}, 400)
                io.cond.notify_all()
            return self._json({"ok": True})
        if self.path == "/api/restart":
            game.restart(bool(body.get("new")))
            return self._json({"ok": True})
        self._json({"error": "not found"}, 404)


def main(argv=None) -> int:
    global GAME
    ap = argparse.ArgumentParser(description="神里绫华 · 网页版")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--save", default=None)
    ap.add_argument("--new", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--incognito", action="store_true", help="无痕：不留存档、记忆和 trace")
    ap.add_argument("--share", action="store_true", help="分享给朋友：一人一局，要邀请码")
    ap.add_argument("--code", default="", help="分享模式的邀请码（可以用逗号隔开给多个）")
    ap.add_argument("--max-games", type=int, default=5, help="分享模式同时最多几局")
    ap.add_argument("--budget", type=int, default=0, help="每局 token 上限（分享模式默认 600000）")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址；用隧道分享时保持 127.0.0.1")
    args = ap.parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))      # 被外部停掉时也走 atexit 清理
    if args.share:
        global SHARE
        if not args.code:
            ap.error("--share 需要 --code 邀请码")
        args.budget = args.budget or 600000
        SHARE = Share(args)
        atexit.register(SHARE.stop_all)
        print(f"分享模式：每人一局，最多 {args.max_games} 局，每局 token 上限 {args.budget}", flush=True)
    else:
        GAME = Game(args)
        atexit.register(GAME.cleanup)
        GAME.start(args.new)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"神里绫华 · 网页版：http://127.0.0.1:{args.port}  "
          + ("（无痕模式：不留存档和记录，退出即删）" if args.incognito else "（Ctrl-C 停止，进度会保存）"), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if SHARE is not None:
            SHARE.stop_all()
        else:
            GAME.io.stop = True
            if GAME.engine and not args.incognito:
                GAME.engine.save()
            GAME.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
