"""带连接复用的 HTTP 小客户端（只用标准库）。

为什么要有它：标准库的 urllib.request.urlopen 每个请求都新建一条连接（TCP + TLS 握手），不复用。
一回合游戏要调八九次模型（评估、说话、核对、记录、替玩家想台词、给台词打标签、核对声明……），就是八九次握手。
在代理 / VPN 不稳的网络上，新连接的 TLS 握手经常卡几秒甚至超时：实测 openrouter.ai 的握手 6 次里有
0.4 / 7.8 / 1.5 秒和一次 10 秒超时；于是明明只输出几个 token 的小请求也要等 10~20 秒。

做法：
  · 每个线程、每个主机留一条保持连接，后面的请求直接复用，一回合只握手一次；
  · 握手单独用短超时（默认 4 秒）并立刻重连，不等 10 秒、不退避；读响应才用长超时（模型生成要时间）；
  · 复用的旧连接被对方关了（空闲太久）：悄悄换一条新的重发一次——这种失败请求根本没送达，重发是安全的；
  · 报错和 urllib 保持一致（HTTPError / URLError / TimeoutError），调用方的错误处理不用改；
  · 环境里设了代理（HTTPS_PROXY 等）就退回 urllib，保证不改变现有行为。
"""
from __future__ import annotations

import http.client
import io
import socket
import ssl
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

CONNECT_TIMEOUT = 4.0       # 握手 / 建连的超时，卡住就立刻重连
CONNECT_TRIES = 3
_UA = f"Python-urllib/{sys.version_info.major}.{sys.version_info.minor}"    # 和 urllib 发的一样，免得被前面的防护层当成陌生客户端

_local = threading.local()
_stats = {"connects": 0, "reused": 0, "connect_retries": 0}       # 给测试和排查看


def stats() -> dict:
    return dict(_stats)


def _pool() -> dict:
    if not hasattr(_local, "conns"):
        _local.conns = {}
    return _local.conns


def _connect(scheme: str, host: str, port: int, timeout: float, connect_timeout: float, tries: int):
    last: Optional[BaseException] = None
    for i in range(max(1, tries)):
        cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
        conn = cls(host, port, timeout=connect_timeout)
        try:
            conn.connect()
            conn.sock.settimeout(timeout)            # 建连用短超时，读响应换回长超时
            _stats["connects"] += 1
            return conn
        except (OSError, ssl.SSLError) as e:         # socket.timeout / 握手超时 / SSL EOF 都是 OSError
            last = e
            conn.close()
            _stats["connect_retries"] += 1
    raise urllib.error.URLError(last)


def _proxied(scheme: str, host: str) -> bool:
    return bool(urllib.request.getproxies().get(scheme)) and not urllib.request.proxy_bypass(host)


def _legacy(method: str, url: str, data: Optional[bytes], headers: dict, timeout: float) -> bytes:
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def request(method: str, url: str, data: Optional[bytes] = None, headers: Optional[dict] = None, timeout: float = 120.0,
            connect_timeout: float = CONNECT_TIMEOUT, connect_tries: int = CONNECT_TRIES) -> bytes:
    """发一个请求，返回响应体。HTTP 错误抛 urllib.error.HTTPError（可 .read()），连不上抛 URLError，读超时抛 TimeoutError。"""
    u = urllib.parse.urlsplit(url)
    scheme, host = u.scheme, u.hostname or ""
    port = u.port or (443 if scheme == "https" else 80)
    hdrs = {"User-Agent": _UA, **(headers or {})}
    if _proxied(scheme, host):
        return _legacy(method, url, data, hdrs, timeout)
    path = (u.path or "/") + (f"?{u.query}" if u.query else "")
    key = (scheme, host, port)
    pool = _pool()
    for attempt in (0, 1):
        conn = pool.pop(key, None)                    # 用的时候先从池里取走，避免同一线程里重入
        reused = conn is not None
        if conn is None:
            conn = _connect(scheme, host, port, timeout, connect_timeout, connect_tries)
        else:
            _stats["reused"] += 1
            conn.sock.settimeout(timeout)
        try:
            conn.request(method, path, body=data, headers=hdrs)
            resp = conn.getresponse()
            body = resp.read()
        except (socket.timeout, TimeoutError) as e:
            conn.close()
            raise TimeoutError(str(e) or "读响应超时") from e
        except (http.client.HTTPException, ConnectionError, ssl.SSLError, OSError) as e:
            conn.close()
            if reused and attempt == 0:
                continue                              # 复用的旧连接被对方关了：换一条新的重来（请求没送达，重发安全）
            raise urllib.error.URLError(e) from e
        if resp.will_close:
            conn.close()
        else:
            pool[key] = conn
        if resp.status >= 400:
            raise urllib.error.HTTPError(url, resp.status, resp.reason, resp.headers, io.BytesIO(body))
        return body
    raise urllib.error.URLError("连接反复失败")        # 不会走到这里，给类型检查看


def close_all() -> None:
    """关掉当前线程留着的连接（测试用）。"""
    for c in _pool().values():
        c.close()
    _pool().clear()
