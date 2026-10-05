"""Who may call what, when the server is opened to a LAN.

One rule, used by every route that reads another session's data or changes
state every session shares: from off this machine, refuse. Reading engine
names and the voice list stays open, because the test page needs them.

The peer address alone is not enough. A browser ON the server host — the one
the lab page needs — sends every request from loopback, including the ones a
foreign page it happens to have open makes. A form post or a no-cors fetch
needs no CORS preflight, so such a page could swap the LLM endpoint for its
own server. Hence the second rule: a browser request (it carries `Origin`)
from any origin other than this server's own is refused too.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from fastapi import Request
from fastapi.responses import JSONResponse

from ..core.config import Config, ServerConfig

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}
_DEFAULT_PORT = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def is_loopback(host: str) -> bool:
    if host in _LOOPBACK:
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def must_be_private(server: ServerConfig) -> bool:
    """Bound off loopback, or serving TLS (which only exists for other machines)."""
    return not is_loopback(server.host) or bool(server.ssl_certfile)


def _origin_key(scheme: str, netloc: str) -> tuple[str, str, int | None]:
    parts = urlsplit(f"{scheme}://{netloc}")
    try:
        port = parts.port
    except ValueError:
        port = -1
    return scheme, (parts.hostname or "").lower(), port or _DEFAULT_PORT.get(scheme)


def foreign_origin(request: Request, config: Config) -> bool:
    """A browser request made by a page this server did not serve.

    No `Origin` header: curl, the measurement scripts, or a same-origin GET —
    browsers attach Origin to every POST and to every cross-origin fetch.
    Origins named in `server.cors_origins` count as ours; the wildcard does
    not, so "*" can open the public routes without opening these.
    """
    origin = request.headers.get("origin")
    if origin is None:
        return False
    if origin in config.server.cors_origins and origin != "*":
        return False
    parts = urlsplit(origin)
    if not parts.scheme or not parts.netloc:
        return True                    # "null", file://, sandboxed frames
    own = _origin_key(request.url.scheme, request.headers.get("host", ""))
    return _origin_key(parts.scheme, parts.netloc) != own


def local_only(request: Request, config: Config) -> JSONResponse | None:
    """None if allowed, else the 403 to return.

    Only meaningful because nothing sits in front of this server: behind a
    reverse proxy every request would look local and this would wave the
    whole LAN through.
    """
    if foreign_origin(request, config):
        return JSONResponse({"detail": "yêu cầu từ trang khác bị từ chối"}, status_code=403)
    if not config.server.private_introspection:
        return None
    host = request.client.host if request.client else ""
    if host in _LOOPBACK:
        return None
    return JSONResponse(
        {"detail": "chỉ làm được từ chính máy chạy server"}, status_code=403
    )
