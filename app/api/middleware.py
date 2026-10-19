"""Origin checks, demo access, response headers and request timing."""

import logging
import time
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.access import DemoAccess
from app.config import Settings

logger = logging.getLogger("support.api")


def register_guards(app: FastAPI, config: Settings, access: DemoAccess):
    @app.middleware("http")
    async def origin_guard(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method in {"POST", "PUT", "DELETE", "PATCH"} and origin:
            parsed = urlparse(origin)
            allowed = (
                origin.rstrip("/") == config.app_public_origin.rstrip("/")
                if config.app_public_origin
                else parsed.hostname in {"127.0.0.1", "localhost", "[::1]", "::1"}
            )
            if not allowed:
                return JSONResponse({"detail": "请求来源不被允许。"}, status_code=403)
        rejected = access.guard(request)
        if rejected is not None:
            return rejected
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        if access.enabled or request.url.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        if request.url.path.startswith("/api"):
            logger.info(
                "request method=%s path=%s status=%s duration_ms=%.1f",
                request.method,
                request.url.path,
                response.status_code,
                (time.perf_counter() - start) * 1000,
            )
        return response
