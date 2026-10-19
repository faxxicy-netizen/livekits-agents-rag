"""Server-side password gate for a shared demo; no password is sent to the frontend."""

import asyncio
import hashlib
import hmac
import secrets
import time
from collections import OrderedDict, deque

import jwt
from fastapi import Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

COOKIE = "support_access"
PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>知声 · 演示登录</title>
<style>body{margin:0;background:#f5f7fa;color:#182230;font:16px system-ui;display:grid;
place-items:center;min-height:100vh}main{background:white;padding:36px;width:320px;
border:1px solid #e5e7eb;border-radius:16px}h1{margin:0 0 12px;font-size:24px}
p{font-size:14px;color:#667085;line-height:1.6}input,button{box-sizing:border-box;width:100%;
padding:12px;border-radius:8px;margin-top:12px;font:inherit}input{border:1px solid #d0d5dd}
button{border:0;background:#2563eb;color:white;cursor:pointer}.error{color:#b42318}</style>
<main><h1>知声</h1><p>知识库语音助手演示。请输入访问密码后继续。</p>
<form method="post" action="/login"><label for="password">访问密码</label>
<input id="password" name="password" type="password" required maxlength="256"
autocomplete="current-password" autofocus><button type="submit">进入演示</button></form>
__ERROR__</main></html>"""


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 310000).hex()
    return f"pbkdf2_sha256$310000${salt}${digest}"


def verify_password(password: str, value: str) -> bool:
    try:
        scheme, rounds, salt, expected = value.split("$")
        if scheme != "pbkdf2_sha256" or not 100000 <= int(rounds) <= 1000000:
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(rounds)).hex()
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


class DemoAccess:
    def __init__(self, config):
        self.config = config
        self.enabled = bool(config.app_access_password_hash)
        if self.enabled and len(config.app_session_secret) < 32:
            raise ValueError("启用访问密码时，APP_SESSION_SECRET 至少需要32个字符。")
        self.password_tag = hashlib.sha256(config.app_access_password_hash.encode()).hexdigest()[
            :16
        ]
        self.attempts = OrderedDict()
        self.global_attempts = deque()

    def authorized(self, request: Request):
        if not self.enabled:
            return True
        try:
            data = jwt.decode(
                request.cookies.get(COOKIE, ""),
                self.config.app_session_secret,
                algorithms=["HS256"],
                options={"require": ["exp", "iat", "scope"]},
            )
            return data.get("scope") == "demo" and data.get("password_tag") == self.password_tag
        except jwt.InvalidTokenError:
            return False

    def limited(self, request):
        now = time.monotonic()
        while self.global_attempts and self.global_attempts[0] < now - 60:
            self.global_attempts.popleft()
        identity = request.client.host if request.client else "unknown"
        started, count = self.attempts.pop(identity, (now, 0))
        if started < now - 600:
            started, count = now, 0
        self.attempts[identity] = started, count + 1
        while len(self.attempts) > 1000:
            self.attempts.popitem(last=False)
        if count >= 5 or len(self.global_attempts) >= 30:
            return True
        self.global_attempts.append(now)
        return False

    def guard(self, request):
        if request.url.path in {"/login", "/healthz"} or self.authorized(request):
            return None
        if request.url.path.startswith("/api"):
            return JSONResponse({"detail": "请先输入访问密码。"}, status_code=401)
        return RedirectResponse("/login", status_code=303)

    def register(self, app):
        @app.get("/healthz", include_in_schema=False)
        def health():
            return {"status": "ok"}

        @app.get("/login", include_in_schema=False)
        def login_page(request: Request):
            if self.authorized(request):
                return RedirectResponse("/", status_code=303)
            return HTMLResponse(
                PAGE.replace("__ERROR__", ""), headers={"Cache-Control": "no-store"}
            )

        @app.post("/login", include_in_schema=False)
        async def login(request: Request, password: str = Form(max_length=256)):
            if not self.enabled:
                return RedirectResponse("/", status_code=303)
            if self.limited(request):
                return HTMLResponse(
                    PAGE.replace("__ERROR__", '<p class="error">尝试过多，请稍后重试。</p>'),
                    status_code=429,
                    headers={"Retry-After": "600"},
                )
            if not await asyncio.to_thread(
                verify_password, password, self.config.app_access_password_hash
            ):
                return HTMLResponse(
                    PAGE.replace("__ERROR__", '<p class="error">密码不正确。</p>'), status_code=401
                )
            now = int(time.time())
            identity = request.client.host if request.client else "unknown"
            self.attempts.pop(identity, None)
            token = jwt.encode(
                {
                    "scope": "demo",
                    "iat": now,
                    "exp": now + 8 * 3600,
                    "password_tag": self.password_tag,
                    "nonce": secrets.token_hex(16),
                },
                self.config.app_session_secret,
                algorithm="HS256",
            )
            response = RedirectResponse("/", status_code=303)
            response.set_cookie(
                COOKIE,
                token,
                httponly=True,
                secure=self.config.app_secure_cookies,
                samesite="strict",
                max_age=8 * 3600,
            )
            response.headers["Cache-Control"] = "no-store"
            return response

        @app.post("/logout", include_in_schema=False)
        def logout():
            response = RedirectResponse("/login", status_code=303)
            response.delete_cookie(
                COOKIE, secure=self.config.app_secure_cookies, httponly=True, samesite="strict"
            )
            return response
