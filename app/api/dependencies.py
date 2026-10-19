"""Request dependencies for application resources and browser ownership."""

import re
import secrets

from fastapi import Depends, Request, Response

from app.api.context import AppContext


def get_context(request: Request) -> AppContext:
    return request.app.state.context


def owner(request: Request, response: Response, ctx: AppContext = Depends(get_context)) -> str:
    identity = request.cookies.get("support_owner", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{43}", identity):
        identity = secrets.token_urlsafe(32)
        response.set_cookie(
            "support_owner",
            identity,
            httponly=True,
            secure=ctx.config.app_secure_cookies,
            samesite="strict",
            max_age=60 * 60 * 24 * 30,
        )
    return identity
