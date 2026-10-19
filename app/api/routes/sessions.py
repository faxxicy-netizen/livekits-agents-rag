import asyncio
import json
import logging
import time
import uuid
from datetime import timedelta

import aiohttp
from fastapi import APIRouter, Depends, HTTPException
from livekit import api

from app.api.context import AppContext
from app.api.dependencies import get_context, owner
from app.conversation.engine import ModelError
from app.knowledge.retrieval import RetrievalError
from app.models import ChatRequest, CreateSession

logger = logging.getLogger("support.api")


router = APIRouter()


@router.get("/api/sessions")
def list_sessions(identity=Depends(owner), ctx: AppContext = Depends(get_context)):
    return ctx.store.sessions(identity)


@router.post("/api/sessions")
async def create_session(
    body: CreateSession, identity=Depends(owner), ctx: AppContext = Depends(get_context)
):
    async with ctx.registry.manager.lock:
        bound = ctx.knowledge(body.knowledge_base_id)
        if (body.backend or ctx.config.rag_backend) == "dify":
            raise HTTPException(400, "Dify 尚未配置独立知识库映射，请选择本地检索引擎。")
        return ctx.store.create_session(
            identity,
            body.backend or ctx.config.rag_backend,
            "连接诊断" if body.diagnostic else "新的知识库咨询",
            knowledge_base_id=body.knowledge_base_id,
            knowledge_profile={
                "id": bound["id"],
                "name": bound["name"],
                "profile": bound["profile"],
                "updated_at": bound["updated_at"],
            },
        )


@router.get("/api/sessions/{sid}")
def get_session(sid: str, identity=Depends(owner), ctx: AppContext = Depends(get_context)):
    return {**ctx.session(sid, identity), "messages": ctx.store.messages(sid)}


@router.delete("/api/sessions/{sid}")
async def delete_session(sid: str, identity=Depends(owner), ctx: AppContext = Depends(get_context)):
    ctx.session(sid, identity)
    lock = ctx.locks.setdefault(sid, asyncio.Lock())
    if lock.locked():
        raise HTTPException(409, "正在处理此会话的问题，请等待回答完成后再删除。")
    async with lock:
        try:
            ctx.store.delete_session(sid, identity)
        except KeyError:
            raise HTTPException(404, "会话不存在或无权访问。") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
    ctx.locks.pop(sid, None)
    return {"status": "deleted", "session_id": sid}


@router.post("/api/sessions/{sid}/chat")
async def chat(
    sid: str, body: ChatRequest, identity=Depends(owner), ctx: AppContext = Depends(get_context)
):
    current = ctx.session(sid, identity)
    ctx.knowledge(current["knowledge_base_id"])
    if current["status"] != "idle":
        raise HTTPException(409, "通话中请直接说话；历史会话请新建后继续。")
    lock = ctx.locks.setdefault(sid, asyncio.Lock())
    if lock.locked():
        raise HTTPException(409, "上一条问题正在处理中。")
    async with lock:
        history = ctx.store.messages(sid)
        turn_id = uuid.uuid4().hex
        started = time.perf_counter()
        ctx.store.save_message(sid, turn_id, "user", body.text, answer_mode="text")
        try:
            prepared = await ctx.engine.prepare(
                body.text,
                history,
                current["backend"],
                current["knowledge_base_id"],
                current["knowledge_profile"],
            )
            answer, sources, mode, metrics = await ctx.engine.answer(
                prepared, history, body.text, body.preview
            )
        except (RetrievalError, ModelError) as exc:
            ctx.store.save_message(sid, turn_id, "assistant", str(exc), answer_mode="error")
            raise HTTPException(502, str(exc)) from None
        # Discard results from a turn whose session ended while generation was pending.
        if ctx.store.session(sid)["status"] != "idle":
            raise HTTPException(409, "此会话已结束，旧任务结果已丢弃。")
        metrics["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
        message = ctx.store.save_message(
            sid, turn_id, "assistant", answer, sources, metrics, answer_mode=mode
        )
        logger.info(
            "turn_completed session=%s turn=%s metrics=%s", sid, turn_id, json.dumps(metrics)
        )
        return message


@router.post("/api/sessions/{sid}/token")
async def token(sid: str, identity=Depends(owner), ctx: AppContext = Depends(get_context)):
    current = ctx.session(sid, identity)
    ctx.knowledge(current["knowledge_base_id"])
    if current["status"] == "ended":
        raise HTTPException(409, "此通话已结束，请创建新会话。")
    if ctx.config.max_active_calls:
        with ctx.store.connect() as db:
            active = db.execute(
                "SELECT COUNT(*) FROM sessions WHERE status='active' AND id!=?", (sid,)
            ).fetchone()[0]
        if active >= ctx.config.max_active_calls:
            raise HTTPException(409, "演示通话正在使用中，请稍后重试。")
    missing = ctx.config.missing_voice_resources()
    if missing:
        raise HTTPException(424, "缺少资源：" + "；".join(missing))
    room = "support-" + sid
    # Dispatch only this application's named worker to its room.
    dispatch = api.RoomAgentDispatch(
        agent_name=ctx.config.livekit_agent_name, metadata=json.dumps({"session_id": sid})
    )
    jwt = (
        api.AccessToken(ctx.config.livekit_api_key, ctx.config.livekit_api_secret)
        .with_identity("user-" + sid)
        .with_name("访客")
        .with_ttl(timedelta(minutes=10))
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .with_room_config(api.RoomConfiguration(agents=[dispatch]))
        .to_jwt()
    )
    ctx.store.set_status(sid, "active")
    return {
        "url": ctx.config.livekit_public_url or ctx.config.livekit_url,
        "token": jwt,
        "room": room,
    }


@router.post("/api/sessions/{sid}/end")
async def end_session(sid: str, identity=Depends(owner), ctx: AppContext = Depends(get_context)):
    ctx.session(sid, identity)
    ctx.store.set_status(sid, "ended")
    try:
        async with api.LiveKitAPI(
            ctx.config.livekit_url,
            ctx.config.livekit_api_key,
            ctx.config.livekit_api_secret,
            timeout=httpx_timeout(),
        ) as lk:
            await lk.room.delete_room(api.DeleteRoomRequest(room="support-" + sid))
    except Exception as exc:
        logger.info("room_cleanup session=%s result=%s", sid, type(exc).__name__)
    return {"status": "ended"}


def httpx_timeout():
    # LiveKit server APIs use aiohttp rather than httpx.
    return aiohttp.ClientTimeout(total=3)
