import asyncio

import httpx
from fastapi import APIRouter, Depends

from app.api.context import AppContext
from app.api.dependencies import get_context, owner

router = APIRouter()


@router.get("/api/status")
async def status(identity=Depends(owner), ctx: AppContext = Depends(get_context)):

    async def reachable(url):
        try:
            async with httpx.AsyncClient(timeout=0.7) as client:
                r = await client.get(url)
                return r.status_code < 500
        except httpx.HTTPError:
            return False

    lk_http = ctx.config.livekit_url.replace("ws://", "http://").replace("wss://", "https://")
    livekit, worker, vector_server = await asyncio.gather(
        reachable(lk_http),
        reachable("http://127.0.0.1:8081"),
        reachable(ctx.config.qdrant_url + "/readyz")
        if ctx.config.vector_enabled
        else asyncio.sleep(0, result=False),
    )
    missing = ctx.config.missing_voice_resources()
    if ctx.config.rag_backend in {"vector", "hybrid"} and (
        ctx.knowledge_error or not vector_server
    ):
        missing.append(ctx.knowledge_error or "本地 Qdrant 尚未启动。")
    document_list = ctx.registry.repository.list()
    vector_ready = vector_server and (not ctx.knowledge_error)
    return {
        "default_backend": ctx.config.rag_backend,
        "livekit_ready": livekit,
        "agent_ready": worker,
        "voice_ready": livekit and worker and (not missing),
        "llm_configured": bool(ctx.config.llm_api_key),
        "llm_model": ctx.config.llm_model,
        "stt_provider": ctx.config.stt_provider,
        "tts_provider": ctx.config.tts_provider,
        "stt_model": ctx.config.dashscope_asr_model
        if ctx.config.stt_provider == "dashscope"
        else "SenseVoice Small"
        if ctx.config.stt_provider == "local"
        else ctx.config.stt_model,
        "tts_model": ctx.config.dashscope_tts_model
        if ctx.config.tts_provider == "dashscope"
        else ctx.config.local_tts_backend
        if ctx.config.tts_provider == "local"
        else ctx.config.tts_model,
        "missing_resources": missing,
        "document_count": len(document_list),
        "chunk_count": sum(d["chunks"] for d in document_list),
        "vector_ready": vector_ready,
        "embedding_model": ctx.config.embedding_model_name,
        "embedding_dimension": ctx.config.embedding_dimension,
        "embedding_provider": ctx.config.embedding_provider,
        "deployment": "server" if ctx.config.app_public_origin else "local",
        "livekit_public_url": ctx.config.livekit_public_url or ctx.config.livekit_url,
        "vector_database": "Qdrant",
        "vector_dashboard_url": ""
        if ctx.config.app_public_origin
        else ctx.config.qdrant_url + "/dashboard",
        "knowledge_error": ctx.knowledge_error,
        "backends": [
            {"id": "bm25", "name": "BM25 关键词检索", "ready": True},
            {"id": "vector", "name": "向量语义检索 · Qdrant", "ready": vector_ready},
            {"id": "hybrid", "name": "混合检索 · BM25 + 向量", "ready": vector_ready},
            {"id": "tfidf", "name": "TF-IDF 词项检索", "ready": True},
            {"id": "dify", "name": "Dify 知识库", "ready": False},
        ],
    }
