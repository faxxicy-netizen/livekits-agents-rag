from fastapi import APIRouter, Depends, HTTPException

from app.api.context import AppContext
from app.api.dependencies import get_context, owner
from app.conversation.engine import ModelError
from app.knowledge.repository import DEFAULT_KNOWLEDGE_BASE
from app.knowledge.vector import VectorError
from app.models import KnowledgeInput

router = APIRouter()


@router.get("/api/knowledge-bases")
def knowledge_bases(ctx: AppContext = Depends(get_context)):
    documents = ctx.registry.repository.list()
    return [
        {**kb, "document_count": sum(d["knowledge_base_id"] == kb["id"] for d in documents)}
        for kb in ctx.registry.repository.knowledge.list()
    ]


@router.post("/api/knowledge-bases")
async def create_knowledge(
    body: KnowledgeInput, identity=Depends(owner), ctx: AppContext = Depends(get_context)
):
    async with ctx.registry.manager.lock:
        return ctx.registry.repository.knowledge.save(body)


@router.put("/api/knowledge-bases/{knowledge_base_id}")
async def update_knowledge(
    knowledge_base_id: str,
    body: KnowledgeInput,
    identity=Depends(owner),
    ctx: AppContext = Depends(get_context),
):
    async with ctx.registry.manager.lock:
        ctx.knowledge(knowledge_base_id)
        return ctx.registry.repository.knowledge.save(body, knowledge_base_id)


@router.delete("/api/knowledge-bases/{knowledge_base_id}")
async def delete_knowledge(
    knowledge_base_id: str, identity=Depends(owner), ctx: AppContext = Depends(get_context)
):
    async with ctx.registry.manager.lock:
        ctx.knowledge(knowledge_base_id)
        if knowledge_base_id == DEFAULT_KNOWLEDGE_BASE:
            raise HTTPException(409, "默认知识库作为入口保留，可以删除其中的文档。")
        with ctx.store.connect() as db:
            sessions = db.execute(
                "SELECT id,status FROM sessions WHERE knowledge_base_id=?", (knowledge_base_id,)
            ).fetchall()
        if any(s["status"] == "active" for s in sessions):
            raise HTTPException(409, "此知识库有进行中的通话，请先结束通话。")
        if any(ctx.locks.get(s["id"]) and ctx.locks[s["id"]].locked() for s in sessions):
            raise HTTPException(409, "此知识库正在生成回答，请等待回答完成后再删除。")
        # Reject new turns/token requests while the external vector deletion is pending.
        ctx.deleting_knowledge.add(knowledge_base_id)
        try:
            if ctx.config.vector_enabled:
                await ctx.registry.manager.index.delete_knowledge_base(knowledge_base_id)
            ctx.registry.repository.delete_knowledge_base(knowledge_base_id)
            ctx.registry.reload()
        except VectorError as exc:
            raise HTTPException(503, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        finally:
            ctx.deleting_knowledge.discard(knowledge_base_id)
        return {"status": "deleted", "knowledge_base_id": knowledge_base_id}


@router.post("/api/knowledge-bases/{knowledge_base_id}/profile-draft")
async def profile_draft(
    knowledge_base_id: str, identity=Depends(owner), ctx: AppContext = Depends(get_context)
):
    ctx.knowledge(knowledge_base_id)
    try:
        return {"profile": await ctx.engine.draft_profile(knowledge_base_id)}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    except ModelError as exc:
        raise HTTPException(502, str(exc)) from None
