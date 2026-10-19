from fastapi import APIRouter, Depends, HTTPException

from app.api.context import AppContext
from app.api.dependencies import get_context
from app.knowledge.retrieval import RetrievalError
from app.models import SearchRequest

router = APIRouter()


@router.post("/api/search")
async def search(body: SearchRequest, ctx: AppContext = Depends(get_context)):
    ctx.knowledge(body.knowledge_base_id)
    try:
        return await ctx.registry.search(body.query, body.backend, body.knowledge_base_id)
    except RetrievalError as exc:
        raise HTTPException(502, str(exc)) from None
