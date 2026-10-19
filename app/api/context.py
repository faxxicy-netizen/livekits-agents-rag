"""Resources shared by routes within one FastAPI application instance."""

import asyncio
from dataclasses import dataclass, field

from fastapi import HTTPException

from app.config import Settings
from app.conversation.engine import SupportEngine
from app.knowledge.retrieval import RetrievalRegistry
from app.store import Store


@dataclass
class AppContext:
    config: Settings
    store: Store
    registry: RetrievalRegistry
    engine: SupportEngine
    locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    deleting_knowledge: set[str] = field(default_factory=set)
    knowledge_error: str = ""

    def session(self, sid: str, identity: str) -> dict:
        try:
            return self.store.session(sid, identity)
        except KeyError:
            raise HTTPException(404, "会话不存在或无权访问。") from None

    def knowledge(self, knowledge_base_id: str) -> dict:
        if knowledge_base_id in self.deleting_knowledge:
            raise HTTPException(409, "知识库正在删除，请稍后重试。")
        try:
            return self.registry.repository.knowledge.get(knowledge_base_id)
        except KeyError:
            raise HTTPException(404, "知识库不存在。") from None
