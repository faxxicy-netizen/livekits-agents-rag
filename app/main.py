"""Assemble the web application, shared resources, routes and static frontend."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.access import DemoAccess
from app.api.context import AppContext
from app.api.middleware import register_guards
from app.api.routes import documents, knowledge_bases, search, sessions, status
from app.config import ROOT, Settings, settings
from app.conversation.engine import SupportEngine
from app.knowledge.retrieval import RetrievalRegistry
from app.knowledge.vector import VectorError
from app.store import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("support.api")


def create_app(config: Settings | None = None) -> FastAPI:
    config = config or settings()
    store = Store(config.database_path)
    registry = RetrievalRegistry(config)
    store.freeze_legacy_profiles(registry.repository.knowledge)
    engine = SupportEngine(config, registry)
    ctx = AppContext(config, store, registry, engine)

    @asynccontextmanager
    async def lifespan(app):
        if config.vector_enabled:
            try:
                await registry.manager.index.initialize()
            except (VectorError, OSError, RuntimeError) as exc:
                ctx.knowledge_error = "知识库初始化失败：" + str(exc)
                logger.warning("knowledge_index_unavailable reason=%s", type(exc).__name__)
        try:
            yield
        finally:
            await engine.aclose()
            await registry.manager.index.aclose()

    app = FastAPI(title="知声 · 本地知识库语音助手", version="0.2.0", lifespan=lifespan)
    app.state.context = ctx
    app.state.store, app.state.registry, app.state.engine = store, registry, engine
    access = DemoAccess(config)
    access.register(app)
    register_guards(app, config, access)
    for module in (knowledge_bases, status, sessions, documents, search):
        app.include_router(module.router)

    dist = ROOT / "frontend/dist"
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/")
        def index():
            return FileResponse(dist / "index.html")
    else:

        @app.get("/")
        def index():
            return {
                "message": "前端尚未构建，请在 frontend 中运行 npm run build，"
                "或访问 Vite 开发地址。"
            }

    return app


app = create_app()
