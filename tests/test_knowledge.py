import json
import sqlite3
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

from app.conversation.engine import SYSTEM, SupportEngine
from app.knowledge.retrieval import RetrievalRegistry, normalize
from app.knowledge.vector import KnowledgeIndex
from app.main import create_app
from app.models import KnowledgeInput, KnowledgeProfile
from app.store import Store
from app.voice.agent import SupportAgent


def create_base(client, name, role="IoT 平台使用助手"):
    response = client.post("/api/knowledge-bases", json={"name": name, "profile": {"role": role}})
    response.raise_for_status()
    return response.json()


def test_binding_isolation_same_filename_snapshot_and_empty_base(config):
    with TestClient(create_app(config)) as client:
        first = create_base(client, "IoT 平台 A")
        second = create_base(client, "IoT 平台 B")
        for kb, value in [(first, "星河青鸟"), (second, "银色海豚")]:
            response = client.post(
                "/api/documents/upload",
                data={"knowledge_base_id": kb["id"]},
                files={
                    "file": (
                        "使用手册.txt",
                        f"SDK初始化密钥是{value}，调用初始化接口后建立连接。".encode(),
                    )
                },
            )
            response.raise_for_status()
        session = client.post("/api/sessions", json={"knowledge_base_id": first["id"]}).json()
        assert session["knowledge_profile"]["profile"]["role"] == "IoT 平台使用助手"
        for backend in ["bm25", "tfidf"]:
            for kb, value, other in [
                (first, "星河青鸟", "银色海豚"),
                (second, "银色海豚", "星河青鸟"),
            ]:
                result = client.post(
                    "/api/search",
                    json={
                        "query": "SDK初始化密钥",
                        "backend": backend,
                        "knowledge_base_id": kb["id"],
                    },
                ).json()
                assert result["chunks"]
                assert all(
                    c["knowledge_base_id"] == kb["id"]
                    and value in c["text"]
                    and other not in c["text"]
                    for c in result["chunks"]
                )
        client.put(
            "/api/knowledge-bases/" + first["id"],
            json={"name": first["name"], "profile": {"role": "新的平台助手"}},
        ).raise_for_status()
        old = client.get("/api/sessions/" + session["id"]).json()
        new = client.post("/api/sessions", json={"knowledge_base_id": first["id"]}).json()
        assert old["knowledge_profile"]["profile"]["role"] == "IoT 平台使用助手"
        assert new["knowledge_profile"]["profile"]["role"] == "新的平台助手"
        empty = create_base(client, "空知识库")
        result = client.post(
            "/api/search", json={"query": "SDK初始化密钥", "knowledge_base_id": empty["id"]}
        ).json()
        assert result["chunks"] == []
        assert (
            client.post("/api/sessions", json={"knowledge_base_id": "missing"}).status_code == 404
        )
        assert (
            client.get("/api/documents?knowledge_base_id=" + first["id"]).json()[0]["filename"]
            == "使用手册.txt"
        )


async def test_business_prompt_and_voice_share_bound_scope(config, monkeypatch):
    config.llm_api_key = "test-only"
    registry = RetrievalRegistry(config)
    kb = registry.repository.knowledge.save(
        KnowledgeInput(
            name="IoT 接入指南",
            profile=KnowledgeProfile(role="IoT 平台使用助手", aliases="爱欧踢=IoT"),
        )
    )
    store = Store(config.database_path)
    bound = store.create_session("owner", "bm25", knowledge_base_id=kb["id"], knowledge_profile=kb)
    agent = SupportAgent(SimpleNamespace(room=None), bound["id"], store, config)
    assert "IoT 平台使用助手" in agent.instructions
    engine = SupportEngine(config, registry)
    prepared = await engine.prepare("如何初始化SDK？", [], "bm25", kb["id"], kb)
    assert prepared.metrics["knowledge_empty"] and prepared.result.chunks == []
    assert prepared.query == "如何初始化SDK？"
    alias_turn = await engine.prepare("爱欧踢设备接入", [], "bm25", kb["id"], kb)
    assert alias_turn.query == "爱欧踢设备接入"
    original = httpx.AsyncClient

    def handler(request):
        body = json.loads(request.content)
        system = body["messages"][0]["content"]
        assert "IoT 平台使用助手" in system and "青岚" not in system and "R100" not in system
        assert "爱欧踢=IoT" in system
        assert "当前知识库为空" in body["messages"][-1]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "请先上传平台手册。",
                                    "used_chunk_ids": [],
                                    "answer_type": "unknown",
                                }
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw)
    )
    answer, sources, _, _ = await engine.answer(prepared, [], "如何初始化SDK？")
    assert "上传" in answer and sources == []


def test_generic_normalization_preserves_platform_terms():
    assert normalize("SDK初始化、删除设备、错误码100") == "sdk初始化、删除设备、错误码100"
    assert normalize("100。") == "100。"
    assert "青岚" not in SYSTEM and "R100" not in SYSTEM


def test_generated_profile_is_a_draft_until_saved(config, monkeypatch):
    config.llm_api_key = "test-only"
    config.llm_base_url = "https://api.deepseek.com"
    original = httpx.AsyncClient

    def handler(request):
        body = json.loads(request.content)
        assert body["thinking"] == {"type": "disabled"}
        assert "SDK初始化" in body["messages"][1]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                KnowledgeProfile(
                                    role="设备接入助手", scope="帮助用户初始化SDK并完成设备接入。"
                                ).model_dump()
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw)
    )
    with TestClient(create_app(config)) as client:
        kb = create_base(client, "平台手册")
        client.post(
            "/api/documents",
            json={
                "filename": "SDK.txt",
                "content": "SDK初始化需要先创建接入密钥，再调用初始化接口。",
                "knowledge_base_id": kb["id"],
            },
        ).raise_for_status()
        response = client.post("/api/knowledge-bases/" + kb["id"] + "/profile-draft")
        response.raise_for_status()
        assert response.json()["profile"]["role"] == "设备接入助手"
        stored = next(k for k in client.get("/api/knowledge-bases").json() if k["id"] == kb["id"])
        assert stored["profile"]["role"] == "IoT 平台使用助手"


async def test_qdrant_filters_scope_before_ranking(config, monkeypatch):
    registry = RetrievalRegistry(config)
    kb = registry.repository.knowledge.save(KnowledgeInput(name="IoT"))
    from app.knowledge.documents import build_document

    registry.repository.save(
        build_document(
            "手册.txt", "SDK初始化参数为平台颁发的接入密钥。", knowledge_base_id=kb["id"]
        )
    )
    index = KnowledgeIndex(config, registry.repository)
    monkeypatch.setattr(index, "encode", lambda *a, **kw: [[0.0] * 512])

    async def request(method, path, body):
        filters = body["filter"]["must"]
        # Protocol/version-looking text must not become an extra semantic filter.
        assert len(filters) == 2
        assert {"key": "knowledge_base_id", "match": {"value": kb["id"]}} in filters
        assert all(
            condition["must"][0]["match"]["value"]
            in {d["id"] for d in registry.repository.list(knowledge_base_id=kb["id"])}
            for condition in filters[0]["should"]
        )
        return {"points": []}

    monkeypatch.setattr(index, "request", request)
    await index.search("SDK3.0通过HTTP2初始化", knowledge_base_id=kb["id"])


def test_legacy_document_and_session_migrate_without_losing_ids(config):
    # Reproduce the previous single-base schema, including its global filename constraint.
    with sqlite3.connect(config.database_path) as db:
        db.executescript("""CREATE TABLE kb_documents (
            id TEXT PRIMARY KEY, filename TEXT UNIQUE COLLATE NOCASE, title TEXT NOT NULL,
            content TEXT NOT NULL, version TEXT NOT NULL, indexed_version TEXT NOT NULL DEFAULT '',
            deleted INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL);
            INSERT INTO kb_documents VALUES ('old-doc','手册.txt','旧资料',
            '原有资料内容保持不变，用于迁移验证。','v1','v1',0,'now');
            CREATE TABLE sessions (id TEXT PRIMARY KEY, owner TEXT NOT NULL, title TEXT NOT NULL,
            backend TEXT NOT NULL, status TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            INSERT INTO sessions VALUES ('old-session','owner','旧会话','bm25','idle','now','now');
        """)
    registry = RetrievalRegistry(config)
    assert registry.repository.get("old-doc")["knowledge_base_id"] == "default"
    assert registry.repository.get("old-doc")["indexed_version"] == ""
    assert Store(config.database_path).session("old-session")["knowledge_base_id"] == "default"
