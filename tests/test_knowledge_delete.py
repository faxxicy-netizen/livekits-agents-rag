import asyncio

import httpx
from fastapi.testclient import TestClient

from app.knowledge.vector import VectorError
from app.main import create_app


def create_base(client, name):
    response = client.post("/api/knowledge-bases", json={"name": name})
    response.raise_for_status()
    return response.json()["id"]


def upload(client, kb):
    response = client.post(
        "/api/documents",
        json={
            "knowledge_base_id": kb,
            "filename": "手册.txt",
            "content": "平台设备接入需要先创建凭据，再登记设备并建立连接。",
        },
    )
    response.raise_for_status()
    return response.json()["document"]


def test_delete_cleans_only_bound_data_and_preserves_readable_history(config):
    app = create_app(config)
    with TestClient(app) as client:
        kb, other = create_base(client, "平台资料"), create_base(client, "其他资料")
        doc, other_doc = upload(client, kb), upload(client, other)
        client.put(
            "/api/documents/" + doc["id"],
            json={
                "content": "平台设备接入需要先创建凭据，再登记设备，最后建立安全连接。",
            },
        ).raise_for_status()
        sid = client.post("/api/sessions", json={"knowledge_base_id": kb}).json()["id"]
        client.post(
            f"/api/sessions/{sid}/chat",
            json={
                "text": "平台设备如何接入",
                "preview": True,
            },
        ).raise_for_status()
        before = client.get(f"/api/sessions/{sid}").json()["messages"]
        assert before[-1]["citations"]
        # Include soft-deleted documents and their archived versions in the cleanup.
        client.delete("/api/documents/" + doc["id"]).raise_for_status()
        client.delete("/api/knowledge-bases/" + kb).raise_for_status()
        assert all(k["id"] != kb for k in client.get("/api/knowledge-bases").json())
        history = client.get(f"/api/sessions/{sid}").json()
        assert history["status"] == "ended" and history["messages"] == before
        assert history["knowledge_profile"]["name"] == "平台资料"
        assert client.post(f"/api/sessions/{sid}/token").status_code == 404
        assert client.post(f"/api/sessions/{sid}/chat", json={"text": "继续"}).status_code == 404
        assert client.post("/api/sessions", json={"knowledge_base_id": kb}).status_code == 404
        assert (
            client.post(
                "/api/search",
                json={
                    "knowledge_base_id": kb,
                    "query": "设备接入",
                },
            ).status_code
            == 404
        )
        assert client.get("/api/documents/" + other_doc["id"]).status_code == 200
        assert client.delete("/api/knowledge-bases/" + kb).status_code == 404
        assert client.delete("/api/knowledge-bases/default").status_code == 409
        with app.state.store.connect() as db:
            assert (
                db.execute(
                    "SELECT count(*) FROM kb_documents WHERE knowledge_base_id=?", (kb,)
                ).fetchone()[0]
                == 0
            )
            assert (
                db.execute(
                    "SELECT count(*) FROM kb_document_versions WHERE document_id=?", (doc["id"],)
                ).fetchone()[0]
                == 0
            )
    with TestClient(create_app(config)) as reopened:
        assert all(k["id"] != kb for k in reopened.get("/api/knowledge-bases").json())


def test_vector_failure_leaves_knowledge_and_documents_usable(config, monkeypatch):
    app = create_app(config)
    with TestClient(app) as client:
        kb = create_base(client, "不能误删")
        doc = upload(client, kb)
        config.vector_enabled = True

        async def fail(*args):
            raise VectorError("向量服务不可用")

        monkeypatch.setattr(app.state.registry.manager.index, "delete_knowledge_base", fail)
        assert client.delete("/api/knowledge-bases/" + kb).status_code == 503
        assert client.get("/api/documents/" + doc["id"]).status_code == 200
        assert kb not in app.state.context.deleting_knowledge
        assert client.post("/api/sessions", json={"knowledge_base_id": kb}).status_code == 200


async def test_active_calls_or_inflight_answers_block_deletion(config, monkeypatch):
    app = create_app(config)
    entered, release = asyncio.Event(), asyncio.Event()
    original = app.state.engine.prepare

    async def delayed(*args):
        entered.set()
        await release.wait()
        return await original(*args)

    monkeypatch.setattr(app.state.engine, "prepare", delayed)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost"
    ) as client:
        kb = (await client.post("/api/knowledge-bases", json={"name": "通话资料"})).json()["id"]
        sid = (await client.post("/api/sessions", json={"knowledge_base_id": kb})).json()["id"]
        app.state.store.set_status(sid, "active")
        assert (await client.delete("/api/knowledge-bases/" + kb)).status_code == 409
        app.state.store.set_status(sid, "idle")
        pending = asyncio.create_task(
            client.post(
                f"/api/sessions/{sid}/chat",
                json={
                    "text": "如何使用",
                    "preview": True,
                },
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert (await client.delete("/api/knowledge-bases/" + kb)).status_code == 409
        finally:
            release.set()
            (await pending).raise_for_status()
        assert (await client.delete("/api/knowledge-bases/" + kb)).status_code == 200


async def test_scoped_vector_deletion_blocks_new_turns_and_writes(config, monkeypatch):
    app = create_app(config)
    entered, release = asyncio.Event(), asyncio.Event()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost"
    ) as client:
        kb = (await client.post("/api/knowledge-bases", json={"name": "删除中的资料"})).json()["id"]
        sid = (await client.post("/api/sessions", json={"knowledge_base_id": kb})).json()["id"]
        config.vector_enabled = True

        async def request(method, path, body):
            assert method == "POST" and path.endswith("/points/delete?wait=true")
            assert body == {
                "filter": {"must": [{"key": "knowledge_base_id", "match": {"value": kb}}]}
            }
            entered.set()
            await release.wait()
            return {}

        monkeypatch.setattr(app.state.registry.manager.index, "request", request)
        deleting = asyncio.create_task(client.delete("/api/knowledge-bases/" + kb))
        await asyncio.wait_for(entered.wait(), 2)
        try:
            assert (await client.post(f"/api/sessions/{sid}/token")).status_code == 409
            assert (
                await client.post(
                    f"/api/sessions/{sid}/chat",
                    json={
                        "text": "如何使用",
                        "preview": True,
                    },
                )
            ).status_code == 409
            creating = asyncio.create_task(
                client.post("/api/sessions", json={"knowledge_base_id": kb})
            )
            uploading = asyncio.create_task(
                client.post(
                    "/api/documents",
                    json={
                        "knowledge_base_id": kb,
                        "filename": "新文档.txt",
                        "content": "删除过程中不能写入新的文档和向量。",
                    },
                )
            )
            await asyncio.sleep(0)
            assert not creating.done()
        finally:
            release.set()
            (await deleting).raise_for_status()
        assert (await creating).status_code == 404
        assert (await uploading).status_code in {404, 409}
        assert kb not in app.state.context.deleting_knowledge
