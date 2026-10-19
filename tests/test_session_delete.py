import asyncio

import httpx
from fastapi.testclient import TestClient

from app.main import create_app


def test_delete_is_owner_scoped_and_removes_only_session_history(config):
    app = create_app(config)
    with TestClient(app) as alice:
        first = alice.post("/api/sessions", json={}).json()["id"]
        second = alice.post("/api/sessions", json={}).json()["id"]
        alice.post(
            f"/api/sessions/{first}/chat",
            json={"text": "R100如何恢复出厂设置", "preview": True},
        ).raise_for_status()
        assert len(app.state.store.messages(first)) == 2
        documents = alice.get("/api/documents").json()
        with TestClient(app) as bob:
            assert bob.delete(f"/api/sessions/{first}").status_code == 404
        assert alice.get(f"/api/sessions/{first}").status_code == 200
        response = alice.delete(f"/api/sessions/{first}")
        assert response.status_code == 200 and response.json()["status"] == "deleted"
        assert alice.get(f"/api/sessions/{first}").status_code == 404
        assert alice.delete(f"/api/sessions/{first}").status_code == 404
        assert app.state.store.messages(first) == []
        assert [s["id"] for s in alice.get("/api/sessions").json()] == [second]
        assert alice.get("/api/documents").json() == documents
    with TestClient(create_app(config)) as reopened:
        assert reopened.get(f"/api/sessions/{first}").status_code == 404
        assert reopened.app.state.store.messages(first) == []


async def test_active_or_generating_session_cannot_be_deleted(config, monkeypatch):
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
        sid = (await client.post("/api/sessions", json={})).json()["id"]
        app.state.store.set_status(sid, "active")
        response = await client.delete(f"/api/sessions/{sid}")
        assert response.status_code == 409 and "结束通话" in response.json()["detail"]
        app.state.store.set_status(sid, "idle")
        pending = asyncio.create_task(
            client.post(
                f"/api/sessions/{sid}/chat", json={"text": "R100如何恢复出厂设置", "preview": True}
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            response = await client.delete(f"/api/sessions/{sid}")
            assert response.status_code == 409 and "等待回答" in response.json()["detail"]
        finally:
            release.set()
            (await pending).raise_for_status()
        assert (await client.delete(f"/api/sessions/{sid}")).status_code == 200
