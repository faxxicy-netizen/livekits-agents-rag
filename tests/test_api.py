import jwt
import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.mark.usefixtures("seeded_knowledge")
def test_history_persists_and_is_owner_scoped(config):
    app = create_app(config)
    with TestClient(app) as alice:
        sid = alice.post("/api/sessions", json={"backend": "bm25"}).json()["id"]
        response = alice.post(
            f"/api/sessions/{sid}/chat", json={"text": "R100怎么恢复出厂设置？", "preview": True}
        )
        assert response.status_code == 200
        answer = response.json()
        assert answer["text"].startswith("资料摘录：")
        assert answer["citations"][0]["document_name"] == "04-重置与密码.md"
        assert answer["answer_mode"] == "extractive"
        assert len(alice.get(f"/api/sessions/{sid}").json()["messages"]) == 2
        cookie = alice.cookies.get("support_owner")
    with TestClient(create_app(config)) as reopened:
        reopened.cookies.set("support_owner", cookie)
        assert len(reopened.get(f"/api/sessions/{sid}").json()["messages"]) == 2
    with TestClient(app) as bob:
        assert bob.get(f"/api/sessions/{sid}").status_code == 404
        assert bob.get("/api/sessions").json() == []


def test_voice_missing_resources_are_actionable(config):
    with TestClient(create_app(config)) as client:
        sid = client.post("/api/sessions", json={}).json()["id"]
        response = client.post(f"/api/sessions/{sid}/token")
        assert response.status_code == 424
        assert "LLM_API_KEY" in response.json()["detail"]


def test_tokens_are_short_lived_and_scoped(config):
    config.llm_api_key = "test-only"
    config.local_asr_dir.mkdir()
    (config.local_asr_dir / "model.int8.onnx").touch()
    config.local_tts_dir.mkdir()
    for filename in ["config.json", "kokoro-v1_1-zh.pth", "voices/zf_001.pt"]:
        target = config.local_tts_dir / filename
        target.parent.mkdir(exist_ok=True)
        target.touch()
    with TestClient(create_app(config)) as client:
        sid = client.post("/api/sessions", json={}).json()["id"]
        response = client.post(f"/api/sessions/{sid}/token")
        assert response.status_code == 200
        payload = jwt.decode(
            response.json()["token"], config.livekit_api_secret, algorithms=["HS256"]
        )
        assert payload["video"]["room"] == "support-" + sid
        assert payload["exp"] - payload["nbf"] <= 600
        assert not payload["video"].get("roomAdmin")
        assert payload["roomConfig"]["agents"][0]["agentName"] == config.livekit_agent_name


def test_cross_origin_and_path_traversal_are_rejected(config):
    with TestClient(create_app(config)) as client:
        response = client.post("/api/sessions", json={}, headers={"Origin": "https://evil.example"})
        assert response.status_code == 403
        response = client.post("/api/documents", json={"filename": "../.env", "content": "x" * 20})
        assert response.status_code == 400


def test_failed_model_is_not_reported_as_no_evidence(config):
    with TestClient(create_app(config)) as client:
        sid = client.post("/api/sessions", json={}).json()["id"]
        response = client.post(f"/api/sessions/{sid}/chat", json={"text": "R100怎么恢复出厂设置"})
        assert response.status_code == 502
        assert "LLM_API_KEY" in response.json()["detail"]
        last = client.get(f"/api/sessions/{sid}").json()["messages"][-1]
        assert last["answer_mode"] == "error" and last["citations"] == []


def test_plain_text_import_update_and_version_backup(config):
    with TestClient(create_app(config)) as client:
        first = "R100测试设备的保修期限是十二个月，申请保修需要购买凭证。"
        response = client.post(
            "/api/documents", json={"filename": "导入测试.txt", "content": first}
        )
        assert response.status_code == 200
        original = client.get("/api/documents").json()[0]
        assert original["chunks"] == 1
        search = client.post(
            "/api/search", json={"query": "R100保修期限", "backend": "bm25"}
        ).json()
        assert search["chunks"][0]["document_name"] == "导入测试.txt"
        updated = "R100测试设备的保修期限是二十四个月，申请保修需要设备序列号。"
        client.put(
            "/api/documents/" + original["id"],
            json={"content": updated, "expected_version": original["version"]},
        ).raise_for_status()
        current = client.get("/api/documents").json()[0]
        assert current["version"] != original["version"]
        with client.app.state.registry.repository.connect() as db:
            backup = db.execute("SELECT content FROM kb_document_versions").fetchall()
        assert len(backup) == 1 and backup[0][0] == first
