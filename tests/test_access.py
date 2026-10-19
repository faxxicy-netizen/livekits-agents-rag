from fastapi.testclient import TestClient

from app.api.access import COOKIE, hash_password
from app.main import create_app


def protected(config):
    config.app_access_password_hash = hash_password("test-demo-password")
    config.app_session_secret = "test-session-secret-with-at-least-32-characters"
    config.app_secure_cookies = True
    config.app_public_origin = "https://demo.example"
    return create_app(config)


def test_password_guards_pages_and_all_business_endpoints(config):
    with TestClient(
        protected(config), base_url="https://demo.example", follow_redirects=False
    ) as client:
        assert client.get("/").status_code == 303
        for path in ("/api/status", "/api/knowledge-bases", "/api/documents", "/api/sessions"):
            assert client.get(path).status_code == 401
        assert client.post("/api/search", json={"query": "hi"}).status_code == 401
        assert client.post("/api/sessions/fake/token").status_code == 401
        assert client.get("/healthz").status_code == 200
        assert client.get("/docs").status_code == 303
        assert "test-demo-password" not in client.get("/login").text
        assert client.post("/login", data={"password": "wrong"}).status_code == 401
        login = client.post("/login", data={"password": "test-demo-password"})
        assert login.status_code == 303
        cookie = login.headers["set-cookie"]
        assert "HttpOnly" in cookie and "Secure" in cookie and "SameSite=strict" in cookie
        assert client.get("/api/sessions").status_code == 200
        client.post("/logout")
        assert client.get("/api/sessions").status_code == 401


def test_tampered_cookie_and_foreign_origin_are_rejected(config):
    with TestClient(protected(config), base_url="https://demo.example") as client:
        client.cookies.set(COOKIE, "forged-session")
        assert client.get("/api/status").status_code == 401
        assert (
            client.post(
                "/login",
                data={"password": "test-demo-password"},
                headers={"Origin": "https://evil.example"},
            ).status_code
            == 403
        )
        client.post(
            "/login",
            data={"password": "test-demo-password"},
            headers={"Origin": "https://demo.example"},
        )
        assert client.get("/api/sessions").status_code == 200


def test_password_attempts_are_limited(config):
    with TestClient(protected(config), base_url="https://demo.example") as client:
        for _ in range(5):
            assert client.post("/login", data={"password": "wrong"}).status_code == 401
        assert client.post("/login", data={"password": "wrong"}).status_code == 429


def test_public_url_and_one_call_limit(config):
    config.llm_api_key = "test-only"
    config.stt_provider = config.tts_provider = "dashscope"
    config.dashscope_api_key = "test-only"
    config.dashscope_workspace_id = "llm-test"
    config.livekit_api_secret = "test-livekit-secret-with-32-characters"
    config.livekit_public_url = "wss://demo.example/livekit"
    config.max_active_calls = 1
    with TestClient(protected(config), base_url="https://demo.example") as client:
        client.post("/login", data={"password": "test-demo-password"})
        first = client.post("/api/sessions", json={}).json()["id"]
        response = client.post(f"/api/sessions/{first}/token")
        assert response.status_code == 200
        assert response.json()["url"] == config.livekit_public_url
        second = client.post("/api/sessions", json={}).json()["id"]
        assert client.post(f"/api/sessions/{second}/token").status_code == 409


def test_successful_login_resets_failed_attempt_count(config):
    with TestClient(
        protected(config), base_url="https://demo.example", follow_redirects=False
    ) as client:
        for _ in range(4):
            assert client.post("/login", data={"password": "wrong"}).status_code == 401
        assert client.post("/login", data={"password": "test-demo-password"}).status_code == 303
        client.post("/logout")
        assert client.post("/login", data={"password": "wrong"}).status_code == 401
