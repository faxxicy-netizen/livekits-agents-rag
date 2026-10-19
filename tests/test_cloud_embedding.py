import json

import httpx
import pytest

from app.config import Settings
from app.knowledge.vector import KnowledgeIndex, VectorError


def test_cloud_embedding_configuration_loads_from_dotenv(tmp_path):
    env = tmp_path / "server.env"
    env.write_text("EMBEDDING_PROVIDER=dashscope\nEMBEDDING_DIMENSIONS='1024'\n", encoding="utf-8")
    config = Settings(_env_file=env)
    assert config.embedding_dimension == 1024


async def test_cloud_embedding_batches_and_matches_response_order(config, monkeypatch):
    config.embedding_provider = "dashscope"
    config.embedding_dimensions = 256
    config.dashscope_api_key = "test-only"
    config.dashscope_workspace_id = "llm-test"
    original = httpx.AsyncClient
    sizes = []

    def handler(request):
        data = json.loads(request.content)
        assert request.url.path == "/compatible-mode/v1/embeddings"
        assert request.headers["Authorization"] == "Bearer test-only"
        assert data["model"] == "qwen3.7-text-embedding-flash"
        assert not any(t.startswith("为这个句子") for t in data["input"])
        sizes.append(len(data["input"]))
        return httpx.Response(
            200,
            json={
                "data": list(
                    reversed(
                        [
                            {"index": i, "embedding": [float(i + 1)] * 256}
                            for i in range(len(data["input"]))
                        ]
                    )
                )
            },
        )

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw)
    )
    index = KnowledgeIndex(config, None)
    try:
        vectors = await index.embed(["查询文本"] * 23, query=True)
        assert sizes == [20, 3]
        assert [v[0] for v in vectors[:20]] == list(range(1, 21))
    finally:
        await index.aclose()


async def test_invalid_cloud_vector_fails_without_provider_response_leak(config, monkeypatch):
    config.embedding_provider = "dashscope"
    config.dashscope_api_key = "test-only"
    config.dashscope_workspace_id = "llm-test"
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: original(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 2]}]})
            ),
            **kw,
        ),
    )
    index = KnowledgeIndex(config, None)
    try:
        with pytest.raises(VectorError):
            await index.embed(["查询"])
    finally:
        await index.aclose()


async def test_same_dimensions_but_different_embedding_model_refuses_old_collection(config):
    index = KnowledgeIndex(config, None)

    async def request(method, path, body=None, **kw):
        if method == "GET":
            return {"config": {"params": {"vectors": {"size": 512}}}, "points_count": 1}
        return {"points": [{"payload": {"embedding_model": "a-different-model"}}]}

    index.request = request
    with pytest.raises(VectorError, match="embedding 模型不同"):
        await index.initialize()
