import json

import httpx
import pytest

from app.conversation.engine import ModelError, SupportEngine
from app.knowledge.retrieval import DifyRetriever, RetrievalError, RetrievalRegistry


def mock_http(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )


async def test_dify_maps_evidence_and_preserves_dataset_search_settings(config, monkeypatch):
    config.dify_api_key, config.dify_dataset_id = "test-only", "dataset"

    def handler(request):
        assert request.url.path == "/v1/datasets/dataset/retrieve"
        assert json.loads(request.content) == {"query": "R100重置"}
        return httpx.Response(
            200,
            json={
                "records": [
                    {
                        "score": 0.9,
                        "segment": {
                            "id": "chunk-a",
                            "document_id": "doc-a",
                            "content": "按住八秒",
                            "document": {"name": "说明.md"},
                        },
                    }
                ]
            },
        )

    mock_http(monkeypatch, handler)
    result = await DifyRetriever(config).search("R100重置")
    assert result.chunks[0].document_name == "说明.md"
    assert result.chunks[0].chunk_id == "chunk-a"


async def test_dify_outage_is_not_empty_retrieval(config, monkeypatch):
    config.dify_api_key, config.dify_dataset_id = "test-only", "dataset"
    mock_http(monkeypatch, lambda request: httpx.Response(503))
    with pytest.raises(RetrievalError, match="检索失败"):
        await DifyRetriever(config).search("R100重置")


@pytest.mark.usefixtures("seeded_knowledge")
async def test_model_cannot_fabricate_citations(config, monkeypatch):
    config.llm_api_key = "test-only"
    engine = SupportEngine(config, RetrievalRegistry(config))
    prepared = await engine.prepare("R100如何恢复出厂设置", [], "bm25")
    mock_http(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"answer": "长按重置。", "used_chunk_ids": ["fabricated"]}
                            )
                        }
                    }
                ]
            },
        ),
    )
    with pytest.raises(ModelError):
        await engine.answer(prepared, [], "R100如何恢复出厂设置")


@pytest.mark.usefixtures("seeded_knowledge")
async def test_model_response_links_only_returned_sources(config, monkeypatch):
    config.llm_api_key = "test-only"
    engine = SupportEngine(config, RetrievalRegistry(config))
    prepared = await engine.prepare("R100如何恢复出厂设置", [], "bm25")
    first = prepared.result.chunks[0]
    mock_http(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "通电时按住 RESET 八秒，松开后等待两分钟。",
                                    "used_chunk_ids": [first.chunk_id],
                                }
                            )
                        }
                    }
                ]
            },
        ),
    )
    text, citations, mode, metrics = await engine.answer(prepared, [], "R100如何恢复出厂设置")
    assert "八秒" in text and citations == [first] and mode == "llm"
    assert metrics["generation_ms"] >= 0


@pytest.mark.usefixtures("seeded_knowledge")
async def test_deepseek_thinking_disabled_and_json_history(config, monkeypatch):
    from app.models import Message

    config.llm_base_url = "https://api.deepseek.com"
    config.llm_model = "deepseek-flash"
    config.llm_api_key = "test-only"
    engine = SupportEngine(config, RetrievalRegistry(config))
    prepared = await engine.prepare("R100如何恢复出厂设置", [], "bm25")
    chunk = prepared.result.chunks[0]

    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "deepseek-flash"
        assert body["thinking"] == {"type": "disabled"}
        assert body["response_format"] == {"type": "json_object"}
        assert json.loads(body["messages"][1]["content"])["answer"] == "上一条实际播报"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"answer": "按住八秒", "used_chunk_ids": [chunk.chunk_id]}
                            )
                        }
                    }
                ]
            },
        )

    mock_http(monkeypatch, handler)
    history = [
        Message(
            id="m",
            session_id="s",
            turn_id="t",
            role="assistant",
            text="上一条实际播报",
            created_at="2026-10-14T00:00:00Z",
        )
    ]
    await engine.answer(prepared, history, "R100如何恢复出厂设置")
