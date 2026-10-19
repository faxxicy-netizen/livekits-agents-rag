import json

import httpx
import pytest

from app.conversation.engine import ModelError, SupportEngine
from app.knowledge.retrieval import RetrievalRegistry
from app.models import Message


def message(role, text, **metrics):
    return Message(
        id="test",
        session_id="s",
        turn_id="t",
        role=role,
        text=text,
        metrics=metrics,
        created_at="2026-10-15T00:00:00Z",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text", ["如何请假？", "如何初始化SDK？", "100。", "那怎么重置？", "SDK3.0", "拜拜。"]
)
async def test_query_is_never_rewritten_by_model_or_followup_rules(config, text):
    engine = SupportEngine(config, RetrievalRegistry(config))
    history = [
        message("user", "R100如何重启？"),
        message("assistant", "请补充使用环境。", requested_slot="context"),
    ]
    prepared = await engine.prepare(text, history, "bm25")
    assert prepared.query == text
    assert prepared.metrics["retrieval_query"] == text
    assert "product" not in prepared.metrics and "model_only" not in prepared.metrics


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["你好", "你都可以干什么？", "青蓝100。", "100。", "拜拜。"])
async def test_dialogue_is_sent_to_llm_without_domain_gates(config, monkeypatch, text):
    config.llm_api_key = "test-only"
    engine = SupportEngine(config, RetrievalRegistry(config))
    prepared = await engine.prepare(text, [], "bm25")
    assert prepared.direct_answer == ""
    original = httpx.AsyncClient

    def handler(request):
        body = json.loads(request.content)
        assert body["messages"][-1]["content"].startswith(text)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "自然对话回复",
                                    "used_chunk_ids": [],
                                    "answer_type": "conversation",
                                    "requested_slot": None,
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
    answer, sources, mode, metrics = await engine.answer(prepared, [], text)
    assert answer == "自然对话回复" and sources == [] and mode == "llm"
    assert metrics["answer_type"] == "conversation"


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["打开卫星联网开关。", "无法连接时打开卫星联网开关。"])
async def test_factual_answer_still_requires_returned_evidence(config, monkeypatch, answer):
    config.llm_api_key = "test-only"
    engine = SupportEngine(config, RetrievalRegistry(config))
    prepared = await engine.prepare("R100卫星联网怎么用", [], "bm25")
    original = httpx.AsyncClient
    response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "answer": answer,
                            "used_chunk_ids": [],
                            "answer_type": "answer",
                        }
                    )
                }
            }
        ]
    }
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: original(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response)), **kw
        ),
    )
    with pytest.raises(ModelError):
        await engine.answer(prepared, [], "R100卫星联网怎么用")
