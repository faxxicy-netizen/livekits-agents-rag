import asyncio
import json

import httpx
import pytest

from app.conversation.engine import ModelError, PreparedTurn, SupportEngine
from app.conversation.streaming import SentenceDecoder
from app.models import Citation, SearchResult


def sentence(answer="第一句。", ids=None, kind="answer"):
    return {
        "answer": answer,
        "used_chunk_ids": ["S1"] if ids is None else ids,
        "answer_type": kind,
        "requested_slot": None,
    }


def encoded(value):
    return json.dumps(value, ensure_ascii=False)


def sse(content="", *, finish=None, usage=None):
    data = {"choices": [{"delta": {"content": content}, "finish_reason": finish}]}
    if usage:
        data["usage"] = usage
    return ("data: " + encoded(data) + "\r\n\r\n").encode()


class Body(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks, self.closed = chunks, False

    async def __aiter__(self):
        async for chunk in self.chunks():
            yield chunk

    async def aclose(self):
        self.closed = True


def engine_with_body(config, monkeypatch, body):
    config.llm_api_key = "test-only"
    config.llm_base_url = "https://api.deepseek.com"
    config.llm_model = "deepseek-flash"
    original = httpx.AsyncClient

    def handler(request):
        data = json.loads(request.content)
        assert data["stream"] and data["thinking"] == {"type": "disabled"}
        assert data["response_format"] == {"type": "json_object"}
        return httpx.Response(200, stream=body, headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw)
    )
    engine = SupportEngine(config, None)
    source = Citation(chunk_id="actual-id", document_id="doc", document_name="手册", text="证据")
    prepared = PreparedTurn("问题", SearchResult(query="问题", backend="bm25", chunks=[source]))
    return engine, prepared, source


def test_decoder_handles_escaped_quotes_backslashes_unicode_and_split_boundaries():
    item = sentence('路径 C:\\文档，提示“你好”，字符 "}]" 与 emoji 😀。')
    raw = json.dumps({"segments": [item, sentence("第二句。")]}, ensure_ascii=True)
    parser = SentenceDecoder()
    result = []
    for char in raw:  # every JSON escape can be split across deltas
        result.extend(parser.feed(char))
    parser.finish()
    assert result == [item, sentence("第二句。")]


@pytest.mark.parametrize(
    "raw", ['{"segments":[', '{"segments":[]}', '{"segments":[' + encoded(sentence()) + ",]}"]
)
def test_decoder_rejects_incomplete_and_invalid_envelopes(raw):
    parser = SentenceDecoder()
    with pytest.raises(ValueError):
        list(parser.feed(raw))
        parser.finish()


async def test_first_sentence_arrives_before_response_finishes(config, monkeypatch):
    release = asyncio.Event()

    async def chunks():
        data = sse('{"segments":[' + encoded(sentence()))
        yield b": keepalive\r\n\r\n"
        for offset in range(0, len(data), 7):
            yield data[offset : offset + 7]  # split UTF-8/SSE boundaries too
        await release.wait()
        yield sse("," + encoded(sentence("第二句。")) + "]}")
        yield sse(finish="stop", usage={"total_tokens": 80})
        yield b"data: [DONE]\r\n\r\n"

    body = Body(chunks)
    engine, prepared, source = engine_with_body(config, monkeypatch, body)
    stream = engine.stream_answer(prepared, [], "问题")
    try:
        first = await asyncio.wait_for(anext(stream), 1)
        assert first.answer == "第一句。" and first.citations == [source]
        assert not release.is_set() and not first.metrics["stream_completed"]
        release.set()
        assert [s.answer async for s in stream] == ["第二句。"]
        assert first.metrics["stream_completed"] and first.metrics["tokens"] == 80
        assert body.closed
    finally:
        await stream.aclose()
        await engine.aclose()


@pytest.mark.parametrize(
    "item", [sentence(ids=["fabricated"]), sentence(ids=[]), sentence(kind="bogus")]
)
async def test_bad_sentence_is_blocked_before_any_text_is_yielded(config, monkeypatch, item):
    async def chunks():
        yield sse(encoded({"segments": [item]}))
        yield sse(finish="stop")
        yield b"data: [DONE]\n\n"

    body = Body(chunks)
    engine, prepared, _ = engine_with_body(config, monkeypatch, body)
    try:
        with pytest.raises(ModelError):
            await anext(engine.stream_answer(prepared, [], "问题"))
        assert body.closed
    finally:
        await engine.aclose()


async def test_cancellation_closes_live_http_response_without_replaying(config, monkeypatch):
    waiting = asyncio.Event()

    async def chunks():
        yield sse('{"segments":[' + encoded(sentence()))
        waiting.set()
        await asyncio.Event().wait()

    body = Body(chunks)
    engine, prepared, _ = engine_with_body(config, monkeypatch, body)
    stream = engine.stream_answer(prepared, [], "问题")
    try:
        first = await anext(stream)
        pending = asyncio.create_task(anext(stream))
        await waiting.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert body.closed and first.metrics["stream_cancelled"]
    finally:
        await stream.aclose()
        await engine.aclose()


async def test_network_cut_after_first_sentence_keeps_prefix_without_retry(config, monkeypatch):
    async def chunks():
        yield sse('{"segments":[' + encoded(sentence()))
        raise httpx.ReadError("provider disconnected")

    body = Body(chunks)
    engine, prepared, _ = engine_with_body(config, monkeypatch, body)
    stream = engine.stream_answer(prepared, [], "问题")
    try:
        assert (await anext(stream)).answer == "第一句。"
        with pytest.raises(ModelError):
            await anext(stream)
        assert body.closed
    finally:
        await stream.aclose()
        await engine.aclose()
