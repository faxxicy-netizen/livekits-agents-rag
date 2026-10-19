import asyncio
from contextlib import asynccontextmanager

import pytest
from aiohttp import WSMsgType, web
from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, stt

from app.voice.dashscope import DashScopeSTT, DashScopeTTS

OPTIONS = APIConnectOptions(max_retry=0)
PCM = b"\x01\x00" * 2400


async def event(ws, task_id, kind, payload=None, **header):
    await ws.send_json(
        {"header": {"event": kind, "task_id": task_id, **header}, "payload": payload or {}}
    )


@asynccontextmanager
async def service(handler):
    app = web.Application()
    app.router.add_get("/speech", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/speech"
    finally:
        await runner.cleanup()


async def test_asr_live_updates_wait_for_start_and_ignore_heartbeat(config):
    received_audio = []

    async def handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        run = await ws.receive_json()
        tid = run["header"]["task_id"]
        # No PCM is sent until the service acknowledges run-task.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(ws.receive(), 0.03)
        await event(ws, tid, "task-started")
        async for message in ws:
            if message.type == WSMsgType.BINARY:
                received_audio.append(message.data)
                sentence = {"sentence_id": 0, "begin_time": 0, "text": "如何上传文档"}
                await event(ws, tid, "result-generated", {"output": {"sentence": sentence}})
                sentence.update(sentence_end=True, end_time=100)
                payload = {"output": {"sentence": sentence}, "usage": {"input_tokens": 3}}
                await event(ws, tid, "result-generated", payload)
                await event(ws, tid, "result-generated", payload)  # duplicate final
                await event(
                    ws,
                    tid,
                    "result-generated",
                    {"output": {"sentence": {"heartbeat": True, "text": "ignored"}}},
                )
            elif message.type == WSMsgType.TEXT:
                await event(ws, tid, "task-finished", {"usage": {"input_tokens": 3}})
                break
        return ws

    async with service(handler) as url:
        provider = DashScopeSTT(config, url=url)
        stream = provider.stream(conn_options=OPTIONS)
        stream.push_frame(rtc.AudioFrame(PCM, 16000, 1, 2400))
        stream.end_input()
        try:
            events = [item async for item in stream]
            assert len(received_audio) == 1
            finals = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
            assert [e.alternatives[0].text for e in finals] == ["如何上传文档"]
            assert any(e.type == stt.SpeechEventType.INTERIM_TRANSCRIPT for e in events)
            assert stream.usage["input_tokens"] == 3  # not accumulated three times
        finally:
            await stream.aclose()
            await provider.aclose()


async def test_tts_streams_before_text_finishes_and_reuses_socket(config):
    connections, received = [], []

    async def handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        connections.append(ws)
        tid = None
        async for message in ws:
            if message.type != WSMsgType.TEXT:
                continue
            data = message.json()
            action, tid = data["header"]["action"], data["header"]["task_id"]
            if action == "run-task":
                await event(ws, tid, "task-started")
            elif action == "continue-task":
                received.append(data["payload"]["input"]["text"])
                await ws.send_bytes(PCM)
            elif action == "finish-task":
                await event(
                    ws, tid, "task-finished", {"usage": {"input_tokens": 2, "output_tokens": 3}}
                )
        return ws

    async with service(handler) as url:
        provider = DashScopeTTS(config, url=url)
        try:
            stream = provider.stream(conn_options=OPTIONS)
            stream.push_text("你好。")
            # Receiving a frame without calling flush/end_input proves duplex streaming.
            first = await asyncio.wait_for(anext(stream), 2)
            assert first.frame.sample_rate == 24000
            stream.push_text("请问。")
            stream.end_input()
            assert [chunk async for chunk in stream]
            await stream.aclose()
            async with provider.synthesize("再见。", conn_options=OPTIONS) as second:
                assert [chunk async for chunk in second]
            assert len(connections) == 1
            assert received == ["你好。", "请问。", "再见。"]
            assert provider.last_usage == {"input_tokens": 2, "output_tokens": 3}
        finally:
            await provider.aclose()


async def test_cancel_discards_old_audio_connection(config):
    connections, cancelled = [], asyncio.Event()

    async def handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        connections.append(ws)
        async for message in ws:
            if message.type != WSMsgType.TEXT:
                continue
            data = message.json()
            tid, action = data["header"]["task_id"], data["header"]["action"]
            if action == "run-task":
                await event(ws, tid, "task-started")
            elif action == "continue-task":
                await ws.send_bytes(PCM)
            elif data["payload"]["input"].get("directive") == "cancel":
                cancelled.set()
            else:
                await event(ws, tid, "task-finished")
        return ws

    async with service(handler) as url:
        provider = DashScopeTTS(config, url=url)
        try:
            stream = provider.stream(conn_options=OPTIONS)
            stream.push_text("需要打断的回答。")
            await asyncio.wait_for(anext(stream), 2)
            await asyncio.wait_for(stream.aclose(), 2)
            await asyncio.wait_for(cancelled.wait(), 2)
            async with provider.synthesize("新的回答。", conn_options=OPTIONS) as second:
                assert [chunk async for chunk in second]
            assert len(connections) == 2
        finally:
            await provider.aclose()


async def test_provider_failure_does_not_expose_message_or_key(config):
    async def handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        run = await ws.receive_json()
        await event(
            ws,
            run["header"]["task_id"],
            "task-failed",
            error_code="InvalidApiKey",
            error_message="secret-do-not-leak",
        )
        return ws

    async with service(handler) as url:
        provider = DashScopeTTS(config, url=url)
        try:
            with pytest.raises(APIStatusError) as caught:
                async with provider.synthesize("你好。", conn_options=OPTIONS) as stream:
                    await anext(stream)
            assert "secret-do-not-leak" not in str(caught.value)
            assert caught.value.status_code == 401
        finally:
            await provider.aclose()
