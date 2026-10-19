"""Native DashScope run-task WebSockets, PCM streaming, and LiveKit adapters."""

import asyncio
import contextlib
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiohttp
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    stt,
    tts,
)

from app.config import Settings

logger = logging.getLogger("support.dashscope")
ASR_RATE = 16000
TTS_RATE = 24000


def command(action: str, task_id: str, payload: dict) -> dict:
    return {
        "header": {"action": action, "task_id": task_id, "streaming": "duplex"},
        "payload": payload,
    }


def usage_from(event: dict) -> dict:
    payload = event.get("payload", {})
    raw = payload.get("usage") or payload.get("output", {}).get("usage") or event.get("usage") or {}
    return {
        k: v
        for k, v in raw.items()
        if k in {"input_tokens", "output_tokens", "total_tokens", "duration", "characters"}
        and isinstance(v, (int, float))
        and v >= 0
    }


def check_event(event: dict, task_id: str):
    header = event.get("header", {})
    if header.get("task_id") != task_id:
        raise APIConnectionError("阿里云语音任务编号不一致，请重试。", retryable=False)
    if header.get("event") == "task-failed":
        code = str(header.get("error_code", "Unknown"))
        code = code if re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", code) else "Unknown"
        status = 403 if "Quota" in code or "FreeTier" in code else 400
        if "ApiKey" in code or "Authentication" in code:
            status = 401
        # Provider messages and exception headers can contain credentials; expose only the code.
        raise APIStatusError(
            f"阿里云语音请求失败（{code}），请检查密钥、地域、模型权限或免费额度。",
            status_code=status,
            request_id=task_id,
            retryable=False,
        )


async def receive(ws, timeout: float | None):
    try:
        message = await asyncio.wait_for(ws.receive(), timeout=timeout)
    except (TimeoutError, aiohttp.ClientError, OSError):
        raise APIConnectionError("阿里云语音连接超时或中断，请重试。", retryable=False) from None
    if message.type == aiohttp.WSMsgType.TEXT:
        try:
            return json.loads(message.data)
        except (ValueError, TypeError):
            raise APIConnectionError("阿里云语音响应格式无效。", retryable=False) from None
    if message.type == aiohttp.WSMsgType.BINARY:
        return message.data
    raise APIConnectionError("阿里云语音连接提前关闭，请重试。", retryable=False)


async def start_task(ws, task_id: str, payload: dict, timeout: float):
    await ws.send_json(command("run-task", task_id, payload))
    event = await receive(ws, timeout)
    if not isinstance(event, dict):
        raise APIConnectionError("阿里云语音任务未正确启动。", retryable=False)
    check_event(event, task_id)
    if event["header"].get("event") != "task-started":
        raise APIConnectionError("阿里云语音任务未就绪。", retryable=False)


class DashScopeSocket:
    """One reusable TTS socket per agent; cancelled or failed tasks always discard it."""

    def __init__(self, config: Settings, *, url: str | None = None):
        self.url = url or config.dashscope_ws_url
        self.key = config.dashscope_api_key
        self.timeout = config.dashscope_speech_timeout
        self._session: aiohttp.ClientSession | None = None
        self._idle = None
        self._idle_since = 0.0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def connection(self, *, reuse=False):
        async with self._lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=None, sock_connect=self.timeout)
                )
            ws, self._idle = self._idle, None
            if ws is not None and (ws.closed or time.monotonic() - self._idle_since > 15):
                await ws.close()
                ws = None
            try:
                if ws is None:
                    ws = await self._session.ws_connect(
                        self.url,
                        headers={"Authorization": "Bearer " + self.key},
                        max_msg_size=8 * 1024 * 1024,
                        heartbeat=20 if not reuse else None,
                    )
                yield ws
                if reuse and not ws.closed:
                    self._idle, ws = ws, None
                    self._idle_since = time.monotonic()
            except aiohttp.WSServerHandshakeError as exc:
                raise APIStatusError(
                    f"阿里云语音握手失败（HTTP {exc.status}），请检查密钥和业务空间。",
                    status_code=exc.status,
                    retryable=False,
                ) from None
            except (aiohttp.ClientError, OSError, TimeoutError):
                raise APIConnectionError(
                    "无法连接阿里云语音服务，请检查网络。", retryable=False
                ) from None
            finally:
                if ws is not None:
                    await ws.close()

    async def aclose(self):
        if self._idle is not None:
            await self._idle.close()
            self._idle = None
        if self._session is not None:
            await self._session.close()


class DashScopeSTT(stt.STT):
    def __init__(self, config: Settings, *, vocabulary: dict | None = None, url=None):
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))
        self.config = config
        self.vocabulary = vocabulary or {}
        self.transport = DashScopeSocket(config, url=url)

    @property
    def model(self):
        return self.config.dashscope_asr_model

    @property
    def provider(self):
        return "aliyun-dashscope"

    def stream(self, *, language=None, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        return DashScopeSpeechStream(stt=self, conn_options=conn_options)

    async def _recognize_impl(
        self, buffer, *, language=None, conn_options=DEFAULT_API_CONNECT_OPTIONS
    ):
        stream = self.stream(conn_options=conn_options)
        try:
            frames = [buffer] if isinstance(buffer, rtc.AudioFrame) else buffer
            for frame in frames:
                stream.push_frame(frame)
            stream.end_input()
            results = [ev async for ev in stream if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=stream.request_id,
                alternatives=[
                    stt.SpeechData(
                        language="zh",
                        text="".join(ev.alternatives[0].text for ev in results),
                        metadata={"usage": stream.usage},
                    )
                ],
            )
        finally:
            await stream.aclose()

    async def aclose(self):
        await self.transport.aclose()


class DashScopeSpeechStream(stt.SpeechStream):
    def __init__(self, *, stt: DashScopeSTT, conn_options: APIConnectOptions):
        self.request_id = uuid.uuid4().hex
        self.usage: dict = {}
        self._speaking = False
        self._last_final = None
        self._last_interim = None
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=ASR_RATE)

    async def _run(self):
        provider = self._stt
        config = provider.config
        params = {
            "format": "pcm",
            "sample_rate": ASR_RATE,
            "language_hints": ["zh", "en"],
            "heartbeat": True,
            "max_sentence_silence": config.dashscope_asr_silence_ms,
        }
        if "3.1" in provider.model:
            params.update(vad_model="near_meeting_16k", keep_dialect=False)
        if provider.vocabulary:
            params["vocabulary"] = provider.vocabulary
        async with provider.transport.connection() as ws:
            await start_task(
                ws,
                self.request_id,
                {
                    "task_group": "audio",
                    "task": "asr",
                    "function": "recognition",
                    "model": provider.model,
                    "parameters": params,
                    "input": {},
                },
                config.dashscope_speech_timeout,
            )

            async def send():
                async for frame in self._input_ch:
                    if isinstance(frame, self._FlushSentinel):
                        continue  # Server VAD ends sentences; end_input ends the whole task.
                    if frame.num_channels != 1:
                        raise APIConnectionError("阿里云 ASR 需要单声道音频。", retryable=False)
                    await ws.send_bytes(frame.data.tobytes())
                await ws.send_json(command("finish-task", self.request_id, {"input": {}}))

            async def read():
                while True:
                    # A healthy microphone may be silent indefinitely. WebSocket ping/pong
                    # detects broken connections; task start/finish still have deadlines.
                    event = await receive(ws, None)
                    if not isinstance(event, dict):
                        continue
                    check_event(event, self.request_id)
                    usage = usage_from(event)
                    if usage:
                        self.usage.update(usage)  # ASR usage is cumulative, never sum snapshots.
                    if event["header"].get("event") == "task-finished":
                        return
                    sentence = event.get("payload", {}).get("output", {}).get("sentence", {})
                    self._sentence(sentence)

            tasks = [asyncio.create_task(send()), asyncio.create_task(read())]
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
                if tasks[1] in done and not tasks[0].done():
                    raise APIConnectionError("阿里云 ASR 任务提前结束，请重试。", retryable=False)
                try:
                    await asyncio.wait_for(tasks[1], config.dashscope_speech_timeout)
                except TimeoutError:
                    raise APIConnectionError(
                        "阿里云 ASR 结束任务超时，请重试。", retryable=False
                    ) from None
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.info(
                    "speech_usage kind=asr request=%s usage=%s", self.request_id, self.usage
                )

    def _sentence(self, sentence: dict):
        if not sentence or sentence.get("heartbeat"):
            return
        text = sentence.get("text", "").strip()
        final = sentence.get("sentence_end", False)
        key = (sentence.get("sentence_id"), sentence.get("begin_time"), text)
        if final and key == self._last_final:
            return
        if not self._speaking and (sentence.get("sentence_begin") or text):
            self._speaking = True
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
        if text and (final or key != self._last_interim):
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT
                    if final
                    else stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=self.request_id,
                    alternatives=[
                        stt.SpeechData(
                            language="zh",
                            text=text,
                            start_time=(sentence.get("begin_time") or 0) / 1000,
                            end_time=(sentence.get("end_time") or 0) / 1000,
                            metadata={"usage": dict(self.usage)},
                        )
                    ],
                )
            )
            self._last_interim = key
        if final:
            self._last_final = key
            if self._speaking:
                self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
                self._speaking = False


class DashScopeTTS(tts.TTS):
    def __init__(self, config: Settings, *, url=None):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True), sample_rate=TTS_RATE, num_channels=1
        )
        self.config = config
        self.transport = DashScopeSocket(config, url=url)
        self.last_usage: dict = {}

    @property
    def model(self):
        return self.config.dashscope_tts_model

    @property
    def provider(self):
        return "aliyun-dashscope"

    def synthesize(self, text: str, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        return DashScopeChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(self, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        return DashScopeSynthesizeStream(tts=self, conn_options=conn_options)

    async def audio(self, texts: AsyncIterator[str], task_id: str) -> AsyncIterator[bytes]:
        config = self.config
        usage: dict = {}
        async with self.transport.connection(reuse=True) as ws:
            await start_task(
                ws,
                task_id,
                {
                    "task_group": "audio",
                    "task": "tts",
                    "function": "SpeechSynthesizer",
                    "model": self.model,
                    "input": {},
                    "parameters": {
                        "text_type": "PlainText",
                        "voice": config.dashscope_tts_voice,
                        "format": "pcm",
                        "sample_rate": TTS_RATE,
                        "rate": config.dashscope_tts_rate,
                        "language_hints": ["zh"],
                    },
                },
                config.dashscope_speech_timeout,
            )

            async def send():
                async for text in texts:
                    if text:
                        await ws.send_json(
                            command("continue-task", task_id, {"input": {"text": text}})
                        )
                await ws.send_json(command("finish-task", task_id, {"input": {}}))

            sender = asyncio.create_task(send())
            complete, audio_bytes = False, 0
            try:
                while True:
                    event = await receive(ws, config.dashscope_speech_timeout)
                    if sender.done():
                        sender.result()
                    if isinstance(event, bytes):
                        audio_bytes += len(event)
                        yield event
                        continue
                    check_event(event, task_id)
                    usage.update(usage_from(event))
                    if event["header"].get("event") == "task-finished":
                        await sender
                        if not audio_bytes:
                            raise APIConnectionError("阿里云 TTS 未返回音频。", retryable=False)
                        complete = True
                        return
            finally:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
                self.last_usage = usage
                logger.info("speech_usage kind=tts request=%s usage=%s", task_id, usage)
                if not complete:
                    with contextlib.suppress(Exception):
                        async with asyncio.timeout(1):
                            await ws.send_json(
                                command("finish-task", task_id, {"input": {"directive": "cancel"}})
                            )
                    # Do not pool a socket containing unconsumed audio from a cancelled task.
                    await ws.close()

    async def aclose(self):
        await self.transport.aclose()


class DashScopeChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter):
        task_id = uuid.uuid4().hex
        output_emitter.initialize(
            request_id=task_id,
            sample_rate=TTS_RATE,
            num_channels=1,
            mime_type="audio/pcm",
            frame_size_ms=20,
        )

        async def text():
            yield self.input_text

        async with contextlib.aclosing(self._tts.audio(text(), task_id)) as audio:
            async for pcm in audio:
                output_emitter.push(pcm)
        self._input_tokens = self._tts.last_usage.get("input_tokens", 0)
        self._output_tokens = self._tts.last_usage.get("output_tokens", 0)
        output_emitter.flush()


class DashScopeSynthesizeStream(tts.SynthesizeStream):
    async def _run(self, output_emitter):
        output_emitter.initialize(
            request_id=uuid.uuid4().hex,
            sample_rate=TTS_RATE,
            num_channels=1,
            mime_type="audio/pcm",
            frame_size_ms=20,
            stream=True,
        )
        async for first in self._input_ch:
            if not isinstance(first, str) or not first:
                continue
            task_id = uuid.uuid4().hex

            async def texts():
                yield first
                async for item in self._input_ch:
                    if isinstance(item, self._FlushSentinel):
                        return
                    yield item

            self._mark_started()
            output_emitter.start_segment(segment_id=task_id)
            async with contextlib.aclosing(self._tts.audio(texts(), task_id)) as audio:
                async for pcm in audio:
                    output_emitter.push(pcm)
            self._input_tokens += self._tts.last_usage.get("input_tokens", 0)
            self._output_tokens += self._tts.last_usage.get("output_tokens", 0)
            output_emitter.end_segment()
