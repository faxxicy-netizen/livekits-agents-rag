"""One short TTS -> ASR check, without the browser, WebRTC or LLM calls.

Consumes a small amount of configured DashScope quota; credentials are never printed.
"""

import asyncio
import json
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livekit import rtc  # noqa: E402
from livekit.agents import APIConnectOptions, stt  # noqa: E402

from app.config import ROOT, Settings  # noqa: E402
from app.voice.dashscope import DashScopeSTT, DashScopeTTS  # noqa: E402


async def main():
    config = Settings()
    speaker, recognizer = DashScopeTTS(config), DashScopeSTT(config)
    options = APIConnectOptions(max_retry=0, timeout=config.dashscope_speech_timeout)
    text = "你好，我是知声。你可以上传文档，然后通过语音提问。"
    output = ROOT / "runtime/dashscope-check.wav"
    report = {
        "asr_model": recognizer.model,
        "tts_model": speaker.model,
        "voice": config.dashscope_tts_voice,
        "input_text": text,
    }
    try:
        started = time.perf_counter()
        pcm = []
        async with speaker.synthesize(text, conn_options=options) as stream:
            async for chunk in stream:
                if not pcm:
                    report["tts_first_frame_ms"] = round((time.perf_counter() - started) * 1000)
                pcm.append(chunk.frame.data.tobytes())
        report["tts_total_ms"] = round((time.perf_counter() - started) * 1000)
        report["tts_usage"] = speaker.last_usage
        audio = b"".join(pcm)
        report["audio_seconds"] = round(len(audio) / 48000, 3)
        output.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(audio)
        stream = recognizer.stream(conn_options=options)
        started = time.perf_counter()

        input_finished = 0.0

        async def feed():
            nonlocal input_finished
            # SDK resamples 24 kHz TTS to the ASR's 16 kHz; mimic a live microphone.
            for offset in range(0, len(audio), 4800):
                chunk = audio[offset : offset + 4800]
                stream.push_frame(rtc.AudioFrame(chunk, 24000, 1, len(chunk) // 2))
                await asyncio.sleep(0.1)
            stream.end_input()
            input_finished = time.perf_counter()

        sender = asyncio.create_task(feed())
        texts = []
        try:
            async for event in stream:
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    texts.append(event.alternatives[0].text)
            await sender
            report["asr_total_ms"] = round((time.perf_counter() - started) * 1000)
            report["asr_after_input_ms"] = round((time.perf_counter() - input_finished) * 1000)
            report["asr_usage"] = stream.usage
            report["transcript"] = "".join(texts)
            if not texts:
                raise RuntimeError("ASR returned no final transcript")
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            await stream.aclose()
        (ROOT / "runtime/dashscope-check.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        await asyncio.gather(speaker.aclose(), recognizer.aclose())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"Speech check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
