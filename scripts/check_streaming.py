"""One real LLM stream plus first-sentence TTS; no browser, ASR or WebRTC test."""

import asyncio
import contextlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livekit.agents import APIConnectOptions  # noqa: E402

from app.config import ROOT, Settings  # noqa: E402
from app.conversation.engine import SupportEngine  # noqa: E402
from app.knowledge.retrieval import RetrievalRegistry  # noqa: E402
from app.voice.providers import build_tts  # noqa: E402


async def main():
    config = Settings()
    engine = SupportEngine(config, RetrievalRegistry(config))
    speaker = build_tts(config)
    report, audio_task = {}, None
    question = "R100如何恢复出厂设置？请简要说明操作前的注意事项、步骤和等待时间。"
    try:
        await engine.warmup()
        prepared = await engine.prepare(question, [], "bm25")
        start = time.perf_counter()

        async def first_audio(text):
            options = APIConnectOptions(max_retry=0)
            async with speaker.synthesize(text, conn_options=options) as stream:
                first = True
                async for chunk in stream:
                    if first:
                        report["first_audio_ms"] = round((time.perf_counter() - start) * 1000)
                        first = False

        async with contextlib.aclosing(engine.stream_answer(prepared, [], question)) as stream:
            answers = []
            async for segment in stream:
                answers.append(segment.answer)
                if audio_task is None:
                    report["first_sentence_ms"] = round((time.perf_counter() - start) * 1000)
                    audio_task = asyncio.create_task(first_audio(segment.answer))
            report["llm_complete_ms"] = round((time.perf_counter() - start) * 1000)
            report["llm_metrics"] = segment.metrics
            report["sentences"] = answers
        if audio_task:
            await audio_task
        report["audio_before_llm_complete"] = (
            report.get("first_audio_ms", float("inf")) < report["llm_complete_ms"]
        )
        (ROOT / "runtime/streaming-check.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        if audio_task:
            audio_task.cancel()
            await asyncio.gather(audio_task, return_exceptions=True)
        await engine.aclose()
        await speaker.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"Streaming check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
