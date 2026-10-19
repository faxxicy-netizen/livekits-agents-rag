"""Real local WebRTC + STT + DeepSeek + TTS smoke test using synthetic microphone audio.

Uses the configured LLM API (small number of calls). Does not access a microphone/speaker.
Run the app first. The test's cookie-isolated session is marked ended in finally.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings
from app.voice.local import build_local_tts

import httpx
import numpy as np
from livekit import rtc


async def main():
    config = settings()
    speech = build_local_tts(config)
    questions = [
        "我用的是R一百，橙色指示灯一直亮，该怎么办？",
        "那我应该怎么恢复出厂设置？",
        "重置后会清除什么设置？",
        "换成R两百，恢复出厂应该长按多久？",
        "这个型号保修多久？",
        "这个型号支持卫星联网吗？",
    ]
    audio_inputs = []
    for question in questions:
        output = await asyncio.to_thread(speech.generate, question)
        samples = np.interp(
            np.arange(int(len(output.samples) * 16000 / output.sample_rate))
            * output.sample_rate
            / 16000,
            np.arange(len(output.samples)),
            output.samples,
        )
        pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
        audio_inputs.append(pcm)
    print("Synthetic inputs ready. No hardware microphone or speaker is used.", flush=True)
    room = rtc.Room()
    events, tasks = [], set()
    changed = asyncio.Event()
    received_audio_frames = 0

    async def receive_text(reader):
        try:
            event = json.loads(await reader.read_all())
        except rtc.StreamError:
            return
        events.append(event)
        changed.set()
        if event.get("type") == "message":
            m = event["message"]
            print(
                f"{m['role']}: {m['text']} (interrupted={m['interrupted']}, citations={len(m['citations'])})",
                flush=True,
            )
        elif event.get("type") == "error":
            print("Agent error: " + event["detail"], flush=True)

    def background(coroutine):
        task = asyncio.create_task(coroutine)
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    def text_handler(reader, participant_identity):
        background(receive_text(reader))

    async def receive_audio(track):
        nonlocal received_audio_frames
        stream = rtc.AudioStream(track, sample_rate=16000, num_channels=1)
        try:
            async for frame in stream:
                received_audio_frames += 1
        finally:
            await stream.aclose()

    @room.on("track_subscribed")
    def subscribed(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            background(receive_audio(track))

    room.register_text_stream_handler("support.events", text_handler)

    async def wait_until(predicate, timeout=55):
        async with asyncio.timeout(timeout):
            while not predicate():
                changed.clear()
                await changed.wait()

    def messages(role=None):
        return [
            e["message"]
            for e in events
            if e.get("type") == "message" and (role is None or e["message"]["role"] == role)
        ]

    source = rtc.AudioSource(16000, 1)

    async def speak(pcm):
        pad = np.zeros(16000, dtype=np.int16)
        combined = np.concatenate((np.zeros(3200, dtype=np.int16), pcm, pad))
        for start in range(0, len(combined), 320):
            block = combined[start : start + 320]
            if len(block) < 320:
                block = np.pad(block, (0, 320 - len(block)))
            await source.capture_frame(rtc.AudioFrame(block.tobytes(), 16000, 1, len(block)))
        await source.wait_for_playout()

    started = time.perf_counter()
    sid = None
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=15) as client:
        try:
            await client.get("/api/status")
            response = await client.post("/api/sessions", json={"backend": "bm25"})
            response.raise_for_status()
            sid = response.json()["id"]
            response = await client.post(f"/api/sessions/{sid}/token")
            response.raise_for_status()
            connection = response.json()
            await room.connect(connection["url"], connection["token"])
            track = rtc.LocalAudioTrack.create_audio_track("synthetic-microphone", source)
            await room.local_participant.publish_track(
                track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            )
            await wait_until(lambda: any(e.get("type") == "ready" for e in events))
            await wait_until(lambda: any(m["turn_id"] == "greeting" for m in messages("assistant")))
            print("Agent connected and greeting audio received.", flush=True)
            await speak(audio_inputs[0])
            await wait_until(lambda: len(messages("user")) >= 1)
            first_turn = messages("user")[0]["turn_id"]
            await wait_until(
                lambda: any(
                    e.get("type") == "agent_state" and e.get("state") == "speaking"
                    for e in events[
                        events.index(
                            next(
                                e
                                for e in events
                                if e.get("type") == "message"
                                and e["message"]["turn_id"] == first_turn
                            )
                        ) :
                    ]
                )
            )
            # Start a second utterance while the first answer is still playing.
            await speak(audio_inputs[1])
            await wait_until(lambda: len(messages("user")) >= 2)
            second_turn = messages("user")[-1]["turn_id"]
            await wait_until(
                lambda: any(m["turn_id"] == second_turn for m in messages("assistant"))
            )
            for index in range(2, len(questions)):
                await speak(audio_inputs[index])
                await wait_until(lambda: len(messages("user")) >= index + 1)
                turn_id = messages("user")[-1]["turn_id"]
                await wait_until(
                    lambda: any(m["turn_id"] == turn_id for m in messages("assistant"))
                )
            assert received_audio_frames > 20, "No synthesized audio received over WebRTC"
            assert len(messages("user")) >= 5, "Less than five voice turns completed"
            assert any(m["interrupted"] for m in messages("assistant")), (
                "Barge-in did not interrupt playback"
            )
            assert any(m["citations"] for m in messages("assistant")), (
                "Voice replies lost citations"
            )
            assert not any(m["answer_mode"] == "error" for m in messages("assistant")), (
                "Voice pipeline reported an error"
            )
            history = (await client.get(f"/api/sessions/{sid}")).json()
            report = {
                "passed": True,
                "model": config.llm_model,
                "thinking": config.llm_thinking,
                "user_turns": len(messages("user")),
                "assistant_messages": len(messages("assistant")),
                "interrupted_messages": sum(m["interrupted"] for m in messages("assistant")),
                "received_audio_frames": received_audio_frames,
                "duration_seconds": round(time.perf_counter() - started, 2),
                "messages": history["messages"],
            }
            path = config.database_path.parent / "voice-smoke.json"
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {k: v for k, v in report.items() if k != "messages"},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        finally:
            await room.disconnect()
            await source.aclose()
            if sid:
                await client.post(f"/api/sessions/{sid}/end")
            for task in list(tasks):
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
