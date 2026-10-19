import asyncio
import json
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings
from app.voice.local import SenseVoiceSTT, build_local_tts


async def main():
    config = settings()
    started = time.perf_counter()
    tts = build_local_tts(config)
    asr = SenseVoiceSTT(config)
    load_ms = (time.perf_counter() - started) * 1000
    text = "青岚路由器恢复出厂设置会清除网络配置，请先准备宽带账号和密码。"
    started = time.perf_counter()
    audio = await asyncio.to_thread(tts.generate, text)
    tts_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    recognized = await asyncio.to_thread(asr.decode, audio.samples, audio.sample_rate)
    stt_ms = (time.perf_counter() - started) * 1000
    import numpy as np

    path = config.database_path.parent / "speech-check.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(audio.sample_rate)
        wav.writeframes((np.clip(audio.samples, -1, 1) * 32767).astype(np.int16).tobytes())
    report = {
        "model_load_ms": round(load_ms),
        "tts_ms": round(tts_ms),
        "stt_ms": round(stt_ms),
        "audio_seconds": round(len(audio.samples) / audio.sample_rate, 2),
        "input": text,
        "recognized": recognized,
        "audio_path": str(path),
    }
    assert len(audio.samples) > audio.sample_rate and "恢复" in recognized
    (config.database_path.parent / "speech-check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
