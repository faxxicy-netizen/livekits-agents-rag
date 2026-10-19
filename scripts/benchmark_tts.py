"""Benchmark actual local TTS on this machine and write comparable Chinese samples."""

import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import ROOT, settings
from app.voice.local import ChineseVitsTTS, KokoroChineseTTS


def save_wav(path, audio):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(audio.sample_rate)
        wav.writeframes((np.clip(audio.samples, -1, 1) * 32767).astype(np.int16).tobytes())


def main():
    import torch

    config = settings()
    output = ROOT / "runtime/tts-comparison"
    output.mkdir(exist_ok=True)
    sample = (
        "你好，我是知声。恢复出厂设置会清除网络配置，请先准备好宽带账号和密码。"
        "长按重置按钮八秒，等待路由器重新启动。"
    )
    report = {"text": sample, "gpu": torch.cuda.get_device_name(), "results": []}
    for voice in ("zf_001", "zf_017", "zm_010"):
        selected = config.model_copy(update={"local_tts_voice": voice})
        started = time.perf_counter()
        model = KokoroChineseTTS(selected)
        model.generate("你好。")
        warmup_ms = (time.perf_counter() - started) * 1000
        torch.cuda.reset_peak_memory_stats()
        measurements = []
        for _ in range(3):
            started = time.perf_counter()
            chunks = []
            first_ms = None
            for chunk in model.iter_audio(sample):
                if first_ms is None:
                    first_ms = (time.perf_counter() - started) * 1000
                chunks.append(chunk.samples)
            elapsed = time.perf_counter() - started
            samples = np.concatenate(chunks)
            measurements.append(
                {
                    "first_chunk_ms": round(first_ms),
                    "total_ms": round(elapsed * 1000),
                    "audio_seconds": round(len(samples) / 24000, 2),
                    "rtf": round(elapsed / (len(samples) / 24000), 3),
                }
            )
        path = output / f"kokoro-{voice}-speed{model.speed:.2f}.wav"
        from types import SimpleNamespace

        save_wav(path, SimpleNamespace(samples=samples, sample_rate=24000))
        report["results"].append(
            {
                "model": model.model,
                "device": model.device,
                "voice": voice,
                "speed": model.speed,
                "warmup_ms": round(warmup_ms),
                "measurements": measurements,
                "audio_path": str(path),
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2),
            }
        )
        print(json.dumps(report["results"][-1], ensure_ascii=False), flush=True)
    legacy_dir = ROOT / "runtime/models/vits-zh"
    if legacy_dir.exists():
        legacy = ChineseVitsTTS(config.model_copy(update={"local_tts_dir": legacy_dir}))
        started = time.perf_counter()
        audio = legacy.generate(sample)
        elapsed = time.perf_counter() - started
        path = output / "previous-vits-speed1.05.wav"
        save_wav(path, audio)
        report["previous_vits"] = {
            "tts_ms": round(elapsed * 1000),
            "audio_seconds": round(len(audio.samples) / audio.sample_rate, 2),
            "audio_path": str(path),
        }
    (output / "benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
