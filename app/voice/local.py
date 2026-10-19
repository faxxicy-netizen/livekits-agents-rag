"""Local Chinese speech adapters implementing the public LiveKit STT/TTS APIs."""

import asyncio
import os
import re
import threading
import uuid
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sherpa_onnx
from livekit import rtc
from livekit.agents import APIConnectOptions, stt, tts

from app.config import Settings


def native_path(path) -> str:
    """OpenFST on Windows needs an ANSI-compatible path; relative paths work."""
    value = str(path)
    if os.name == "nt" and not value.isascii():
        relative = os.path.relpath(path)
        if relative.isascii():
            return relative
        import ctypes

        buffer = ctypes.create_unicode_buffer(32768)
        if ctypes.windll.kernel32.GetShortPathNameW(value, buffer, len(buffer)):
            return buffer.value
        raise ValueError("请从 support-agent 项目根目录启动，以兼容本地语音模型的文件路径。")
    return value


@lru_cache(maxsize=2)
def load_asr(directory: str, threads: int):
    from pathlib import Path

    folder = Path(directory)
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=native_path(folder / "model.int8.onnx"),
        tokens=native_path(folder / "tokens.txt"),
        language="zh",
        use_itn=True,
        num_threads=threads,
        provider="cpu",
    )
    return recognizer, threading.Lock()


@lru_cache(maxsize=2)
def load_tts(directory: str, threads: int):
    from pathlib import Path

    folder = Path(directory)
    onnx = next(folder.glob("*.onnx"))
    vits = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=native_path(onnx),
        lexicon=native_path(folder / "lexicon.txt"),
        tokens=native_path(folder / "tokens.txt"),
    )
    rules = ",".join(
        native_path(folder / f)
        for f in ("phone.fst", "date.fst", "number.fst")
        if (folder / f).exists()
    )
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=threads, provider="cpu"),
        rule_fsts=rules,
        max_num_sentences=1,
    )
    if not config.validate():
        raise ValueError("本地 TTS 模型配置无效，请重新运行资源安装脚本。")
    return sherpa_onnx.OfflineTts(config), threading.Lock()


class SenseVoiceSTT(stt.STT):
    def __init__(self, config: Settings):
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self.recognizer, self.lock = load_asr(str(config.local_asr_dir), config.speech_threads)

    @property
    def model(self):
        return "SenseVoice-Small-int8"

    @property
    def provider(self):
        return "local-sherpa-onnx"

    def decode(self, samples: np.ndarray, sample_rate: int) -> str:
        with self.lock:
            stream = self.recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            self.recognizer.decode_stream(stream)
            return re.sub(r"<\|.*?\|>", "", stream.result.text).strip()

    async def _recognize_impl(self, buffer, *, language=None, conn_options: APIConnectOptions):
        frame = rtc.combine_audio_frames(buffer)
        samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        if frame.num_channels > 1:
            samples = samples.reshape(-1, frame.num_channels).mean(axis=1)
        text = await asyncio.to_thread(self.decode, samples, frame.sample_rate)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=uuid.uuid4().hex,
            alternatives=[stt.SpeechData(language="zh", text=text)],
        )


class ChineseVitsTTS(tts.TTS):
    def __init__(self, config: Settings):
        self.engine, self.lock = load_tts(str(config.local_tts_dir), config.speech_threads)
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=self.engine.sample_rate,
            num_channels=1,
        )

    @property
    def model(self):
        return "vits-zh-hf-fanchen-C"

    @property
    def provider(self):
        return "local-sherpa-onnx"

    def generate(self, text: str):
        with self.lock:
            return self.engine.generate(text, sid=0, speed=1.05)

    def synthesize(self, text: str, *, conn_options=None):
        from livekit.agents import DEFAULT_API_CONNECT_OPTIONS

        return VitsStream(
            tts=self, input_text=text, conn_options=conn_options or DEFAULT_API_CONNECT_OPTIONS
        )


class VitsStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter):
        audio = await asyncio.to_thread(self._tts.generate, self.input_text)
        pcm = (np.clip(audio.samples, -1, 1) * 32767).astype(np.int16).tobytes()
        output_emitter.initialize(
            request_id=uuid.uuid4().hex,
            sample_rate=audio.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            frame_size_ms=20,
        )
        output_emitter.push(pcm)
        output_emitter.flush()


def speech_segments(text: str) -> list[str]:
    """Bound synthesis work and emit the first Chinese sentence early."""
    text = re.sub(r"[*#`]+", "", text).strip()
    sentences = re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", text)
    result = []
    for sentence in sentences:
        # Keep normal short sentences intact for prosody; split long ones at commas.
        clauses = re.findall(r"[^，,]+[，,]?", sentence) if len(sentence) > 42 else [sentence]
        pending = ""
        for clause in clauses:
            if len(pending) + len(clause) > 42 and pending:
                result.append(pending)
                pending = ""
            while len(clause) > 60:
                result.append(clause[:42])
                clause = clause[42:]
            pending += clause
        if pending.strip():
            result.append(pending.strip())
    return result


@lru_cache(maxsize=2)
def load_kokoro(directory: str, device: str, threads: int):
    import shutil

    import espeakng_loader
    import torch
    from kokoro import KModel, KPipeline
    from misaki.espeak import EspeakG2P
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    # eSpeak's Windows C runtime cannot read the Chinese workspace path.
    # Its wrapper resolves paths, so a relative path cannot bypass this issue.
    if os.name == "nt":
        data = Path(os.environ["LOCALAPPDATA"]) / "local-support-agent/espeak-ng-data"
        if not (data / "phontab").is_file():
            shutil.copytree(espeakng_loader.get_data_path(), data, dirs_exist_ok=True)
        EspeakWrapper.set_data_path(str(data))

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Kokoro CUDA 不可用。请安装 CUDA 版 PyTorch，或将 LOCAL_TTS_DEVICE 改为 cpu。"
        )
    torch.set_num_threads(threads)
    folder = Path(directory)
    repo = "hexgrad/Kokoro-82M-v1.1-zh"
    model = (
        KModel(
            repo_id=repo,
            config=str(folder / "config.json"),
            model=str(folder / "kokoro-v1_1-zh.pth"),
        )
        .to(device)
        .eval()
    )
    english = EspeakG2P(language="en-us")
    pipeline = KPipeline(
        lang_code="z", repo_id=repo, model=model, en_callable=lambda text: english(text)[0]
    )
    return pipeline, threading.Lock()


class KokoroChineseTTS(tts.TTS):
    """Kokoro v1.1 Mandarin, sentence-at-a-time synthesis on the local GPU."""

    def __init__(self, config: Settings):
        self.pipeline, self.lock = load_kokoro(
            str(config.local_tts_dir), config.local_tts_device, config.speech_threads
        )
        self.voice = str(config.local_tts_dir / "voices" / f"{config.local_tts_voice}.pt")
        self.speed = config.local_tts_speed
        self.voice_name = config.local_tts_voice
        self.device = config.local_tts_device
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False), sample_rate=24000, num_channels=1
        )
        # Load voice embedding now; the normal synthesis path never needs a network call.
        with self.lock:
            self.pipeline.load_voice(self.voice)

    @property
    def model(self):
        return "Kokoro-82M-v1.1-zh"

    @property
    def provider(self):
        return f"local-pytorch-{self.device}"

    def iter_audio(self, text: str):
        for segment in speech_segments(text):
            with self.lock:
                chunks = [
                    item.audio.numpy()
                    for item in self.pipeline(segment, voice=self.voice, speed=self.speed)
                    if item.audio is not None
                ]
            for samples in chunks:
                yield SimpleNamespace(samples=samples, sample_rate=24000)

    def generate(self, text: str):
        chunks = [audio.samples for audio in self.iter_audio(text)]
        return SimpleNamespace(
            samples=np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32),
            sample_rate=24000,
        )

    def synthesize(self, text: str, *, conn_options=None):
        from livekit.agents import DEFAULT_API_CONNECT_OPTIONS

        return KokoroStream(
            tts=self, input_text=text, conn_options=conn_options or DEFAULT_API_CONNECT_OPTIONS
        )


class KokoroStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter):
        output_emitter.initialize(
            request_id=uuid.uuid4().hex,
            sample_rate=24000,
            num_channels=1,
            mime_type="audio/pcm",
            frame_size_ms=20,
        )
        iterator = self._tts.iter_audio(self.input_text)
        while True:
            audio = await asyncio.to_thread(next, iterator, None)
            if audio is None:
                break
            pcm = (np.clip(audio.samples, -1, 1) * 32767).astype(np.int16).tobytes()
            output_emitter.push(pcm)
            output_emitter.flush()


def build_local_tts(config: Settings):
    if config.local_tts_backend == "kokoro":
        return KokoroChineseTTS(config)
    if config.local_tts_backend == "vits":
        return ChineseVitsTTS(config)
    raise ValueError(f"Unsupported local TTS backend: {config.local_tts_backend}")
