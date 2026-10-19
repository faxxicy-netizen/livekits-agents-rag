import threading
from types import SimpleNamespace

import numpy as np
import pytest
from livekit.agents import tts

from app.voice.local import ChineseVitsTTS, KokoroChineseTTS, speech_segments


def test_local_synthesis_preserves_domain_terms():
    text = "R100、R200、Wi-Fi、RESET、WAN、LAN与SDK3.0。"
    calls = []

    def pipeline(segment, **kwargs):
        calls.append(segment)
        return []

    fake = SimpleNamespace(lock=threading.Lock(), pipeline=pipeline, voice="test", speed=1.0)
    assert list(KokoroChineseTTS.iter_audio(fake, text)) == []
    assert calls == [text]
    fake.engine = SimpleNamespace(generate=lambda value, **kwargs: value)
    assert ChineseVitsTTS.generate(fake, text) == text


def test_chinese_segments_preserve_content_and_bound_synthesis_work():
    text = "你好，我是知声。" + "请先准备宽带账号和密码，" * 15 + "恢复出厂后重新配置。"
    pieces = speech_segments(text)
    assert "".join(pieces) == text
    assert pieces[0] == "你好，我是知声。"
    assert all(0 < len(piece) <= 60 for piece in pieces)


@pytest.mark.asyncio
async def test_first_audio_does_not_wait_for_remaining_sentences():
    release = threading.Event()

    class FakeTTS(KokoroChineseTTS):
        def __init__(self):
            self.device = "test"
            tts.TTS.__init__(
                self,
                capabilities=tts.TTSCapabilities(streaming=False),
                sample_rate=24000,
                num_channels=1,
            )

        def iter_audio(self, text):
            yield SimpleNamespace(samples=np.zeros(2400), sample_rate=24000)
            assert release.wait(5), "First audio was blocked by later synthesis"
            yield SimpleNamespace(samples=np.zeros(2400), sample_rate=24000)

    stream = FakeTTS().synthesize("第一句话。第二句话。")
    try:
        first = await anext(stream)
        assert first.frame.samples_per_channel > 0
        assert not release.is_set()
    finally:
        release.set()
        await stream.aclose()
