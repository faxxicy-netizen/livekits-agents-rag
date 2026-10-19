"""Select speech providers without loading local model libraries for cloud speech."""

from app.config import Settings


def build_stt(config: Settings):
    if config.stt_provider == "local":
        from app.voice.local import SenseVoiceSTT

        return SenseVoiceSTT(config)
    if config.stt_provider == "dashscope":
        from app.voice.dashscope import DashScopeSTT

        return DashScopeSTT(config)
    from livekit.plugins import openai

    return openai.STT(
        model=config.stt_model,
        language="zh",
        use_realtime=False,
        api_key=config.stt_api_key,
        base_url=config.stt_base_url,
    )


def build_tts(config: Settings):
    if config.tts_provider == "local":
        from app.voice.local import build_local_tts

        return build_local_tts(config)
    if config.tts_provider == "dashscope":
        from app.voice.dashscope import DashScopeTTS

        return DashScopeTTS(config)
    from livekit.plugins import openai

    return openai.TTS(
        model=config.tts_model,
        voice=config.tts_voice,
        api_key=config.tts_api_key,
        base_url=config.tts_base_url,
    )
