from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")
    livekit_url: str = "ws://127.0.0.1:7880"
    livekit_public_url: str = ""
    livekit_api_key: str = "devkey"
    livekit_api_secret: str = "secret"
    livekit_agent_name: str = "knowledge-assistant"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4.1-mini"
    llm_thinking: str = "disabled"
    stt_provider: Literal["local", "api", "dashscope"] = "local"
    stt_base_url: str = "https://api.openai.com/v1"
    stt_api_key: str = ""
    stt_model: str = "gpt-4o-mini-transcribe"
    tts_provider: Literal["local", "api", "dashscope"] = "local"
    tts_base_url: str = "https://api.openai.com/v1"
    tts_api_key: str = ""
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "coral"
    dashscope_api_key: str = Field(default="", repr=False)
    dashscope_workspace_id: str = ""
    dashscope_region: Literal["cn-beijing", "ap-southeast-1"] = "cn-beijing"
    dashscope_asr_model: str = "qwen-audio-3.1-asr-flash-streaming"
    dashscope_tts_model: str = "qwen-audio-3.1-tts-flash"
    dashscope_tts_voice: str = "yuxiaoyun_v3.1"
    dashscope_tts_rate: float = Field(default=1.05, ge=0.5, le=2.0)
    dashscope_asr_silence_ms: int = Field(default=400, ge=200, le=6000)
    dashscope_speech_timeout: float = Field(default=30, ge=3, le=120)
    local_asr_dir: Path = ROOT / "runtime/models/sensevoice"
    local_tts_backend: str = "kokoro"
    local_tts_dir: Path = ROOT / "runtime/models/kokoro-zh"
    local_tts_voice: str = "zf_001"
    local_tts_device: str = "cuda"
    local_tts_speed: float = Field(default=1.10, ge=0.7, le=1.5)
    speech_threads: int = 4
    rag_backend: str = "hybrid"
    rag_top_k: int = 3
    retrieval_timeout_seconds: float = 8
    model_timeout_seconds: float = 35
    dify_base_url: str = "http://127.0.0.1/v1"
    dify_api_key: str = ""
    dify_dataset_id: str = ""
    database_path: Path = ROOT / "runtime/support.db"
    vector_enabled: bool = True
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "support_knowledge"
    embedding_model_dir: Path = ROOT / "runtime/models/bge-small-zh-v1.5"
    embedding_provider: Literal["local", "dashscope"] = "local"
    embedding_api_model: str = "qwen3.7-text-embedding-flash"
    embedding_dimensions: int = 1024
    embedding_timeout_seconds: float = 20
    embedding_threads: int = 4
    vector_min_score: float = 0.42
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    app_public_origin: str = ""
    app_access_password_hash: str = Field(default="", repr=False)
    app_session_secret: str = Field(default="", repr=False)
    app_secure_cookies: bool = False
    max_active_calls: int = 0
    agent_idle_processes: int = Field(default=1, ge=0, le=2)

    @field_validator("embedding_dimensions")
    @classmethod
    def supported_embedding_dimension(cls, value: int) -> int:
        if value not in {256, 512, 768, 1024}:
            raise ValueError("embedding dimension must be 256, 512, 768 or 1024")
        return value

    @property
    def embedding_model_name(self) -> str:
        return (
            self.embedding_api_model
            if self.embedding_provider == "dashscope"
            else ("BAAI/bge-small-zh-v1.5")
        )

    @property
    def embedding_dimension(self) -> int:
        return self.embedding_dimensions if self.embedding_provider == "dashscope" else 512

    @property
    def dashscope_ws_url(self) -> str:
        # Workspace-only host construction prevents sending credentials to an arbitrary URL.
        import re

        if not re.fullmatch(r"[A-Za-z0-9-]{1,100}", self.dashscope_workspace_id):
            raise ValueError("请填写有效的 DASHSCOPE_WORKSPACE_ID。")
        return (
            f"wss://{self.dashscope_workspace_id}.{self.dashscope_region}.maas.aliyuncs.com"
            "/api-ws/v1/inference"
        )

    def missing_voice_resources(self) -> list[str]:
        missing = []
        if not self.llm_api_key:
            missing.append("LLM_API_KEY（以及对应 LLM_BASE_URL / LLM_MODEL）")
        if self.stt_provider == "local":
            if not (self.local_asr_dir / "model.int8.onnx").is_file():
                missing.append("本地 SenseVoice 模型：uv run python scripts/setup_resources.py")
        elif self.stt_provider == "api" and not self.stt_api_key:
            missing.append("STT_API_KEY")
        if self.tts_provider == "local":
            if self.local_tts_backend == "kokoro":
                required = [
                    "config.json",
                    "kokoro-v1_1-zh.pth",
                    f"voices/{self.local_tts_voice}.pt",
                ]
                if not all((self.local_tts_dir / name).is_file() for name in required):
                    missing.append("本地 Kokoro 中文模型：uv run python scripts/setup_resources.py")
            elif not list(self.local_tts_dir.glob("*.onnx")):
                missing.append(
                    "本地中文 VITS 模型：uv run python scripts/setup_resources.py --legacy-vits"
                )
        elif self.tts_provider == "api" and not self.tts_api_key:
            missing.append("TTS_API_KEY")
        if "dashscope" in {self.stt_provider, self.tts_provider, self.embedding_provider}:
            if not self.dashscope_api_key:
                missing.append("DASHSCOPE_API_KEY（阿里云百炼语音）")
            try:
                self.dashscope_ws_url
            except ValueError:
                missing.append("DASHSCOPE_WORKSPACE_ID（免费额度所在地域的业务空间 ID）")
        if (
            self.rag_backend in {"vector", "hybrid"}
            and self.embedding_provider == "local"
            and not (self.embedding_model_dir / "model.safetensors").is_file()
        ):
            missing.append("本地 BGE embedding 模型：python scripts/setup_knowledge.py")
        return missing


@lru_cache
def settings() -> Settings:
    return Settings()
