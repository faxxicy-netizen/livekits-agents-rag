from typing import Any, Literal

from pydantic import BaseModel, Field

Backend = Literal["bm25", "tfidf", "vector", "hybrid", "dify"]


class KnowledgeProfile(BaseModel):
    role: str = Field(default="知识库使用助手", max_length=120)
    scope: str = Field(default="帮助用户理解当前知识库中的资料并解决相关问题。", max_length=1200)
    context_fields: str = Field(default="相关对象、版本、操作阶段、错误信息", max_length=500)
    clarification_policy: str = Field(
        default="信息足够就直接回答；确实影响答案时，只询问一个关键条件。", max_length=1200
    )
    response_style: str = Field(default="先给结论，再给简短步骤和验证方法。", max_length=1200)
    aliases: str = Field(default="", max_length=2000)


class KnowledgeInput(BaseModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"\S")
    profile: KnowledgeProfile = Field(default_factory=KnowledgeProfile)


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    heading: str = ""
    text: str
    score: float = 0
    version: str = ""
    scores: dict[str, float] = Field(default_factory=dict)
    knowledge_base_id: str = "default"


class SearchResult(BaseModel):
    query: str
    backend: str
    chunks: list[Citation] = Field(default_factory=list)
    latency_ms: float = 0
    details: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    id: str
    session_id: str
    turn_id: str
    role: Literal["user", "assistant"]
    text: str
    citations: list[Citation] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    interrupted: bool = False
    answer_mode: str = "llm"
    created_at: str


class CreateSession(BaseModel):
    backend: Backend | None = None
    diagnostic: bool = False
    knowledge_base_id: str = "default"


class ChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    preview: bool = False


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    backend: Backend = "bm25"
    knowledge_base_id: str = "default"


class DocumentInput(BaseModel):
    filename: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=10, max_length=1_000_000)
    knowledge_base_id: str = "default"


class DocumentUpdate(BaseModel):
    content: str = Field(min_length=10, max_length=1_000_000)
    expected_version: str | None = None
