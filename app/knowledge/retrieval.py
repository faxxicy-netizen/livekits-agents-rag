import asyncio
import logging
import math
import re
import time
import unicodedata
from collections import Counter
from typing import Protocol

import httpx
import jieba

from app.config import Settings
from app.knowledge.documents import DocumentRepository
from app.models import Citation, SearchResult

jieba.setLogLevel(logging.WARNING)
STOP = set(
    (
        "的 了 吗 呢 啊 是 在 和 与 怎么 如何 什么 请 帮 我 一下 那 这个 那个 可以 是否 有 支持"
    ).split()
)


def normalize(text: str) -> str:
    # Text normalization only; no domain-specific identifiers or substitutions.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def tokenize(text: str) -> list[str]:
    return [
        w.strip()
        for w in jieba.lcut(normalize(text))
        if w.strip() not in STOP and re.search(r"[\w\u4e00-\u9fff]", w) and len(w.strip()) > 1
    ]


class Retriever(Protocol):
    async def search(self, query: str, top_k: int = 3) -> SearchResult: ...


class LocalRetriever:
    """Two real sparse retrieval algorithms sharing the same document chunks."""

    def __init__(self, backend: str = "bm25", *, snapshot):
        self.backend = backend
        self.documents, self.chunks = snapshot
        self.terms = [Counter(tokenize(c.heading + " " + c.text)) for c in self.chunks]
        self.df = Counter(t for terms in self.terms for t in terms)
        self.avg_length = sum(sum(t.values()) for t in self.terms) / max(len(self.terms), 1)

    async def search(
        self, query: str, top_k: int = 3, knowledge_base_id: str | None = None
    ) -> SearchResult:
        started = time.perf_counter()
        query_terms = Counter(tokenize(query))
        n = len(self.terms)
        scored = []
        for chunk, terms in zip(self.chunks, self.terms):
            if knowledge_base_id is not None and chunk.knowledge_base_id != knowledge_base_id:
                continue
            if not set(query_terms).intersection(terms):
                continue
            if self.backend == "bm25":
                score = 0.0
                length = sum(terms.values())
                for term in query_terms:
                    tf = terms[term]
                    idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                    score += (
                        idf
                        * tf
                        * 2.5
                        / (tf + 1.5 * (0.25 + 0.75 * length / max(self.avg_length, 1)))
                    )
            else:

                def weights(tokens):
                    return {
                        t: (1 + math.log(v)) * (math.log((n + 1) / (self.df[t] + 1)) + 1)
                        for t, v in tokens.items()
                    }

                q, d = weights(query_terms), weights(terms)
                norm = math.sqrt(sum(v * v for v in q.values()) * sum(v * v for v in d.values()))
                score = sum(v * d.get(t, 0) for t, v in q.items()) / max(norm, 1e-9)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return SearchResult(
            query=query,
            backend=self.backend,
            chunks=[
                chunk.model_copy(update={"score": round(score, 4)})
                for score, chunk in scored[:top_k]
            ],
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )


class RetrievalError(Exception):
    pass


class DifyRetriever:
    def __init__(self, config: Settings):
        self.config = config

    async def search(self, query: str, top_k: int = 3) -> SearchResult:
        c = self.config
        if not c.dify_api_key or not c.dify_dataset_id:
            raise RetrievalError("Dify 未配置：请填写 DIFY_API_KEY 和 DIFY_DATASET_ID。")
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=c.retrieval_timeout_seconds) as client:
                response = await client.post(
                    f"{c.dify_base_url.rstrip('/')}/datasets/{c.dify_dataset_id}/retrieve",
                    headers={"Authorization": f"Bearer {c.dify_api_key}"},
                    # Preserve the knowledge base's configured search/reranking settings.
                    json={"query": query[:250]},
                )
                response.raise_for_status()
                records = response.json().get("records", [])
        except (httpx.HTTPError, ValueError) as exc:
            raise RetrievalError("Dify 检索失败，请检查连接、知识库状态和服务端配置。") from exc
        chunks = []
        for record in records[:top_k]:
            s = record.get("segment", {})
            if not s.get("content"):
                continue
            doc = s.get("document", {})
            chunks.append(
                Citation(
                    chunk_id=s.get("id", ""),
                    document_id=s.get("document_id", ""),
                    document_name=doc.get("name", "未知文档"),
                    text=s["content"][:1600],
                    score=record.get("score", 0),
                )
            )
        return SearchResult(
            query=query,
            backend="dify",
            chunks=chunks,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )


class RetrievalRegistry:
    def __init__(self, config: Settings):
        from app.knowledge.vector import KnowledgeManager

        self.config = config
        self.repository = DocumentRepository(config)
        self.manager = KnowledgeManager(config, self.repository)
        self.revision = -1
        self.reload()

    def reload(self):
        documents, chunks, revision = self.repository.snapshot(include_revision=True)
        snapshot = documents, chunks
        self.local = {name: LocalRetriever(name, snapshot=snapshot) for name in ("bm25", "tfidf")}
        self.revision = revision

    async def search(
        self, query: str, backend: str, knowledge_base_id: str = "default"
    ) -> SearchResult:
        from app.knowledge.vector import VectorError

        if self.repository.revision() != self.revision:
            self.reload()
        try:
            self.repository.knowledge.get(knowledge_base_id)
        except KeyError:
            raise RetrievalError("绑定的知识库不存在。") from None
        if backend == "dify":
            raise RetrievalError("Dify 尚未配置独立知识库映射，请选择本地检索引擎。")
        try:
            async with asyncio.timeout(self.config.retrieval_timeout_seconds):
                if backend in {"vector", "hybrid"}:
                    if not self.config.vector_enabled:
                        raise RetrievalError("向量检索未启用。")
                    dense = await self.manager.index.search(
                        query,
                        self.config.rag_top_k if backend == "vector" else 12,
                        knowledge_base_id,
                    )
                    if backend == "vector":
                        return dense
                    lexical = await self.local["bm25"].search(query, 12, knowledge_base_id)
                    combined = {}
                    for name, result in (("vector", dense), ("bm25", lexical)):
                        for rank, chunk in enumerate(result.chunks, 1):
                            entry = combined.setdefault(chunk.chunk_id, [chunk, 0.0, {}])
                            entry[1] += 1 / (60 + rank)
                            entry[2][name] = chunk.score
                    ranked = sorted(combined.values(), key=lambda item: item[1], reverse=True)
                    return SearchResult(
                        query=query,
                        backend="hybrid",
                        chunks=[
                            chunk.model_copy(update={"score": round(score, 6), "scores": scores})
                            for chunk, score, scores in ranked[: self.config.rag_top_k]
                        ],
                        latency_ms=round(dense.latency_ms + lexical.latency_ms, 2),
                        details={**dense.details, "fusion": "RRF", "rrf_k": 60},
                    )
                return await self.local[backend].search(
                    query, self.config.rag_top_k, knowledge_base_id
                )
        except TimeoutError as exc:
            raise RetrievalError("检索超时，请重试。") from exc
        except VectorError as exc:
            raise RetrievalError(str(exc)) from exc
