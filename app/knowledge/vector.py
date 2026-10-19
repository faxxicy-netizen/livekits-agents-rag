"""Configurable local/cloud embeddings with Qdrant storage on the application server."""

import asyncio
import logging
import math
import threading
import time
import uuid
from functools import lru_cache

import httpx

from app.config import Settings
from app.knowledge.documents import DocumentRepository, document_chunks
from app.models import Citation, SearchResult

MODEL = "BAAI/bge-small-zh-v1.5"
DIMENSION = 512


class VectorError(RuntimeError):
    pass


@lru_cache(maxsize=2)
def load_embedding(directory: str, threads: int):
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    model = AutoModel.from_pretrained(directory, local_files_only=True).to("cpu").eval()
    return tokenizer, model, threading.Lock()


class KnowledgeIndex:
    def __init__(self, config: Settings, repository: DocumentRepository):
        self.config, self.repository = config, repository
        self.path = f"/collections/{config.qdrant_collection}"
        self._embedding_client = None
        self._embedding_lock = asyncio.Lock()

    async def aclose(self):
        if self._embedding_client is not None:
            await self._embedding_client.aclose()

    async def embed(self, texts: list[str], *, query=False) -> list[list[float]]:
        if self.config.embedding_provider == "local":
            return await asyncio.to_thread(self.encode, texts, query=query)
        if not texts:
            return []
        config = self.config
        if not config.dashscope_api_key:
            raise VectorError("缺少 DASHSCOPE_API_KEY。")
        async with self._embedding_lock:
            if self._embedding_client is None:
                self._embedding_client = await asyncio.to_thread(
                    httpx.AsyncClient, timeout=config.embedding_timeout_seconds
                )
        url = config.dashscope_ws_url.replace("wss://", "https://").replace(
            "/api-ws/v1/inference", "/compatible-mode/v1/embeddings"
        )
        vectors = []
        try:
            for offset in range(0, len(texts), 20):
                batch = texts[offset : offset + 20]
                response = await self._embedding_client.post(
                    url,
                    headers={"Authorization": "Bearer " + config.dashscope_api_key},
                    json={
                        "model": config.embedding_api_model,
                        "input": batch,
                        "dimensions": config.embedding_dimensions,
                        "encoding_format": "float",
                    },
                )
                response.raise_for_status()
                data = sorted(response.json()["data"], key=lambda row: row["index"])
                if [row["index"] for row in data] != list(range(len(batch))):
                    raise ValueError("embedding response count mismatch")
                for row in data:
                    vector = row["embedding"]
                    if (
                        len(vector) != config.embedding_dimension
                        or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in vector)
                        or not any(vector)
                    ):
                        raise ValueError("invalid embedding vector")
                    vectors.append(vector)
            return vectors
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            logging.getLogger(__name__).warning(
                "cloud_embedding_failed kind=%s", type(exc).__name__
            )
            raise VectorError("阿里云向量化失败，请检查网络、密钥、模型权限和额度。") from None

    def encode(self, texts: list[str], *, query=False) -> list[list[float]]:
        import torch
        import torch.nn.functional as functional

        if not (self.config.embedding_model_dir / "model.safetensors").is_file():
            raise VectorError("缺少 BGE 模型，请运行 python scripts/setup_knowledge.py。")
        tokenizer, model, lock = load_embedding(
            str(self.config.embedding_model_dir), self.config.embedding_threads
        )
        if query:
            texts = ["为这个句子生成表示以用于检索相关文章：" + text for text in texts]
        vectors = []
        with lock, torch.inference_mode():
            for offset in range(0, len(texts), 16):
                inputs = tokenizer(
                    texts[offset : offset + 16],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                # The official BGE recipe uses normalized CLS embeddings.
                outputs = model(**inputs).last_hidden_state[:, 0]
                vectors += functional.normalize(outputs, p=2, dim=1).cpu().tolist()
        return vectors

    async def request(self, method, path, body=None, *, allow_missing=False):
        try:
            async with httpx.AsyncClient(base_url=self.config.qdrant_url, timeout=30) as client:
                response = await client.request(method, path, json=body)
                if allow_missing and response.status_code == 404:
                    return None
                response.raise_for_status()
                return response.json().get("result")
        except (httpx.HTTPError, ValueError) as exc:
            raise VectorError(
                "Qdrant 操作失败，请检查本地向量服务及 runtime/qdrant 日志。"
            ) from exc

    async def initialize(self):
        info = await self.request("GET", self.path, allow_missing=True)
        if info is None:
            await self.request(
                "PUT",
                self.path,
                {
                    "vectors": {"size": self.config.embedding_dimension, "distance": "Cosine"},
                    "optimizers_config": {"indexing_threshold": 1000},
                },
            )
        elif info["config"]["params"]["vectors"]["size"] != self.config.embedding_dimension:
            raise VectorError("向量维度不匹配，请为此 embedding 模型设置独立的 collection。")
        if info and info.get("points_count", 0):
            sample = await self.request(
                "POST",
                self.path + "/points/scroll",
                {
                    "limit": 1,
                    "with_payload": ["embedding_model"],
                    "with_vector": False,
                },
            )
            if any(
                p.get("payload", {}).get("embedding_model") != self.config.embedding_model_name
                for p in sample.get("points", [])
            ):
                raise VectorError("索引的 embedding 模型不同，请使用独立 collection 并重建向量。")
        docs, chunks = self.repository.snapshot()
        if (
            info is None
            or info.get("points_count") != len(chunks)
            or any(d["index_status"] != "indexed" for d in docs)
        ):
            revision = self.repository.revision()
            await self.upsert(chunks)
            ids = [self.point_id(c.chunk_id) for c in chunks]
            # Remove old versions and orphaned points from interrupted writes.
            body = {"filter": {"must_not": [{"has_id": ids}]}} if ids else {"filter": {}}
            await self.request("POST", self.path + "/points/delete?wait=true", body)
            self.repository.mark_indexed(revision)
        elif self.config.embedding_provider == "local":
            await asyncio.to_thread(self.encode, ["预热"], query=True)

    @staticmethod
    def point_id(chunk_id: str):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, "support-knowledge:" + chunk_id))

    async def upsert(self, chunks: list[Citation]):
        if not chunks:
            return
        vectors = await self.embed([c.heading[:80] + "\n" + c.text for c in chunks])
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            points.append(
                {
                    "id": self.point_id(chunk.chunk_id),
                    "vector": vector,
                    "payload": {
                        **chunk.model_dump(),
                        "embedding_model": self.config.embedding_model_name,
                    },
                }
            )
        for offset in range(0, len(points), 64):
            await self.request(
                "PUT", self.path + "/points?wait=true", {"points": points[offset : offset + 64]}
            )

    async def delete(self, document_id: str, version: str | None = None):
        conditions = [{"key": "document_id", "match": {"value": document_id}}]
        if version:
            conditions.append({"key": "version", "match": {"value": version}})
        await self.request(
            "POST", self.path + "/points/delete?wait=true", {"filter": {"must": conditions}}
        )

    async def delete_knowledge_base(self, knowledge_base_id: str):
        await self.request(
            "POST",
            self.path + "/points/delete?wait=true",
            {
                "filter": {
                    "must": [{"key": "knowledge_base_id", "match": {"value": knowledge_base_id}}]
                }
            },
        )

    async def search(self, query: str, top_k=3, knowledge_base_id="default") -> SearchResult:
        started = time.perf_counter()
        documents = self.repository.list(knowledge_base_id=knowledge_base_id)
        if not documents:
            return SearchResult(query=query, backend="vector")
        # Only committed, currently visible document versions are searchable.
        active = {
            "should": [
                {
                    "must": [
                        {"key": "document_id", "match": {"value": d["id"]}},
                        {"key": "version", "match": {"value": d["version"]}},
                    ]
                }
                for d in documents
            ]
        }
        filters = [active, {"key": "knowledge_base_id", "match": {"value": knowledge_base_id}}]
        vector = (await self.embed([query], query=True))[0]
        embedding_ms = (time.perf_counter() - started) * 1000
        result = await self.request(
            "POST",
            self.path + "/points/query",
            {
                "query": vector,
                "filter": {"must": filters},
                "limit": top_k,
                "with_payload": True,
                "score_threshold": self.config.vector_min_score,
            },
        )
        chunks = [
            Citation(
                **{
                    **p["payload"],
                    "score": round(p["score"], 4),
                    "scores": {"vector": round(p["score"], 4)},
                }
            )
            for p in result.get("points", [])
        ]
        return SearchResult(
            query=query,
            backend="vector",
            chunks=chunks,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            details={
                "embedding_model": self.config.embedding_model_name,
                "dimension": self.config.embedding_dimension,
                "embedding_ms": round(embedding_ms, 2),
                "database": "Qdrant",
            },
        )


class KnowledgeManager:
    """One API writer; stage vectors before publishing new document versions."""

    def __init__(self, config: Settings, repository: DocumentRepository):
        self.config, self.repository = config, repository
        self.index = KnowledgeIndex(config, repository)
        self.lock = asyncio.Lock()

    async def save(self, doc: dict, old: dict | None = None):
        chunks = document_chunks(doc)
        if not chunks:
            raise ValueError("文档需要包含正文。")
        if self.config.vector_enabled:
            await self.index.upsert(chunks)
        saved = self.repository.save(doc, indexed=self.config.vector_enabled)
        if old and old["version"] != doc["version"] and self.config.vector_enabled:
            try:
                await self.index.delete(doc["id"], old["version"])
            except VectorError:
                # Active-version filtering hides stale vectors; initialization retries cleanup.
                logging.getLogger(__name__).warning(
                    "old_vectors_cleanup_pending document=%s", doc["id"]
                )
                saved["cleanup_pending"] = True
        return saved

    async def delete(self, document_id: str):
        self.repository.get(document_id)
        if self.config.vector_enabled:
            await self.index.delete(document_id)
        self.repository.delete(document_id)
