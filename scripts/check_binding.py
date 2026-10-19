"""Small real-Qdrant scope check; isolated SQLite/collection, no LLM or voice calls."""

import asyncio
import json
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.knowledge.documents import build_document  # noqa: E402
from app.models import KnowledgeInput  # noqa: E402
from app.knowledge.retrieval import RetrievalRegistry  # noqa: E402


async def main():
    original = settings()
    checks = []
    with tempfile.TemporaryDirectory(
        prefix="binding-check-", dir=original.database_path.parent
    ) as tmp:
        assert Path(tmp).resolve().is_relative_to(original.database_path.parent.resolve())
        config = original.model_copy(
            update={
                "database_path": Path(tmp) / "check.db",
                "qdrant_collection": "binding_check_" + uuid.uuid4().hex[:10],
                "llm_api_key": "",
            }
        )
        registry = RetrievalRegistry(config)
        index = registry.manager.index
        try:
            await index.initialize()
            bases = [
                registry.repository.knowledge.save(KnowledgeInput(name=n))
                for n in ["IoT 平台 A", "IoT 平台 B"]
            ]
            saved = []
            for base, value in zip(bases, ["星河青鸟", "银色海豚"], strict=True):
                doc = build_document(
                    "使用手册.txt",
                    f"IoT平台的SDK初始化密钥是{value}。设备首次上线前需要完成SDK初始化。",
                    knowledge_base_id=base["id"],
                )
                saved.append(await registry.manager.save(doc))
            assert saved[0]["id"] != saved[1]["id"]
            checks.append("same filename in two knowledge bases has independent document IDs")
            for backend in ["vector", "hybrid", "bm25", "tfidf"]:
                for base, doc in zip(bases, saved, strict=True):
                    result = await registry.search("SDK初始化密钥", backend, base["id"])
                    assert result.chunks, (backend, base["name"])
                    assert all(
                        c.document_id == doc["id"] and c.knowledge_base_id == base["id"]
                        for c in result.chunks
                    )
            checks.append("vector/hybrid/BM25/TF-IDF return only the bound base")
            points = await index.request(
                "POST", index.path + "/points/scroll", {"limit": 100, "with_payload": True}
            )
            assert {p["payload"]["knowledge_base_id"] for p in points["points"]} == {
                b["id"] for b in bases
            }
            checks.append("Qdrant persists knowledge_base_id on every point")
            await registry.manager.delete(saved[0]["id"])
            for backend in ["vector", "hybrid", "bm25", "tfidf"]:
                assert not (await registry.search("SDK初始化密钥", backend, bases[0]["id"])).chunks
                assert (await registry.search("SDK初始化密钥", backend, bases[1]["id"])).chunks
            checks.append("empty base never falls through to another base")
        finally:
            await index.request("DELETE", index.path, allow_missing=True)
    report = {"passed": True, "checks": checks, "llm_calls": 0, "voice_calls": 0}
    (original.database_path.parent / "binding-check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
