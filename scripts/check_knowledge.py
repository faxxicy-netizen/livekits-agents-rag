"""Short real Qdrant CRUD check. No LLM call, microphone, TTS or WebRTC."""

import asyncio
import json
import sys
import uuid
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings
from app.knowledge.retrieval import RetrievalRegistry


async def main():
    config = settings()
    reader = RetrievalRegistry(config)  # Keep alive across edits, like the voice worker.
    filename = "临时验收-" + uuid.uuid4().hex[:8] + ".md"
    original = (
        "# 明川园区会议预约\n\n## 会议室预约\n"
        "明川园区会议室需要提前三小时预约，预约入口是内部门户的空间预约。"
    )
    updated = original.replace("三小时", "五小时")
    document_id = None
    checks = []
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=60) as client:
        before = (await client.get("/api/documents")).json()
        try:
            response = await client.post(
                "/api/documents/upload",
                files={"file": (filename, original.encode(), "text/markdown")},
            )
            response.raise_for_status()
            doc = response.json()["document"]
            document_id = doc["id"]
            assert doc["index_status"] == "indexed"
            checks.append("multipart upload -> persisted document and embeddings")
            result = await reader.search("明川园区开会需要提前多久订房间", "hybrid")
            assert any(c.document_id == document_id and "三小时" in c.text for c in result.chunks)
            checks.append("semantic/hybrid search from another long-lived registry")
            response = await client.put(
                "/api/documents/" + document_id,
                json={
                    "content": updated,
                    "expected_version": doc["version"],
                },
            )
            response.raise_for_status()
            current_version = response.json()["document"]["version"]
            result = await reader.search("明川园区会议室预约", "vector")
            hits = [c for c in result.chunks if c.document_id == document_id]
            assert hits and all("五小时" in c.text and "三小时" not in c.text for c in hits)
            points = await reader.manager.index.request(
                "POST",
                reader.manager.index.path + "/points/scroll",
                {
                    "filter": {"must": [{"key": "document_id", "match": {"value": document_id}}]},
                    "limit": 100,
                    "with_payload": True,
                },
            )
            assert points["points"] and all(
                p["payload"]["version"] == current_version for p in points["points"]
            )
            checks.append("edit -> new embeddings visible; old vectors physically removed")
            (await client.delete("/api/documents/" + document_id)).raise_for_status()
            result = await reader.search("明川园区会议室预约", "vector")
            assert all(c.document_id != document_id for c in result.chunks)
            assert (await client.get("/api/documents/" + document_id)).status_code == 404
            assert len((await client.get("/api/documents")).json()) == len(before)
            checks.append("delete -> document hidden and no longer retrievable")
        finally:
            if document_id:
                await client.delete("/api/documents/" + document_id)
    report = {"passed": True, "database": "Qdrant", "checks": checks}
    path = config.database_path.parent / "knowledge-check.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
