import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings
from app.conversation.engine import SupportEngine
from app.knowledge.retrieval import RetrievalRegistry


async def main():
    config = settings()
    engine = SupportEngine(config, RetrievalRegistry(config))
    question = "R100怎么恢复出厂设置？会清除什么？"
    prepared = await engine.prepare(question, [], "bm25")
    text, citations, mode, metrics = await engine.answer(prepared, [], question)
    report = {
        "model": config.llm_model,
        "thinking": config.llm_thinking,
        "answer_mode": mode,
        "answer": text,
        "sources": [c.document_name for c in citations],
        "metrics": metrics,
    }
    (config.database_path.parent / "llm-check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
