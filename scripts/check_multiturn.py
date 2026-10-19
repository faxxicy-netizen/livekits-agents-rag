"""Exercise five real model turns and inspect provider format without logging credentials."""
import asyncio
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings
from app.conversation.engine import SupportEngine
from app.models import Message
from app.knowledge.retrieval import RetrievalRegistry

import httpx


async def inspect_response(response):
    await response.aread()
    try:
        data = response.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        print(json.dumps({"http_status": response.status_code,
                          "finish_reason": choice.get("finish_reason"),
                          "content_length": len(content) if content else 0,
                          "content_prefix": content[:120] if isinstance(content, str) else "not-text"},
                         ensure_ascii=False), flush=True)
    except ValueError:
        print(json.dumps({"http_status": response.status_code,
                          "content_type": response.headers.get("content-type"),
                          "body_prefix": response.text[:160]}, ensure_ascii=False), flush=True)


async def main():
    original = httpx.AsyncClient
    httpx.AsyncClient = lambda **kwargs: original(event_hooks={"response": [inspect_response]}, **kwargs)
    engine = SupportEngine(settings(), RetrievalRegistry(settings()))
    history = []
    for question in ["R100橙色指示灯常亮怎么办？", "那我怎么恢复出厂设置？", "重置后会清除什么设置？",
                     "换成R200，恢复出厂长按多久？", "这个型号支持卫星联网吗？"]:
        prepared = await engine.prepare(question, history, "bm25")
        answer, sources, mode, metrics = await engine.answer(prepared, history, question)
        print(answer, flush=True)
        for role, text in [("user", question), ("assistant", answer)]:
            history.append(Message(id=uuid.uuid4().hex, session_id="check", turn_id="check",
                                   role=role, text=text, answer_mode=mode,
                                   created_at="2026-10-14T00:00:00Z"))


if __name__ == "__main__":
    asyncio.run(main())
