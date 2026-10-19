import asyncio
from types import SimpleNamespace

import pytest
from livekit.agents import FlushSentinel, llm

from app.conversation.engine import AnswerSegment
from app.models import Citation
from app.store import Store
from app.voice.agent import SupportAgent, spoken_turn_data


async def test_superseded_voice_generation_never_yields_old_answer(config):
    store = Store(config.database_path)
    sid = store.create_session("test-owner", "bm25")["id"]
    agent = SupportAgent(SimpleNamespace(room=None), sid, store, config)
    agent.epoch = 1
    entered, release = asyncio.Event(), asyncio.Event()

    class SlowEngine:
        async def prepare(self, *args):
            entered.set()
            await release.wait()
            return None

        async def stream_answer(self, *args):
            yield AnswerSegment("obsolete answer", [], {})

    agent.engine = SlowEngine()
    agent.turns["t1"] = {"text": "R100如何重置", "history": [], "epoch": 1}
    context = llm.ChatContext()
    context.add_message(role="user", content="R100如何重置", extra={"support_turn_id": "t1"})
    response = agent.llm_node(context, [], None)
    pending = asyncio.create_task(anext(response))
    await entered.wait()
    agent.epoch = 2  # new user speech started while the request was in flight
    release.set()
    with pytest.raises(StopAsyncIteration):
        await pending


async def test_ended_session_never_yields_pending_answer(config):
    store = Store(config.database_path)
    sid = store.create_session("test-owner", "bm25")["id"]
    agent = SupportAgent(SimpleNamespace(room=None), sid, store, config)

    class ReadyEngine:
        async def prepare(self, *args):
            return None

        async def stream_answer(self, *args):
            store.set_status(sid, "ended")
            yield AnswerSegment("obsolete answer", [], {})

    agent.engine = ReadyEngine()
    agent.turns["t1"] = {"text": "R100如何重置", "history": [], "epoch": 0}
    context = llm.ChatContext()
    context.add_message(role="user", content="R100如何重置", extra={"support_turn_id": "t1"})
    assert [chunk async for chunk in agent.llm_node(context, [], None)] == []


async def test_sentence_is_flushed_before_generation_finishes_and_close_cancels(config):
    store = Store(config.database_path)
    sid = store.create_session("test-owner", "bm25")["id"]
    agent = SupportAgent(SimpleNamespace(room=None), sid, store, config)
    closed = asyncio.Event()

    class StreamingEngine:
        async def prepare(self, *args):
            return None

        async def stream_answer(self, *args):
            try:
                yield AnswerSegment("先回答第一句。", [], {"answer_type": "conversation"})
                await asyncio.Event().wait()  # the remaining response is still generating
            finally:
                closed.set()

    agent.engine = StreamingEngine()
    agent.turns["t1"] = {"text": "你好", "history": [], "epoch": 0}
    context = llm.ChatContext()
    context.add_message(role="user", content="你好", extra={"support_turn_id": "t1"})
    stream = agent.llm_node(context, [], None)
    first = await asyncio.wait_for(anext(stream), 1)
    assert first.delta.content == "先回答第一句。"
    assert isinstance(await anext(stream), FlushSentinel)
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await stream.aclose()
    assert closed.is_set()


def test_interrupted_history_only_keeps_citations_and_slots_from_spoken_sentences():
    source = Citation(chunk_id="c", document_id="d", document_name="手册", text="操作")
    turn = {
        "metrics": {"requested_slot": "context"},
        "segments": [
            {
                "text": "第一句。",
                "citations": [],
                "dialogue": {"answer_type": "conversation", "requested_slot": None},
            },
            {
                "text": "尚未播放的追问。",
                "citations": [source],
                "dialogue": {"answer_type": "clarify", "requested_slot": "context"},
            },
        ],
    }
    citations, metrics = spoken_turn_data(turn, "第一句。")
    assert citations == [] and metrics["requested_slot"] is None
    assert spoken_turn_data(turn, "第一句。尚未")[0] == [source]
