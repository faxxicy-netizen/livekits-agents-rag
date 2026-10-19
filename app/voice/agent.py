import asyncio
import contextlib
import dataclasses
import json
import logging
import time
import uuid

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    FlushSentinel,
    JobContext,
    MetricsCollectedEvent,
    TurnHandlingOptions,
    llm,
)
from livekit.plugins import openai, silero

from app.config import settings
from app.conversation.engine import ModelError, SupportEngine, build_system
from app.knowledge.retrieval import RetrievalError, RetrievalRegistry
from app.store import Store
from app.voice.providers import build_stt, build_tts

logger = logging.getLogger("support.voice")
config = settings()
server = AgentServer(
    ws_url=config.livekit_url,
    api_key=config.livekit_api_key,
    api_secret=config.livekit_api_secret,
    host="127.0.0.1",
    port=8081,
    initialize_process_timeout=60,
    num_idle_processes=config.agent_idle_processes,
)


def prewarm(proc):
    if config.stt_provider == "local":
        build_stt(config)
    if config.tts_provider == "local":
        model = build_tts(config)
        model.generate("你好。")
        logger.info("local_tts_warmed model=%s provider=%s", model.model, model.provider)
    if config.vector_enabled and config.embedding_provider == "local":
        from app.knowledge.documents import DocumentRepository
        from app.knowledge.vector import KnowledgeIndex

        KnowledgeIndex(config, DocumentRepository(config)).encode(["预热"], query=True)
        logger.info("local_embedding_warmed model=bge-small-zh-v1.5 device=cpu")


server.setup_fnc = prewarm


def spoken_turn_data(turn: dict, spoken_text: str):
    """Keep sources/clarification state for the played prefix, including a cut last sentence."""
    metrics = dict(turn.get("metrics", {}))
    if "segments" not in turn:
        return turn.get("citations", []), metrics
    sources, cursor = {}, 0
    for segment in turn["segments"]:
        if cursor >= len(spoken_text):
            break
        for source in segment["citations"]:
            sources[source.chunk_id] = source
        metrics.update(segment["dialogue"])
        cursor += len(segment["text"])
    return list(sources.values()), metrics


class SupportAgent(Agent):
    def __init__(self, ctx: JobContext, sid: str, store: Store, configuration=None):
        configuration = configuration or config
        self.registry = RetrievalRegistry(configuration)
        bound = store.session(sid)
        self.knowledge = bound["knowledge_profile"] or self.registry.repository.knowledge.get(
            bound["knowledge_base_id"]
        )
        super().__init__(instructions=build_system(self.knowledge))
        self.ctx, self.sid, self.store = ctx, sid, store
        self.engine = SupportEngine(configuration, self.registry)
        self.turns: dict[str, dict] = {}
        self.epoch = 0
        self.tasks: set[asyncio.Task] = set()

    def background(self, coro):
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

        def done(t):
            if not t.cancelled() and t.exception():
                logger.warning("event_delivery_failed kind=%s", type(t.exception()).__name__)

        task.add_done_callback(done)

    async def publish(self, event: dict):
        if self.ctx.room.isconnected():
            await self.ctx.room.local_participant.send_text(
                json.dumps(event, ensure_ascii=False), topic="support.events"
            )

    async def on_enter(self):
        self.background(self.engine.warmup())
        await self.publish({"type": "ready", "session_id": self.sid})
        self.session.say(
            f"你好，我是知声。我会根据{self.knowledge['name'][:40]}的资料回答，你想了解什么？",
            allow_interruptions=True,
        )

    async def on_user_turn_completed(self, turn_ctx, new_message):
        turn_id = uuid.uuid4().hex
        history = self.store.messages(self.sid)
        text = new_message.text_content or ""
        new_message.extra["support_turn_id"] = turn_id
        self.turns[turn_id] = {
            "text": text,
            "history": history,
            "epoch": self.epoch,
            "started": time.perf_counter(),
            "citations": [],
            "metrics": {},
            "mode": "llm",
            "segments": [],
        }
        message = self.store.save_message(
            self.sid, turn_id, "user", text, message_id=new_message.id, answer_mode="voice"
        )
        await self.publish({"type": "message", "message": message.model_dump()})

    async def llm_node(self, chat_ctx, tools, model_settings):
        user = next(
            (
                m
                for m in reversed(chat_ctx.items)
                if isinstance(m, llm.ChatMessage) and m.role == "user"
            ),
            None,
        )
        turn_id = user.extra.get("support_turn_id") if user else None
        if not turn_id or turn_id not in self.turns:
            return
        turn = self.turns[turn_id]

        def current():
            return turn["epoch"] == self.epoch and self.store.session(self.sid)["status"] != "ended"

        def chunk(text):
            return llm.ChatChunk(
                id=turn_id,
                delta=llm.ChoiceDelta(
                    role="assistant", content=text, extra={"support_turn_id": turn_id}
                ),
            )

        if not current():
            return
        try:
            bound = self.store.session(self.sid)
            prepared = await self.engine.prepare(
                turn["text"],
                turn["history"],
                bound["backend"],
                bound["knowledge_base_id"],
                self.knowledge,
            )
            if not current():
                return
            async with contextlib.aclosing(
                self.engine.stream_answer(prepared, turn["history"], turn["text"])
            ) as answers:
                async for segment in answers:
                    if not current():
                        return
                    turn.update(metrics=segment.metrics, mode=segment.mode)
                    turn.setdefault("segments", []).append(
                        {
                            "text": segment.answer,
                            "citations": segment.citations,
                            "dialogue": {
                                k: segment.metrics.get(k) for k in ("answer_type", "requested_slot")
                            },
                        }
                    )
                    yield chunk(segment.answer)
                    # A sentence is an independent playout segment. The next sentence's
                    # synthesis can overlap this one's playback, under LiveKit cancellation.
                    if not current():
                        return
                    yield FlushSentinel()
        except (RetrievalError, ModelError) as exc:
            if not current():
                return
            turn["mode"] = "error"
            turn.setdefault("metrics", {})["stream_failed"] = True
            self.background(self.publish({"type": "error", "detail": str(exc)}))
            turn.setdefault("segments", []).append(
                {
                    "text": str(exc),
                    "citations": [],
                    "dialogue": {"answer_type": "unknown", "requested_slot": None},
                }
            )
            yield chunk(str(exc))
            yield FlushSentinel()


@server.rtc_session(agent_name=config.livekit_agent_name)
async def entrypoint(ctx: JobContext):
    metadata = json.loads(ctx.job.metadata or "{}")
    sid = metadata.get("session_id")
    if not sid or ctx.job.room.name != "support-" + sid:
        raise ValueError("Invalid application session metadata")
    store = Store(config.database_path)
    store.session(sid)
    missing = config.missing_voice_resources()
    if missing:
        raise ValueError("Missing resources: " + "; ".join(missing))
    await ctx.connect()
    agent = SupportAgent(ctx, sid, store)
    stt_model = (
        await asyncio.to_thread(build_stt, config)
        if config.stt_provider == "local"
        else build_stt(config)
    )
    tts_model = (
        await asyncio.to_thread(build_tts, config)
        if config.tts_provider == "local"
        else build_tts(config)
    )
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=stt_model,
        tts=tts_model,
        # The llm_node uses the same endpoint through SupportEngine so it can validate citations.
        llm=openai.LLM(
            model=config.llm_model, api_key=config.llm_api_key, base_url=config.llm_base_url
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            endpointing={"min_delay": 0.45, "max_delay": 2.5},
            interruption={
                "enabled": True,
                "mode": "vad",
                "min_duration": 0.3,
                "resume_false_interruption": False,
            },
            preemptive_generation={"enabled": False},
        ),
    )

    @session.on("user_state_changed")
    def user_state(event):
        if event.new_state == "speaking":
            agent.epoch += 1
        agent.background(agent.publish({"type": "user_state", "state": event.new_state}))

    @session.on("agent_state_changed")
    def agent_state(event):
        agent.background(agent.publish({"type": "agent_state", "state": event.new_state}))

    @session.on("conversation_item_added")
    def conversation(event: ConversationItemAddedEvent):
        item = event.item
        if not isinstance(item, llm.ChatMessage) or item.role != "assistant":
            return
        turn_id = item.extra.get("support_turn_id", "greeting")
        turn = agent.turns.get(turn_id, {})
        citations, engine_metrics = spoken_turn_data(turn, item.text_content or "")
        timings = {**engine_metrics, **dict(item.metrics)}
        if turn.get("started"):
            timings["turn_until_playback_end_ms"] = round(
                (time.perf_counter() - turn["started"]) * 1000, 1
            )
        message = store.save_message(
            sid,
            turn_id,
            "assistant",
            item.text_content or "",
            citations,
            timings,
            item.interrupted,
            turn.get("mode", "policy"),
            message_id=item.id,
        )
        agent.background(agent.publish({"type": "message", "message": message.model_dump()}))
        agent.turns.pop(turn_id, None)

    @session.on("metrics_collected")
    def collected(event: MetricsCollectedEvent):
        if event.metrics.type == "vad_metrics":
            return
        metric = (
            dataclasses.asdict(event.metrics)
            if dataclasses.is_dataclass(event.metrics)
            else event.metrics.model_dump(mode="json")
            if hasattr(event.metrics, "model_dump")
            else str(event.metrics)
        )
        logger.info("voice_metric session=%s data=%s", sid, json.dumps(metric, default=str))

    @session.on("error")
    def error(event):
        logger.warning("voice_error session=%s source=%s", sid, type(event.source).__name__)
        agent.background(
            agent.publish(
                {"type": "error", "detail": "语音或模型服务暂时不可用，请结束通话后重试。"}
            )
        )

    async def shutdown():
        store.set_status(sid, "ended")
        await session.aclose()
        await asyncio.gather(stt_model.aclose(), tts_model.aclose(), return_exceptions=True)
        await agent.engine.aclose()
        await agent.registry.manager.index.aclose()
        if agent.tasks:
            await asyncio.gather(*agent.tasks, return_exceptions=True)

    ctx.add_shutdown_callback(shutdown)
    await session.start(agent=agent, room=ctx.room)
