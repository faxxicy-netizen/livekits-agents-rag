import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from app.models import Citation, Message


def now() -> str:
    return datetime.now(UTC).isoformat()


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, title TEXT NOT NULL,
                    backend TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'idle',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL,
                    citations TEXT NOT NULL, metrics TEXT NOT NULL,
                    interrupted INTEGER NOT NULL, answer_mode TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id, created_at);
            """)
            db.execute("BEGIN IMMEDIATE")
            columns = {r[1] for r in db.execute("PRAGMA table_info(sessions)")}
            if "knowledge_base_id" not in columns:
                db.execute(
                    "ALTER TABLE sessions ADD COLUMN knowledge_base_id "
                    "TEXT NOT NULL DEFAULT 'default'"
                )
            if "knowledge_profile" not in columns:
                db.execute(
                    "ALTER TABLE sessions ADD COLUMN knowledge_profile TEXT NOT NULL DEFAULT '{}'"
                )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def create_session(
        self,
        owner: str,
        backend: str,
        title: str = "新的知识库咨询",
        *,
        knowledge_base_id: str = "default",
        knowledge_profile: dict | None = None,
    ) -> dict:
        sid = uuid.uuid4().hex
        with self.connect() as db:
            db.execute(
                "INSERT INTO sessions "
                "(id,owner,title,backend,status,created_at,updated_at,knowledge_base_id,"
                "knowledge_profile) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sid,
                    owner,
                    title,
                    backend,
                    "idle",
                    now(),
                    now(),
                    knowledge_base_id,
                    json.dumps(knowledge_profile or {}, ensure_ascii=False),
                ),
            )
        return self.session(sid)

    def freeze_legacy_profiles(self, knowledge_repository):
        with self.connect() as db:
            rows = db.execute(
                "SELECT id,knowledge_base_id FROM sessions WHERE knowledge_profile='{}'"
            ).fetchall()
            for row in rows:
                try:
                    kb = knowledge_repository.get(row["knowledge_base_id"])
                except KeyError:
                    continue
                snapshot = {k: kb[k] for k in ("id", "name", "profile", "updated_at")}
                db.execute(
                    "UPDATE sessions SET knowledge_profile=? WHERE id=?",
                    (json.dumps(snapshot, ensure_ascii=False), row["id"]),
                )

    def session(self, sid: str, owner: str | None = None) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
        if row is None or (owner is not None and row["owner"] != owner):
            raise KeyError(sid)
        result = dict(row)
        result.pop("owner")
        result["knowledge_profile"] = json.loads(result["knowledge_profile"])
        return result

    def sessions(self, owner: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM sessions WHERE owner=? ORDER BY updated_at DESC", (owner,)
            ).fetchall()
        return [
            {
                **{k: v for k, v in dict(row).items() if k not in {"owner", "knowledge_profile"}},
                "knowledge_profile": json.loads(row["knowledge_profile"]),
            }
            for row in rows
        ]

    def set_status(self, sid: str, status: str):
        with self.connect() as db:
            db.execute(
                "UPDATE sessions SET status=?, updated_at=? WHERE id=?", (status, now(), sid)
            )

    def delete_session(self, sid: str, owner: str):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT owner,status FROM sessions WHERE id=?", (sid,)).fetchone()
            if row is None or row["owner"] != owner:
                raise KeyError(sid)
            if row["status"] == "active":
                raise ValueError("请先结束通话，再删除会话。")
            # Foreign-key cascade removes messages, including their citation/metrics snapshots.
            db.execute("DELETE FROM sessions WHERE id=? AND owner=?", (sid, owner))

    def save_message(
        self,
        sid: str,
        turn_id: str,
        role: str,
        text: str,
        citations: list[Citation] | None = None,
        metrics: dict | None = None,
        interrupted: bool = False,
        answer_mode: str = "llm",
        message_id: str | None = None,
    ) -> Message:
        msg = Message(
            id=message_id or uuid.uuid4().hex,
            session_id=sid,
            turn_id=turn_id,
            role=role,
            text=text,
            citations=citations or [],
            metrics=metrics or {},
            interrupted=interrupted,
            answer_mode=answer_mode,
            created_at=now(),
        )
        with self.connect() as db:
            db.execute(
                """INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET text=excluded.text,
                        citations=excluded.citations, metrics=excluded.metrics,
                        interrupted=excluded.interrupted""",
                (
                    msg.id,
                    sid,
                    turn_id,
                    role,
                    text,
                    json.dumps([c.model_dump() for c in msg.citations], ensure_ascii=False),
                    json.dumps(msg.metrics),
                    int(interrupted),
                    answer_mode,
                    msg.created_at,
                ),
            )
            if role == "user":
                db.execute(
                    "UPDATE sessions SET title=? WHERE id=? "
                    "AND title IN ('新的产品咨询','新的知识库咨询')",
                    (text[:24], sid),
                )
            db.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now(), sid))
        return msg

    def messages(self, sid: str) -> list[Message]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM messages WHERE session_id=? ORDER BY created_at, rowid", (sid,)
            ).fetchall()
        return [
            Message(
                **{
                    **dict(row),
                    "citations": json.loads(row["citations"]),
                    "metrics": json.loads(row["metrics"]),
                }
            )
            for row in rows
        ]
