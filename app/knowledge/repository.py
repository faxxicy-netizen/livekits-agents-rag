"""Knowledge-base metadata and per-base business guidance."""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from app.models import KnowledgeInput, KnowledgeProfile

DEFAULT_KNOWLEDGE_BASE = "default"


class KnowledgeRepository:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS knowledge_bases (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, profile TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )""")
            profile = KnowledgeProfile()
            name = "默认知识库"
            stamp = datetime.now(UTC).isoformat()
            db.execute(
                "INSERT OR IGNORE INTO knowledge_bases VALUES (?,?,?,?,?)",
                (DEFAULT_KNOWLEDGE_BASE, name, profile.model_dump_json(), stamp, stamp),
            )

    @staticmethod
    def decode(row):
        value = dict(row)
        value["profile"] = json.loads(value["profile"])
        return value

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def list(self):
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT * FROM knowledge_bases ORDER BY created_at, id").fetchall()
        return [self.decode(row) for row in rows]

    def get(self, knowledge_base_id: str):
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT * FROM knowledge_bases WHERE id=?", (knowledge_base_id,)
            ).fetchone()
        if row is None:
            raise KeyError(knowledge_base_id)
        return self.decode(row)

    def save(self, body: KnowledgeInput, knowledge_base_id: str | None = None):
        stamp = datetime.now(UTC).isoformat()
        if knowledge_base_id:
            self.get(knowledge_base_id)
        else:
            knowledge_base_id = uuid.uuid4().hex
        with self.connect() as db:
            db.execute(
                "INSERT INTO knowledge_bases VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                "name=excluded.name, profile=excluded.profile, updated_at=excluded.updated_at",
                (
                    knowledge_base_id,
                    body.name.strip(),
                    body.profile.model_dump_json(),
                    stamp,
                    stamp,
                ),
            )
        return self.get(knowledge_base_id)
