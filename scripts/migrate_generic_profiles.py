"""One-time migration of the old, unedited factory profile; never used during conversation.

Only the exact previous factory name/profile is replaced. Custom profiles, documents,
messages and session IDs remain intact. A SQLite backup is saved before any change.
"""

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.models import KnowledgeProfile

LEGACY_NAME = "青岚路由器示例"
LEGACY_PROFILE_HASH = "951b6981bd3e6ad9fbeb75f1029e7e7e50988531bc6bdcb66429a430da103f84"


def is_factory(name, profile):
    canonical = json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return name == LEGACY_NAME and hashlib.sha256(canonical.encode()).hexdigest() == (
        LEGACY_PROFILE_HASH
    )


def migrate(path: Path):
    path = path.resolve(strict=True)
    with sqlite3.connect(path, timeout=20) as db:
        db.row_factory = sqlite3.Row
        base = db.execute("SELECT * FROM knowledge_bases WHERE id='default'").fetchone()
        change_base = bool(base and is_factory(base["name"], json.loads(base["profile"])))
        snapshots = []
        for row in db.execute(
            "SELECT id,knowledge_profile FROM sessions WHERE knowledge_base_id='default'"
        ):
            snapshot = json.loads(row["knowledge_profile"])
            if is_factory(snapshot.get("name"), snapshot.get("profile")):
                snapshots.append((row["id"], snapshot))
        if not change_base and not snapshots:
            return {"knowledge_bases": 0, "sessions": 0, "backup": None}
        stamp = datetime.now(UTC)
        backup = path.with_name(path.name + ".before-generic-" + stamp.strftime("%Y%m%dT%H%M%S%fZ"))
        with sqlite3.connect(backup) as copy:
            db.backup(copy)
        backup.chmod(0o600)
        profile = KnowledgeProfile().model_dump()
        if change_base:
            db.execute(
                "UPDATE knowledge_bases SET name=?,profile=?,updated_at=? WHERE id='default'",
                ("默认知识库", json.dumps(profile, ensure_ascii=False), stamp.isoformat()),
            )
        for sid, snapshot in snapshots:
            snapshot.update(name="默认知识库", profile=profile, updated_at=stamp.isoformat())
            db.execute(
                "UPDATE sessions SET knowledge_profile=? WHERE id=?",
                (json.dumps(snapshot, ensure_ascii=False), sid),
            )
    return {"knowledge_bases": int(change_base), "sessions": len(snapshots), "backup": str(backup)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    print(json.dumps(migrate(parser.parse_args().database), ensure_ascii=False))
