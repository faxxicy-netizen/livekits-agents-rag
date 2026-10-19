import json
import sqlite3

import pytest

from app.knowledge.repository import KnowledgeRepository
from app.knowledge.retrieval import LocalRetriever, RetrievalRegistry
from app.models import Citation, KnowledgeInput, KnowledgeProfile
from app.store import Store
from scripts.migrate_generic_profiles import LEGACY_NAME, migrate


def test_new_install_has_empty_generic_base(config):
    registry = RetrievalRegistry(config)
    assert registry.repository.list() == []
    default = registry.repository.knowledge.get("default")
    assert default["name"] == "默认知识库"
    assert default["profile"] == KnowledgeProfile().model_dump()


@pytest.mark.parametrize("backend", ["bm25", "tfidf"])
async def test_sparse_retrieval_does_not_exclude_other_codes_or_identifier_only_hits(
    config, backend
):
    chunks = [
        Citation(
            chunk_id="c",
            document_id="d",
            document_name="SDK.txt",
            heading="HTTP2",
            text="SDK3.0支持HTTP2与HTTP3两种协议，错误ERR1001表示鉴权失败。",
        )
    ]
    retriever = LocalRetriever(backend, snapshot=([], chunks))
    for query in ["SDK3.0 HTTP2", "ERR1001"]:
        assert (await retriever.search(query)).chunks[0].chunk_id == "c"


def test_factory_migration_is_backed_up_idempotent_and_preserves_custom_profiles(config):
    # Historical factory data is a migration fixture, not an application business rule.
    old = {
        "role": "网络设备支持助手",
        "scope": "帮助用户理解路由器资料，完成安装、联网排查和设备维护。",
        "context_fields": "设备型号、指示灯状态、连接方式",
        "clarification_policy": "仅当资料中的操作存在型号差异时询问型号；信息足够就直接回答。",
        "response_style": "先给下一步操作，再说明预期现象；必要时分步排查。",
        "aliases": "青蓝100=R100\n青岚100=R100\n青岚一百=R100\n阿尔一百=R100\n"
        "青蓝200=R200\n青岚200=R200\n青岚两百=R200\n阿尔两百=R200",
    }
    repository = KnowledgeRepository(config.database_path)
    base = repository.save(
        KnowledgeInput(name=LEGACY_NAME, profile=KnowledgeProfile(**old)), "default"
    )
    custom = repository.save(KnowledgeInput(name="IoT", profile=KnowledgeProfile(role="IoT助手")))
    store = Store(config.database_path)
    session = store.create_session("owner", "bm25", knowledge_profile=base)
    edited_snapshot = {**base, "profile": {**old, "scope": "用户自定义范围"}}
    edited = store.create_session("owner", "bm25", knowledge_profile=edited_snapshot)
    result = migrate(config.database_path)
    assert result["knowledge_bases"] == result["sessions"] == 1
    assert repository.get("default")["profile"] == KnowledgeProfile().model_dump()
    assert store.session(session["id"])["knowledge_profile"]["name"] == "默认知识库"
    assert store.session(edited["id"])["knowledge_profile"] == edited_snapshot
    assert repository.get(custom["id"]) == custom
    with sqlite3.connect(result["backup"]) as db:
        assert (
            json.loads(
                db.execute("SELECT profile FROM knowledge_bases WHERE id='default'").fetchone()[0]
            )
            == old
        )
    assert migrate(config.database_path) == {"knowledge_bases": 0, "sessions": 0, "backup": None}
