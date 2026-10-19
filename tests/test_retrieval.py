import pytest

from app.conversation.engine import SupportEngine
from app.knowledge.retrieval import RetrievalRegistry


@pytest.mark.usefixtures("seeded_knowledge")
@pytest.mark.parametrize("backend", ["bm25", "tfidf"])
@pytest.mark.parametrize(
    "query,heading",
    [
        ("R100如何恢复出厂设置", "R100 恢复出厂设置"),
        ("R200如何恢复出厂设置", "R200 恢复出厂设置"),
        ("R100橙色指示灯一直亮", "R100 橙色指示灯常亮"),
        ("R100管理员密码忘了怎么办", "忘记管理员密码"),
        ("R200和R100可以混合Mesh组网吗", "R200 产品定位与接口"),
        ("R100怎么修改Wi-Fi密码", "修改 Wi-Fi 密码"),
        ("R100固件怎么升级", "R100 和 R200 固件升级"),
    ],
)
async def test_known_answers_retrieve_relevant_evidence(config, backend, query, heading):
    registry = RetrievalRegistry(config)
    result = await registry.search(query, backend)
    assert any(c.heading == heading for c in result.chunks), result
    assert all(c.chunk_id and c.document_name and c.text for c in result.chunks)


@pytest.mark.parametrize("backend", ["bm25", "tfidf"])
async def test_query_without_any_matching_terms_is_empty(config, backend):
    result = await RetrievalRegistry(config).search("咖啡研磨刀盘参数", backend)
    assert result.chunks == []


async def test_unknown_and_ambiguous_questions_do_not_trigger_fixed_answers(config):
    engine = SupportEngine(config, RetrievalRegistry(config))
    unknown = await engine.prepare("咖啡研磨刀盘参数", [], "bm25")
    assert unknown.result.chunks == []
    assert "不能凭常识编造事实答案" in unknown.evidence
    ambiguous = await engine.prepare("灯一直亮怎么办", [], "bm25")
    assert ambiguous.direct_answer == ""
    assert "product" not in ambiguous.metrics
