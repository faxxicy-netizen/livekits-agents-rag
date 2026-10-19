from pathlib import Path

import pytest

from app.config import Settings


@pytest.fixture
def config(tmp_path: Path):
    return Settings(
        _env_file=None,
        database_path=tmp_path / "test.db",
        llm_api_key="",
        local_asr_dir=tmp_path / "asr",
        local_tts_dir=tmp_path / "tts",
        vector_enabled=False,
        rag_backend="bm25",
    )


@pytest.fixture
def seeded_knowledge(config):
    """Small synthetic retrieval corpus, explicitly inserted only by tests that need it."""
    from app.knowledge.documents import DocumentRepository, build_document

    content = """# 测试资料

## R100 恢复出厂设置
R100 通电时按住 RESET 八秒，松开后等待两分钟，恢复出厂设置会清除网络配置。

## R200 恢复出厂设置
R200 通电时长按 RESET 十秒恢复出厂设置。

## R100 橙色指示灯常亮
R100 橙色指示灯一直亮表示联网异常，先检查接线。

## 忘记管理员密码
R100 忘记管理员密码时，需要恢复出厂设置后重新配置。

## R200 产品定位与接口
R200 和 R100 不可以混合 Mesh 组网。

## 修改 Wi-Fi 密码
R100 可以在管理页面修改 Wi-Fi 密码。

## R100 和 R200 固件升级
R100 固件升级前需要备份设置，在管理页面上传固件。
"""
    DocumentRepository(config).save(build_document("04-重置与密码.md", content))
