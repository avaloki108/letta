from types import SimpleNamespace

import pytest

from letta.schemas.agent import AgentType
from letta.schemas.letta_message_content import TextContent
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message
from letta.services.summarizer.compact import (
    MORPH_COMPACTION_MODEL,
    build_morph_query,
    call_morph_compact,
    compact_messages,
    morph_compact_all_messages,
)
from letta.services.summarizer.summarizer_config import CompactionSettings


def make_message(role: str, text: str, *, step_id: str | None = None) -> Message:
    return Message(role=role, content=[TextContent(text=text)], step_id=step_id)


def make_llm_config() -> LLMConfig:
    return LLMConfig(
        model="gpt-4o-mini",
        model_endpoint_type="openai",
        context_window=128000,
        handle="openai/gpt-4o-mini",
    )


def test_build_morph_query_prefers_latest_user_message_and_focus() -> None:
    messages = [
        make_message("user", "first request"),
        make_message("assistant", "reply"),
        make_message("user", "latest request"),
    ]

    query = build_morph_query(messages, "keep auth details")

    assert query.startswith("latest request")
    assert "Focus:\nkeep auth details" in query


@pytest.mark.asyncio
async def test_call_morph_compact_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MORPH_API_KEY", raising=False)

    with pytest.raises(Exception, match="MORPH_API_KEY"):
        await call_morph_compact(transcript="[user] hi", query="hi")


@pytest.mark.asyncio
async def test_morph_compact_all_messages_keeps_approval_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_morph_compact_prefix(messages, custom_instructions=None):
        assert [message.role for message in messages] == ["user"]
        return "compressed-prefix"

    monkeypatch.setattr(
        "letta.services.summarizer.compact.morph_compact_prefix",
        fake_morph_compact_prefix,
    )

    messages = [
        make_message("system", "system"),
        make_message("user", "user-turn"),
        make_message("assistant", "assistant-tail", step_id="step-1"),
        make_message("approval", "approval-tail", step_id="step-1"),
    ]

    summary, compacted_messages = await morph_compact_all_messages(messages, "focus")

    assert summary == "compressed-prefix"
    assert [message.role for message in compacted_messages] == ["system", "assistant", "approval"]


@pytest.mark.asyncio
async def test_compact_messages_routes_morph_all_without_loading_summarizer_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_build(*args, **kwargs):
        raise AssertionError("build_summarizer_llm_config should not run for Morph compaction")

    async def fake_morph_all(messages, custom_instructions=None):
        return "compressed-history", [messages[0]]

    async def fake_count_tokens_with_tools(*args, **kwargs):
        return 42

    monkeypatch.setattr(
        "letta.services.summarizer.compact.build_summarizer_llm_config",
        fail_build,
    )
    monkeypatch.setattr(
        "letta.services.summarizer.compact.morph_compact_all_messages",
        fake_morph_all,
    )
    monkeypatch.setattr(
        "letta.services.summarizer.compact.count_tokens_with_tools",
        fake_count_tokens_with_tools,
    )

    result = await compact_messages(
        actor=SimpleNamespace(id="user-1", organization_id="org-1"),
        agent_id="agent-1",
        agent_llm_config=make_llm_config(),
        telemetry_manager=SimpleNamespace(),
        llm_client=SimpleNamespace(),
        agent_type=AgentType.letta_v1_agent,
        messages=[make_message("system", "system"), make_message("user", "hello")],
        timezone="UTC",
        compaction_settings=CompactionSettings(
            model=MORPH_COMPACTION_MODEL,
            mode="all",
            sliding_window_percentage=0.3,
        ),
        use_summary_role=True,
    )

    assert result.summary_text == "compressed-history"
    assert result.context_token_estimate == 42
    assert result.summary_message.role == "summary"
    assert len(result.compacted_messages) == 2


@pytest.mark.asyncio
async def test_compact_messages_routes_morph_sliding_window(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_build(*args, **kwargs):
        raise AssertionError("build_summarizer_llm_config should not run for Morph compaction")

    async def fake_morph_sliding_window(actor, agent_llm_config, summarizer_config, in_context_messages):
        return "compressed-window", [in_context_messages[0], in_context_messages[-1]]

    async def fake_count_tokens_with_tools(*args, **kwargs):
        return 17

    monkeypatch.setattr(
        "letta.services.summarizer.compact.build_summarizer_llm_config",
        fail_build,
    )
    monkeypatch.setattr(
        "letta.services.summarizer.compact.morph_compact_via_sliding_window",
        fake_morph_sliding_window,
    )
    monkeypatch.setattr(
        "letta.services.summarizer.compact.count_tokens_with_tools",
        fake_count_tokens_with_tools,
    )

    messages = [
        make_message("system", "system"),
        make_message("user", "older"),
        make_message("assistant", "middle"),
        make_message("user", "latest"),
    ]

    result = await compact_messages(
        actor=SimpleNamespace(id="user-1", organization_id="org-1"),
        agent_id="agent-1",
        agent_llm_config=make_llm_config(),
        telemetry_manager=SimpleNamespace(),
        llm_client=SimpleNamespace(),
        agent_type=AgentType.letta_v1_agent,
        messages=messages,
        timezone="UTC",
        compaction_settings=CompactionSettings(
            model=MORPH_COMPACTION_MODEL,
            mode="sliding_window",
            prompt="keep the latest ask",
            sliding_window_percentage=0.3,
        ),
        use_summary_role=True,
    )

    assert result.summary_text == "compressed-window"
    assert result.context_token_estimate == 17
    assert [message.role for message in result.compacted_messages] == ["system", "summary", "user"]
