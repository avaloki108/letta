"""Standalone compaction functions for message summarization."""

from dataclasses import dataclass
import os
from typing import List, Optional

import httpx

from letta.constants import TOOL_RETURN_TRUNCATION_CHARS
from letta.errors import ContextWindowExceededError, LettaConfigurationError
from letta.helpers.message_helper import convert_message_creates_to_messages
from letta.llm_api.llm_client import LLMClient
from letta.log import get_logger
from letta.otel.tracing import trace_method
from letta.schemas.agent import AgentType
from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import TextContent
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message, MessageCreate
from letta.schemas.provider_trace import BillingContext
from letta.schemas.user import User
from letta.services.summarizer.self_summarizer import self_summarize_all, self_summarize_sliding_window
from letta.services.summarizer.summarizer_all import summarize_all
from letta.services.summarizer.summarizer_config import CompactionSettings, get_default_prompt_for_mode, get_default_summarizer_model
from letta.services.summarizer.summarizer import format_messages_as_compact_plaintext
from letta.services.summarizer.summarizer_sliding_window import (
    count_tokens,
    count_tokens_with_tools,
    summarize_via_sliding_window,
)
from letta.services.telemetry_manager import TelemetryManager
from letta.system import package_summarize_message_no_counts

logger = get_logger(__name__)

MORPH_COMPACTION_MODEL = "morph/compact"
MORPH_COMPACT_API_URL = "https://api.morphllm.com/v1/compact"
MORPH_COMPACT_TIMEOUT_SECONDS = 60.0


@dataclass
class CompactResult:
    """Result of a compaction operation."""

    summary_message: Message
    compacted_messages: list[Message]
    summary_text: str
    context_token_estimate: Optional[int]


def is_morph_compaction_model(model: Optional[str]) -> bool:
    return (model or "").strip().lower() == MORPH_COMPACTION_MODEL


def _extract_text_from_message(message: Message) -> str:
    if not message.content:
        return ""

    parts: list[str] = []
    for part in message.content:
        if isinstance(part, TextContent):
            parts.append(part.text)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text") or ""))
        else:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return " ".join(part for part in parts if part).strip()


def build_morph_query(messages: List[Message], custom_instructions: Optional[str] = None) -> str:
    last_user_message = next((message for message in reversed(messages) if message.role == MessageRole.user), None)
    base_query = _extract_text_from_message(last_user_message) if last_user_message else "Continue the current conversation accurately."
    custom = (custom_instructions or "").strip()
    return f"{base_query}\n\nFocus:\n{custom}" if custom else base_query


def build_morph_transcript(messages: List[Message]) -> str:
    return format_messages_as_compact_plaintext(
        messages,
        include_system=False,
        tool_return_truncation_chars=TOOL_RETURN_TRUNCATION_CHARS,
    )


async def call_morph_compact(*, transcript: str, query: str) -> str:
    api_key = (os.getenv("MORPH_API_KEY") or "").strip()
    if not api_key:
        raise LettaConfigurationError("Morph compaction requires MORPH_API_KEY to be set.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": transcript,
        "query": query,
        "preserve_recent": 0,
    }

    async with httpx.AsyncClient(timeout=MORPH_COMPACT_TIMEOUT_SECONDS) as client:
        response = await client.post(MORPH_COMPACT_API_URL, headers=headers, json=payload)

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip()
        raise LettaConfigurationError(
            f"Morph compaction request failed ({exc.response.status_code}): {detail or exc.response.reason_phrase}"
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise LettaConfigurationError("Morph compaction returned a non-JSON response.") from exc

    output = data.get("output")
    if not isinstance(output, str) or not output.strip():
        raise LettaConfigurationError("Morph compaction response did not include a valid output string.")
    return output


async def morph_compact_prefix(messages: List[Message], custom_instructions: Optional[str] = None) -> str:
    transcript = build_morph_transcript(messages)
    query = build_morph_query(messages, custom_instructions)
    return await call_morph_compact(transcript=transcript, query=query)


async def morph_compact_all_messages(
    messages: List[Message],
    custom_instructions: Optional[str] = None,
) -> tuple[str, List[Message]]:
    logger.info(
        f"Morph-compacting all messages (index 1 to {len(messages) - 2}), keeping last message: {messages[-1].role}"
    )
    if messages[-1].role == MessageRole.approval:
        protected_messages = [messages[-1]]
        if len(messages) >= 2:
            potential_assistant = messages[-2]
            approval_request = messages[-1]
            if potential_assistant.role == MessageRole.assistant and potential_assistant.step_id == approval_request.step_id:
                protected_messages = [potential_assistant, approval_request]
                messages_to_compact = messages[1:-2]
            else:
                messages_to_compact = messages[1:-1]
        else:
            messages_to_compact = messages[1:-1]
    else:
        messages_to_compact = messages[1:]
        protected_messages = []

    compacted = await morph_compact_prefix(messages_to_compact, custom_instructions)
    return compacted, [messages[0], *protected_messages]


async def morph_compact_via_sliding_window(
    actor: User,
    agent_llm_config: LLMConfig,
    summarizer_config: CompactionSettings,
    in_context_messages: List[Message],
) -> tuple[str, List[Message]]:
    system_prompt = in_context_messages[0]
    total_message_count = len(in_context_messages)

    if in_context_messages[-1].role == MessageRole.approval:
        maximum_message_index = total_message_count - 2
    else:
        maximum_message_index = total_message_count - 1

    eviction_percentage = summarizer_config.sliding_window_percentage
    assert summarizer_config.sliding_window_percentage <= 1.0, "Sliding window percentage must be less than or equal to 1.0"
    assistant_message_index = None

    goal_tokens = (1 - summarizer_config.sliding_window_percentage) * agent_llm_config.context_window
    approx_token_count = agent_llm_config.context_window

    def is_valid_cutoff(message: Message):
        if message.role == MessageRole.assistant:
            return True
        if message.role == MessageRole.approval:
            return message.tool_calls is not None and len(message.tool_calls) > 0
        return False

    while approx_token_count >= goal_tokens and eviction_percentage < 1.0:
        eviction_percentage += 0.10
        message_cutoff_index = round(eviction_percentage * total_message_count)

        assistant_message_index = next(
            (
                i
                for i in reversed(range(1, message_cutoff_index + 1))
                if i < len(in_context_messages) and is_valid_cutoff(in_context_messages[i])
            ),
            None,
        )
        if assistant_message_index is None:
            logger.warning(
                f"No assistant/approval message found for evicting up to index {message_cutoff_index}, incrementing eviction percentage"
            )
            continue

        logger.info(f"Attempting to Morph-compact messages index 1:{assistant_message_index} messages")
        post_compaction_buffer = [system_prompt, *in_context_messages[assistant_message_index:]]
        approx_token_count = await count_tokens(
            actor=actor,
            llm_config=agent_llm_config,
            messages=post_compaction_buffer,
        )
        logger.info(
            f"Morph-compacting messages index 1:{assistant_message_index} messages resulted in {approx_token_count} tokens, goal is {goal_tokens}"
        )

    if assistant_message_index is None or eviction_percentage >= 1.0:
        raise ValueError("No assistant message found for sliding window summarization")

    if assistant_message_index >= maximum_message_index:
        raise ValueError(f"Assistant message index {assistant_message_index} is at the end of the message buffer, skipping summarization")

    messages_to_compact = in_context_messages[1:assistant_message_index]
    logger.info(
        f"Morph-compacting {len(messages_to_compact)} messages, from index 1 to {assistant_message_index} (out of {total_message_count})"
    )

    compacted = await morph_compact_prefix(messages_to_compact, summarizer_config.prompt)
    updated_in_context_messages = in_context_messages[assistant_message_index:]
    return compacted, [system_prompt, *updated_in_context_messages]


async def build_summarizer_llm_config(
    agent_llm_config: LLMConfig,
    summarizer_config: CompactionSettings,
    actor: User,
) -> LLMConfig:
    """Derive an LLMConfig for summarization from a model handle.

    This mirrors the agent-creation path: start from the agent's LLMConfig,
    override provider/model/handle from ``compaction_settings.model``, and
    then apply any explicit ``compaction_settings.model_settings`` via
    ``_to_legacy_config_params``.

    For auto mode agents, routes summarization to Haiku 4.5 instead of the
    agent's model, falling back to zai/glm-5 if Haiku is unavailable.

    Args:
        agent_llm_config: The agent's LLM configuration to use as base.
        summarizer_config: Compaction settings with optional model override.
        actor: The user performing the operation.

    Returns:
        LLMConfig configured for summarization.
    """
    # Auto mode agents: route summarization to Haiku 4.5 instead of the LLM router's
    # default (GLM-5). Haiku is cheaper and well-suited for summarization.
    if agent_llm_config.handle and agent_llm_config.handle.startswith("letta/auto"):
        from letta.services.provider_manager import ProviderManager

        try:
            return await ProviderManager().get_llm_config_from_handle("anthropic/claude-haiku-4-5", actor)
        except Exception as e:
            logger.warning(f"Failed to resolve haiku for auto mode summarizer: {e}. Falling back to zai/glm-5.")
            try:
                return await ProviderManager().get_llm_config_from_handle("zai/glm-5", actor)
            except Exception:
                pass

    # If no summarizer model specified, use lightweight provider-specific defaults
    if not summarizer_config.model:
        provider_name = agent_llm_config.provider_name or agent_llm_config.model_endpoint_type
        default_model = get_default_summarizer_model(provider_name)
        if default_model:
            summarizer_config = summarizer_config.model_copy(update={"model": default_model})

    # If still no model after defaults, use agent's model
    if not summarizer_config.model:
        return agent_llm_config

    try:
        # Load default config for the summarizer model handle, using the agent's context window
        from letta.services.provider_manager import ProviderManager

        provider_manager = ProviderManager()

        # If the summarizer model is an auto mode handle, resolve to haiku
        # (safety net for stale compaction_settings that still reference letta/auto)
        if summarizer_config.model and summarizer_config.model.startswith("letta/auto"):
            try:
                base = await provider_manager.get_llm_config_from_handle("anthropic/claude-haiku-4-5", actor)
            except Exception as e:
                logger.warning(
                    f"Failed to resolve haiku for auto mode summarizer handle '{summarizer_config.model}': {e}. Falling back to zai/glm-5."
                )
                base = await provider_manager.get_llm_config_from_handle("zai/glm-5", actor)
        else:
            try:
                base = await provider_manager.get_llm_config_from_handle(
                    handle=summarizer_config.model,
                    actor=actor,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load LLM config for summarizer handle '{summarizer_config.model}': {e}. Falling back to agent's LLM config."
                )
                return agent_llm_config

        # If explicit model_settings are provided for the summarizer, apply
        # them just like server.create_agent_async does for agents.
        if summarizer_config.model_settings is not None:
            update_params = summarizer_config.model_settings._to_legacy_config_params()
            # Don't clobber max_tokens with the Pydantic default when the caller
            # didn't explicitly provide max_output_tokens.
            if "max_output_tokens" not in summarizer_config.model_settings.model_fields_set:
                update_params.pop("max_tokens", None)
            return base.model_copy(update=update_params)

        return base
    except Exception:
        # On any error, do not break the agent – just fall back
        return agent_llm_config


@trace_method
async def compact_messages(
    actor: User,
    agent_id: str,
    agent_llm_config: LLMConfig,
    telemetry_manager: TelemetryManager,
    llm_client: LLMClient,
    agent_type: AgentType,
    messages: List[Message],
    timezone: str,
    compaction_settings: Optional[CompactionSettings] = None,
    agent_tags: Optional[List[str]] = None,
    tools: Optional[List[dict]] = None,  # Tool json schemas
    trigger_threshold: Optional[int] = None,
    run_id: Optional[str] = None,
    step_id: Optional[str] = None,
    use_summary_role: bool = True,
    trigger: Optional[str] = None,
    context_tokens_before: Optional[int] = None,
    messages_count_before: Optional[int] = None,
    billing_context: Optional[BillingContext] = None,
) -> CompactResult:
    """Compact in-context messages using summarization.

    Args:
        actor: The user performing the operation.
        agent_id: The agent's ID.
        agent_llm_config: The agent's LLM configuration.
        messages: The in-context messages to compact.
        timezone: The agent's timezone for message formatting.
        compaction_settings: Optional compaction settings override.
        agent_model_handle: The agent's model handle (used if compaction_settings is None).
        agent_tags: The agent's tags for telemetry.
        tools: The agent's tools (for token counting).
        trigger_threshold: If provided, verify context stays below this after compaction.
        run_id: Optional run ID for telemetry.
        step_id: Optional step ID for telemetry.
        use_summary_role: If True, create summary message with role=summary.
        trigger: What triggered the compaction (for stats).
        context_tokens_before: Token count before compaction (for stats).
        messages_count_before: Message count before compaction (for stats).

    Returns:
        CompactResult containing the summary message, compacted messages, summary text,
        and updated context token estimate.
    """
    summarizer_config = compaction_settings if compaction_settings else CompactionSettings()

    use_morph_compaction = is_morph_compaction_model(summarizer_config.model)
    summarizer_llm_config = None
    if not use_morph_compaction:
        # Build the LLMConfig used for summarization
        summarizer_llm_config = await build_summarizer_llm_config(
            agent_llm_config=agent_llm_config,  # used to set default compaction model
            summarizer_config=summarizer_config,
            actor=actor,
        )

    summarization_mode_used = summarizer_config.mode
    if summarizer_config.prompt is None:
        summarizer_config.prompt = get_default_prompt_for_mode(summarizer_config.mode)
    if use_morph_compaction:
        if summarizer_config.mode in {"all", "self_compact_all"}:
            summary, compacted_messages = await morph_compact_all_messages(
                messages=messages,
                custom_instructions=summarizer_config.prompt,
            )
        elif summarizer_config.mode in {"sliding_window", "self_compact_sliding_window"}:
            summary, compacted_messages = await morph_compact_via_sliding_window(
                actor=actor,
                agent_llm_config=agent_llm_config,
                summarizer_config=summarizer_config,
                in_context_messages=messages,
            )
        else:
            raise ValueError(f"Invalid summarizer mode: {summarizer_config.mode}")
    elif summarizer_config.mode == "self_compact_all":
        try:
            summary, compacted_messages = await self_summarize_all(
                actor=actor,
                agent_id=agent_id,
                agent_llm_config=agent_llm_config,
                telemetry_manager=telemetry_manager,
                llm_client=llm_client,
                agent_type=agent_type,
                messages=messages,
                compaction_settings=summarizer_config,
                run_id=run_id,
                step_id=step_id,
                timezone=timezone,
                agent_tags=agent_tags,
                tools=tools,
                billing_context=billing_context,
            )
        except Exception as e:
            logger.warning(f"Self summarization failed with exception: {str(e)}. Falling back to self sliding window mode.")
            try:
                fallback_config = summarizer_config.model_copy(
                    update={
                        "mode": "self_compact_sliding_window",
                        "prompt": get_default_prompt_for_mode("self_compact_sliding_window"),
                    }
                )
                summary, compacted_messages = await self_summarize_sliding_window(
                    actor=actor,
                    agent_id=agent_id,
                    agent_llm_config=agent_llm_config,
                    telemetry_manager=telemetry_manager,
                    llm_client=llm_client,
                    agent_type=agent_type,
                    messages=messages,
                    compaction_settings=fallback_config,
                    run_id=run_id,
                    step_id=step_id,
                    timezone=timezone,
                    agent_tags=agent_tags,
                    tools=tools,
                    billing_context=billing_context,
                )
                summarization_mode_used = "self_compact_sliding_window"
            except Exception as e:
                logger.warning(f"Self sliding window summarization failed with exception: {str(e)}. Falling back to all mode.")
                fallback_config = summarizer_config.model_copy(
                    update={
                        "mode": "all",
                        "prompt": get_default_prompt_for_mode("all"),
                    }
                )
                summary, compacted_messages = await summarize_all(
                    actor=actor,
                    llm_config=summarizer_llm_config,
                    summarizer_config=fallback_config,
                    in_context_messages=messages,
                    agent_id=agent_id,
                    agent_tags=agent_tags,
                    run_id=run_id,
                    step_id=step_id,
                    billing_context=billing_context,
                )
                summarization_mode_used = "all"
    elif summarizer_config.mode == "self_compact_sliding_window":
        try:
            summary, compacted_messages = await self_summarize_sliding_window(
                actor=actor,
                agent_id=agent_id,
                agent_llm_config=agent_llm_config,
                telemetry_manager=telemetry_manager,
                llm_client=llm_client,
                agent_type=agent_type,
                messages=messages,
                compaction_settings=summarizer_config,
                run_id=run_id,
                step_id=step_id,
                timezone=timezone,
                agent_tags=agent_tags,
                tools=tools,
                billing_context=billing_context,
            )
        except ContextWindowExceededError:
            raise
        except Exception as e:
            # Prompts for all and self mode should be similar --> can use original prompt
            logger.warning(f"Self sliding window summarization failed with exception: {str(e)}. Falling back to all mode.")
            fallback_config = summarizer_config.model_copy(
                update={
                    "mode": "all",
                    "prompt": get_default_prompt_for_mode("all"),
                }
            )
            summary, compacted_messages = await summarize_all(
                actor=actor,
                llm_config=summarizer_llm_config,
                summarizer_config=fallback_config,
                in_context_messages=messages,
                agent_id=agent_id,
                agent_tags=agent_tags,
                run_id=run_id,
                step_id=step_id,
                billing_context=billing_context,
            )
            summarization_mode_used = "all"
    elif summarizer_config.mode == "all":
        summary, compacted_messages = await summarize_all(
            actor=actor,
            llm_config=summarizer_llm_config,
            summarizer_config=summarizer_config,
            in_context_messages=messages,
            agent_id=agent_id,
            agent_tags=agent_tags,
            run_id=run_id,
            step_id=step_id,
            billing_context=billing_context,
        )
    elif summarizer_config.mode == "sliding_window":
        try:
            summary, compacted_messages = await summarize_via_sliding_window(
                actor=actor,
                llm_config=summarizer_llm_config,
                agent_llm_config=agent_llm_config,
                summarizer_config=summarizer_config,
                in_context_messages=messages,
                agent_id=agent_id,
                agent_tags=agent_tags,
                run_id=run_id,
                step_id=step_id,
                billing_context=billing_context,
            )
        except ContextWindowExceededError:
            # If sliding window failed because the transcript was too large for
            # the summarizer's context window, falling back to all mode will fail harder.
            raise
        except Exception as e:
            logger.warning(f"Sliding window summarization failed with exception: {str(e)}. Falling back to all mode.")
            fallback_config = summarizer_config.model_copy(
                update={
                    "mode": "all",
                    "prompt": get_default_prompt_for_mode("all"),
                }
            )
            summary, compacted_messages = await summarize_all(
                actor=actor,
                llm_config=summarizer_llm_config,
                summarizer_config=fallback_config,
                in_context_messages=messages,
                agent_id=agent_id,
                agent_tags=agent_tags,
                run_id=run_id,
                step_id=step_id,
                billing_context=billing_context,
            )
            summarization_mode_used = "all"
    else:
        raise ValueError(f"Invalid summarizer mode: {summarizer_config.mode}")

    # Update the token count (including tools for accurate comparison with LLM's prompt_tokens)
    context_token_estimate = await count_tokens_with_tools(
        actor=actor,
        llm_config=agent_llm_config,
        messages=compacted_messages,
        tools=tools or [],
    )
    logger.info(f"Context token estimate after summarization: {context_token_estimate}")

    # If the trigger_threshold is provided, verify the new token count is below it
    if trigger_threshold is not None and context_token_estimate is not None and context_token_estimate >= trigger_threshold:
        logger.warning(
            "Summarization failed to sufficiently reduce context size: "
            f"post-summarization tokens={context_token_estimate}, "
            f"threshold={trigger_threshold}. "
            "Attempting fallback strategies.",
        )

        # If we used the sliding window mode, try to summarize again with the all mode
        if summarization_mode_used == "sliding_window":
            if use_morph_compaction:
                summary, compacted_messages = await morph_compact_all_messages(
                    messages=compacted_messages,
                    custom_instructions=summarizer_config.prompt,
                )
            else:
                summary, compacted_messages = await summarize_all(
                    actor=actor,
                    llm_config=summarizer_llm_config,
                    summarizer_config=summarizer_config,
                    in_context_messages=compacted_messages,
                    agent_id=agent_id,
                    agent_tags=agent_tags,
                    run_id=run_id,
                    step_id=step_id,
                    billing_context=billing_context,
                )
            summarization_mode_used = "all"

        context_token_estimate = await count_tokens_with_tools(
            actor=actor,
            llm_config=agent_llm_config,
            messages=compacted_messages,
            tools=tools or [],
        )

        # Final edge case: check if we're still over threshold
        if context_token_estimate is not None and context_token_estimate >= trigger_threshold:
            # Check if system prompt is the cause
            system_prompt_token_estimate = await count_tokens(
                actor=actor,
                llm_config=agent_llm_config,
                messages=[compacted_messages[0]],
            )
            if system_prompt_token_estimate is not None and system_prompt_token_estimate >= agent_llm_config.context_window:
                from letta.errors import SystemPromptTokenExceededError

                logger.warning(
                    f"System prompt ({system_prompt_token_estimate} tokens) exceeds context window ({agent_llm_config.context_window})"
                )
                raise SystemPromptTokenExceededError(
                    system_prompt_token_estimate=system_prompt_token_estimate,
                    context_window=agent_llm_config.context_window,
                )

            # Log error but don't brick the agent
            logger.critical(f"Failed to summarize messages after fallback: {context_token_estimate} > {trigger_threshold}")
        else:
            logger.info(f"Summarization fallback succeeded: {context_token_estimate} < {trigger_threshold}")

    # Build compaction stats if we have the before values
    compaction_stats = None
    if trigger and context_tokens_before is not None and messages_count_before is not None:
        compaction_stats = {
            "trigger": trigger,
            "context_tokens_before": context_tokens_before,
            "context_tokens_after": context_token_estimate,
            "context_window": agent_llm_config.context_window,
            "messages_count_before": messages_count_before,
            "messages_count_after": len(compacted_messages) + 1,
        }

    # Create the summary message
    summary_message_str_packed = package_summarize_message_no_counts(
        summary=summary,
        timezone=timezone,
        compaction_stats=compaction_stats,
        mode=summarization_mode_used,
    )

    if use_summary_role:
        # New behavior: Create Message directly with role=summary
        summary_message_obj = Message(
            role=MessageRole.summary,
            content=[TextContent(text=summary_message_str_packed)],
            agent_id=agent_id,
            run_id=run_id,
            step_id=step_id,
        )
    else:
        # Legacy behavior: Use convert_message_creates_to_messages with role=user
        summary_messages = await convert_message_creates_to_messages(
            message_creates=[
                MessageCreate(
                    role=MessageRole.user,
                    content=[TextContent(text=summary_message_str_packed)],
                )
            ],
            agent_id=agent_id,
            timezone=timezone,
            wrap_user_message=False,
            wrap_system_message=False,
            run_id=run_id,
        )
        if len(summary_messages) != 1:
            logger.error(f"Expected only one summary message, got {len(summary_messages)}")
        summary_message_obj = summary_messages[0]

    # Build final messages: [system] + [summary] + remaining compacted messages
    final_messages = [compacted_messages[0], summary_message_obj]
    if len(compacted_messages) > 1:
        final_messages += compacted_messages[1:]

    return CompactResult(
        summary_message=summary_message_obj,
        compacted_messages=final_messages,
        summary_text=summary,
        context_token_estimate=context_token_estimate,
    )
