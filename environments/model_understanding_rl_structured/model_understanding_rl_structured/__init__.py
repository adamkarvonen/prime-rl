"""Prime RL Verifiers env: model-understanding with structured-judge reward (v2).

Per-rollout scoring with a programmatic format gate + lookup-table judge reward.

Reward pipeline (see investigations/model_understanding_prime_rl/REWARD_SIGNAL.md):

  parsed model report → format gate → (fail: reward=0, skip judge)
                                    → (pass: judge call → per-item LOOKUP sum
                                                      − overproduction penalty
                                                      − length penalty)

All anti-hack term breakdowns and per-cell (mq, ea) item counts are logged as
wandb component metrics at weight=0 (no gradient effect, full observability).
Negative rewards are allowed — the env's reward range is roughly [−1.8, +1.8]
for the modal (2,1) entry, wider for entries with more GT items.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .format_gate import check_format
from .judge import (
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_THINKING_BUDGET,
    parse_structured_report,
    score_response,
)
from .reductions import (
    COMPONENT_NAMES,
    DEFAULT_ALPHA_CORE,
    DEFAULT_ALPHA_REF,
    DEFAULT_FORMAT_FAIL_REWARD,
    REDUCTIONS,
    RewardContext,
    compute_components,
)

__all__ = [
    "COMPONENT_NAMES",
    "DEFAULT_JUDGE_MAX_TOKENS",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_JUDGE_THINKING_BUDGET",
    "REDUCTIONS",
    "RewardContext",
    "check_format",
    "compute_components",
    "load_environment",
    "parse_structured_report",
    "score_response",
]


# ---------------------------------------------------------------------------
# Prompt / completion shape helpers
# ---------------------------------------------------------------------------


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part["text"]
                assert isinstance(text, str)
                parts.append(text)
            elif hasattr(part, "type") and getattr(part, "type") == "text":
                text = getattr(part, "text")
                assert isinstance(text, str)
                parts.append(text)
        return "".join(parts)
    return str(content)


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return message["role"]
    role = getattr(message, "role")
    assert isinstance(role, str)
    return role


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message["content"]
    return getattr(message, "content")


def _messages_to_dicts(messages: Any) -> list[dict[str, str]]:
    assert isinstance(messages, list), f"Expected message list, got {type(messages)}"
    return [
        {
            "role": _message_role(m),
            "content": _content_to_text(_message_content(m)),
        }
        for m in messages
    ]


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    assert isinstance(completion, list), f"Unexpected completion type: {type(completion)}"
    for message in reversed(completion):
        if _message_role(message) == "assistant":
            return _content_to_text(_message_content(message)).strip()
    raise AssertionError("Completion did not contain an assistant message")


def _split_transcript_and_question(prompt: Any) -> tuple[str, str]:
    """RL prompt = [orig_user..., assistant_chosen, explanation_question_user].

    Returns (transcript_text formatted for judge, last-user-message question).
    """
    msgs = _messages_to_dicts(prompt)
    assert msgs, "Empty prompt"
    assert msgs[-1]["role"] == "user", (
        f"Expected final prompt message to be the explanation question (role=user), got {msgs[-1]['role']}"
    )
    question = msgs[-1]["content"].strip()
    transcript_messages = msgs[:-1]
    transcript_text = _format_transcript_for_prompt(transcript_messages)
    return transcript_text, question


def _format_transcript_for_prompt(messages: list[dict[str, str]]) -> str:
    role_names = {"system": "System", "user": "User", "assistant": "Assistant"}
    return "\n\n".join(f"{role_names[m['role']]}: {m['content']}" for m in messages)


def _render_prompt_for_transcript(
    tokenizer: Any,
    messages: list[dict[str, str]],
    chat_template_kwargs: dict[str, Any],
) -> tuple[str, list[int]]:
    text_kwargs = {**chat_template_kwargs, "tokenize": False, "add_generation_prompt": True}
    rendered = tokenizer.apply_chat_template(messages, **text_kwargs)
    assert isinstance(rendered, str)
    token_kwargs = {**chat_template_kwargs, "tokenize": True, "return_dict": False, "add_generation_prompt": True}
    token_ids = tokenizer.apply_chat_template(messages, **token_kwargs)
    assert isinstance(token_ids, list) and all(isinstance(t, int) for t in token_ids)
    return rendered, token_ids


def _single_turn_tokens_from_state(state: Any) -> dict[str, Any]:
    trajectory = state["trajectory"]
    assert isinstance(trajectory, list) and len(trajectory) == 1
    tokens = trajectory[0]["tokens"]
    assert isinstance(tokens, dict)
    return tokens


def _token_ids_from_state(tokens: dict[str, Any], key: str) -> list[int]:
    ids = tokens[key]
    assert isinstance(ids, list) and all(isinstance(t, int) for t in ids)
    return ids


def _completion_to_record(completion: Any) -> Any:
    if isinstance(completion, str):
        return completion
    return _messages_to_dicts(completion)


async def _append_transcript(
    lock: asyncio.Lock,
    path: Path,
    record: dict[str, Any],
) -> None:
    line = json.dumps(record, ensure_ascii=False)
    async with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Environment entrypoint
# ---------------------------------------------------------------------------


def load_environment(
    dataset_path: str,
    dataset_split: str,
    tokenizer_name: str,
    tokenizer_trust_remote_code: bool,
    chat_template_kwargs: dict[str, Any],
    judge_concurrency: int,
    transcript_dir: str,
    reward_reduction: str = "core_anti_hack_v2",
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
    judge_thinking_budget: int = DEFAULT_JUDGE_THINKING_BUDGET,
    reward_alpha_core: float = DEFAULT_ALPHA_CORE,
    reward_alpha_ref: float = DEFAULT_ALPHA_REF,
    reward_format_fail_reward: float = DEFAULT_FORMAT_FAIL_REWARD,
    reward_padding_penalty: float = 0.5,  # used only by the v1 core_minus_padding reduction
    reward_min_sanity: float = -10.0,     # sanity bound, not a clip
    reward_max_sanity: float = 10.0,
    **kwargs: Any,
):
    # Deferred imports so the reductions/format_gate modules can be unit-tested
    # without pulling in verifiers / datasets / transformers / anthropic.
    import anthropic
    import verifiers as vf
    from datasets import load_dataset
    from transformers import AutoTokenizer

    path = Path(dataset_path)
    assert path.exists(), f"Dataset path does not exist: {path}"
    transcript_dir_path = Path(transcript_dir)
    transcript_dir_path.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_dir_path / f"judge_transcripts_pid{os.getpid()}.jsonl"

    assert reward_reduction in REDUCTIONS, (
        f"reward_reduction={reward_reduction!r} not in {sorted(REDUCTIONS)}"
    )
    assert isinstance(chat_template_kwargs, dict)
    assert judge_concurrency > 0
    assert judge_max_tokens > judge_thinking_budget, (
        f"judge_max_tokens ({judge_max_tokens}) must exceed judge_thinking_budget ({judge_thinking_budget})"
    )
    assert reward_alpha_core >= 0 and reward_alpha_ref >= 0

    dataset = load_dataset("json", data_files={dataset_split: str(path)}, split=dataset_split)
    tokenizer: Any = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=tokenizer_trust_remote_code,
    )
    parser = vf.Parser()
    semaphore = asyncio.Semaphore(judge_concurrency)
    transcript_lock = asyncio.Lock()
    client = anthropic.AsyncAnthropic()
    reduction_fn = REDUCTIONS[reward_reduction]
    reduction_kwargs: dict[str, Any] = {
        "alpha_core": reward_alpha_core,
        "alpha_ref": reward_alpha_ref,
        "format_fail_reward": reward_format_fail_reward,
        "padding_penalty": reward_padding_penalty,
    }

    async def structured_reward_func(prompts, completions, answers, states) -> list[float]:
        async def _score_one(i: int) -> tuple[int, float]:
            prompt = prompts[i]
            completion = completions[i]
            gt_text = answers[i]

            transcript_text, question_text = _split_transcript_and_question(prompt)
            model_text = _completion_text(completion)

            # Parse both reports up-front (cheap; needed for format gate + count features).
            gt_report = parse_structured_report(gt_text)
            model_report = parse_structured_report(model_text)

            m_core = len(model_report.core_causes)
            m_ref = len(model_report.refuted_hypotheses)
            gt_core = len(gt_report.core_causes)
            gt_ref = len(gt_report.refuted_hypotheses)

            # Token counts (need these for length penalty + transcript).
            state = states[i]
            state_tokens = _single_turn_tokens_from_state(state)
            vllm_prompt_token_ids = _token_ids_from_state(state_tokens, "prompt_ids")
            completion_token_ids = _token_ids_from_state(state_tokens, "completion_ids")
            completion_tokens = len(completion_token_ids)

            # Format gate — hard structural checks.
            format_pass, format_fail_reason = check_format(model_report)

            # Judge call only when format passes (saves API spend on malformed rollouts).
            judge_score = None
            judge_call_skipped = not format_pass
            if format_pass:
                _gt_parsed, _model_parsed, judge_score = await score_response(
                    client,
                    semaphore,
                    question_text=question_text,
                    transcript_text=transcript_text,
                    gt_text=gt_text,
                    model_text=model_text,
                    model=judge_model,
                    max_tokens=judge_max_tokens,
                    thinking_budget=judge_thinking_budget,
                )

            ctx = RewardContext(
                score=judge_score,
                m_core=m_core,
                m_ref=m_ref,
                gt_core=gt_core,
                gt_ref=gt_ref,
                completion_tokens=completion_tokens,
                format_pass=format_pass,
                format_fail_reason=format_fail_reason,
            )
            reward = reduction_fn(ctx, **reduction_kwargs)
            components = compute_components(ctx)

            assert reward_min_sanity <= reward <= reward_max_sanity, (
                f"reduction {reward_reduction!r} returned reward={reward}, outside sanity bounds "
                f"[{reward_min_sanity}, {reward_max_sanity}]"
            )

            state["structured_judge_score"] = asdict(judge_score) if judge_score is not None else None
            state["structured_judge_components"] = components
            state["structured_judge_reward"] = reward
            state["format_pass"] = format_pass
            state["format_fail_reason"] = format_fail_reason
            state["judge_call_skipped"] = judge_call_skipped

            prompt_messages = _messages_to_dicts(prompt)
            rendered_prompt, rendered_prompt_token_ids = _render_prompt_for_transcript(
                tokenizer, prompt_messages, chat_template_kwargs,
            )

            await _append_transcript(
                transcript_lock,
                transcript_path,
                {
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    "rollout_index_in_batch": i,
                    "group_size": len(prompts),
                    "judge_model": judge_model,
                    "judge_thinking_budget": judge_thinking_budget,
                    "judge_max_tokens": judge_max_tokens,
                    "judge_call_skipped": judge_call_skipped,
                    "reward_reduction": reward_reduction,
                    "reward_alpha_core": reward_alpha_core,
                    "reward_alpha_ref": reward_alpha_ref,
                    "reward_padding_penalty": reward_padding_penalty,
                    "format_pass": format_pass,
                    "format_fail_reason": format_fail_reason,
                    "prompt_messages": prompt_messages,
                    "rollout_tokenizer_name": tokenizer_name,
                    "rollout_chat_template_kwargs": chat_template_kwargs,
                    "rollout_rendered_prompt": rendered_prompt,
                    "rollout_rendered_prompt_token_ids": rendered_prompt_token_ids,
                    "rollout_vllm_prompt_token_ids": vllm_prompt_token_ids,
                    "rollout_prompt_token_ids_match": rendered_prompt_token_ids == vllm_prompt_token_ids,
                    "rollout_vllm_prompt_decoded": tokenizer.decode(vllm_prompt_token_ids, skip_special_tokens=False),
                    "rollout_prompt_token_count": len(vllm_prompt_token_ids),
                    "completion": _completion_to_record(completion),
                    "completion_token_ids": completion_token_ids,
                    "completion_token_count": completion_tokens,
                    "question_text": question_text,
                    "transcript_text": transcript_text,
                    "gt_structured_text": gt_text,
                    "model_response_text": model_text,
                    "m_core_count": m_core,
                    "m_refuted_count": m_ref,
                    "gt_core_count": gt_core,
                    "gt_refuted_count": gt_ref,
                    "structured_judge_score": asdict(judge_score) if judge_score is not None else None,
                    "structured_judge_components": components,
                    "reward": reward,
                },
            )
            return i, reward

        results = await asyncio.gather(*(_score_one(i) for i in range(len(prompts))))
        rewards = [0.0] * len(prompts)
        for i, r in results:
            rewards[i] = r
        return rewards

    # Per-component metric funcs read from state["structured_judge_components"].
    # All run at weight=0 — purely observability.
    def _make_component_metric(component_key: str) -> Callable[[list[Any]], list[float]]:
        def _metric(states) -> list[float]:
            return [float(s["structured_judge_components"][component_key]) for s in states]
        _metric.__name__ = f"metric_{component_key}"
        return _metric

    component_funcs = [_make_component_metric(name) for name in COMPONENT_NAMES]
    n_components = len(component_funcs)

    rubric = vf.Rubric(
        funcs=[structured_reward_func, *component_funcs],
        weights=[1.0] + [0.0] * n_components,
        parser=parser,
    )
    return vf.SingleTurnEnv(dataset=dataset, parser=parser, rubric=rubric, **kwargs)
