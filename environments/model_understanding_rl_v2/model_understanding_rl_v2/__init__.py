from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import anthropic
import verifiers as vf
from datasets import load_dataset
from transformers import AutoTokenizer


JUDGE_TOOL = {
    "name": "record_batch_scores",
    "description": "Record comparative judge scores for all model responses in the batch.",
    "input_schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "description": "One entry per response, in the same order they were presented.",
                "items": {
                    "type": "object",
                    "properties": {
                        "response_index": {
                            "type": "integer",
                            "description": "0-based index of the response being scored.",
                        },
                        "correctness": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "description": "Correctness score from 1 to 10.",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of the grading decision for this response.",
                        },
                    },
                    "required": ["response_index", "correctness", "reasoning"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["scores"],
        "additionalProperties": False,
    },
}

JUDGE_SYSTEM_PROMPT = """\
You are grading whether a model produced good explanations of its behavior in context.

You will be given:
1. The QUESTION that was asked
2. The REFERENCE ANSWER to use as ground truth
3. Multiple MODEL RESPONSES to grade comparatively

You are seeing ALL responses to this question at once so you can compare them \
against each other and the reference. Use the full 1-10 scale to differentiate \
between responses. Responses that are clearly better should get higher scores; \
responses that are clearly worse should get lower scores. Spread your scores out \
to reflect meaningful differences.

Use the reference answer to judge correctness, but be flexible about how the explanation \
is framed. Different wording or framing of the same underlying mechanism should count as \
correct (e.g. "I interpret 'check grammar' narrowly" vs "I enter a mechanical correction \
mode" -- these describe the same behavior differently and both count).

Approximate percentages are fine as long as they point to the same qualitative conclusion. \
30% vs 60% both mean "happens a meaningful fraction of the time" -- that's a match. \
The important boundaries are between "never" (0%), "sometimes" (10-70%), \
"almost always" (80-100%), not the exact numbers within those bands.

Focus on whether the response identifies the correct core causal factor, not whether it \
captures every secondary detail from the reference.

CORRECTNESS (1-10):
 1 = Completely wrong, irrelevant, or contradicts the reference
 2 = Mostly wrong, may contain a vaguely relevant keyword
 3 = Identifies a plausible but incorrect core cause
 4 = Hints at the right direction but gets the mechanism wrong
 5 = Identifies the correct core cause but explanation is vague or substantially incomplete
 6 = Correct core cause with moderate detail, some notable gaps
 7 = Correct core cause with good detail, only minor omissions
 8 = Mostly correct and specific, with only trivial inaccuracies
 9 = Essentially correct and well-explained, matches the reference closely
10 = Fully correct, specific, and well-grounded — matches or exceeds the reference

Use the record_batch_scores tool to record your scores for ALL responses."""


def _build_judge_user_message(
    question_text: str,
    reference_answer: str,
    responses: list[str],
) -> str:
    parts = [
        f"QUESTION:\n{question_text}",
        f"\nREFERENCE ANSWER:\n{reference_answer}",
        f"\nThere are {len(responses)} MODEL RESPONSES to grade below.\n",
    ]
    for i, response in enumerate(responses):
        parts.append(f"--- RESPONSE {i} ---\n{response}\n")
    return "\n".join(parts)


@dataclass(frozen=True)
class BatchJudgeResult:
    scores: list[SingleScore]


@dataclass(frozen=True)
class SingleScore:
    response_index: int
    correctness: int
    reasoning: str


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
        role = message["role"]
    else:
        role = getattr(message, "role")
    assert isinstance(role, str)
    return role


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message["content"]
    return getattr(message, "content")


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    assert isinstance(completion, list), f"Unexpected completion type: {type(completion)}"
    for message in reversed(completion):
        if _message_role(message) == "assistant":
            return _content_to_text(_message_content(message)).strip()
    raise AssertionError("Completion did not contain an assistant message")


def _messages_to_dicts(messages: Any) -> list[dict[str, str]]:
    assert isinstance(messages, list), f"Expected message list, got {type(messages)}"
    return [
        {
            "role": _message_role(message),
            "content": _content_to_text(_message_content(message)),
        }
        for message in messages
    ]


def _render_prompt_for_transcript(
    tokenizer: Any,
    messages: list[dict[str, str]],
    chat_template_kwargs: dict[str, Any],
) -> tuple[str, list[int]]:
    text_kwargs = {**chat_template_kwargs, "tokenize": False, "add_generation_prompt": True}
    rendered = tokenizer.apply_chat_template(messages, **text_kwargs)
    assert isinstance(rendered, str), f"Expected rendered prompt string, got {type(rendered)}"

    token_kwargs = {
        **chat_template_kwargs,
        "tokenize": True,
        "return_dict": False,
        "add_generation_prompt": True,
    }
    token_ids = tokenizer.apply_chat_template(messages, **token_kwargs)
    assert isinstance(token_ids, list), f"Expected prompt token id list, got {type(token_ids)}"
    assert all(isinstance(token_id, int) for token_id in token_ids), "Prompt token ids must be integers"
    return rendered, token_ids


def _single_turn_tokens_from_state(state: Any) -> dict[str, Any]:
    trajectory = state["trajectory"]
    assert isinstance(trajectory, list), f"Expected trajectory list, got {type(trajectory)}"
    assert len(trajectory) == 1, f"Expected one trajectory step for single-turn env, got {len(trajectory)}"
    tokens = trajectory[0]["tokens"]
    assert isinstance(tokens, dict), f"Expected trajectory token dict, got {type(tokens)}"
    return tokens


def _token_ids_from_state(tokens: dict[str, Any], key: str) -> list[int]:
    token_ids = tokens[key]
    assert isinstance(token_ids, list), f"Expected {key} list, got {type(token_ids)}"
    assert all(isinstance(token_id, int) for token_id in token_ids), f"{key} must contain integer token ids"
    return token_ids


def _completion_to_record(completion: Any) -> Any:
    if isinstance(completion, str):
        return completion
    return _messages_to_dicts(completion)


def _question_from_prompt(prompt: Any) -> str:
    assert isinstance(prompt, list), f"Expected prompt messages, got {type(prompt)}"
    assert prompt, "Prompt messages are empty"
    last_message = prompt[-1]
    assert _message_role(last_message) == "user", "Final prompt message must be the explanation question"
    return _content_to_text(_message_content(last_message)).strip()


def _parse_tool_result(response: Any, expected_count: int) -> BatchJudgeResult:
    tool_blocks = [
        block
        for block in response.content
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == JUDGE_TOOL["name"]
    ]
    assert len(tool_blocks) == 1, f"Expected exactly one {JUDGE_TOOL['name']} tool call, got {len(tool_blocks)}"
    payload = tool_blocks[0].input
    raw_scores = payload["scores"]
    assert isinstance(raw_scores, list), f"Expected scores list, got {type(raw_scores)}"
    assert len(raw_scores) == expected_count, (
        f"Expected {expected_count} scores, got {len(raw_scores)}"
    )
    scores = []
    for entry in raw_scores:
        score = SingleScore(
            response_index=int(entry["response_index"]),
            correctness=int(entry["correctness"]),
            reasoning=str(entry["reasoning"]),
        )
        assert 1 <= score.correctness <= 10, f"Invalid correctness: {score.correctness}"
        scores.append(score)
    seen_indices = {s.response_index for s in scores}
    assert seen_indices == set(range(expected_count)), (
        f"Expected response_index values 0..{expected_count - 1}, got {sorted(seen_indices)}"
    )
    return BatchJudgeResult(scores=sorted(scores, key=lambda s: s.response_index))


async def _append_transcript(
    transcript_lock: asyncio.Lock,
    transcript_path: Path,
    record: dict[str, Any],
) -> None:
    line = json.dumps(record, ensure_ascii=False)
    async with transcript_lock:
        with transcript_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


async def _judge_batch(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    *,
    model: str,
    thinking_budget: int,
    max_tokens: int,
    question_text: str,
    reference_answer: str,
    response_texts: list[str],
) -> BatchJudgeResult:
    user_message = _build_judge_user_message(
        question_text=question_text,
        reference_answer=reference_answer,
        responses=response_texts,
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            async with semaphore:
                response = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=JUDGE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                    thinking={"type": "enabled", "budget_tokens": thinking_budget},
                    tools=cast(Any, [JUDGE_TOOL]),
                    tool_choice=cast(Any, {"type": "auto"}),
                )
                return _parse_tool_result(response, expected_count=len(response_texts))
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                await asyncio.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def load_environment(
    dataset_path: str,
    dataset_split: str,
    tokenizer_name: str,
    tokenizer_trust_remote_code: bool,
    chat_template_kwargs: dict[str, Any],
    judge_model: str,
    judge_concurrency: int,
    judge_thinking_budget: int,
    judge_max_tokens: int,
    transcript_dir: str,
    **kwargs: Any,
) -> vf.Environment:
    path = Path(dataset_path)
    assert path.exists(), f"Dataset path does not exist: {path}"
    transcript_dir_path = Path(transcript_dir)
    transcript_dir_path.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_dir_path / f"judge_transcripts_pid{os.getpid()}.jsonl"
    assert isinstance(chat_template_kwargs, dict), f"chat_template_kwargs must be a dict, got {type(chat_template_kwargs)}"
    assert judge_concurrency > 0, "judge_concurrency must be positive"
    assert judge_max_tokens > 0, "judge_max_tokens must be positive"
    assert judge_thinking_budget > 0, "judge_thinking_budget must be positive"
    assert judge_max_tokens > judge_thinking_budget, "judge_max_tokens must exceed judge_thinking_budget"

    dataset = load_dataset("json", data_files={dataset_split: str(path)}, split=dataset_split)
    tokenizer: Any = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=tokenizer_trust_remote_code,
    )
    parser = vf.Parser()
    semaphore = asyncio.Semaphore(judge_concurrency)
    transcript_lock = asyncio.Lock()
    client = anthropic.AsyncAnthropic()

    async def correctness_reward_func(prompts, completions, answers, states) -> list[float]:
        question_text = _question_from_prompt(prompts[0])
        response_texts = [_completion_text(c) for c in completions]
        reference_answer = answers[0]

        result = await _judge_batch(
            client,
            semaphore,
            model=judge_model,
            thinking_budget=judge_thinking_budget,
            max_tokens=judge_max_tokens,
            question_text=question_text,
            reference_answer=reference_answer,
            response_texts=response_texts,
        )

        rewards: list[float] = []
        for i, (prompt, completion, answer, state, score) in enumerate(
            zip(prompts, completions, answers, states, result.scores)
        ):
            reward = (score.correctness - 1) / 9
            state["model_understanding_judge"] = score

            prompt_messages = _messages_to_dicts(prompt)
            rendered_prompt, rendered_prompt_token_ids = _render_prompt_for_transcript(
                tokenizer, prompt_messages, chat_template_kwargs,
            )
            state_tokens = _single_turn_tokens_from_state(state)
            vllm_prompt_token_ids = _token_ids_from_state(state_tokens, "prompt_ids")
            completion_token_ids = _token_ids_from_state(state_tokens, "completion_ids")
            prompt_token_ids_match = rendered_prompt_token_ids == vllm_prompt_token_ids

            judge_user_message = _build_judge_user_message(
                question_text=question_text,
                reference_answer=reference_answer,
                responses=response_texts,
            )

            await _append_transcript(
                transcript_lock,
                transcript_path,
                {
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    "group_size": len(prompts),
                    "response_index": i,
                    "judge_model": judge_model,
                    "judge_thinking_budget": judge_thinking_budget,
                    "judge_max_tokens": judge_max_tokens,
                    "prompt_messages": prompt_messages,
                    "rollout_tokenizer_name": tokenizer_name,
                    "rollout_chat_template_kwargs": chat_template_kwargs,
                    "rollout_rendered_prompt": rendered_prompt,
                    "rollout_rendered_prompt_token_ids": rendered_prompt_token_ids,
                    "rollout_vllm_prompt_token_ids": vllm_prompt_token_ids,
                    "rollout_prompt_token_ids_match": prompt_token_ids_match,
                    "rollout_vllm_prompt_decoded": tokenizer.decode(vllm_prompt_token_ids, skip_special_tokens=False),
                    "rollout_prompt_token_count": len(vllm_prompt_token_ids),
                    "completion": _completion_to_record(completion),
                    "completion_token_ids": completion_token_ids,
                    "completion_token_count": len(completion_token_ids),
                    "question_text": question_text,
                    "reference_answer": answer,
                    "model_response": response_texts[i],
                    "judge_system_prompt": JUDGE_SYSTEM_PROMPT,
                    "judge_user_message": judge_user_message,
                    "all_response_texts": response_texts,
                    "correctness": score.correctness,
                    "reward": reward,
                    "judge_reasoning": score.reasoning,
                    "all_scores": [
                        {"response_index": s.response_index, "correctness": s.correctness, "reasoning": s.reasoning}
                        for s in result.scores
                    ],
                },
            )
            rewards.append(reward)

        return rewards

    def correctness_metric(states) -> list[float]:
        return [float(state["model_understanding_judge"].correctness) for state in states]

    rubric = vf.Rubric(
        funcs=[correctness_reward_func, correctness_metric],
        weights=[1.0, 0.0],
        parser=parser,
    )
    return vf.SingleTurnEnv(dataset=dataset, parser=parser, rubric=rubric, **kwargs)
