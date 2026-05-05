from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Selection = Literal["first", "random", "score-diverse", "indices"]


@dataclass(frozen=True)
class TranscriptRecord:
    index: int
    timestamp: float
    batch_index: int
    judge_model: str
    judge_mode: str
    judge_thinking_budget: int
    judge_max_tokens: int
    prompt_messages: list[dict[str, str]]
    rollout_tokenizer_name: str
    rollout_chat_template_kwargs: dict[str, object]
    rollout_rendered_prompt: str
    rollout_prompt_token_ids_match: bool
    rollout_prompt_token_count: int
    completion_token_count: int
    question_text: str
    reference_answer: str
    model_response: str
    judge_system_prompt: str
    judge_user_message: str
    specificity: int
    correctness: int
    reward: float
    judge_reasoning: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render full model-understanding RL rollout transcripts as a Markdown spot check.",
    )
    parser.add_argument("--transcript-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--selection", choices=["first", "random", "score-diverse", "indices"], required=True)
    parser.add_argument("--count", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--indices", type=str)
    args = parser.parse_args()

    if args.selection == "indices":
        assert args.indices is not None, "--indices is required with --selection indices"
        assert args.count is None, "--count must not be passed with --selection indices"
        assert args.seed is None, "--seed must not be passed with --selection indices"
    elif args.selection == "first":
        assert args.count is not None, "--count is required with --selection first"
        assert args.seed is None, "--seed must not be passed with --selection first"
        assert args.indices is None, "--indices must not be passed with --selection first"
    else:
        assert args.count is not None, f"--count is required with --selection {args.selection}"
        assert args.seed is not None, f"--seed is required with --selection {args.selection}"
        assert args.indices is None, f"--indices must not be passed with --selection {args.selection}"

    if args.count is not None:
        assert args.count > 0, "--count must be positive"
    return args


def load_records(transcript_path: Path) -> list[TranscriptRecord]:
    records: list[TranscriptRecord] = []
    with transcript_path.open(encoding="utf-8") as f:
        for index, line in enumerate(f):
            payload = json.loads(line)
            records.append(
                TranscriptRecord(
                    index=index,
                    timestamp=float(payload["timestamp"]),
                    batch_index=int(payload["batch_index"]),
                    judge_model=str(payload["judge_model"]),
                    judge_mode=str(payload["judge_mode"]),
                    judge_thinking_budget=int(payload["judge_thinking_budget"]),
                    judge_max_tokens=int(payload["judge_max_tokens"]),
                    prompt_messages=payload["prompt_messages"],
                    rollout_tokenizer_name=str(payload["rollout_tokenizer_name"]),
                    rollout_chat_template_kwargs=payload["rollout_chat_template_kwargs"],
                    rollout_rendered_prompt=str(payload["rollout_rendered_prompt"]),
                    rollout_prompt_token_ids_match=bool(payload["rollout_prompt_token_ids_match"]),
                    rollout_prompt_token_count=int(payload["rollout_prompt_token_count"]),
                    completion_token_count=int(payload["completion_token_count"]),
                    question_text=str(payload["question_text"]),
                    reference_answer=str(payload["reference_answer"]),
                    model_response=str(payload["model_response"]),
                    judge_system_prompt=str(payload["judge_system_prompt"]),
                    judge_user_message=str(payload["judge_user_message"]),
                    specificity=int(payload["specificity"]),
                    correctness=int(payload["correctness"]),
                    reward=float(payload["reward"]),
                    judge_reasoning=str(payload["judge_reasoning"]),
                )
            )
    assert records, f"No transcript records found in {transcript_path}"
    return records


def parse_indices(indices_text: str, record_count: int) -> list[int]:
    indices = [int(part) for part in indices_text.split(",")]
    assert indices, "--indices did not contain any indices"
    for index in indices:
        assert 0 <= index < record_count, f"Index {index} is outside transcript range 0..{record_count - 1}"
    return indices


def choose_indices(records: list[TranscriptRecord], selection: Selection, count: int | None, seed: int | None, indices: str | None) -> list[int]:
    if selection == "indices":
        assert indices is not None
        return parse_indices(indices, len(records))

    assert count is not None
    assert count <= len(records), f"Requested {count} records but transcript only has {len(records)}"

    if selection == "first":
        return list(range(count))

    assert seed is not None
    rng = random.Random(seed)

    if selection == "random":
        return sorted(rng.sample(range(len(records)), count))

    by_score: dict[int, list[int]] = {score: [] for score in range(1, 6)}
    for record in records:
        by_score[record.correctness].append(record.index)
    for indices_for_score in by_score.values():
        rng.shuffle(indices_for_score)

    selected: list[int] = []
    while len(selected) < count:
        before_round = len(selected)
        for score in range(1, 6):
            if by_score[score] and len(selected) < count:
                selected.append(by_score[score].pop())
        assert len(selected) > before_round, "Could not select more records from score groups"
    return sorted(selected)


def fence(text: str, language: str = "text") -> str:
    longest_backtick_run = 0
    current_run = 0
    for character in text:
        if character == "`":
            current_run += 1
            longest_backtick_run = max(longest_backtick_run, current_run)
        else:
            current_run = 0
    marker = "`" * max(3, longest_backtick_run + 1)
    return f"{marker}{language}\n{text}\n{marker}"


def format_message_list(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message_index, message in enumerate(messages):
        role = message["role"]
        content = message["content"]
        parts.append(f"### Message {message_index}: `{role}`\n\n{fence(content)}")
    return "\n\n".join(parts)


def format_record(record: TranscriptRecord) -> str:
    chat_template_kwargs = json.dumps(record.rollout_chat_template_kwargs, ensure_ascii=False, sort_keys=True)
    metadata = "\n".join(
        [
            f"- Transcript index: `{record.index}`",
            f"- Batch index: `{record.batch_index}`",
            f"- Judge: `{record.judge_model}` / `{record.judge_mode}`",
            f"- Judge thinking budget: `{record.judge_thinking_budget}`",
            f"- Judge max tokens: `{record.judge_max_tokens}`",
            f"- Specificity: `{record.specificity}`",
            f"- Correctness: `{record.correctness}`",
            f"- Reward: `{record.reward}`",
            f"- Rollout tokenizer: `{record.rollout_tokenizer_name}`",
            f"- Chat template kwargs: `{chat_template_kwargs}`",
            f"- Prompt token count: `{record.rollout_prompt_token_count}`",
            f"- Completion token count: `{record.completion_token_count}`",
            f"- Local rendered prompt tokens match vLLM prompt tokens: `{record.rollout_prompt_token_ids_match}`",
        ]
    )
    return "\n\n".join(
        [
            f"## Rollout {record.index}",
            metadata,
            "### Prompt Messages",
            format_message_list(record.prompt_messages),
            "### Exact Rendered Prompt Fed To Rollout Model",
            fence(record.rollout_rendered_prompt),
            "### Rollout Model Response",
            fence(record.model_response),
            "### Explanation Question",
            fence(record.question_text),
            "### Reference Answer",
            fence(record.reference_answer),
            "### Judge System Prompt",
            fence(record.judge_system_prompt),
            "### Judge User Message",
            fence(record.judge_user_message),
            "### Judge Reasoning",
            fence(record.judge_reasoning),
        ]
    )


def write_spot_check(
    transcript_path: Path,
    output_path: Path,
    records: list[TranscriptRecord],
    selected_indices: list[int],
    selection: Selection,
    seed: int | None,
) -> None:
    selected_set = set(selected_indices)
    selected_records = [record for record in records if record.index in selected_set]
    assert len(selected_records) == len(selected_indices), "Selected records were not all found"

    correctness_counts: dict[int, int] = {score: 0 for score in range(1, 6)}
    for record in records:
        correctness_counts[record.correctness] += 1

    header_lines = [
        "# Model Understanding RL Rollout Spot Check",
        f"- Transcript path: `{transcript_path}`",
        f"- Total transcript records: `{len(records)}`",
        f"- Selection: `{selection}`",
        f"- Seed: `{seed}`" if seed is not None else "- Seed: none",
        f"- Selected indices: `{','.join(str(index) for index in selected_indices)}`",
        f"- Correctness counts: `{json.dumps(correctness_counts, sort_keys=True)}`",
    ]
    output = "\n".join(header_lines) + "\n\n" + "\n\n".join(format_record(record) for record in selected_records) + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = load_records(args.transcript_path)
    selected_indices = choose_indices(records, args.selection, args.count, args.seed, args.indices)
    write_spot_check(args.transcript_path, args.output_path, records, selected_indices, args.selection, args.seed)
    print(f"Wrote {len(selected_indices)} full rollout(s) to {args.output_path}")


if __name__ == "__main__":
    main()
