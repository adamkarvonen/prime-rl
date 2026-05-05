# Model Understanding RL Environment

This package defines the `model_understanding` Verifiers environment used by the
PrimeRL configs in `examples/model_understanding/`.

The task is single-turn RL on model-understanding explanations: the rollout
model receives a prior conversation plus a question about its behavior, writes an
explanation, and an Anthropic judge scores that explanation against a reference
answer.

For the full runnable workflow, Slurm commands, known runs, and spot-check
commands, see:

```text
examples/model_understanding/README.md
```

## Files

```text
environments/model_understanding_rl/
|-- README.md
|-- pyproject.toml
`-- model_understanding_rl/
    `-- __init__.py
```

- `pyproject.toml` makes this environment installable as a local Python package.
- `model_understanding_rl/__init__.py` contains the environment implementation.

PrimeRL loads the package through the TOML config's environment section. The
environment entrypoint is:

```python
load_environment(...)
```

## Dataset Contract

The environment expects a JSONL dataset path and split, usually supplied by a run
config such as:

```text
examples/model_understanding/rl_3gpu_thinking_100.toml
```

The current intended dataset is investigation-only and lives in the
activation-oracles repo:

```text
/workspace-vast/adamk/activation_oracles_dev/investigations/model_understanding_prime_rl/results/prime_mu_investigation_only_dataset/train.jsonl
```

It is exported from:

```text
/workspace-vast/adamk/activation_oracles_dev/data_pipelines/model_understanding/runs/qwen3_8b_50k_train/investigations.json
```

joined with the pipeline's screening and verification files. Synthetic data is
deliberately excluded.

Each dataset row must provide the prompt/messages consumed by Verifiers and the
reference answer used as the ground truth answer. The code is intentionally
strict: missing paths, malformed prompt structure, unknown judge modes, and
invalid judge scores assert rather than silently falling back.

## Prompt Rendering

The environment receives explicit tokenizer settings:

```text
tokenizer_name
tokenizer_trust_remote_code
chat_template_kwargs
```

For every judged rollout, it renders the prompt locally with
`tokenizer.apply_chat_template(...)` and compares those token IDs against the
prompt IDs reported by vLLM. The transcript includes both versions so prompt
formatting issues can be audited after a run.

## Judge And Reward

The judge prompt asks for two 1-5 scores:

- `specificity`
- `correctness`

The RL reward uses correctness only:

```text
reward = (correctness - 1) / 4
```

So:

```text
correctness 1 -> reward 0.00
correctness 2 -> reward 0.25
correctness 3 -> reward 0.50
correctness 4 -> reward 0.75
correctness 5 -> reward 1.00
```

Raw `specificity` and `correctness` are also registered as zero-weight rubric
metrics, so they are logged during training but do not affect reward or
advantages.

The preferred judge mode is:

```text
thinking_json
```

That mode enables Anthropic thinking and asks the judge to return JSON text. The
existing configs use:

```text
judge_model = "claude-haiku-4-5-20251001"
judge_thinking_budget = 1024
judge_max_tokens = 1400
```

There is also a `forced_tool` mode in the code from earlier latency experiments.
Treat it as a historical ablation, not the preferred path for new runs.

## Transcript Output

Every judged rollout is appended to:

```text
<output_dir>/judge_transcripts/judge_transcripts_pid<PID>.jsonl
```

Each JSONL record includes:

- prompt messages
- exact rendered rollout prompt
- local rendered prompt token IDs
- vLLM prompt token IDs
- whether local and vLLM prompt IDs match
- vLLM-decoded prompt
- completion text and completion token IDs
- question text
- reference answer
- model response
- judge system prompt
- judge user message
- specificity, correctness, reward, and judge reasoning

These transcripts are the main artifact for qualitative review. The spot-check
renderer in `examples/model_understanding/spot_check_rollouts.py` turns them into
Markdown without truncating prompt, response, reference answer, or judge
reasoning.

## Operational Notes

- Source the activation-oracles `.env` before local runs that call Anthropic, or
  use the Slurm template that sources it for you.
- Keep judge API calls asynchronous with a concurrency semaphore.
- Do not set a fixed rollout sampling seed for real RL runs with multiple
  rollouts per prompt; that made repeated rollouts nearly identical.
- Keep `max_model_len` explicit in vLLM configs.
- Keep `gpu_memory_utilization = 0.90`.
- Do not commit generated `outputs/`, checkpoints, merged model weights, logs, or
  transcripts.

## Related Context

Most dataset and SFT context lives outside this PrimeRL fork:

```text
/workspace-vast/adamk/activation_oracles_dev/data_pipelines/model_understanding/GUIDE.md
/workspace-vast/adamk/activation_oracles_dev/AGENTS.md
/workspace-vast/adamk/activation_oracles_dev/investigations/model_understanding_prime_rl/
```

This package should stay small: it is the RL environment wrapper around those
assets, not the source of truth for the model-understanding data pipeline.
