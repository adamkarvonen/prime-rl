# Model Understanding RL

This is a PrimeRL environment for RL fine-tuning model-understanding explanations.

## Current Status

Verified runs on the Slurm cluster:

- `mu-qwen3-8b-rl-2gpu-forced-tool-noseed-v2`
  - W&B: `https://wandb.ai/adam-karvonen/model-understanding-rl/runs/bcf6a85eeb634a6b8356d11dabf0b0d7`
  - Slurm job: `1504030`
  - Status: completed cleanly
  - Layout: 1 inference GPU, 1 trainer GPU
  - Judge mode: forced tool, no judge thinking

- `mu-qwen3-8b-rl-3gpu-thinking-split`
  - W&B: `https://wandb.ai/adam-karvonen/model-understanding-rl/runs/7ea94700f0b640f3b629a2f44a62778a`
  - Slurm job: `1504393`
  - Status: completed cleanly
  - Layout: 2 independent inference GPU replicas, 1 trainer GPU
  - Judge mode: Haiku 4.5 with thinking enabled, 1024-token thinking budget
  - Final local checkpoint: `outputs/model-understanding-rl-qwen3-8b-3gpu-thinking-split/checkpoints/step_100`

The 3-GPU run finished 100 steps in 33m39s. Online train reward was noisy and did not show a clear monotonic increase:

- Steps 0-9: mean reward 0.3015
- Steps 50-59: mean reward 0.4641
- Steps 90-99: mean reward 0.3414
- Overall: mean reward 0.3353

Treat the online reward curve as a smoke signal only. A fixed held-out eval is needed before drawing conclusions about learning.

## Running

Dry run:

```bash
.venv/bin/rl @ examples/model_understanding/rl_3gpu_thinking_100.toml --dry-run
```

Submit via Slurm:

```bash
.venv/bin/rl @ examples/model_understanding/rl_3gpu_thinking_100.toml
```

The Slurm template requests partial nodes with `--gres=gpu:N` and `--qos=high`.

## Spot Checks

Full transcript records are saved as JSONL under each run's `judge_transcripts/` directory. To render readable Markdown:

```bash
.venv/bin/python examples/model_understanding/spot_check_rollouts.py \
  --transcript-path outputs/model-understanding-rl-qwen3-8b-3gpu-thinking-split/judge_transcripts/judge_transcripts_pid2467211.jsonl \
  --output-path outputs/model-understanding-rl-qwen3-8b-3gpu-thinking-split/spot_checks/rollouts_score_diverse_seed0_count8.md \
  --selection score-diverse \
  --count 8 \
  --seed 0
```

The spot check includes:

- Prompt messages
- Exact rendered chat-template prompt fed to the rollout model
- Model response
- Reference answer
- Judge system prompt
- Judge user message
- Judge scores and reasoning
- Tokenization sanity metadata

## Launcher Note

The 3-GPU LoRA run needs a small local-launcher change in `src/prime_rl/entrypoints/rl.py`.

PrimeRL's default local launcher starts one inference service for the inference GPU group. With LoRA and multiple inference GPUs, that goes through vLLM's data-parallel LoRA path, which failed in this setup. The local patch starts independent one-GPU inference replicas instead and gives the orchestrator both base URLs.

This changes launch plumbing only. It does not change trainer loss, advantage logic, reward computation, or the RL algorithm.
