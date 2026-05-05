# Model Understanding RL

This directory contains the PrimeRL setup for RL fine-tuning model-understanding
explanations. The task is: given a prior conversation and a question about the
model's behavior, train the rollout model to produce an explanation that matches
a reference answer. Rewards come from an Anthropic LLM judge.

The current target model is a Qwen3-8B SFT checkpoint merged from:

```text
checkpoints_text_sft/mu_qwen3_8b_50k_s7_synth_e1_kl1
```

The merged model used by these configs is local to this machine:

```text
/workspace-vast/adamk/prime-rl/outputs/mu_qwen3_8b_50k_s7_synth_e1_kl1_merged
```

## Upstream Context

Most of the model-understanding data and experiment context lives in the
activation-oracles repo, not in this PrimeRL fork:

```text
/workspace-vast/adamk/activation_oracles_dev
```

Useful pointers:

- `/workspace-vast/adamk/activation_oracles_dev/data_pipelines/model_understanding/GUIDE.md`
  - Main guide for the model-understanding data pipeline and SFT setup.

- `/workspace-vast/adamk/activation_oracles_dev/AGENTS.md`
  - Repo-specific operational instructions, Slurm conventions, Anthropic API
    conventions, and experiment footguns.

- `/workspace-vast/adamk/activation_oracles_dev/investigations/model_understanding_prime_rl/`
  - One-off PrimeRL integration scripts and benchmark outputs.
  - Contains dataset export, SFT adapter merge, and judge latency benchmark code.

- `/workspace-vast/adamk/activation_oracles_dev/investigations/model_understanding_prime_rl/results/prime_mu_investigation_only_dataset/`
  - Exported train/eval JSONL files consumed by this PrimeRL environment.

This PrimeRL fork should be read as the RL execution wrapper around those
activation-oracles assets. It intentionally does not duplicate the source
dataset, merged model weights, checkpoints, transcripts, or activation-oracles
pipeline code.

## File Map

- `environments/model_understanding_rl/model_understanding_rl/__init__.py`
  - PrimeRL environment implementation.
  - Loads the train/eval JSONL dataset.
  - Builds rollout prompts from the stored message list.
  - Calls the Anthropic judge asynchronously with a concurrency semaphore.
  - Converts judge correctness scores to rewards.
  - Saves full judge/rollout transcripts.

- `environments/model_understanding_rl/pyproject.toml`
  - Makes the environment installable as a local Python package.

- `examples/model_understanding/*.toml`
  - Run configs for smoke, profile, 2-GPU, and 3-GPU runs.

- `examples/model_understanding/spot_check_rollouts.py`
  - Renders transcript JSONL files into full Markdown spot checks.

- `cluster_templates/single_node_rl_shared.sbatch.j2`
  - Slurm template for partial-node jobs.
  - Requests `--gres=gpu:N`, not a full node.
  - Uses `--qos=high`.
  - Sources `/workspace-vast/adamk/activation_oracles_dev/.env` and local `.env`.
  - Logs GPU utilization to `outputs/.../logs/gpu_util_${SLURM_JOB_ID}.csv`.

- `src/prime_rl/entrypoints/rl.py`
  - PrimeRL local launcher.
  - Patched only for multi-inference-GPU LoRA runs. See "Launcher Patch" below.

## Dataset

The configs point at the exported PrimeRL-format dataset:

```text
/workspace-vast/adamk/activation_oracles_dev/investigations/model_understanding_prime_rl/results/prime_mu_investigation_only_dataset/train.jsonl
/workspace-vast/adamk/activation_oracles_dev/investigations/model_understanding_prime_rl/results/prime_mu_investigation_only_dataset/eval.jsonl
/workspace-vast/adamk/activation_oracles_dev/investigations/model_understanding_prime_rl/results/prime_mu_investigation_only_dataset/metadata.json
```

The export/prep scripts live in the activation-oracles repo:

```text
/workspace-vast/adamk/activation_oracles_dev/investigations/model_understanding_prime_rl/export_prime_dataset.py
/workspace-vast/adamk/activation_oracles_dev/investigations/model_understanding_prime_rl/merge_sft_adapter.py
/workspace-vast/adamk/activation_oracles_dev/investigations/model_understanding_prime_rl/benchmark_judge_latency.py
```

The environment expects each dataset row to contain the conversation prompt,
question, and reference answer. It fails loudly if required fields are missing.

The current dataset is investigation-only:

```text
num_train_examples: 22011
num_eval_examples: 0
synthetic_data_paths: null
```

It is exported from `qwen3_8b_50k_train/investigations.json` joined with
`screening.json` and `verification.json`. It deliberately excludes
`synthetic_data.json`; an earlier PrimeRL export accidentally inherited the SFT
config's synthetic data path and produced a mixed dataset with 109,965 examples.

The export command used for the investigation-only dataset was:

```bash
cd /workspace-vast/adamk/activation_oracles_dev
.venv/bin/python investigations/model_understanding_prime_rl/export_prime_dataset.py \
  --sft-config checkpoints_text_sft/mu_qwen3_8b_50k_s7_synth_e1_kl1/final/training_config.json \
  --run-dir data_pipelines/model_understanding/runs/qwen3_8b_50k_train \
  --exclude-synthetic \
  --output-dir investigations/model_understanding_prime_rl/results/prime_mu_investigation_only_dataset \
  --train-filename train.jsonl \
  --eval-filename eval.jsonl
```

## Reward/Judge

The judge prompt asks for two 1-5 scores:

- `specificity`
- `correctness`

The RL reward currently uses correctness only:

```text
correctness 1 -> reward 0.00
correctness 2 -> reward 0.25
correctness 3 -> reward 0.50
correctness 4 -> reward 0.75
correctness 5 -> reward 1.00
```

Raw judge `specificity` and `correctness` are also logged as zero-weight
PrimeRL metrics during training. They do not affect reward or advantage.

The main judge model used so far:

```text
claude-haiku-4-5-20251001
```

Two judge modes are implemented:

- `thinking_json`
  - Anthropic thinking enabled.
  - The judge returns JSON text.
  - Current 3-GPU run used `judge_thinking_budget = 1024` and
    `judge_max_tokens = 1400`.

- `forced_tool`
  - Uses a tool schema to force structured output.
  - Used as a faster/no-thinking ablation.
  - This was useful experimentally, but the preferred current mode is
    `thinking_json` because we want judge reasoning enabled.

The environment saves the exact judge system prompt and user message in every
transcript row.

## Transcript Contents

Each judged rollout is appended to JSONL under the run's `judge_transcripts/`
directory. Each record includes:

- Prompt messages as a list of chat messages.
- Exact rendered chat-template prompt fed to the rollout model.
- Local tokenizer prompt token IDs.
- vLLM-reported prompt token IDs.
- Boolean check that local prompt tokens match vLLM prompt tokens.
- vLLM-decoded prompt.
- Completion text and completion token IDs.
- Question text.
- Reference answer.
- Judge prompt.
- Judge scores and reasoning.

This is meant to support manual review without rerunning inference.

## Run Configs

### Recommended short run: 3-GPU thinking judge

```text
examples/model_understanding/rl_3gpu_thinking_100.toml
```

This is the current best smoke/trial configuration:

- 3 GPUs total.
- 2 independent one-GPU vLLM inference replicas.
- 1 trainer GPU.
- Haiku judge with thinking enabled.
- `batch_size = 32`.
- `rollouts_per_example = 16`.
- `max_inflight_rollouts = 80`.
- `temperature = 1.0`.
- `max_completion_tokens = 512`.
- LoRA rank 64, alpha 128, dropout 0.05.
- `max_model_len = 5000`.
- `gpu_memory_utilization = 0.90`.
- 100 RL steps.

Run a dry run:

```bash
cd /workspace-vast/adamk/prime-rl
.venv/bin/rl @ examples/model_understanding/rl_3gpu_thinking_100.toml --dry-run
```

Submit to Slurm:

```bash
cd /workspace-vast/adamk/prime-rl
.venv/bin/rl @ examples/model_understanding/rl_3gpu_thinking_100.toml
```

### 2-GPU forced-tool ablation

```text
examples/model_understanding/rl_2gpu_forced_tool_100.toml
```

This uses:

- 1 inference GPU.
- 1 trainer GPU.
- Forced-tool judge output.
- No judge thinking.
- 100 RL steps.

This completed successfully, but it is not the preferred current judge setup.

### Older/diagnostic configs

```text
examples/model_understanding/rl_2gpu.toml
examples/model_understanding/rl_2gpu_smoke.toml
examples/model_understanding/rl_2gpu_profile.toml
```

These are useful as references and for quick debugging, but note that some of
these still set a sampling seed. A fixed sampling seed caused repeated rollouts
for the same prompt to be nearly identical. Prefer the no-seed configs for real
training runs.

## Slurm Notes

The Slurm template is:

```text
cluster_templates/single_node_rl_shared.sbatch.j2
```

Important details:

- Requests partial nodes with `#SBATCH --gres=gpu:{{ gpus_per_node }}`.
- Uses `#SBATCH --qos=high`.
- Does not use `--exclusive`.
- Does not use `--get-user-env`.
- Does not set `CUDA_VISIBLE_DEVICES` globally.
- PrimeRL's launcher sets `CUDA_VISIBLE_DEVICES` only for child inference/trainer
  processes inside the Slurm allocation.
- Logs `nvidia-smi` and periodic GPU utilization.

Before submitting, check existing jobs/GPU usage:

```bash
squeue -u "$USER" -o "%.10i %.12j %.2t %.10M %.20R %.10b"
```

If a job fails at vLLM startup with a low free-memory error, assume a bad/dirty
node and resubmit excluding that node. Do not lower `gpu_memory_utilization` for
that failure mode.

## Trained Runs So Far

Important: the completed runs below used the earlier mixed dataset export,
which included synthetic counterfactual examples. They are useful systems tests
for the launcher, transcript logging, and judge throughput, but should not be
treated as the intended investigation-only RL runs.

### Superseded 2-GPU forced-tool run

```text
W&B name: mu-qwen3-8b-rl-2gpu-forced-tool-noseed-v2
W&B run:  https://wandb.ai/adam-karvonen/model-understanding-rl/runs/bcf6a85eeb634a6b8356d11dabf0b0d7
Slurm job: 1504030
Status: completed cleanly
Layout: 1 inference GPU, 1 trainer GPU
Judge mode: forced_tool
Output dir: outputs/model-understanding-rl-qwen3-8b-2gpu-forced-tool-noseed-v2
Transcript: outputs/model-understanding-rl-qwen3-8b-2gpu-forced-tool-noseed-v2/judge_transcripts/judge_transcripts_pid127394.jsonl
```

Transcript sanity checks passed:

- Full prompt messages are saved.
- Exact rendered prompts are saved.
- Local rendered prompt token IDs match vLLM prompt token IDs.
- Repeated prompt groups had 16 unique responses after removing the sampling
  seed.

### Superseded 3-GPU thinking-judge run

```text
W&B name: mu-qwen3-8b-rl-3gpu-thinking-split
W&B run:  https://wandb.ai/adam-karvonen/model-understanding-rl/runs/7ea94700f0b640f3b629a2f44a62778a
Slurm job: 1504393
Status: completed cleanly
Layout: 2 independent inference GPU replicas, 1 trainer GPU
Judge mode: thinking_json
Output dir: outputs/model-understanding-rl-qwen3-8b-3gpu-thinking-split
Final checkpoint: outputs/model-understanding-rl-qwen3-8b-3gpu-thinking-split/checkpoints/step_100
Transcript: outputs/model-understanding-rl-qwen3-8b-3gpu-thinking-split/judge_transcripts/judge_transcripts_pid2467211.jsonl
```

This run finished 100 RL steps in 33m39s. It wrote final checkpoints at:

```text
outputs/model-understanding-rl-qwen3-8b-3gpu-thinking-split/checkpoints/step_75
outputs/model-understanding-rl-qwen3-8b-3gpu-thinking-split/checkpoints/step_100
```

Online train reward was noisy and did not show a clear monotonic increase:

```text
steps 00-09: mean reward 0.3015
steps 10-19: mean reward 0.3437
steps 20-29: mean reward 0.3586
steps 30-39: mean reward 0.3313
steps 40-49: mean reward 0.3133
steps 50-59: mean reward 0.4641
steps 60-69: mean reward 0.2898
steps 70-79: mean reward 0.3047
steps 80-89: mean reward 0.3047
steps 90-99: mean reward 0.3414
overall:     mean reward 0.3353
```

Treat this as a systems/integration run, not evidence that RL improved the model.
A fixed held-out eval is needed for learning conclusions.

## Spot Checks

Generate a score-diverse Markdown spot check from a transcript:

```bash
cd /workspace-vast/adamk/prime-rl
.venv/bin/python examples/model_understanding/spot_check_rollouts.py \
  --transcript-path outputs/model-understanding-rl-qwen3-8b-3gpu-thinking-split/judge_transcripts/judge_transcripts_pid2467211.jsonl \
  --output-path outputs/model-understanding-rl-qwen3-8b-3gpu-thinking-split/spot_checks/rollouts_score_diverse_seed0_count8.md \
  --selection score-diverse \
  --count 8 \
  --seed 0
```

The generated artifact from the first 3-GPU run is:

```text
outputs/model-understanding-rl-qwen3-8b-3gpu-thinking-split/spot_checks/rollouts_score_diverse_seed0_count8.md
```

Other selection modes:

```bash
# First N records
.venv/bin/python examples/model_understanding/spot_check_rollouts.py \
  --transcript-path <transcript.jsonl> \
  --output-path <spot_check.md> \
  --selection first \
  --count 5

# Random N records with explicit seed
.venv/bin/python examples/model_understanding/spot_check_rollouts.py \
  --transcript-path <transcript.jsonl> \
  --output-path <spot_check.md> \
  --selection random \
  --count 8 \
  --seed 0

# Exact transcript indices
.venv/bin/python examples/model_understanding/spot_check_rollouts.py \
  --transcript-path <transcript.jsonl> \
  --output-path <spot_check.md> \
  --selection indices \
  --indices 0,17,258
```

The script intentionally prints full text, not abbreviated snippets.

## Launcher Patch

The only PrimeRL internal change is in:

```text
src/prime_rl/entrypoints/rl.py
```

The change is launch plumbing for local single-node LoRA runs with more than one
inference GPU. PrimeRL's default local path starts one inference service across
the inference GPU group. With LoRA enabled, that enters vLLM's data-parallel LoRA
path, which failed here.

The patch detects:

```text
config.inference.enable_lora == true
deployment.num_infer_gpus > inference.parallel.tp
```

and starts independent one-GPU inference replicas instead:

```text
inference_0 -> port 19183 -> one GPU
inference_1 -> port 19184 -> one GPU
trainer     -> remaining GPU
```

The orchestrator receives both base URLs and uses PrimeRL's static inference
pool round-robin behavior.

This patch does not change:

- Reward computation.
- Advantage computation.
- Trainer loss.
- Optimizer behavior.
- Checkpoint format.
- The RL algorithm.

For a simple 2-GPU run with 1 inference GPU and 1 trainer GPU, this launcher
patch is not needed.

## Current Read On Efficiency

In the 3-GPU thinking run, the system was mostly trainer-bound after startup.
Rollout generation and judging often completed quickly, and the orchestrator
waited for trainer checkpoints. That suggests adding more inference GPUs is not
obviously useful at this batch/sequence length unless trainer throughput changes.

The trainer GPU showed useful bursts but modest MFU, generally around the
mid-20% range after warmup. The inference GPUs were bursty and mostly idle
between rollout waves.

Possible next experiments:

- Run a fixed held-out eval on the SFT baseline and RL checkpoint.
- Try longer runs only after held-out eval plumbing is in place.
- Try higher trainer batch/effective batch if memory and PrimeRL config allow it.
- Compare `thinking_json` versus faster structured no-thinking judging on a fixed
  sample before choosing a long-run reward path.
- Consider a 2-GPU thinking run if the trainer remains the bottleneck.

## Operational Caveats

- Use the local `.venv`.
- Source `/workspace-vast/adamk/activation_oracles_dev/.env` or rely on the
  Slurm template sourcing it.
- Keep `gpu_memory_utilization = 0.90`.
- Keep `max_model_len` explicit.
- Do not set a rollout sampling seed for real RL runs with multiple rollouts per
  prompt.
- Do not commit `outputs/`, checkpoints, merged model weights, or transcripts.
- Treat local paths in these configs as cluster-local paths, not portable defaults.

### Output directory naming

**When changing hyperparameters, always change `output_dir` and the W&B `name`.**
If two runs share the same `output_dir`, `clean_output_dir = true` will
permanently delete the previous run's checkpoints, weights, and transcripts.
Encode key hyperparameters (at minimum the LR) in the output dir name.
