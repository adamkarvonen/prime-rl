# Model Understanding RL

This directory contains the PrimeRL setup for RL fine-tuning model-understanding
explanations. The task is: given a prior conversation and a question about the
model's behavior, train the rollout model to produce an explanation that matches
a reference answer. Rewards come from an Anthropic LLM judge.

The older configs target a Qwen3-8B SFT checkpoint merged from:

```text
checkpoints_text_sft/mu_qwen3_8b_50k_s7_synth_e1_kl1
```

The LoRA warm-start config below instead uses `Qwen/Qwen3-8B` plus the saved
SFT adapter directly. The merged model used by the older configs is local to
this machine:

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
  - `judge_thinking_budget = 1024`, `judge_max_tokens = 1400`.
  - ~10.6s mean latency per call.
  - Better for offline evals and spot checks where judge reasoning is useful.

- `forced_tool` (recommended for online RL)
  - Uses a forced tool schema with required `reasoning`, `specificity`, and
    `correctness` fields.
  - No Anthropic thinking.
  - ~3.4s mean latency per call (~3x faster).
  - `judge_max_tokens = 600`.
  - Slightly lower correctness scores on average (~0.06-0.08 points), so treat
    as a slightly different reward definition. Do not switch judge modes mid-run.

The judge prompt also includes a hypothesis cap: if the response proposes more
than 6 distinct causal explanations, correctness is capped at 2. This prevents
reward hacking via shotgun-style responses that list many hypotheses.

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

### Canonical: 4-GPU no-think, forced-tool judge, bs256

```text
examples/model_understanding/rl_4gpu_nothink_bs256_forced_tool_canonical.toml
```

This is the current recommended configuration:

- 4 GPUs total: 2 inference + 2 training.
- `batch_size = 256`, `rollouts_per_example = 16` (16 prompts per batch).
- Haiku judge with `forced_tool` mode (no thinking, ~3x faster than
  `thinking_json`).
- `judge_max_tokens = 600`.
- `max_completion_tokens = 512`, `enable_thinking = false`.
- LoRA rank 64, alpha 128, dropout 0.05, LR 5e-6.
- `max_model_len = 5000`, `gpu_memory_utilization = 0.90`.
- Checkpoint every 100 steps.
- `skip_gather_master_weights = true`, `weights_only = false`. This saves
  distributed trainer checkpoints (LoRA weights + optimizer state + scheduler +
  progress) for resumability, but skips gathering and saving the full merged
  model weights, which would be tens of GB per checkpoint. The LoRA + optimizer
  checkpoint is ~3 GB for 32B and ~1 GB for 8B. To export HF-compatible
  weights for inference, merge the LoRA adapter after training (see "Combining
  SFT + RL LoRAs" below). **Do not set both `skip_gather_master_weights = true`
  and `weights_only = true`** — that combination silently disables all trainer
  checkpoint saves, leaving only orchestrator progress (no model weights or
  optimizer state).
- All rollout filters set to `enforce = false` (monitor only).
- Judge prompt includes a hypothesis cap: responses with more than 6 distinct
  causal hypotheses are capped at correctness 2.
- 1000 RL steps.

```bash
cd /workspace-vast/adamk/prime-rl
source /workspace-vast/adamk/activation_oracles_dev/.env
.venv/bin/rl @ examples/model_understanding/rl_4gpu_nothink_bs256_forced_tool_canonical.toml
```

### Canonical: 3-GPU think, thinking judge

```text
examples/model_understanding/rl_3gpu_judge_think_qwen_think_canonical.toml
```

This is the thinking variant from the May 5 successful 1000-step run:

- 3 GPUs total: 2 inference + 1 training.
- `batch_size = 32`, `rollouts_per_example = 16` (2 prompts per batch).
- `enable_thinking = true`, `strip_thinking = true`.
- `thinking_json` judge mode.
- `max_completion_tokens = 2048`.

Note: the small batch size (2 prompts) makes this config fragile with the
`zero_advantage` filter enforced. The May 5 run survived by luck (hit 32/32
all-filtered 5 times but never 3 consecutive). Consider increasing batch_size
or disabling zero_advantage enforcement for new runs.

### Archived configs

Older and experimental configs are in `examples/model_understanding/archive/`.
These include 2-GPU, profiling, LoRA warm-start, opus batch judge, and mixed
data experiment configs.

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

**Always use a 3-day time limit** (`time = "3-00:00:00"`). RL runs should never
die because of a Slurm timeout — the job should run until `max_steps` completes
or a real error occurs. A 12-hour limit is not enough for 1000-step runs,
especially with 32B models or large batch sizes.

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

## Combining SFT + RL LoRAs

Prime RL requires the warm-start model to be a normal HF model path, so the SFT
LoRA was merged into the base before RL training. The result is two stacked
adapters: the SFT LoRA inside the merged model, and the RL LoRA on top. To serve
the final model with the original `Qwen/Qwen3-8B` base via vLLM, the two LoRAs
need to be concatenated into a single adapter.

### Scripts

The combination and verification scripts live in the activation-oracles repo:

```text
/workspace-vast/adamk/activation_oracles_dev/investigations/model_understanding_prime_rl/lora_tools/combine_lora_adapters.py
/workspace-vast/adamk/activation_oracles_dev/investigations/model_understanding_prime_rl/lora_tools/verify_combined_lora.py
```

Run them with the activation-oracles `.venv`. They have no Slurm dependencies
and run in a few seconds (combine) to a few minutes (verify, since it loads
model shards).

### Combine step

For each target module, the combine script concatenates the rank-64 SFT LoRA
with the rank-64 RL LoRA into a single rank-128 LoRA. The math is exact: A
matrices stack along dim 0, B matrices stack along dim 1, and the combined
adapter uses `alpha=256, r=128` to preserve the original `alpha/r=2.0` scaling.

```bash
cd /workspace-vast/adamk/activation_oracles_dev
.venv/bin/python investigations/model_understanding_prime_rl/lora_tools/combine_lora_adapters.py \
  --sft-adapter checkpoints_text_sft/mu_qwen3_8b_50k_s7_synth_e1_kl1/final \
  --rl-adapter /workspace-vast/adamk/prime-rl/outputs/<RL_RUN_DIR>/run_default/broadcasts/step_<N> \
  --output-dir checkpoints_text_sft/<OUTPUT_NAME>_combined \
  --base-model Qwen/Qwen3-8B \
  --from-sft-weights
```

Current runs use `skip_gather_master_weights = true`, so full merged weights are
never written to disk. The RL LoRA is instead found under
`run_default/broadcasts/step_<N>/` (not `weights/step_<N>/lora_adapters/`). The
combine script handles both key-prefix formats automatically.

`--from-sft-weights` uses the raw stored SFT LoRA weights (fast, simple). Omit
the flag and pass `--merged-model-dir <path>` to instead extract the SFT delta
from the actual merged model via SVD. Both approaches produce adapters of
equivalent accuracy in practice.

The output directory contains:

- `adapter_model.safetensors` — combined LoRA weights (bfloat16, ~667MB)
- `adapter_config.json` — `r=128, alpha=256, base_model_name_or_path=Qwen/Qwen3-8B`
- Tokenizer files copied from the SFT adapter
- `combine_metadata.json` — pointers to source adapters and config

### Verification

There are two verification levels.

**Weight-level (CPU, definitive)**: directly compares the effective merged
weights of both setups for every target module. Differences come only from
bfloat16 storage of LoRA factors. Expected max abs diff: ~2.5e-3, max rel diff:
~3e-3.

```bash
cd /workspace-vast/adamk/activation_oracles_dev
.venv/bin/python investigations/model_understanding_prime_rl/lora_tools/verify_combined_lora.py \
  --base-model-dir <HF_CACHE_PATH_TO_Qwen3-8B> \
  --merged-model-dir /workspace-vast/adamk/prime-rl/outputs/mu_qwen3_8b_50k_s7_synth_e1_kl1_merged \
  --rl-adapter /workspace-vast/adamk/prime-rl/outputs/<RL_RUN_DIR>/weights/step_<N>/lora_adapters \
  --combined-adapter checkpoints_text_sft/<OUTPUT_NAME>_combined \
  --output checkpoints_text_sft/<OUTPUT_NAME>_combined/verify_results.json
```

**Forward-pass logprob diffs (Slurm GPU, qualitative)**: optional. Loads both
configs in vLLM and compares per-token prompt logprobs across a sample of
training prompts. Submit via Slurm:

```bash
sbatch <PATH_TO>/kl_check.sbatch  # template at outputs/<combined>/kl_check.sbatch
```

The kl-check sbatch template runs:

```text
investigations/model_understanding_prime_rl/kl_check_combined_lora.py
```

with `--n-prompts 100` against the investigation-only training set.

### Expected accuracy

On 100 training prompts (~25k token positions):

```text
Per-token logprob |Δ|:  median 0.018, mean 0.056, p95 0.18, p99 0.37
                        max 25 (a few extreme single-token outliers per run)
Per-prompt mean Δ:      median 0.05, max 0.17
```

This is larger than pure bf16 noise because the combined LoRA applies the SFT
delta as a factored low-rank computation (`B @ (A @ x)`) instead of a
pre-merged full-rank matmul. The 0.3% per-module weight diff compounds across
36 residual layers. For generation it's qualitatively fine; for exact logprob
evaluation expect noticeable shifts on a small fraction of tokens.

### Using with vLLM

Point vLLM at `Qwen/Qwen3-8B` as the base, enable LoRA with `max_lora_rank=128`,
and load the combined adapter as the LoRA request:

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="Qwen/Qwen3-8B", enable_lora=True, max_lora_rank=128, dtype="bfloat16")
adapter = LoRARequest("mu", 1, "/path/to/<OUTPUT_NAME>_combined")
out = llm.generate(prompts, SamplingParams(...), lora_request=adapter)
```

### Currently exported adapters

```text
/workspace-vast/adamk/activation_oracles_dev/checkpoints_text_sft/mu_qwen3_8b_sft_rl_think_5e6_step1000_combined
/workspace-vast/adamk/activation_oracles_dev/checkpoints_text_sft/mu_qwen3_8b_sft_rl_no_think_5e6_step1000_combined
```

These are the combined SFT + RL LoRAs from the 5e-6 runs at the final step
(1000), one for the thinking-judge variant and one for the no-think variant
(`outputs/mu-rl-no-think-lr5e-6`). Replace `<RL_RUN_DIR>` and `<N>` in the
commands above to combine other RL checkpoints.

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

With `batch_size = 256` and `forced_tool` judge mode on 4 GPUs (2 train + 2
infer), the orchestrator and trainer are roughly balanced at ~50-90s per step.
The judge API calls (~3.4s each, concurrency 80) and vLLM rollout generation
share the orchestrator step time. The trainer processes ~128 micro-batches per
step via gradient accumulation.

Key batch size finding (May 7): `batch_size` in prime-rl is the total number
of rollouts per step, not the number of prompts. With `rollouts_per_example =
16`, the number of unique prompts per batch is `batch_size / 16`. The original
bs=32 configs had only 2 prompts per batch, which made the zero-advantage
filter statistically fragile. The canonical bs=256 config has 16 prompts per
batch.

Startup takes ~25-30 min on a fresh node (uv resolution, model loading, CUDA
graph capture). See `may_7_findings.markdown` for a detailed breakdown and
optimization ideas.

## Operational Caveats

- Use the local `.venv`.
- **Source `.env` before running the `rl` CLI locally.** The `rl` command
  pre-downloads the trainer model before submitting to Slurm. Without
  `HF_HOME=/workspace-vast/pretrained_ckpts` (set in the `.env`), it defaults to
  `~/.cache/huggingface/`, which is on `/home/` — not accessible from Slurm nodes
  and not where the cluster's shared model cache lives. Either source the
  activation-oracles `.env` or export `HF_HOME` manually:
  ```bash
  source /workspace-vast/adamk/activation_oracles_dev/.env
  # or: export HF_HOME=/workspace-vast/pretrained_ckpts
  ```
  The Slurm template already sources this `.env`, so jobs run correctly once
  submitted — the issue is only with the local `rl` CLI pre-download step.
- **Submission: prefer `uv run rl` (one command).** Both `uv run rl` and the
  `--dry-run` + `sbatch` two-step path spend ~9 minutes in `uv` dependency
  resolution, which dominates submission time. The dry-run only skips
  `pre_download_model()`, which is fast for local model paths. Use the
  dry-run path when you need to edit the generated sbatch before submitting,
  or for rapid resubmission of the same config (just `sbatch outputs/.../rl.sbatch`
  again, skipping `uv` resolve entirely).
  ```bash
  # Default (one command):
  source /workspace-vast/adamk/activation_oracles_dev/.env
  uv run rl @ examples/model_understanding/<config>.toml

  # Fast resubmit (skip uv resolve):
  sbatch --qos=high outputs/<output_dir>/rl.sbatch
  ```
  **Caveat:** `--dry-run` is not read-only. The launcher still validates the
  output directory, writes resolved configs and `rl.sbatch`, and cleans stale
  rollout/broadcast step directories for fresh runs. Use it for fresh output
  directories, not against an active or valuable existing run directory.
- **Always set `skip_model_check = true`** in `[orchestrator.client]`. Without
  this, the orchestrator queries `/v1/models` immediately after the vLLM API
  server starts, but there's a race condition where the model isn't registered
  yet. This has killed multiple runs. The check has low value for local model
  paths — if the path is wrong, the first rollout request will fail with a
  clear error anyway. See `may_7_findings.markdown` for details.
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
