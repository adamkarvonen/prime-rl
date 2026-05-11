# May 7 RL Run Findings

Date: 2026-05-07

This note summarizes the May 7 investigation into the model-understanding RL
runs. It is based on local output directories, saved configs, Slurm accounting,
and trainer/orchestrator logs.

## High-Level Read

Three parallel efforts ran overnight on May 7:

1. **Codex** was trying to get LoRA warm-start working (loading an SFT adapter
   into fresh RL training). This produced 6 retry runs that all crashed within
   3-62 steps.
2. **Claude (mixed data)** was adding synthetic data to RL training via a
   multi-environment setup. Hit multi-env spawning issues, then zero-advantage
   filter crashes.
3. **Another Claude agent** was trying to launch a 32B run. Used the warm-start
   converter path unintentionally, producing an invalid run.

The core discovery: **`batch_size = 32` with `rollouts_per_example = 16` means
only 2 unique prompts per training batch** (32 total rollouts / 16 per prompt).
This was not the intended design — the expectation was 32 prompts with 16
rollouts each. With only 2 prompts, the zero-advantage filter is statistically
fragile: if both prompts produce uniform-reward rollouts in the same step, the
entire batch is filtered. Three consecutive all-filtered batches crashes the
run.

The May 5 successful think run hit 32/32 all-filtered **5 times** across 1000
steps but never 3 in a row — it survived by luck. The think configuration is
especially vulnerable because early in training, thinking is broken (the model
hasn't learned `<think>` tags), so `strip_thinking = true` assigns reward 0.0
to most rollouts, making uniform-reward batches very likely.

Additional issues found:

- Some successful May 5 runs were hard to find because their output directory
  names do not exactly match the W&B run names.
- Several May 7 runs were not comparable to the May 5 baselines because they
  used the step-0 LoRA warm-start converter path.

The successful May 5 think run does not appear to have been deleted. It is
present at:

```text
outputs/model-understanding-rl-qwen3-8b-3gpu-qwen-think-inv-only-v2
```

Its W&B run name is:

```text
mu-qwen3-8b-rl-3gpu-qwen-think-investigation-only
```

The similarly named directory below is basically empty and should not be treated
as the successful run:

```text
outputs/model-understanding-rl-qwen3-8b-3gpu-qwen-think-investigation-only
```

## Runs Checked

### May 5 Think Run

Output:

```text
outputs/model-understanding-rl-qwen3-8b-3gpu-qwen-think-inv-only-v2
```

Slurm:

```text
1511641, completed 2026-05-05, node-13, elapsed 05:18:38
```

W&B:

```text
https://wandb.ai/adam-karvonen/model-understanding-rl/runs/2d3d18040d7a47508fa4d965326ee06d
```

This run completed 1000 steps. It used the merged 8B SFT model, 3 GPUs, LoRA
rank 64/alpha 128/dropout 0.05, trainer LR 5e-6, investigation-only dataset,
`enable_thinking = true`, `strip_thinking = true`, and
`max_completion_tokens = 2048`.

The reward curve was sparse early and then recovered. Early examples:

```text
step 0: reward 0.0078
step 1: reward 0.1641
step 2: reward 0.0625
step 5: reward 0.0000
step 6: reward 0.0000
step 20: reward 0.0312
step 62: reward 0.3047
step 999: reward 0.3516
```

This matches the qualitative expectation that thinking was initially damaged and
then recovered, but it also shows the run was already operating close to a
sparse/all-zero-reward regime.

### May 5 No-Think Run

Output:

```text
outputs/mu-rl-no-think-lr5e-6
```

Slurm:

```text
1515164, completed 2026-05-05 to 2026-05-06, node-6, elapsed 04:27:42
```

W&B:

```text
https://wandb.ai/adam-karvonen/model-understanding-rl/runs/fa96192d9cb1455a99e30197f30d792b
```

This run also completed 1000 steps. It used the same merged 8B SFT model and
trainer LR 5e-6, but with `enable_thinking = false` and
`max_completion_tokens = 512`.

It had much less severe early sparsity than the think run:

```text
step 0: reward 0.2188
step 1: reward 0.2109
step 2: reward 0.2109
step 20: reward 0.3281
step 62: reward 0.3359
step 999: reward 0.5234
```

### May 7 8B LoRA Warm-Start Retry

Output:

```text
outputs/mu-rl-lora-warmstart-think-2gpu-lr5e-6-retry5
```

W&B:

```text
https://wandb.ai/adam-karvonen/model-understanding-rl/runs/d99cb16c66464e20b804c4c5f444e026
```

This run used the new converted-checkpoint LoRA warm-start path. It loaded the
adapter and initially had a reward curve with the same broad shape as the May 5
think run:

```text
step 0: reward 0.0234
step 1: reward 0.0156
step 2: reward 0.0078
step 15: reward 0.1094
step 16: reward 0.2031
step 24: reward 0.2969
step 31: reward 0.2188
step 43: reward 0.3906
```

It eventually failed at step 62:

```text
All 32 rollouts were filtered out on 3 consecutive attempts at step 62
```

This makes the run invalid as a clean continuation result. It does not prove
that the adapter failed to load; the early reward curve is compatible with the
adapter having loaded. The immediate failure mechanism was the zero-advantage
filter.

### May 7 Mixed Single-Env Run

Output:

```text
outputs/mu-rl-no-think-lr5e-6-mixed-single
```

This run failed almost immediately:

```text
step 0: reward 0.0000
step 1: reward 0.0000
step 2: All 32 rollouts were filtered out on 3 consecutive attempts
```

The retry with zero-advantage enforcement disabled was much healthier:

```text
outputs/mu-rl-no-think-lr5e-6-mixed-single-v2
```

That run was observed around step 480 with normal moving rewards. This is strong
evidence that the zero-advantage enforcement behavior is enough to kill sparse
reward runs that would otherwise keep training.

### May 7 32B Run

Output:

```text
outputs/mu-rl-32b-think-lr5e-6
```

This was not a clean "normal 32B RL" run. It was a resume-from-step-0 run using
the LoRA checkpoint converter.

Evidence:

```text
outputs/mu-rl-32b-think-lr5e-6/run_default/checkpoints/step_0/conversion_metadata.json
```

contains:

```json
{
  "format": "prime_rl_multi_run_lora_warm_start",
  "run_id": "run_default",
  "step": 0,
  "trainer_rank_count": 1
}
```

The saved config has `resume_step = 0`, and the trainer log says:

```text
Loading checkpoint from outputs/mu-rl-32b-think-lr5e-6/run_default/checkpoints/step_0/trainer
Resumed run run_default from step 0
```

The converted trainer checkpoint contains only:

```text
model
progress
```

It does not contain optimizer or scheduler state.

The trainer initially logs optimizer initialization with `lr=5e-06`, but the
training step logs then show `LR: 1.00e-04` for some updates and `LR: 0.00e+00`
for many others. This lines up suspiciously with the resolved per-run
`orchestrator.optim.lr = 0.0001` default and the trainer's "No runs are ready to
update" warnings.

This run should be treated as contaminated/invalid. It likely used the
warm-start converter partially correctly (adapter weights present), but the
resume/multi-run training state was incomplete or underspecified.

## Zero-Advantage Filter Findings

The zero-advantage filter was introduced/defaulted upstream in commit
`0660a9b3` on 2026-05-03. The default orchestrator filters are:

```text
gibberish: monitor only
repetition: monitor only
zero_advantage: enforce true
```

This is new enough that it plausibly explains why these crashes had not been
seen in older runs.

There is also a subtle accounting/enforcement issue: `apply_filters` uses
"first matching filter wins." Because gibberish and repetition are monitor-only
and come before zero-advantage, a rollout can be both gibberish and
zero-advantage but only counted as gibberish. That can produce confusing logs
like:

```text
Detected 32/32 rollouts (gibberish=1, zero_advantage=31), enforced 31
Only 1/32 rollouts in the batch are trainable
Reward: 0.0000
```

The monitor-only filter is effectively masking the enforcing filter for that
one rollout. This explains the apparently contradictory "reward zero but still
one or two trainable rollouts" behavior.

The May 5 think run survived despite early zero-reward batches. The May 7
warm-start and mixed runs did not. That looks like a stochastic/sparsity
threshold issue, not necessarily a single adapter-loading failure.

## What Looks Self-Contained

The converter code is opt-in. Normal fresh RL runs should not use it unless the
config has `resume_step = 0` and a converted `step_0` checkpoint exists.

The local tracked code changes with broad behavior impact are:

```text
src/prime_rl/orchestrator/envs.py
src/prime_rl/trainer/multi_ckpt.py
```

The `envs.py` change only extends env-server startup timeout. It should not
change reward computation or filtering.

The `multi_ckpt.py` changes are relevant to checkpoint load/save/resume paths.
They should be treated as potentially relevant for converted-checkpoint and
resume experiments, but they do not explain ordinary fresh RL reward collapse by
themselves.

## Current Working Hypotheses

1. The May 5 successful think run was not deleted. It lives under the `...v2`
   output directory.
2. The zero-advantage filter is the immediate cause of the May 7 8B run
   crashes.
3. The filter's first-match behavior makes the logs confusing and can let
   monitor-only detections mask an enforcing zero-advantage detection.
4. The 32B run accidentally/implicitly used the converter warm-start path when
   the intent may have been "just run 32B." That made it non-comparable to a
   clean run.
5. The 32B converter/resume path is incomplete or under-specified around
   optimizer/scheduler/per-run LR state. The observed `1e-4`/`0.0` LR behavior
   makes that run invalid as evidence.

## Suggested Next Steps

1. For reproducing the May 5 8B think curve, use a clean output directory, the
   merged SFT model, no `resume_step`, and no converted `step_0` checkpoint.
2. For sparse-reward experiments, decide explicitly whether
   `zero_advantage.enforce` should be enabled. Do not rely on the default.
3. Fix or at least review filter semantics so monitor-only filters cannot mask
   an enforcing zero-advantage filter.
4. Treat the converter/warm-start path as experimental until it handles the
   optimizer/scheduler/per-run LR state deliberately.
5. For any future warm-start run, record in the run note whether it is:
   fresh RL, merged-model RL, or resume-from-converted-adapter RL.

## Mixed Investigation + Synthetic Data Experiment

### Goal

Mix 50% investigation data and 50% synthetic data (from
`data_pipelines/model_understanding/runs/qwen3_8b_50k_train/synthetic_data.json`)
into RL training to see if the additional diversity helps reward compared to the
investigation-only baseline.

### Data

The synthetic dataset has 87,954 examples (22,393 behavior predictions + 65,561
counterfactual predictions), generated by Sonnet 4.5 from the same investigation
prompts. Each example has a conversation context, a question about model
behavior, and a reference answer.

A merged 50/50 dataset was created by sampling 22,011 synthetic examples to
match the 22,011 investigation examples, shuffled together into a single JSONL:

```text
investigations/model_understanding_prime_rl/results/prime_mu_mixed_50_50_dataset/train.jsonl
```

A synthetic-only export also exists at:

```text
investigations/model_understanding_prime_rl/results/prime_mu_synthetic_only_dataset/train.jsonl
```

Both were exported by:

```text
investigations/model_understanding_prime_rl/export_synthetic_only_dataset.py
```

### Multi-Environment Approach (Failed)

The initial plan was to use two `[[orchestrator.train.env]]` blocks with
`ratio = 0.5` each, so that per-environment metrics would be logged separately
in W&B. This is a supported prime-rl feature (see `configs/math_group/rl.toml`
for a working example).

The first multi-env run (job 1527436) trained successfully for 116 steps before
crashing due to a transient LoRA adapter 404 error during a weight checkpoint.
Both environments loaded and produced rewards.

All subsequent multi-env runs failed because the second environment's worker
process (`mu_synthetic`) silently died before creating its log file. The env
server would start and spawn the worker subprocess via
`mp.get_context("spawn").Process()`, but the worker never produced output.
The orchestrator would then timeout after 10 minutes waiting for the env server
to become healthy.

This happened on multiple nodes (node-3, node-6) and with fresh output
directories, ruling out stale state. The root cause appears to be in how
verifiers spawns multiple env worker subprocesses using the Python "spawn"
multiprocessing context. The first environment's worker starts fine; the second
never does. This needs further investigation but was not pursued further here.

Config used:

```text
examples/model_understanding/rl_3gpu_mixed_inv_synth.toml
```

### Single-Environment Approach

Switched to merging both datasets into a single JSONL and using one environment.
The trade-off: no per-environment W&B metrics, but the `source_kind` field in
each example allows post-hoc splitting from transcripts.

### Zero-Advantage Filter Crash

The single-env mixed run crashed at step 2 with all 32 rollouts filtered by the
zero_advantage filter on 3 consecutive attempts. This happened despite the
transcripts showing varied rewards across the full set of rollouts (mean 0.34,
spread across 0.0-1.0).

With `batch_size = 32` and `rollouts_per_example = 16`, there are only **2
prompts per training batch**. If both prompts happen to produce uniform-reward
rollouts (all 16 get the same judge score), the entire batch is filtered. This
is not rare:

- The May 5 no-think run (1000 steps, investigation-only): 174
  zero_advantage events, but **never** hit 32/32 all-filtered. Stable.
- The May 5 think run (1000 steps, investigation-only): 189
  zero_advantage events, hit 32/32 all-filtered **5 times** but never 3
  consecutive. Lucky.
- The May 7 mixed no-think run: hit 32/32 all-filtered on steps 0, 1, and 2
  consecutively. Crashed.

The filter is statistically incompatible with only 2 prompts per batch. Whether
a run survives depends on luck in the first few steps.

### Current Status

A run with `zero_advantage enforce = false` (monitor only) was started and
reached step 590 before being cancelled to investigate other issues found in
parallel. The training metrics looked healthy (rewards in the 0.2-0.5 range,
stable grad norms).

Config used:

```text
examples/model_understanding/rl_3gpu_mixed_single_env.toml
```

Output:

```text
outputs/mu-rl-no-think-lr5e-6-mixed-single-v2
```

### Judge Rubric

The same judge prompt and rubric (1-5 specificity + correctness) was used for
both investigation and synthetic data. The synthetic data includes both behavior
predictions ("what would you do?") and counterfactual predictions ("how would
your response change if X?"). The rubric focuses on "correct core causal
factor" which applies to both types, since counterfactual questions are still
fundamentally about understanding what drives behavior.

## Actions Taken

### Batch size fix

Launched a new run with `batch_size = 256` (16 prompts per batch instead of 2),
`max_steps = 125` (same total training tokens as the original 1000-step bs=32
run), and `ckpt.interval = 25`. All filters set to `enforce = false`.

Added checkpoint space-saving settings (these go under `[trainer.ckpt]`, NOT
the top-level `[ckpt]` — the top-level one is for the orchestrator):

- `trainer.ckpt.skip_gather_master_weights = true` — skips merging LoRA into
  full model weights at each checkpoint. Without this, every checkpoint saves
  the full merged model (~16GB per checkpoint, ~1TB across 10 checkpoints with
  the original ckpt.interval=100 over 1000 steps). The LoRA adapters are still
  saved separately via the orchestrator broadcast path
  (`run_default/broadcasts/step_X/`), which is all we need.
- `trainer.ckpt.weights_only = true` — skips saving optimizer/scheduler state
  (~17GB `.distcp` files per checkpoint). Not needed since we don't plan to
  resume training from checkpoints.

The trainer handles large batch sizes via gradient accumulation: rollouts are
packed into micro-batches (multiple rollouts per micro-batch up to `seq_len`
tokens), processed sequentially with gradient accumulation, then one optimizer
step. No OOM risk from increasing batch size.

Config:

```text
examples/model_understanding/rl_3gpu_nothink_bs256.toml
```

### Canonical config naming

Created clearly-named copies of the two configs that produced successful
1000-step May 5 runs:

- `rl_3gpu_judge_think_qwen_nothink_canonical.toml` — Qwen generates without
  thinking, LLM judge uses thinking mode. Produced `mu-rl-no-think-lr5e-6`.
- `rl_3gpu_judge_think_qwen_think_canonical.toml` — Qwen generates with
  thinking (`strip_thinking = true`), LLM judge uses thinking mode. Produced
  `mu-qwen3-8b-rl-3gpu-qwen-think-investigation-only`.

The old names (`rl_3gpu_thinking_100.toml` for the no-think run) were confusing
because "thinking" referred to the judge mode, not whether Qwen itself thinks.

### Judge mode latency/stability check

Tested Haiku judge modes on a fixed 300-rollout sample from
`outputs/mu-rl-nothink-bs256`:

- `thinking_json`: Anthropic thinking enabled, JSON text output,
  `judge_thinking_budget = 1024`, `judge_max_tokens = 1400`.
- `forced_tool`: no Anthropic thinking, forced tool schema with required
  `reasoning`, `specificity`, and `correctness` fields.

The fixed-sample repeat used two independent runs of each judge mode and
asserted that paired comparisons used the exact same rollout response text.

Results:

- `forced_tool` latency was about 3.4s mean versus about 10.6s for
  `thinking_json`, roughly a 3x speedup.
- `forced_tool` run-to-run stability was strong: exact match 77%, within-1
  100%, Spearman about 0.81.
- `thinking_json` run-to-run stability was lower: exact match 73%, within-1
  99%, Spearman about 0.70.
- Cross-mode agreement was close to thinking's own repeat noise: exact match
  62-63%, within-1 96-97%, Spearman about 0.67-0.70.
- `forced_tool` scored slightly lower on average (about 0.06-0.08 correctness
  points), so it should be treated as a slightly different reward definition.

Recommendation: use `judge_mode = "forced_tool"` for online RL rewards, keep
the required `reasoning` field, and use `judge_max_tokens = 500` rather than
300 to avoid annoying truncation edge cases. Keep `thinking_json` for heldout
evals, audits, and spot checks. Do not switch judge modes halfway through an
existing run; start a fresh run if changing the reward definition.

## TODO

### 1. Redo baseline run with correct batch size

The original 1000-step runs used `batch_size = 32` with
`rollouts_per_example = 16`, producing only 2 prompts per batch. The intended
design was 32 prompts per batch, which means either:

- Increase `batch_size` to 512 (32 prompts x 16 rollouts), or
- Keep `batch_size = 32` and reduce `rollouts_per_example` (e.g., 4 or 8)

A 16x increase in effective batch size means 16x more judge API calls per step,
so each step takes much longer. Run fewer steps (e.g., 100 instead of 1000) to
keep total compute manageable.

### 2. Synthetic data as separate environment

Get multi-environment training working properly so investigation and synthetic
data are logged as separate environments in W&B. This is a supported prime-rl
feature (see `configs/math_group/rl.toml`) and worked for 116 steps on the
first attempt — the second environment worker subprocess dying on subsequent
runs is likely a bug in how we're configuring it, not a fundamental limitation.

Look at prime-rl documentation and working multi-env examples to figure out
what's different.

### 3. LoRA warm-start

Determine whether the LoRA warm-start path (loading an SFT adapter into fresh
RL) is actually working or broken. The overnight retry runs crashed from the
zero-advantage filter, not necessarily from adapter loading failures — the
early reward curves were consistent with a working adapter. Need to separate
the warm-start question from the filter question by running warm-start with
`zero_advantage enforce = false`.

### 4. 32B run

Get a basic 32B RL run working to verify the GPU setup. The overnight 32B run
accidentally used the warm-start converter path and had LR issues
(`1e-4`/`0.0` alternating). Start with a clean config using the same pattern as
the working 8B canonical configs — merged SFT model, no `resume_step`, no
converted checkpoint.

### 5. Efficiency tuning

Key parameters to investigate:

- **`max_off_policy_steps`**: Currently 3. Higher values reuse rollouts more,
  reducing judge API calls but potentially degrading training signal quality.
- **Train vs inference GPU balance**: Currently 1 train + 2 infer GPUs. With
  larger effective batch sizes (more prompts), inference becomes the bottleneck.
  May need to reconsider the split.
- **Judge concurrency**: Currently 80. With 16x more rollouts per step, judge
  API throughput becomes more important.
- **Judge mode**: For online rewards, prefer `judge_mode = "forced_tool"` with
  `judge_max_tokens = 500`. This keeps structured no-thinking judging fast while
  preserving a required rationale field for debugging.

### 6. Reduce startup time

Observed startup breakdown for a 3-GPU run on a fresh node:

- ~5 min: `uv run` dependency resolution on the compute node
- ~3 min: sbatch script overhead (import checks, nvidia-smi, env sourcing)
- ~7 min: vLLM model loading + LoRA kernel setup
- ~6 min: CUDA graph capture (2 inference replicas x ~3 min each)

Total: ~20-23 min before the first training step.

Possible improvements:

- `enforce_eager = true` skips CUDA graph capture (~6 min saved). Trade-off is
  slightly lower inference throughput, which may not matter when judge API calls
  dominate step time.
- Pre-resolve uv dependencies on the shared filesystem to avoid re-resolving on
  each compute node (~5 min saved).
- CUDA graph caching: subsequent runs on the same node start faster since
  graphs are cached.

### 7. Naming convention

Future runs should include the model name (e.g., `qwen3-8b`, `qwen3-32b`) in
both the W&B run name and the output directory name. Several May 5-7 runs were
hard to identify because the names didn't indicate which model was used.
