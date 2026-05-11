# Qwen3-32B RL Performance Analysis

Date: 2026-05-07

This note summarizes the train/inference balance observed for the Qwen3-32B
model-understanding RL runs using the no-thinking, batch-size-256 forced-tool
configuration.

## Runs Compared

- 4 GPU run: `outputs/mu-rl-32b-nothink-bs256-ft`
  - 2 inference GPUs
  - 2 trainer GPUs
- 6 GPU run: `outputs/mu-rl-32b-nothink-bs256-ft-6gpu`
  - 2 inference GPUs
  - 4 trainer GPUs

Both runs used 256 rollouts per orchestrator step, 16 rollouts per example,
`max_async_level = 1`, and Qwen3-32B LoRA training.

## Main Finding

The 6 GPU run is materially faster wall-clock, but not cheaper in GPU-hours.
Using steps 3 through 20 for both runs, excluding startup and warmup:

| Metric | 4 GPU | 6 GPU | Takeaway |
| --- | ---: | ---: | --- |
| Trainer mean step time | 109.80s | 79.05s | 6 GPU is 28.0% faster |
| Trainer GPU-sec / step | 439.18 | 474.31 | 6 GPU costs 8.0% more |
| Orchestrator mean step time | 116.65s | 84.72s | 6 GPU is 27.4% faster |
| Orchestrator GPU-sec / step | 466.60 | 508.34 | 6 GPU costs 8.9% more |

So the current read is:

- Use 6 GPUs when wall-clock time matters.
- Use 4 GPUs when GPU-hour efficiency matters.
- The 6 GPU run is not wasteful, but the added trainer GPUs do not quite pay for
  themselves in throughput.

## Why The 6 GPU Run Fluctuates

The 6 GPU run shows a visible sawtooth pattern. Trainer steps alternate between
roughly 60-80 seconds and roughly 109-116 seconds, especially early in the run.
This does not look like random GPU slowdown. The logged trainer throughput stays
in the same rough band, while the slow steps process substantially more logged
tokens:

- Fast trainer steps: roughly 300k-400k logged tokens.
- Slow trainer steps: roughly 580k-635k logged tokens.

This suggests the slow steps are larger packed training batches rather than
inefficient GPU execution.

The orchestrator has a related pattern:

- Off-policy level 0 steps are usually much faster.
- Off-policy level 1 steps are much slower.
- Slow steps often include a wait for a trainer checkpoint plus slower generation
  after loading the updated LoRA adapter.

For steps 3 through 20 in the 6 GPU run:

| Orchestrator subset | Mean time |
| --- | ---: |
| Off-policy level 0 | 66.98s |
| Off-policy level 1 | 134.30s |

This is consistent with `max_async_level = 1`: the orchestrator gets ahead,
waits at the checkpoint boundary, generates with a newer adapter, then repeats.

## Train Versus Inference Balance

In the 4 GPU run, the trainer was clearly the bottleneck. Rollout generation
after warmup took about 32-43 seconds, while trainer steps were around 100-130
seconds. The inference GPUs spent a lot of time idle.

In the 6 GPU run, the trainer is faster, and the pipeline is closer to balanced,
but not perfectly so. Some rollout steps still wait for trainer checkpoints, and
some inference steps are slower when a newer adapter/checkpoint is involved.

The practical interpretation:

- 2 inference + 2 trainer GPUs is GPU-hour efficient but trainer-bound.
- 2 inference + 4 trainer GPUs improves wall-clock throughput.
- The 6 GPU run has roughly balanced enough resources to justify using it for
  urgent runs, but not enough scaling efficiency to be the default cheapest
  configuration.

## Current Recommendation

For overnight or high-priority Qwen3-32B model-understanding RL runs, 6 GPUs is a
reasonable choice if finishing sooner is worth about 8-9% more GPU-hours per
step.

For routine sweeps or cost-sensitive runs, the 4 GPU configuration remains the
better default.
