# Adam's prime-rl cheat sheet

Personal launch reference for this fork. Upstream README is at [README.md](README.md).
Deeper model-understanding details at [examples/model_understanding/README.md](examples/model_understanding/README.md).

## Launching an RL run

```bash
cd /workspace-vast/adamk/prime-rl
source /workspace-vast/adamk/activation_oracles_dev/.env

# Validate config + generate sbatch (does NOT submit)
.venv/bin/rl @ <path/to/config.toml> --dry-run

# Submit the generated sbatch (path printed by the dry-run)
sbatch <output_dir>/rl.sbatch
```

Notes:
- The `@` MUST be a separate arg, not glued to the path. `@<path>` fails with "unrecognized options".
- `clean_output_dir = true` in the TOML wipes `output_dir` on each launch. Don't point two configs at the same `output_dir`.
- `--dry-run` writes `<output_dir>/rl.sbatch` and `<output_dir>/configs/rl.toml` so you can inspect both before submitting.

## Job management (auto-mode safe)

Always set a unique `[slurm] job_name` per TOML. Then:

```bash
squeue -u $USER -h -o "%i %j %T %M %R"           # what's running
scancel -u $USER --name=<job_name>               # cancel by name (preferred)
scancel --me                                     # nuke all my jobs
```

Don't cancel by bare job ID — auto-mode permission checker can't verify ownership from the command text. Cancel by `--name` or `--me`.

## Active configs (model understanding RL)

| Config | Purpose |
|---|---|
| [examples/model_understanding/rl_4gpu_32b_structured_anti_hack_v3.toml](examples/model_understanding/rl_4gpu_32b_structured_anti_hack_v3.toml) | 32B with v3 reward (current) |
| [examples/model_understanding/rl_4gpu_32b_structured_anti_hack_v2.toml](examples/model_understanding/rl_4gpu_32b_structured_anti_hack_v2.toml) | 32B with v2 reward (kept for A/B) |
| [examples/model_understanding/rl_4gpu_8b_structured_anti_hack_v2.toml](examples/model_understanding/rl_4gpu_8b_structured_anti_hack_v2.toml) | 8B parallel run |

## Reward reductions

All structured reductions live in [environments/model_understanding_rl_structured/model_understanding_rl_structured/reductions.py](environments/model_understanding_rl_structured/model_understanding_rl_structured/reductions.py). Set `reward_reduction = "<name>"` in the env args in the TOML. Current options:

- `core_anti_hack_v3` — symmetric LOOKUP boost, tiered overshoot, free refuted guess, P80/P95 length thresholds
- `core_anti_hack_v2` — flat α, P50/P99 length thresholds (refuted-collapsed in practice)
- `core_recall_strict` / `core_recall_lenient` / etc — v1 (kept for comparison only)

v2 and v3 lookup sums and penalties are BOTH logged on every run as observability metrics (`*_v3` suffix), so you can A/B from a single run's W&B.

## Single base_url auto-expansion (don't break this)

`[orchestrator.client] base_url = ["http://localhost:19293/v1"]` (ONE entry). prime-rl auto-expands this into one URL per inference replica via `configure_lora_inference_replicas` in [src/prime_rl/entrypoints/rl.py](src/prime_rl/entrypoints/rl.py) (~line 118-153). Listing both ports manually triggers `AssertionError: LoRA inference replica launcher expects one base URL before expansion`.

## Judge-prompt sync check

The structured judge constants are vendored into the env package. Run this before launching a new RL job to confirm they match the source-of-truth in `activation_oracles_dev`:

```bash
.venv/bin/python scripts/check_structured_judge_sync.py
```

Exits non-zero on drift. Re-vendor by hand-editing [environments/model_understanding_rl_structured/model_understanding_rl_structured/judge.py](environments/model_understanding_rl_structured/model_understanding_rl_structured/judge.py) to match the source.

## Common log paths

For a run with `output_dir = "outputs/<name>"`:
- Orchestrator (rollouts, step pacing): `outputs/<name>/logs/orchestrator.log`
- Trainer (loss, grad norms): `outputs/<name>/logs/trainer.log`
- Inference (vLLM): `outputs/<name>/logs/inference_*.log`
- Per-rollout transcripts: `transcripts/<run_subdir>/judge_transcripts_pid*.jsonl`
- W&B: `outputs/<name>/wandb/latest-run/` (live `output_trainer.log` for trainer output)

## Merging a LoRA adapter to a full HF checkpoint

Use [scripts/merge_32b_structured_sft.sh](scripts/merge_32b_structured_sft.sh) (or 8B variant) as a template.

**Critical**: `export CUDA_VISIBLE_DEVICES=""` in the merge sbatch — otherwise PEFT `merge_and_unload` tries CUDA matmul on LoRA weights even with the base on CPU, and OOMs on shared GPUs.
