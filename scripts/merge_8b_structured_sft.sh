#!/bin/bash
#SBATCH --job-name=merge-mu-8b-structured
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-00:30:00
#SBATCH --output=/workspace-vast/adamk/exp/logs/%x_%j.out

set -euo pipefail

export HF_HOME=/workspace-vast/$USER/hf_cache
export PYTHONUNBUFFERED=1

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

cd /workspace-vast/adamk/activation_oracles_dev

srun /workspace-vast/adamk/activation_oracles_dev/.venv/bin/python \
    investigations/model_understanding_prime_rl/lora_tools/merge_sft_adapter.py \
        --base-model Qwen/Qwen3-8B \
        --adapter-path /workspace-vast/adamk/activation_oracles_dev/checkpoints_text_sft/mu_qwen3_8b_50k_structured_synth_e1_kl1/final \
        --output-dir /workspace-vast/adamk/prime-rl/outputs/mu_qwen3_8b_50k_structured_synth_e1_kl1_merged

echo "MERGE COMPLETE"
