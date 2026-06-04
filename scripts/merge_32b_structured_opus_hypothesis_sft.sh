#!/bin/bash
#SBATCH --job-name=merge-mu-32b-structured-opus-hypothesis
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-01:00:00
#SBATCH --output=/workspace-vast/adamk/exp/logs/%x_%j.out

set -euo pipefail

export HF_HOME=/workspace-vast/$USER/hf_cache
export PYTHONUNBUFFERED=1
# Force CPU-only: PEFT's merge_and_unload otherwise tries CUDA matmul on LoRA
# weights even when the base model is on CPU. The 32B 8GB LoRA delta blew up
# the previous attempt when SLURM landed it on a node with a busy GPU.
export CUDA_VISIBLE_DEVICES=""

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

cd /workspace-vast/adamk/activation_oracles_dev

srun /workspace-vast/adamk/activation_oracles_dev/.venv/bin/python \
    investigations/model_understanding_prime_rl/lora_tools/merge_sft_adapter.py \
        --base-model Qwen/Qwen3-32B \
        --adapter-path /workspace-vast/adamk/activation_oracles_dev/checkpoints_text_sft/mu_qwen3_32b_structured_strict_opus_hypothesis_e1_kl0/final \
        --output-dir /workspace-vast/adamk/prime-rl/outputs/mu_qwen3_32b_structured_strict_opus_hypothesis_e1_kl0_merged

echo "MERGE COMPLETE"
