#!/bin/bash
# Check whether the vendored structured judge is in sync with the source.
#
# Exits 0 if synced, 2 if drifted. Drift means JUDGE_SYSTEM_PROMPT,
# JUDGE_USER_TEMPLATE, or JUDGE_TOOL differ between:
#
#   SOURCE:   /workspace-vast/adamk/activation_oracles_dev/nl_probes/open_ended_eval/model_understanding.py
#   VENDORED: /workspace-vast/adamk/prime-rl/environments/model_understanding_rl_structured/model_understanding_rl_structured/judge.py
#
# When drift is detected, manually copy the new sections from source to vendored
# (the parser/dataclasses/call mechanics are stable; only the prompt + tool spec
# typically drift). After editing, re-run this script to confirm.

set -euo pipefail
cd "$(dirname "$0")/.."
exec /workspace-vast/adamk/activation_oracles_dev/.venv/bin/python scripts/check_structured_judge_sync.py
