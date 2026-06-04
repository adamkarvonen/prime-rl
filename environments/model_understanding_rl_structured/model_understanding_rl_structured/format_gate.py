"""Format gate — programmatic structural checks on the parsed model report.

Any check failure → reward=0, skip the (expensive) judge call. The gate is
intentionally strict; the model is expected to learn the four-section format
from SFT initialization.

See investigations/model_understanding_prime_rl/REWARD_SIGNAL.md §1.
"""

from __future__ import annotations

from .judge import ParsedReport

MAX_CORE_CAUSES = 3
MAX_REFUTED_HYPOTHESES = 3
MAX_TOTAL_BULLETS = 8

FORMAT_FAIL_REASONS = (
    "missing_observed",
    "missing_final",
    "too_few_core",
    "too_many_core",
    "too_many_refuted",
    "too_many_bullets",
    "core_no_evidence",
    "refuted_no_evidence",
)


def check_format(report: ParsedReport) -> tuple[bool, str | None]:
    """Return `(passed, fail_reason)`. fail_reason is None when passed.

    fail_reason values are drawn from FORMAT_FAIL_REASONS so the env can log
    a one-hot per-rollout breakdown.
    """
    if not report.observed_behavior.strip():
        return False, "missing_observed"
    if not report.final_answer.strip():
        return False, "missing_final"
    if len(report.core_causes) < 1:
        return False, "too_few_core"
    if len(report.core_causes) > MAX_CORE_CAUSES:
        return False, "too_many_core"
    if len(report.refuted_hypotheses) > MAX_REFUTED_HYPOTHESES:
        return False, "too_many_refuted"
    for c in report.core_causes:
        if len(c.evidence) < 1:
            return False, "core_no_evidence"
    for r in report.refuted_hypotheses:
        if len(r.evidence) < 1:
            return False, "refuted_no_evidence"
    total_bullets = sum(len(c.evidence) for c in report.core_causes) + \
                    sum(len(r.evidence) for r in report.refuted_hypotheses)
    if total_bullets > MAX_TOTAL_BULLETS:
        return False, "too_many_bullets"
    return True, None
