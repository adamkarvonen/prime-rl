"""Format gate tests — programmatic structural checks on the parsed model report.

Covers each FORMAT_FAIL_REASONS case + the happy path.
"""

from __future__ import annotations

import pytest

from model_understanding_rl_structured.format_gate import (
    FORMAT_FAIL_REASONS,
    MAX_CORE_CAUSES,
    MAX_REFUTED_HYPOTHESES,
    MAX_TOTAL_BULLETS,
    check_format,
)
from model_understanding_rl_structured.judge import ParsedCause, ParsedReport


def _cause(name: str = "stub", evidence: tuple[str, ...] = ("ev1",), status: str = "supported") -> ParsedCause:
    return ParsedCause(name=name, status=status, evidence=evidence)


def _report(
    observed: str = "Behavior happens at 70%.",
    core: tuple[ParsedCause, ...] = (_cause("c1"),),
    refuted: tuple[ParsedCause, ...] = (),
    final: str = "Summary of the result.",
) -> ParsedReport:
    return ParsedReport(
        observed_behavior=observed,
        core_causes=core,
        refuted_hypotheses=refuted,
        final_answer=final,
    )


def test_happy_path_minimal() -> None:
    ok, reason = check_format(_report())
    assert ok and reason is None


def test_happy_path_max_legal_shape() -> None:
    """Max core (3) + max refuted (3) with bullets spread to total ≤ 8."""
    core = tuple(_cause(f"c{i}", evidence=("ev",)) for i in range(MAX_CORE_CAUSES))
    refuted = tuple(_cause(f"r{i}", evidence=("ev",)) for i in range(MAX_REFUTED_HYPOTHESES))
    total = MAX_CORE_CAUSES + MAX_REFUTED_HYPOTHESES
    assert total <= MAX_TOTAL_BULLETS, "test fixture invariant"
    ok, reason = check_format(_report(core=core, refuted=refuted))
    assert ok, f"unexpected fail reason: {reason}"


def test_missing_observed() -> None:
    ok, reason = check_format(_report(observed=""))
    assert not ok and reason == "missing_observed"


def test_missing_observed_only_whitespace() -> None:
    ok, reason = check_format(_report(observed="   \n  "))
    assert not ok and reason == "missing_observed"


def test_missing_final() -> None:
    ok, reason = check_format(_report(final=""))
    assert not ok and reason == "missing_final"


def test_too_few_core() -> None:
    ok, reason = check_format(_report(core=()))
    assert not ok and reason == "too_few_core"


def test_too_many_core() -> None:
    core = tuple(_cause(f"c{i}") for i in range(MAX_CORE_CAUSES + 1))
    ok, reason = check_format(_report(core=core))
    assert not ok and reason == "too_many_core"


def test_too_many_refuted() -> None:
    refuted = tuple(_cause(f"r{i}") for i in range(MAX_REFUTED_HYPOTHESES + 1))
    ok, reason = check_format(_report(refuted=refuted))
    assert not ok and reason == "too_many_refuted"


def test_core_no_evidence() -> None:
    core = (_cause("c1", evidence=()),)
    ok, reason = check_format(_report(core=core))
    assert not ok and reason == "core_no_evidence"


def test_refuted_no_evidence() -> None:
    refuted = (_cause("r1", evidence=()),)
    ok, reason = check_format(_report(refuted=refuted))
    assert not ok and reason == "refuted_no_evidence"


def test_too_many_bullets() -> None:
    """3 core × 3 bullets = 9 > 8 cap."""
    core = tuple(_cause(f"c{i}", evidence=("ev1", "ev2", "ev3")) for i in range(3))
    ok, reason = check_format(_report(core=core))
    assert not ok and reason == "too_many_bullets"


def test_empty_refuted_is_legal() -> None:
    """Per the strict prompt, omitting Negative evidence entirely is fine."""
    ok, reason = check_format(_report(refuted=()))
    assert ok and reason is None


def test_all_fail_reasons_are_reachable() -> None:
    """Every reason in FORMAT_FAIL_REASONS should be returnable by some input."""
    # This is a registry sanity check — make sure we haven't added a reason string
    # to FORMAT_FAIL_REASONS that no actual check returns.
    reachable = {
        "missing_observed", "missing_final",
        "too_few_core", "too_many_core",
        "too_many_refuted", "too_many_bullets",
        "core_no_evidence", "refuted_no_evidence",
    }
    assert set(FORMAT_FAIL_REASONS) == reachable


@pytest.mark.parametrize("n_core, n_ref, bullets_per", [
    (1, 0, 1),
    (2, 1, 1),
    (3, 3, 1),  # exactly at total-bullet boundary (6 < 8)
    (1, 3, 2),  # 4 + 3 = 7
])
def test_various_legal_shapes(n_core: int, n_ref: int, bullets_per: int) -> None:
    core = tuple(_cause(f"c{i}", evidence=tuple(f"ev{j}" for j in range(bullets_per))) for i in range(n_core))
    refuted = tuple(_cause(f"r{i}", evidence=tuple(f"ev{j}" for j in range(bullets_per))) for i in range(n_ref))
    ok, reason = check_format(_report(core=core, refuted=refuted))
    assert ok, f"shape (core={n_core}, ref={n_ref}, bullets={bullets_per}) should pass; got reason={reason}"
