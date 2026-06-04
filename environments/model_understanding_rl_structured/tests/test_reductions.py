"""Reward-reduction unit tests against mocked StructuredJudgeScore + ParsedReport fixtures.

No API calls. Covers:
- v1 reductions (kept for comparison) still return float in [0, 1] on representative shapes
- v2 lookup-table reward + count overshoot penalty + length penalty piecewise breakpoints
- compute_components shape stability across format_pass / format_fail
- core_anti_hack_v2 short-circuits to 0 on format_fail
"""

from __future__ import annotations

import pytest

from model_understanding_rl_structured.judge import (
    ParsedReport,
    StructuredItemScore,
    StructuredJudgeScore,
    UnmatchedModelItem,
)
from model_understanding_rl_structured.reductions import (
    ALL_LOOKUP_CELLS,
    COMPONENT_NAMES,
    DEFAULT_ALPHA_CORE,
    DEFAULT_FORMAT_FAIL_REWARD,
    LEN_HIGH,
    LEN_MAX_GT,
    LEN_P50,
    LEN_P99,
    LEN_PEN_MAX,
    LOOKUP,
    REDUCTIONS,
    RewardContext,
    compute_components,
    count_overshoot,
    count_penalty_core,
    count_penalty_ref,
    judge_reward_sum,
    length_penalty,
    lookup_item_score,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _item(mq: int, ea: int, matched: bool = True, section: str = "core", idx: int = 0) -> StructuredItemScore:
    return StructuredItemScore(
        gt_name="stub",
        best_model_match_section=section if matched else None,
        best_model_match_idx=idx if matched else None,
        best_model_match_summary="stub" if matched else None,
        match_quality=mq,
        effect_accuracy=ea,
        reasoning="stub",
    )


def _unmatched(idx: int, section: str = "core") -> UnmatchedModelItem:
    return UnmatchedModelItem(model_section=section, model_idx=idx, model_summary="stub")


def _ctx(
    score: StructuredJudgeScore | None,
    *,
    m_core: int = 2,
    m_ref: int = 1,
    gt_core: int = 2,
    gt_ref: int = 1,
    completion_tokens: int = 700,
    format_pass: bool = True,
    format_fail_reason: str | None = None,
) -> RewardContext:
    return RewardContext(
        score=score,
        m_core=m_core, m_ref=m_ref,
        gt_core=gt_core, gt_ref=gt_ref,
        completion_tokens=completion_tokens,
        format_pass=format_pass,
        format_fail_reason=format_fail_reason,
    )


CLEAN_WIN = StructuredJudgeScore(
    gt_core_causes=(_item(3, 3), _item(3, 3)),
    gt_refuted_hypotheses=(_item(3, 3),),
    unmatched_model_causes=(),
)

PARTIAL = StructuredJudgeScore(
    gt_core_causes=(_item(3, 3), _item(2, 2), _item(1, 1, matched=False)),
    gt_refuted_hypotheses=(_item(3, 2),),
    unmatched_model_causes=(_unmatched(2),),
)

ALL_MISS = StructuredJudgeScore(
    gt_core_causes=(_item(1, 1, matched=False), _item(1, 1, matched=False)),
    gt_refuted_hypotheses=(_item(1, 1, matched=False),),
    unmatched_model_causes=(_unmatched(0), _unmatched(1)),
)

NO_GT_CAUSES = StructuredJudgeScore(
    gt_core_causes=(),
    gt_refuted_hypotheses=(_item(3, 3),),
    unmatched_model_causes=(),
)

MATCHED_BUT_WRONG_EFFECT = StructuredJudgeScore(
    gt_core_causes=(_item(3, 1), _item(3, 1)),
    gt_refuted_hypotheses=(),
    unmatched_model_causes=(),
)

SECTION_FLIP_SINGLE = StructuredJudgeScore(
    gt_core_causes=(_item(3, 1),),  # one item, full section flip
    gt_refuted_hypotheses=(),
    unmatched_model_causes=(),
)


SCORES = {
    "clean_win": CLEAN_WIN,
    "partial": PARTIAL,
    "all_miss": ALL_MISS,
    "no_gt_causes": NO_GT_CAUSES,
    "matched_but_wrong_effect": MATCHED_BUT_WRONG_EFFECT,
    "section_flip_single": SECTION_FLIP_SINGLE,
}


V1_REDUCTIONS = (
    "core_recall_strict",
    "core_recall_lenient",
    "core_effect_strict_given_matched",
    "core_composite_recall_effect",
    "core_plus_refuted_composite",
    "core_minus_padding",
)


# ---------------------------------------------------------------------------
# Lookup table tests
# ---------------------------------------------------------------------------


def test_lookup_table_is_symmetric() -> None:
    """ea=1 (contradiction) is reflected to ea=3 (right direction) with same magnitude."""
    assert LOOKUP[(3, 3)] == -LOOKUP[(3, 1)]
    assert LOOKUP[(2, 2)] == -LOOKUP[(2, 1)]


def test_lookup_table_covers_all_legal_cells() -> None:
    """Only the 6 cells legal under judge constraints (mq=1→ea=1, mq=2→ea≤2) are present."""
    expected = {(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)}
    assert set(LOOKUP) == expected
    assert set(ALL_LOOKUP_CELLS) == expected


def test_lookup_item_score_at_each_cell() -> None:
    for (mq, ea), expected in LOOKUP.items():
        assert lookup_item_score(_item(mq, ea)) == expected


def test_judge_reward_sum_clean_win() -> None:
    # 2 core × 0.6 + 1 refuted × 0.6 = 1.8
    assert judge_reward_sum(CLEAN_WIN) == pytest.approx(1.8)


def test_judge_reward_sum_partial() -> None:
    # core: 0.6 + 0.2 + 0.0 = 0.8; refuted: 0.4. Total = 1.2.
    assert judge_reward_sum(PARTIAL) == pytest.approx(1.2)


def test_judge_reward_sum_section_flip() -> None:
    # one (3,1) item → -0.6
    assert judge_reward_sum(SECTION_FLIP_SINGLE) == pytest.approx(-0.6)


# ---------------------------------------------------------------------------
# Length penalty tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tokens, expected", [
    (0, 0.0),
    (LEN_P50, 0.0),
    (LEN_P99, 0.05),
    (LEN_MAX_GT, 0.25),
    (LEN_HIGH, 0.60),
])
def test_length_penalty_at_breakpoints(tokens: int, expected: float) -> None:
    assert length_penalty(tokens) == pytest.approx(expected, abs=1e-6)


def test_length_penalty_below_p50_is_zero() -> None:
    for t in (0, 100, 500, LEN_P50 - 1, LEN_P50):
        assert length_penalty(t) == 0.0


def test_length_penalty_capped_at_max() -> None:
    assert length_penalty(10_000) == pytest.approx(LEN_PEN_MAX)
    assert length_penalty(100_000) == pytest.approx(LEN_PEN_MAX)


def test_length_penalty_monotone_nondecreasing() -> None:
    last = -1.0
    for t in range(0, 5000, 25):
        v = length_penalty(t)
        assert v >= last - 1e-9, f"penalty decreased at t={t}: {last} → {v}"
        last = v


# ---------------------------------------------------------------------------
# Count overshoot / penalty tests
# ---------------------------------------------------------------------------


def test_count_overshoot_basic() -> None:
    assert count_overshoot(3, 2) == 1
    assert count_overshoot(2, 2) == 0
    assert count_overshoot(1, 3) == 0  # undershoot → 0 (one-sided)
    assert count_overshoot(0, 0) == 0


def test_count_penalty_core_default_alpha() -> None:
    assert count_penalty_core(m_core=3, gt_core=2) == pytest.approx(0.2)
    assert count_penalty_core(m_core=2, gt_core=2) == 0.0
    assert count_penalty_core(m_core=1, gt_core=2) == 0.0


def test_count_penalty_ref_default_alpha() -> None:
    assert count_penalty_ref(m_ref=3, gt_ref=0) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# v2 core_anti_hack_v2 reduction tests
# ---------------------------------------------------------------------------


def test_anti_hack_v2_clean_win_no_penalties() -> None:
    """All matches, count exact, length at p50 → judge_reward_sum, no penalties."""
    ctx = _ctx(CLEAN_WIN, m_core=2, m_ref=1, gt_core=2, gt_ref=1, completion_tokens=LEN_P50)
    reward = REDUCTIONS["core_anti_hack_v2"](ctx)
    assert reward == pytest.approx(1.8)


def test_anti_hack_v2_overshoot_reduces_reward() -> None:
    ctx_match = _ctx(CLEAN_WIN, m_core=2, m_ref=1, gt_core=2, gt_ref=1, completion_tokens=LEN_P50)
    ctx_over = _ctx(CLEAN_WIN, m_core=5, m_ref=3, gt_core=2, gt_ref=1, completion_tokens=LEN_P50)
    r_match = REDUCTIONS["core_anti_hack_v2"](ctx_match)
    r_over = REDUCTIONS["core_anti_hack_v2"](ctx_over)
    # overshoot: 3 extra core (0.6 penalty) + 2 extra refuted (0.4 penalty) = 1.0 total
    assert r_over == pytest.approx(r_match - 1.0)


def test_anti_hack_v2_long_completion_penalized() -> None:
    ctx_short = _ctx(CLEAN_WIN, completion_tokens=LEN_P50)
    ctx_long = _ctx(CLEAN_WIN, completion_tokens=LEN_HIGH)
    r_short = REDUCTIONS["core_anti_hack_v2"](ctx_short)
    r_long = REDUCTIONS["core_anti_hack_v2"](ctx_long)
    assert r_long == pytest.approx(r_short - 0.60)


def test_anti_hack_v2_format_fail_default_is_minus_one() -> None:
    ctx = _ctx(None, format_pass=False, format_fail_reason="too_many_core")
    assert REDUCTIONS["core_anti_hack_v2"](ctx) == DEFAULT_FORMAT_FAIL_REWARD
    assert DEFAULT_FORMAT_FAIL_REWARD == -1.0


def test_anti_hack_v2_format_fail_reward_is_configurable() -> None:
    ctx = _ctx(None, format_pass=False, format_fail_reason="too_many_core")
    assert REDUCTIONS["core_anti_hack_v2"](ctx, format_fail_reward=0.0) == 0.0
    assert REDUCTIONS["core_anti_hack_v2"](ctx, format_fail_reward=-2.5) == -2.5


def test_anti_hack_v2_contradictions_are_negative() -> None:
    ctx = _ctx(SECTION_FLIP_SINGLE, m_core=1, m_ref=0, gt_core=1, gt_ref=0, completion_tokens=LEN_P50)
    reward = REDUCTIONS["core_anti_hack_v2"](ctx)
    assert reward == pytest.approx(-0.6)


def test_anti_hack_v2_with_undershoot_is_not_penalized_for_count() -> None:
    """Emitting fewer than GT items should not trigger overshoot penalty (one-sided)."""
    ctx = _ctx(NO_GT_CAUSES, m_core=0, m_ref=1, gt_core=0, gt_ref=1, completion_tokens=LEN_P50)
    reward = REDUCTIONS["core_anti_hack_v2"](ctx)
    # no GT core, one matched refuted → judge_reward_sum=+0.6, no count penalty, no length penalty
    assert reward == pytest.approx(0.6)


def test_anti_hack_v2_alpha_kwarg_passes_through() -> None:
    ctx = _ctx(CLEAN_WIN, m_core=5, m_ref=1, gt_core=2, gt_ref=1, completion_tokens=LEN_P50)
    r_default = REDUCTIONS["core_anti_hack_v2"](ctx)
    r_zero_alpha = REDUCTIONS["core_anti_hack_v2"](ctx, alpha_core=0.0)
    assert r_zero_alpha > r_default


# ---------------------------------------------------------------------------
# v1 reductions still work (wrapped to accept RewardContext)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", V1_REDUCTIONS)
@pytest.mark.parametrize("score_name", list(SCORES))
def test_v1_reduction_returns_unit_interval(name: str, score_name: str) -> None:
    fn = REDUCTIONS[name]
    ctx = _ctx(SCORES[score_name])
    out = fn(ctx, padding_penalty=0.5)
    assert isinstance(out, float)
    assert 0.0 <= out <= 1.0, f"reduction={name} score={score_name} returned {out}"


def test_v1_clean_win_unchanged() -> None:
    ctx = _ctx(CLEAN_WIN)
    assert REDUCTIONS["core_recall_strict"](ctx) == 1.0
    assert REDUCTIONS["core_recall_lenient"](ctx) == 1.0
    assert REDUCTIONS["core_effect_strict_given_matched"](ctx) == 1.0
    assert REDUCTIONS["core_composite_recall_effect"](ctx) == 1.0
    assert REDUCTIONS["core_plus_refuted_composite"](ctx) == 1.0


def test_v1_all_miss_zero() -> None:
    ctx = _ctx(ALL_MISS)
    for name in V1_REDUCTIONS:
        assert REDUCTIONS[name](ctx) == 0.0


def test_format_fail_returns_format_fail_reward_for_all_reductions() -> None:
    ctx = _ctx(None, format_pass=False, format_fail_reason="missing_observed")
    for name in REDUCTIONS:
        assert REDUCTIONS[name](ctx) == DEFAULT_FORMAT_FAIL_REWARD, f"reduction={name}"
    # Configurable override applies to all reductions
    for name in REDUCTIONS:
        assert REDUCTIONS[name](ctx, format_fail_reward=0.0) == 0.0, f"reduction={name}"
        assert REDUCTIONS[name](ctx, format_fail_reward=-2.0) == -2.0, f"reduction={name}"


# ---------------------------------------------------------------------------
# compute_components tests
# ---------------------------------------------------------------------------


def test_compute_components_includes_all_expected_keys() -> None:
    ctx = _ctx(CLEAN_WIN)
    out = compute_components(ctx)
    for name in COMPONENT_NAMES:
        assert name in out, f"missing component {name!r}"


def test_compute_components_format_pass_and_fail_have_same_keyset() -> None:
    ctx_pass = _ctx(CLEAN_WIN)
    ctx_fail = _ctx(None, format_pass=False, format_fail_reason="missing_observed")
    assert set(compute_components(ctx_pass)) == set(compute_components(ctx_fail))


def test_compute_components_format_pass_flag() -> None:
    ctx_pass = _ctx(CLEAN_WIN)
    ctx_fail = _ctx(None, format_pass=False, format_fail_reason="missing_observed")
    assert compute_components(ctx_pass)["format_pass"] == 1.0
    assert compute_components(ctx_fail)["format_pass"] == 0.0


def test_compute_components_one_hot_fail_reason() -> None:
    ctx = _ctx(None, format_pass=False, format_fail_reason="too_many_core")
    comp = compute_components(ctx)
    assert comp["format_fail_reason_too_many_core"] == 1.0
    assert comp["format_fail_reason_missing_observed"] == 0.0
    assert comp["format_fail_reason_too_many_bullets"] == 0.0


def test_compute_components_per_cell_counts() -> None:
    # CLEAN_WIN: 2 core (3,3) items, 1 refuted (3,3) item
    comp = compute_components(_ctx(CLEAN_WIN))
    assert comp["core_n_items_3_3"] == 2.0
    assert comp["refuted_n_items_3_3"] == 1.0
    assert comp["core_n_items_3_2"] == 0.0
    assert comp["core_n_items_1_1"] == 0.0


def test_compute_components_judge_metrics_zero_when_format_fail() -> None:
    comp = compute_components(_ctx(None, format_pass=False, format_fail_reason="too_many_core"))
    assert comp["core_lookup_sum"] == 0.0
    assert comp["refuted_lookup_sum"] == 0.0
    assert comp["total_lookup_sum"] == 0.0
    for (mq, ea) in ALL_LOOKUP_CELLS:
        assert comp[f"core_n_items_{mq}_{ea}"] == 0.0
        assert comp[f"refuted_n_items_{mq}_{ea}"] == 0.0


def test_compute_components_count_metrics_work_on_format_fail() -> None:
    """Counts and length penalty are computable without judge output."""
    comp = compute_components(_ctx(
        None, format_pass=False, format_fail_reason="too_many_core",
        m_core=5, m_ref=2, gt_core=2, gt_ref=1, completion_tokens=1200,
    ))
    assert comp["m_core"] == 5.0
    assert comp["gt_core"] == 2.0
    assert comp["count_overshoot_core"] == 3.0
    assert comp["count_overshoot_refuted"] == 1.0
    assert comp["count_penalty_core"] == pytest.approx(3 * DEFAULT_ALPHA_CORE)
    assert comp["length_penalty"] > 0.0


def test_compute_components_length_penalty_consistent() -> None:
    """compute_components.length_penalty must match the standalone function."""
    for t in (LEN_P50, LEN_P99, LEN_MAX_GT, LEN_HIGH, 2000):
        comp = compute_components(_ctx(CLEAN_WIN, completion_tokens=t))
        assert comp["length_penalty"] == pytest.approx(length_penalty(t))


# ---------------------------------------------------------------------------
# Smoke for registry stability
# ---------------------------------------------------------------------------


def test_registry_keys_are_documented() -> None:
    """Whoever adds a reduction must also export it in REDUCTIONS."""
    expected = {
        "core_anti_hack_v2",
        "core_recall_strict",
        "core_recall_lenient",
        "core_effect_strict_given_matched",
        "core_composite_recall_effect",
        "core_plus_refuted_composite",
        "core_minus_padding",
    }
    assert set(REDUCTIONS) == expected
