"""Reward reductions: RewardContext -> scalar.

v2 design (default): per-item LOOKUP table replaces the v1 multiplicative
composite. Items are scored independently per the (match_quality, effect_accuracy)
cell, summed across core + refuted GT items (no explicit weighting — natural
refuted share is ~35% from the GT count distribution). Anti-hack terms:
one-sided overproduction penalty per section, piecewise length penalty anchored
to the measured GT length distribution. Negative rewards are allowed.

v1 reductions are kept in REDUCTIONS for comparison and are also logged at
weight=0 as observability metrics in compute_components().

See investigations/model_understanding_prime_rl/REWARD_SIGNAL.md for design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .format_gate import FORMAT_FAIL_REASONS
from .judge import StructuredJudgeScore


# ---------------------------------------------------------------------------
# v2 lookup table — symmetric per-item score
# ---------------------------------------------------------------------------

# (match_quality, effect_accuracy) → reward per GT item.
# Judge constraints (mq=1→ea=1, mq=2→ea≤2) mean only six combinations are legal.
LOOKUP: dict[tuple[int, int], float] = {
    (3, 3): +0.6,   # right mechanism, right direction, right strength
    (3, 2): +0.4,   # right mechanism, right direction, strength off by one
    (2, 2): +0.2,   # partial mechanism match, right direction
    (1, 1):  0.0,   # no meaningful match — neutral (model missed it)
    (2, 1): -0.2,   # partial mechanism, wrong direction (symmetric to (2,2))
    (3, 1): -0.6,   # right mechanism, wrong direction / section flip (symmetric to (3,3))
}
ALL_LOOKUP_CELLS: tuple[tuple[int, int], ...] = tuple(sorted(LOOKUP))


# v3 lookup — bigger positive gradient on confident hits, sign-flip on (3,1) treated as
# magnitude-symmetric with (3,2): "right mechanism, wrong direction" is exactly as bad
# as "right mechanism + direction, strength off" is good. Applied symmetrically to both
# core and refuted sections (refuted gets a boost vs v2 because P(3,3) on refuted is
# ~11% empirically vs ~28% on core — without a boost, refuted attempts are net-negative
# in expectation).
LOOKUP_V3: dict[tuple[int, int], float] = {
    (3, 3): +0.8,
    (3, 2): +0.5,
    (2, 2): +0.2,
    (1, 1):  0.0,
    (2, 1): -0.2,
    (3, 1): -0.5,
}


def lookup_item_score(item, table: dict[tuple[int, int], float] = LOOKUP) -> float:
    key = (item.match_quality, item.effect_accuracy)
    assert key in table, (
        f"Illegal (mq={item.match_quality}, ea={item.effect_accuracy}) — judge constraints "
        f"(mq=1→ea=1, mq=2→ea≤2) should make this unreachable. Got: {item}"
    )
    return table[key]


def core_lookup_sum(score: StructuredJudgeScore, table: dict = LOOKUP) -> float:
    return sum(lookup_item_score(it, table) for it in score.gt_core_causes)


def refuted_lookup_sum(score: StructuredJudgeScore, table: dict = LOOKUP) -> float:
    return sum(lookup_item_score(it, table) for it in score.gt_refuted_hypotheses)


def judge_reward_sum(score: StructuredJudgeScore, table: dict = LOOKUP) -> float:
    return core_lookup_sum(score, table) + refuted_lookup_sum(score, table)


# ---------------------------------------------------------------------------
# Length penalty — piecewise linear, anchored to measured GT distribution
# (qwen3_32b_run_2_train: p50=620, p99=823, max=919; tokenized with the
# Qwen3-32B SFT-merged tokenizer). See REWARD_SIGNAL.md §6.
# ---------------------------------------------------------------------------

LEN_P50 = 630        # below this: no penalty
LEN_P99 = 830        # mild zone ends here
LEN_MAX_GT = 950     # past max-GT support — penalty becomes significant (0.25)
LEN_HIGH = 1500      # ~2.4× p50 — penalty reaches 0.60
LEN_PEN_MAX = 0.80   # absolute cap


def length_penalty(tokens: int) -> float:
    if tokens <= LEN_P50:
        return 0.0
    if tokens <= LEN_P99:
        return (tokens - LEN_P50) / (LEN_P99 - LEN_P50) * 0.05            # 0 → 0.05
    if tokens <= LEN_MAX_GT:
        return 0.05 + (tokens - LEN_P99) / (LEN_MAX_GT - LEN_P99) * 0.20  # 0.05 → 0.25
    if tokens <= LEN_HIGH:
        return 0.25 + (tokens - LEN_MAX_GT) / (LEN_HIGH - LEN_MAX_GT) * 0.35  # 0.25 → 0.60
    return min(LEN_PEN_MAX, 0.60 + (tokens - LEN_HIGH) * 0.0004)          # cap 0.80


# v3 length penalty — more permissive in the middle: zero below P80, weak P80→P95,
# strong past P95. Motivated by v2 data showing the model is at p10 length (487 tokens)
# but missing GT items — we don't want length pressure to discourage adding evidence
# bullets that could lift mq=2 → mq=3.
LEN_V3_P80 = 720
LEN_V3_P95 = 790
LEN_V3_MAX_GT = 950
LEN_V3_HIGH = 1500
LEN_V3_PEN_MAX = 0.80


def length_penalty_v3(tokens: int) -> float:
    if tokens <= LEN_V3_P80:
        return 0.0
    if tokens <= LEN_V3_P95:
        return (tokens - LEN_V3_P80) / (LEN_V3_P95 - LEN_V3_P80) * 0.05            # 0 → 0.05
    if tokens <= LEN_V3_MAX_GT:
        return 0.05 + (tokens - LEN_V3_P95) / (LEN_V3_MAX_GT - LEN_V3_P95) * 0.20  # 0.05 → 0.25
    if tokens <= LEN_V3_HIGH:
        return 0.25 + (tokens - LEN_V3_MAX_GT) / (LEN_V3_HIGH - LEN_V3_MAX_GT) * 0.35  # 0.25 → 0.60
    return min(LEN_V3_PEN_MAX, 0.60 + (tokens - LEN_V3_HIGH) * 0.0004)              # cap 0.80


# ---------------------------------------------------------------------------
# Overproduction penalty (one-sided, per section)
# ---------------------------------------------------------------------------

DEFAULT_ALPHA_CORE = 0.20
DEFAULT_ALPHA_REF = 0.20
DEFAULT_FORMAT_FAIL_REWARD = -1.0   # < 0 closes the "deliberately format-fail to dodge a bad judge outcome" escape hatch


def count_overshoot(m: int, gt: int) -> int:
    return max(0, m - gt)


def count_penalty_core(m_core: int, gt_core: int, alpha: float = DEFAULT_ALPHA_CORE) -> float:
    return alpha * count_overshoot(m_core, gt_core)


def count_penalty_ref(m_ref: int, gt_ref: int, alpha: float = DEFAULT_ALPHA_REF) -> float:
    return alpha * count_overshoot(m_ref, gt_ref)


# v3 — tiered overshoot ("weak first guess, strong for more"). Used by core_anti_hack_v3.
# For refuted we apply the same shape AFTER granting one free guess (so the model
# always speculates one refuted item even when GT has none, with no penalty).
V3_WEAK_PENALTY = 0.10
V3_STRONG_PENALTY = 0.40


def tiered_overshoot_penalty(overshoot: int) -> float:
    if overshoot <= 0:
        return 0.0
    if overshoot == 1:
        return V3_WEAK_PENALTY
    return V3_WEAK_PENALTY + V3_STRONG_PENALTY * (overshoot - 1)


def count_penalty_core_v3(m_core: int, gt_core: int) -> float:
    return tiered_overshoot_penalty(count_overshoot(m_core, gt_core))


def count_overshoot_ref_v3(m_ref: int, gt_ref: int) -> int:
    """One free refuted guess: penalty kicks in only at m > gt + 1."""
    return max(0, m_ref - gt_ref - 1)


def count_penalty_ref_v3(m_ref: int, gt_ref: int) -> float:
    return tiered_overshoot_penalty(count_overshoot_ref_v3(m_ref, gt_ref))


# v4 — symmetric count penalty in BOTH directions (undershoot now penalized too).
# v3 had one-directional overshoot penalty, which combined with the catastrophic (3,1)
# cell created a risk-averse equilibrium: model preferred to undershoot (safe -0.24
# lookup loss) over overshoot (uncertain, could hit -0.60). Empirically v3 sat at
# m_core ≈ gt_core − 0.1, missing easy lookup score. v4 breaks the asymmetry.
V4_WEAK_PENALTY = 0.05
V4_TIER_STEP = 0.15  # so distance=2 → 0.05 + 0.15 = 0.20, distance=3 → 0.35


def symmetric_tiered_penalty(distance: int) -> float:
    """Penalty for |m - gt|, symmetric in either direction.
    distance=0: 0.00 | distance=1: 0.05 | distance=2: 0.20 | distance=3: 0.35
    """
    if distance <= 0:
        return 0.0
    if distance == 1:
        return V4_WEAK_PENALTY
    return V4_WEAK_PENALTY + V4_TIER_STEP * (distance - 1)


def count_penalty_core_v4(m_core: int, gt_core: int) -> float:
    return symmetric_tiered_penalty(abs(m_core - gt_core))


def count_penalty_ref_v4(m_ref: int, gt_ref: int) -> float:
    """Refuted keeps a ±1 free zone around gt_ref (preserves v3's free-guess intent
    in both directions): being off by 1 in either direction is unpenalized, so the
    model isn't pressured to either abstain or speculate when it's uncertain.
    """
    return symmetric_tiered_penalty(max(0, abs(m_ref - gt_ref) - 1))


# ---------------------------------------------------------------------------
# RewardContext — bundles all inputs the v2 reduction + components need
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardContext:
    """All inputs needed to compute v2 reward and observability metrics.

    score is None when format_pass is False (judge was not called).
    """
    score: StructuredJudgeScore | None
    m_core: int
    m_ref: int
    gt_core: int
    gt_ref: int
    completion_tokens: int
    format_pass: bool
    format_fail_reason: str | None  # None when format_pass=True


# ---------------------------------------------------------------------------
# v2 reduction — anti-hack composite (default)
# ---------------------------------------------------------------------------


def core_anti_hack_v2(
    ctx: RewardContext,
    *,
    alpha_core: float = DEFAULT_ALPHA_CORE,
    alpha_ref: float = DEFAULT_ALPHA_REF,
    format_fail_reward: float = DEFAULT_FORMAT_FAIL_REWARD,
    **_,
) -> float:
    if not ctx.format_pass:
        return format_fail_reward
    assert ctx.score is not None, "format_pass=True but score is None"

    judge_reward = judge_reward_sum(ctx.score)
    count_pen = (
        count_penalty_core(ctx.m_core, ctx.gt_core, alpha_core)
      + count_penalty_ref(ctx.m_ref, ctx.gt_ref, alpha_ref)
    )
    len_pen = length_penalty(ctx.completion_tokens)
    return judge_reward - count_pen - len_pen


# ---------------------------------------------------------------------------
# v3 reduction — addresses v2's refuted-collapse failure mode
# ---------------------------------------------------------------------------
#
# v2 outcome after 280 steps of 32B RL: reward +0.16 came ~92% from anti-hack
# penalties shrinking (model dropped refuted entirely, trimmed core overshoot),
# ~8% from actual judge signal. Refuted EV per item under v2 was net-negative
# (~-0.06) so the model rationally abstained.
#
# v3 changes:
#  1. LOOKUP_V3 boosts (3,3) to +0.8, (3,2) to +0.5, deepens (3,1) to -0.5
#     (magnitude-symmetric with the +0.5 of right-direction-strength-off).
#     Applied to BOTH core and refuted (symmetric).
#  2. Tiered overshoot: weak 0.10 for first over, strong 0.40 per additional.
#     Discourages spam without crushing single-item exploration.
#  3. Free refuted guess: count_overshoot_ref_v3 starts at m > gt + 1, so the
#     model can always speculate one refuted item without penalty.
#  4. length_penalty_v3 with thresholds at P80/P95 instead of P50/P99 —
#     more room to write evidence-bullet detail without length pressure.
#
# alpha_core/alpha_ref TOML knobs are intentionally NOT consumed by v3 (the
# tiered helpers replace flat alphas). Pass-through accepted via **_ for
# config compatibility.
def core_anti_hack_v3(
    ctx: RewardContext,
    *,
    format_fail_reward: float = DEFAULT_FORMAT_FAIL_REWARD,
    **_,
) -> float:
    if not ctx.format_pass:
        return format_fail_reward
    assert ctx.score is not None, "format_pass=True but score is None"

    judge_reward = judge_reward_sum(ctx.score, LOOKUP_V3)
    count_pen = (
        count_penalty_core_v3(ctx.m_core, ctx.gt_core)
      + count_penalty_ref_v3(ctx.m_ref, ctx.gt_ref)
    )
    len_pen = length_penalty_v3(ctx.completion_tokens)
    return judge_reward - count_pen - len_pen


# v4 reduction — same LOOKUP_V3 + length_penalty_v3 as v3, but with symmetric
# (bidirectional) count penalty. Addresses v3's risk-averse undershoot equilibrium.
def core_anti_hack_v4(
    ctx: RewardContext,
    *,
    format_fail_reward: float = DEFAULT_FORMAT_FAIL_REWARD,
    **_,
) -> float:
    if not ctx.format_pass:
        return format_fail_reward
    assert ctx.score is not None, "format_pass=True but score is None"

    judge_reward = judge_reward_sum(ctx.score, LOOKUP_V3)
    count_pen = (
        count_penalty_core_v4(ctx.m_core, ctx.gt_core)
      + count_penalty_ref_v4(ctx.m_ref, ctx.gt_ref)
    )
    len_pen = length_penalty_v3(ctx.completion_tokens)
    return judge_reward - count_pen - len_pen


# ---------------------------------------------------------------------------
# v1 reductions — kept for comparison; signature wrapped to accept RewardContext
# ---------------------------------------------------------------------------


def _normalize(x: int) -> float:
    assert 1 <= x <= 3, f"expected score in [1,3], got {x}"
    return (x - 1) / 2.0


def core_recall_strict(score: StructuredJudgeScore) -> float:
    items = score.gt_core_causes
    if not items:
        return 0.0
    return sum(1 for it in items if it.match_quality == 3) / len(items)


def core_recall_lenient(score: StructuredJudgeScore) -> float:
    items = score.gt_core_causes
    if not items:
        return 0.0
    return sum(1 for it in items if it.match_quality >= 2) / len(items)


def per_entry_match_q_mean(score: StructuredJudgeScore) -> float:
    items = score.gt_core_causes
    if not items:
        return 0.0
    return sum(_normalize(it.match_quality) for it in items) / len(items)


def per_entry_effect_mean_matched(score: StructuredJudgeScore) -> float:
    matched = [it for it in score.gt_core_causes if it.best_model_match_idx is not None]
    if not matched:
        return 0.0
    return sum(_normalize(it.effect_accuracy) for it in matched) / len(matched)


def core_effect_strict_given_matched(score: StructuredJudgeScore) -> float:
    matched = [it for it in score.gt_core_causes if it.best_model_match_idx is not None]
    if not matched:
        return 0.0
    return sum(1 for it in matched if it.effect_accuracy == 3) / len(matched)


def core_composite_recall_effect(score: StructuredJudgeScore) -> float:
    return per_entry_match_q_mean(score) * per_entry_effect_mean_matched(score)


def all_items_composite(score: StructuredJudgeScore) -> float:
    all_items = score.gt_core_causes + score.gt_refuted_hypotheses
    if not all_items:
        return 0.0
    match_q_mean = sum(_normalize(it.match_quality) for it in all_items) / len(all_items)
    matched = [it for it in all_items if it.best_model_match_idx is not None]
    if not matched:
        return 0.0
    effect_mean = sum(_normalize(it.effect_accuracy) for it in matched) / len(matched)
    return match_q_mean * effect_mean


def padding_rate(score: StructuredJudgeScore) -> float:
    n_gt_matched = sum(
        1 for it in score.gt_core_causes if it.best_model_match_idx is not None
    ) + sum(
        1 for it in score.gt_refuted_hypotheses if it.best_model_match_idx is not None
    )
    n_unmatched = len(score.unmatched_model_causes)
    total_model_items = n_gt_matched + n_unmatched
    if total_model_items == 0:
        return 0.0
    return n_unmatched / total_model_items


def core_minus_padding(score: StructuredJudgeScore, padding_penalty: float) -> float:
    base = core_composite_recall_effect(score)
    return max(0.0, base - padding_penalty * padding_rate(score))


def _v1_wrap(fn: Callable[[StructuredJudgeScore], float]) -> Callable[..., float]:
    """Wrap a v1 reduction (takes only score, returns float in [0,1]) to accept RewardContext.

    Format failures return `format_fail_reward` (default −1.0), same as v2. v1 reductions
    are kept only for observability / comparison — but they should respect the same
    format-fail policy so all-reductions metrics stay comparable.
    """
    def _wrapped(ctx: RewardContext, *, format_fail_reward: float = DEFAULT_FORMAT_FAIL_REWARD, **_) -> float:
        if not ctx.format_pass or ctx.score is None:
            return format_fail_reward
        return fn(ctx.score)
    _wrapped.__name__ = fn.__name__
    return _wrapped


# ---------------------------------------------------------------------------
# Public reduction registry — keyed by string used in the TOML config
# ---------------------------------------------------------------------------


REDUCTIONS: dict[str, Callable[..., float]] = {
    # v4 — symmetric (bidirectional) count penalty; same LOOKUP_V3 + length_penalty_v3 as v3
    "core_anti_hack_v4":                 core_anti_hack_v4,

    # v3 — addresses refuted-collapse failure mode of v2 (see core_anti_hack_v3 docstring)
    "core_anti_hack_v3":                 core_anti_hack_v3,

    # v2 — previous default, kept for A/B comparison
    "core_anti_hack_v2":                 core_anti_hack_v2,

    # v1 — kept for comparison
    "core_recall_strict":                _v1_wrap(core_recall_strict),
    "core_recall_lenient":               _v1_wrap(core_recall_lenient),
    "core_effect_strict_given_matched":  _v1_wrap(core_effect_strict_given_matched),
    "core_composite_recall_effect":      _v1_wrap(core_composite_recall_effect),
    "core_plus_refuted_composite":       _v1_wrap(all_items_composite),
    "core_minus_padding": lambda ctx, padding_penalty=0.5, format_fail_reward=DEFAULT_FORMAT_FAIL_REWARD, **_: (
        format_fail_reward if not ctx.format_pass or ctx.score is None
        else core_minus_padding(ctx.score, padding_penalty)
    ),
}


# ---------------------------------------------------------------------------
# Granular component metrics for wandb logging (weight=0 in the rubric)
# ---------------------------------------------------------------------------


def _count_items_by_cell(items, mq: int, ea: int) -> int:
    return sum(1 for it in items if it.match_quality == mq and it.effect_accuracy == ea)


def compute_components(ctx: RewardContext) -> dict[str, float]:
    """Return all observability metrics for wandb logging.

    Always returns a dict with the same set of keys regardless of format_pass
    (zeros where judge data is unavailable), so wandb columns stay stable
    across rollouts within a run.
    """
    out: dict[str, float] = {}

    # Format gate
    out["format_pass"] = float(ctx.format_pass)
    for reason in FORMAT_FAIL_REASONS:
        out[f"format_fail_reason_{reason}"] = float(
            (not ctx.format_pass) and ctx.format_fail_reason == reason
        )

    # Length
    out["completion_token_count"] = float(ctx.completion_tokens)
    out["length_penalty"]         = length_penalty(ctx.completion_tokens)
    out["length_penalty_v3"]      = length_penalty_v3(ctx.completion_tokens)

    # Counts and count-derived penalties (computable even on format_fail)
    out["m_core"]                  = float(ctx.m_core)
    out["m_refuted"]               = float(ctx.m_ref)
    out["gt_core"]                 = float(ctx.gt_core)
    out["gt_refuted"]              = float(ctx.gt_ref)
    out["count_distance_core"]     = float(ctx.m_core - ctx.gt_core)
    out["count_distance_refuted"]  = float(ctx.m_ref - ctx.gt_ref)
    out["count_overshoot_core"]    = float(count_overshoot(ctx.m_core, ctx.gt_core))
    out["count_overshoot_refuted"] = float(count_overshoot(ctx.m_ref, ctx.gt_ref))
    out["count_penalty_core"]      = count_penalty_core(ctx.m_core, ctx.gt_core)
    out["count_penalty_refuted"]   = count_penalty_ref(ctx.m_ref, ctx.gt_ref)
    out["count_penalty_total"]     = out["count_penalty_core"] + out["count_penalty_refuted"]
    # v3 versions of count penalties (tiered + free refuted guess)
    out["count_overshoot_refuted_v3"] = float(count_overshoot_ref_v3(ctx.m_ref, ctx.gt_ref))
    out["count_penalty_core_v3"]      = count_penalty_core_v3(ctx.m_core, ctx.gt_core)
    out["count_penalty_refuted_v3"]   = count_penalty_ref_v3(ctx.m_ref, ctx.gt_ref)
    out["count_penalty_total_v3"]     = out["count_penalty_core_v3"] + out["count_penalty_refuted_v3"]
    # v4 versions of count penalties (symmetric, bidirectional + refuted ±1 free zone)
    out["count_penalty_core_v4"]      = count_penalty_core_v4(ctx.m_core, ctx.gt_core)
    out["count_penalty_refuted_v4"]   = count_penalty_ref_v4(ctx.m_ref, ctx.gt_ref)
    out["count_penalty_total_v4"]     = out["count_penalty_core_v4"] + out["count_penalty_refuted_v4"]

    # Judge-dependent metrics
    if ctx.format_pass and ctx.score is not None:
        s = ctx.score
        out["core_lookup_sum"]              = core_lookup_sum(s)
        out["refuted_lookup_sum"]           = refuted_lookup_sum(s)
        out["total_lookup_sum"]             = judge_reward_sum(s)
        out["core_lookup_sum_v3"]           = core_lookup_sum(s, LOOKUP_V3)
        out["refuted_lookup_sum_v3"]        = refuted_lookup_sum(s, LOOKUP_V3)
        out["total_lookup_sum_v3"]          = judge_reward_sum(s, LOOKUP_V3)
        out["n_unmatched_model_core"]       = float(sum(
            1 for u in s.unmatched_model_causes if u.model_section == "core"
        ))
        out["n_unmatched_model_refuted"]    = float(sum(
            1 for u in s.unmatched_model_causes if u.model_section == "refuted"
        ))
        out["core_n_items_total"]           = float(len(s.gt_core_causes))
        out["refuted_n_items_total"]        = float(len(s.gt_refuted_hypotheses))
        for (mq, ea) in ALL_LOOKUP_CELLS:
            out[f"core_n_items_{mq}_{ea}"]    = float(_count_items_by_cell(s.gt_core_causes, mq, ea))
            out[f"refuted_n_items_{mq}_{ea}"] = float(_count_items_by_cell(s.gt_refuted_hypotheses, mq, ea))
        # v1 metrics for comparison
        out["v1_core_recall_strict"]              = core_recall_strict(s)
        out["v1_core_recall_lenient"]             = core_recall_lenient(s)
        out["v1_core_effect_strict_given_matched"] = core_effect_strict_given_matched(s)
        out["v1_core_composite_recall_effect"]    = core_composite_recall_effect(s)
        out["v1_core_plus_refuted_composite"]     = all_items_composite(s)
        out["v1_padding_rate"]                    = padding_rate(s)
    else:
        out["core_lookup_sum"]           = 0.0
        out["refuted_lookup_sum"]        = 0.0
        out["total_lookup_sum"]          = 0.0
        out["core_lookup_sum_v3"]        = 0.0
        out["refuted_lookup_sum_v3"]     = 0.0
        out["total_lookup_sum_v3"]       = 0.0
        out["n_unmatched_model_core"]    = 0.0
        out["n_unmatched_model_refuted"] = 0.0
        out["core_n_items_total"]        = 0.0
        out["refuted_n_items_total"]     = 0.0
        for (mq, ea) in ALL_LOOKUP_CELLS:
            out[f"core_n_items_{mq}_{ea}"]    = 0.0
            out[f"refuted_n_items_{mq}_{ea}"] = 0.0
        out["v1_core_recall_strict"]              = 0.0
        out["v1_core_recall_lenient"]             = 0.0
        out["v1_core_effect_strict_given_matched"] = 0.0
        out["v1_core_composite_recall_effect"]    = 0.0
        out["v1_core_plus_refuted_composite"]     = 0.0
        out["v1_padding_rate"]                    = 0.0

    return out


def component_names() -> list[str]:
    """Canonical ordered list of wandb component keys (must match compute_components output)."""
    names = [
        "format_pass",
        *[f"format_fail_reason_{r}" for r in FORMAT_FAIL_REASONS],
        "completion_token_count",
        "length_penalty", "length_penalty_v3",
        "m_core", "m_refuted", "gt_core", "gt_refuted",
        "count_distance_core", "count_distance_refuted",
        "count_overshoot_core", "count_overshoot_refuted",
        "count_penalty_core", "count_penalty_refuted", "count_penalty_total",
        "count_overshoot_refuted_v3",
        "count_penalty_core_v3", "count_penalty_refuted_v3", "count_penalty_total_v3",
        "count_penalty_core_v4", "count_penalty_refuted_v4", "count_penalty_total_v4",
        "core_lookup_sum", "refuted_lookup_sum", "total_lookup_sum",
        "core_lookup_sum_v3", "refuted_lookup_sum_v3", "total_lookup_sum_v3",
        "n_unmatched_model_core", "n_unmatched_model_refuted",
        "core_n_items_total", "refuted_n_items_total",
    ]
    for (mq, ea) in ALL_LOOKUP_CELLS:
        names.append(f"core_n_items_{mq}_{ea}")
        names.append(f"refuted_n_items_{mq}_{ea}")
    names += [
        "v1_core_recall_strict",
        "v1_core_recall_lenient",
        "v1_core_effect_strict_given_matched",
        "v1_core_composite_recall_effect",
        "v1_core_plus_refuted_composite",
        "v1_padding_rate",
    ]
    return names


COMPONENT_NAMES: list[str] = component_names()
