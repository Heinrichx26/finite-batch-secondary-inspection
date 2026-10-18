"""Exact finite-population kernels for uniform review and inspection limits.

All targets in this kernel have a potential positive referral. A response type
that cannot refer every target has zero joint success and can be handled by
the caller. Internal randomisation is uniform sampling without replacement.
There is no source-data dependency and no plotting or experiment execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb
from operator import index
from typing import Iterable, Sequence


def _integer(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, excluding booleans")
    try:
        return index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _parameters(N: int, m: int, k: int, n: int, z: int) -> tuple[int, ...]:
    N, m, k, n, z = (
        _integer(name, value)
        for name, value in zip(("N", "m", "k", "n", "z"), (N, m, k, n, z))
    )
    if N < 0 or not 0 <= m <= N or not 0 <= k <= N - m:
        raise ValueError("Require N >= 0, 0 <= m <= N, 0 <= k <= N-m")
    if not 0 <= n <= N or z < 0:
        raise ValueError("Require 0 <= n <= N and z >= 0")
    return N, m, k, n, z


def target_inclusion_probability_exact(N: int, m: int, n: int) -> Fraction:
    """Probability that all m targets belong to an n-item uniform sample."""
    N, m, _, n, _ = _parameters(N, m, 0, n, 0)
    return Fraction(comb(n, m), comb(N, m)) if n >= m else Fraction(0)


def fixed_sample_success_exact(N: int, m: int, k: int, n: int, z: int) -> Fraction:
    """Joint probability of referring every target and using at most z searches.

    N: population size; m: target count; k: non-target potential referrals;
    n: fixed number reviewed; z: remaining physical inspection limit.
    The returned Fraction is exact, including empty-population boundaries.
    Detection of referred targets is integrated separately by the caller.
    """
    N, m, k, n, z = _parameters(N, m, k, n, z)
    if n < m or z < m:
        return Fraction(0)
    selected_other = n - m
    low = max(0, selected_other - (N - m - k))
    high = min(k, selected_other, z - m)
    favourable = sum(
        comb(k, referrals) * comb(N - m - k, selected_other - referrals)
        for referrals in range(low, high + 1)
    )
    return Fraction(favourable, comb(N, n))


def fixed_sample_success(N: int, m: int, k: int, n: int, z: int) -> float:
    """Floating-point interface to the exact finite-population kernel."""
    return float(fixed_sample_success_exact(N, m, k, n, z))


@dataclass(frozen=True)
class EffectDecomposition:
    target_inclusion: Fraction
    partial_success: Fraction
    full_success: Fraction
    capacity_relief: Fraction
    coverage_loss: Fraction
    effect: Fraction


def fixed_effect_decomposition_exact(
    N: int, m: int, k: int, n: int, z: int
) -> EffectDecomposition:
    """Exact partial-minus-full effect as capacity relief minus coverage loss."""
    N, m, k, n, z = _parameters(N, m, k, n, z)
    inclusion = target_inclusion_probability_exact(N, m, n)
    partial = fixed_sample_success_exact(N, m, k, n, z)
    full = Fraction(int(m + k <= z))
    relief = partial if m + k > z else Fraction(0)
    loss = 1 - inclusion if m + k <= z else Fraction(0)
    return EffectDecomposition(inclusion, partial, full, relief, loss, partial - full)


def _weight(value: object) -> Fraction:
    if isinstance(value, Fraction):
        return value
    # Decimal text preserves an explicitly supplied decimal mixture weight.
    return Fraction(str(value))


def mixture_success_exact(
    N: int,
    m: int,
    empty_referrals: Sequence[int],
    weights: Sequence[object],
    n: int,
    z: int,
) -> Fraction:
    """Success under an arbitrary finite distribution of potential referrals."""
    if len(empty_referrals) != len(weights) or len(weights) == 0:
        raise ValueError("Referral types and weights must have equal positive lengths")
    probabilities = tuple(_weight(value) for value in weights)
    if any(value < 0 for value in probabilities) or sum(probabilities) != 1:
        raise ValueError("Weights must be nonnegative and sum exactly to one")
    return sum(
        (
            probability * fixed_sample_success_exact(N, m, k, n, z)
            for k, probability in zip(empty_referrals, probabilities)
        ),
        start=Fraction(0),
    )


def _referral_distribution(
    k_values: Sequence[int], weights: Sequence[object], max_k: int | None = None
) -> dict[int, Fraction]:
    """Aggregate a possibly defective distribution of target-complete types."""
    if len(k_values) != len(weights) or len(weights) == 0:
        raise ValueError("Referral types and weights must have equal positive lengths")
    distribution: dict[int, Fraction] = {}
    for value, raw_weight in zip(k_values, weights):
        k = _integer("k", value)
        probability = _weight(raw_weight)
        if k < 0 or (max_k is not None and k > max_k) or probability < 0:
            raise ValueError("Referral counts and weights must be nonnegative and feasible")
        distribution[k] = distribution.get(k, Fraction(0)) + probability
    total = sum(distribution.values(), start=Fraction(0))
    if total > 1:
        raise ValueError("Target-complete response weights must sum to at most one")
    return {k: probability for k, probability in distribution.items() if probability > 0}


@dataclass(frozen=True)
class NearFullCriterion:
    x: int
    boundary_probability: Fraction
    cumulative_probability: Fraction
    capacity_relief: Fraction
    coverage_loss: Fraction
    effect: Fraction


def near_full_criterion_exact(
    N: int, m: int, k_values: Sequence[int], weights: Sequence[object], z: int
) -> NearFullCriterion:
    """Closed-form effect for reviewing N-1 items against reviewing all N.

    Put x=z-m+1 and let p_x be the mass of response types with k=x
    non-target referrals. The effect is [x*p_x-m*F(x-1)]/N. Weights may
    sum to less than one: the remaining mass represents types where at
    least one target cannot be referred, whose joint success is zero.
    """
    N, m, _, _, z = _parameters(N, m, 0, 0, z)
    if N == 0:
        raise ValueError("Near-full review requires N >= 1")
    distribution = _referral_distribution(k_values, weights, N - m)
    x = z - m + 1
    boundary = distribution.get(x, Fraction(0))
    cumulative = sum((probability for k, probability in distribution.items() if k < x),
                     start=Fraction(0))
    relief = Fraction(x, N) * boundary
    loss = Fraction(m, N) * cumulative
    return NearFullCriterion(x, boundary, cumulative, relief, loss, relief - loss)


def near_full_effect_exact(
    N: int, m: int, k_values: Sequence[int], weights: Sequence[object], z: int
) -> Fraction:
    """Return the exact N-1 versus N review effect from its boundary criterion."""
    return near_full_criterion_exact(N, m, k_values, weights, z).effect


def single_crossing_condition(k_values: Sequence[int], weights: Sequence[object]) -> bool:
    """Check a sufficient condition for positive-to-negative effect crossings.

    Positive support must be {0,...,M}, and a_x=x*p_x/p_(x-1) must be
    nonincreasing for x=1,...,M. For positive target count this ensures
    that near-full effects, ordered by capacity, have no negative-to-
    positive change. This is a sufficient distribution condition.
    """
    distribution = _referral_distribution(k_values, weights)
    if not distribution or set(distribution) != set(range(max(distribution) + 1)):
        return False
    ratios = [Fraction(x) * distribution[x] / distribution[x - 1]
              for x in range(1, max(distribution) + 1)]
    return all(left >= right for left, right in zip(ratios, ratios[1:]))


def feasible_review_counts(
    N: int,
    m: int,
    empty_referrals: Sequence[int],
    weights: Sequence[object],
    z: int,
    threshold: object,
) -> tuple[int, ...]:
    """Return the complete feasible set, without assuming it is an interval."""
    N, m, _, _, z = _parameters(N, m, 0, 0, z)
    target = _weight(threshold)
    if not 0 <= target <= 1:
        raise ValueError("Reliability threshold must lie in [0, 1]")
    return tuple(
        n for n in range(N + 1)
        if mixture_success_exact(N, m, empty_referrals, weights, n, z) >= target
    )


def repeated_reversal_rows(L: int) -> list[dict[str, object]]:
    """Construct 2L-1 strict order reversals as the search limit increases.

    N=2L, m=1, n=N-1. There are L equally likely response types, with
    k=1,3,...,2L-1 non-target potential referrals. At z=2j-1 the partial
    effect is j/(LN); at z=2j it is -j/(LN).
    """
    L = _integer("L", L)
    if L < 1:
        raise ValueError("L must be positive")
    N = 2 * L
    referral_types = tuple(2 * r - 1 for r in range(1, L + 1))
    weights = (Fraction(1, L),) * L
    rows = []
    for z in range(1, N + 1):
        partial = mixture_success_exact(N, 1, referral_types, weights, N - 1, z)
        full = mixture_success_exact(N, 1, referral_types, weights, N, z)
        j = (z + 1) // 2
        expected = Fraction(j if z % 2 else -j, L * N)
        rows.append(dict(L=L, N=N, m=1, n=N - 1, z=z,
                         partial_success=partial, full_success=full,
                         effect=partial - full, expected_effect=expected))
    return rows


def disconnected_review_examples() -> tuple[dict[str, object], ...]:
    """Small explicit examples of a disconnected reliability-feasible set.

    N=3 is minimal for positive target count and arbitrary response weights:
    with N<=2 there are only two positive review counts and n=0 fails.
    The N=4 case is minimal among equal-weight, two-type mixtures, as can
    also be checked exhaustively by the companion validation program.
    """
    return (
        dict(name="minimum_population", N=3, m=1, z=1,
             empty_referrals=(0, 2), weights=(Fraction(2, 5), Fraction(3, 5)),
             threshold=Fraction(3, 10), expected_feasible=(1, 3)),
        dict(name="minimum_equal_two_type", N=4, m=1, z=2,
             empty_referrals=(0, 3), weights=(Fraction(1, 2), Fraction(1, 2)),
             threshold=Fraction(1, 2), expected_feasible=(2, 4)),
        dict(name="specified_five_item_smoke", N=5, m=1, z=2,
             empty_referrals=(0, 4), weights=(Fraction(1, 2), Fraction(1, 2)),
             threshold=Fraction(2, 5), expected_feasible=(2, 4, 5)),
    )


@dataclass(frozen=True)
class JointFeasibilityExample:
    N: int
    m: int
    z: int
    empty_referrals: tuple[int, int]
    weights: tuple[Fraction, Fraction]
    reliability_target: Fraction
    miss_limit: Fraction
    expected_feasible: tuple[int, int]


def high_reliability_disconnected_example(N: int) -> JointFeasibilityExample:
    """Disconnected review counts at arbitrarily high joint reliability.

    One target is always referable. A response type with no non-target
    referrals has probability (N-2)/N; the other type refers every item.
    With z=N-2, only review counts N-2 and N meet reliability (N-2)/N
    and mean target-miss limit 2/N. Detection is perfect in this example.
    """
    N = _integer("N", N)
    if N < 3:
        raise ValueError("This construction requires N >= 3")
    p = Fraction(N - 2, N)
    return JointFeasibilityExample(N=N, m=1, z=N - 2,
                                  empty_referrals=(0, N - 1), weights=(p, 1 - p),
                                  reliability_target=p, miss_limit=Fraction(2, N),
                                  expected_feasible=(N - 2, N))


def high_reliability_family_success_exact(N: int, n: int) -> Fraction:
    """Closed-form success for the high-reliability disconnected family."""
    example = high_reliability_disconnected_example(N)
    N, _, _, n, _ = _parameters(N, 1, 0, n, example.z)
    if n <= N - 2:
        return Fraction(n, N)
    return example.weights[0] * Fraction(n, N)


def high_reliability_family_miss_exact(N: int, n: int) -> Fraction:
    """Exact mean target-miss fraction, with every potential referral detected."""
    example = high_reliability_disconnected_example(N)
    N, _, _, n, _ = _parameters(N, 1, 0, n, example.z)
    return Fraction(N - n, N)


def has_internal_gap(values: Iterable[int]) -> bool:
    ordered = sorted(set(values))
    return bool(ordered) and ordered != list(range(ordered[0], ordered[-1] + 1))
