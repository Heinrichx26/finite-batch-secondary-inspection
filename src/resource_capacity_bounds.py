"""Exact full-review resource bounds under fixed finite marginal load laws.

Nonnegative integer loads may contain +inf, which represents failure to refer
every target in a response group. Infinite loads remain unsuccessful for every
capacity, including capacity=+inf (the limit over finite capacities).

The Makarov formulas and transportation aggregation used here are established
mathematical results. These bounds condition on supplied marginal laws; they do
not cover uncertainty about the source population.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DiscreteLoad:
    values: np.ndarray
    weights: np.ndarray

    @property
    def finite_mass(self):
        selected = np.isfinite(self.values)
        return 1. if selected.all() else float(math.fsum(self.weights[selected]))

    def cdf(self, threshold, strict=False):
        selected = self.values < threshold if strict else self.values <= threshold
        selected &= np.isfinite(self.values)
        return 1. if selected.all() else float(math.fsum(self.weights[selected]))


@dataclass(frozen=True)
class SumBounds:
    lower: float
    independent: float
    upper: float
    lower_split: float | None
    upper_split: float | None
    lower_ceiling: float
    upper_ceiling: float


@dataclass(frozen=True)
class ResourceRequirement:
    feasible: bool
    minimum_capacity: int | None
    target: float
    detection: float
    lower_ceiling: float
    upper_ceiling: float
    first_split: int | None
    second_split: int | None


def compressed_load(values, weights):
    """Aggregate identical loads, retaining all positive marginal probability.

    For a kernel f(X_i,Y_j), aggregation preserves both transport extrema.
    Aggregate any original coupling over load classes. Conversely an aggregate
    coupling Q_ab lifts as Q_ab*p_i/P_a*q_j/Q_b within each nonzero class pair.
    This gives the original marginals and the same objective. Extra moment
    constraints require separate compatible aggregation.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.shape != values.shape or not len(values):
        raise ValueError("Loads and weights must be nonempty vectors of equal length")
    if np.any(np.isnan(values)) or np.any(values < 0):
        raise ValueError("Loads must be nonnegative integers or +inf")
    finite = np.isfinite(values)
    if np.any(values[finite] != np.floor(values[finite])):
        raise ValueError("Finite loads must be integers")
    if np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("Weights must be finite and nonnegative")
    if not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0.):
        raise ValueError("Marginal weights must sum to one")
    selected = weights > 0
    support, index = np.unique(values[selected], return_inverse=True)
    mass = np.bincount(index, weights=weights[selected], minlength=len(support))
    mass /= math.fsum(mass)
    return DiscreteLoad(support, mass)


def _capacity(value):
    value = float(value)
    if math.isnan(value) or value == -math.inf:
        raise ValueError("Capacity must be finite or +inf")
    return value


def sum_cdf_bounds(first_values, first_weights, second_values, second_weights, capacity):
    """Sharp P(X+Y<=c, X,Y finite), with the discrete boundary handled exactly.

    L(c)=max(0,max_x[F_X(x)+F_Y(c-x)-1]);
    U(c)=min(P(X finite),P(Y finite),min_x[F_X(x-)+F_Y(c-x)]).
    The split x ranges over finite support of X. The upper formula uses a left
    limit, essential at integer sums. Independent pairing is also returned.
    """
    first = compressed_load(first_values, first_weights)
    second = compressed_load(second_values, second_weights)
    capacity = _capacity(capacity)
    fx, fy = first.finite_mass, second.finite_mass
    low_ceiling, high_ceiling = max(0., fx + fy - 1.), min(fx, fy)
    if capacity == math.inf:
        return SumBounds(low_ceiling, fx*fy, high_ceiling, None, None,
                         low_ceiling, high_ceiling)
    lower, upper, low_split, high_split = 0., high_ceiling, None, None
    for value in first.values[np.isfinite(first.values)]:
        candidate = first.cdf(value) + second.cdf(capacity-value) - 1.
        if candidate > lower:
            lower, low_split = candidate, float(value)
        candidate = first.cdf(value, strict=True) + second.cdf(capacity-value)
        if candidate < upper:
            upper, high_split = candidate, float(value)
    finite = np.isfinite(first.values[:, None]) & np.isfinite(second.values[None, :])
    event = finite & (first.values[:, None]+second.values[None, :] <= capacity)
    independent = float(first.weights @ event @ second.weights)
    return SumBounds(float(np.clip(lower, 0., 1.)), independent,
                     float(np.clip(upper, 0., 1.)), low_split, high_split,
                     low_ceiling, high_ceiling)


def full_review_resource_bounds(first_values, first_weights, second_values,
                                second_weights, capacity, offset=0, detection=1.):
    """Multiply sharp finite-load bounds by the common detection factor.

    The offset contains original mandatory checks plus reviewed targets, while
    X and Y contain the two groups' non-target referrals (or +inf on failure).
    """
    if not 0 <= detection <= 1 or not math.isfinite(detection):
        raise ValueError("Detection factor must lie in [0,1]")
    if offset < 0 or int(offset) != offset:
        raise ValueError("Offset must be a nonnegative integer")
    bound = sum_cdf_bounds(first_values, first_weights, second_values,
                           second_weights, _capacity(capacity)-offset)
    return SumBounds(*(detection*getattr(bound, key)
                       for key in ("lower", "independent", "upper")),
                     bound.lower_split, bound.upper_split,
                     detection*bound.lower_ceiling, detection*bound.upper_ceiling)


def minimum_robust_capacity(first_values, first_weights, second_values,
                            second_weights, target, offset=0, detection=1.):
    """Necessary and sufficient finite capacity for a robust reliability target.

    For target>0, feasibility is target<=D*max(0,p_finite+q_finite-1).
    The minimum capacity is offset + min{x+y:F_X(x)+F_Y(y)>=1+target/D}.
    The resulting pair is an exact split certificate under the fixed marginals.
    For target=0 the minimum nonnegative capacity is defined as zero.
    """
    if not 0 <= target <= 1 or not math.isfinite(target):
        raise ValueError("Target must lie in [0,1]")
    first = compressed_load(first_values, first_weights)
    second = compressed_load(second_values, second_weights)
    ceilings = full_review_resource_bounds(first.values, first.weights,
        second.values, second.weights, math.inf, offset, detection)
    if target == 0:
        return ResourceRequirement(True, 0, target, detection,
            ceilings.lower_ceiling, ceilings.upper_ceiling, None, None)
    if detection == 0 or target > ceilings.lower_ceiling + 1e-12:
        return ResourceRequirement(False, None, target, detection,
            ceilings.lower_ceiling, ceilings.upper_ceiling, None, None)
    wanted = 1. + target/detection
    candidates = []
    for x in first.values[np.isfinite(first.values)]:
        for y in second.values[np.isfinite(second.values)]:
            if first.cdf(x)+second.cdf(y) >= wanted-1e-12:
                candidates.append((int(offset+x+y), int(x), int(y)))
    if not candidates:
        raise ArithmeticError("Feasible ceiling and split certificate disagree")
    capacity, x, y = min(candidates)
    return ResourceRequirement(True, capacity, target, detection,
        ceilings.lower_ceiling, ceilings.upper_ceiling, x, y)
