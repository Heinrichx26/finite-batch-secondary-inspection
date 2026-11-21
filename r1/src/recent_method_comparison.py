"""Explicit core-method reproductions and common-task adaptations.

Sources: Maisano et al. (2026), 10.1002/qre.70291, Eqs. 9 and 14--17;
Yang et al. (2024), 10.1016/j.ress.2024.110240, Algorithms 1--2;
Li et al. (2026), 10.1016/j.ress.2026.112187, Section 2.2.2.
The source studies have different systems and objectives. These functions
reproduce named mathematical cores and state their finite-batch adaptations.
No experiment, document edit, or plotting is performed at import time.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import itertools
from math import comb

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import binom, hypergeom


MAISANO_RATES = {
    "L": dict(false_positive=(.053, .043, .065, .055, .029, .041, .031),
              false_negative=(.026, .042, .060, .036, .047, .024, .030)),
    "H": dict(false_positive=(.174, .134, .141, .136, .132, .155, .255),
              false_negative=(.132, .230, .168, .229, .197, .213, .156)),
}


def poisson_binomial(probabilities):
    """Distribution of independent heterogeneous Bernoulli successes."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or not len(probabilities) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Provide a nonempty vector of Bernoulli probabilities")
    values = np.array([1.])
    for probability in probabilities:
        values = np.convolve(values, [1. - probability, probability])
    return values


def majority_errors(false_positive, false_negative):
    """Maisano Eqs. 16--17, for odd panels and independent inspectors."""
    a = np.asarray(false_positive, dtype=float)
    b = np.asarray(false_negative, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) % 2 != 1:
        raise ValueError("The inspector panel must have odd positive size")
    threshold = len(a) // 2 + 1
    # With odd panels, a majority of error votes is equivalent to failure
    # of the paper's majority-conformity rule.
    return float(poisson_binomial(a)[threshold:].sum()), float(poisson_binomial(b)[threshold:].sum())


def majority_errors_by_subsets(false_positive, false_negative):
    """Independent literal subset sums of the source equations."""
    result = []
    for probabilities in (false_positive, false_negative):
        values = np.asarray(probabilities, dtype=float)
        threshold = len(values) // 2 + 1
        total = 0.
        for votes in itertools.product((False, True), repeat=len(values)):
            if sum(votes) >= threshold:
                total += float(np.prod(np.where(votes, values, 1. - values)))
        result.append(total)
    return tuple(result)


def type_b_acceptance(n, acceptance_number, prevalence, false_positive, false_negative):
    """Maisano Eqs. 14--15: binomial Type-B lot-acceptance probability."""
    apparent = (1. - false_negative) * prevalence + false_positive * (1. - prevalence)
    return float(binom.cdf(acceptance_number, n, apparent))


def full_lot_acceptance(N, defectives, acceptance_number, false_positive, false_negative):
    """Maisano Eq. 9: convolution of apparent true and false defectives."""
    true_count = binom.pmf(np.arange(defectives + 1), defectives, 1. - false_negative)
    false_count = binom.pmf(np.arange(N - defectives + 1), N - defectives, false_positive)
    return float(np.convolve(true_count, false_count)[:acceptance_number + 1].sum())


def panel_batch_success(N, targets, reviews, searches, false_positive, false_negative, detection=1.):
    """Common-task adaptation: capture all targets and meet a search limit.

    Independent panel decisions on different items are assumed. Each sampled
    target must receive a majority-positive decision. False referrals among
    the sampled non-targets consume downstream search capacity. This endpoint
    extends the source error aggregation; it changes its lot-acceptance rule.
    """
    N, m, n, z = map(int, (N, targets, reviews, searches))
    if not 0 <= m <= N or not 0 <= n <= N or z < 0:
        raise ValueError("Invalid finite-batch parameters")
    if n < m or z < m:
        return 0.
    inclusion = comb(n, m) / comb(N, m)
    return float(inclusion * ((1. - false_negative) * detection) ** m
                 * binom.cdf(z - m, n - m, false_positive))


def panel_expected_miss(N, targets, reviews, false_negative, detection=1.):
    """Fraction missed in the complete referral chain, before capacity testing."""
    if targets == 0:
        return 0.
    return float(1. - reviews / N * (1. - false_negative) * detection)


class OrderedBDD:
    """Canonical Shannon graph with shared variable identities.

    The source phase conjunction and success-path representation are retained.
    The probability recursion uses fixed-cardinality selection probabilities,
    replacing the source component-degradation probability model.
    """

    def __init__(self, size):
        self.size = int(size)
        self.nodes = [None, None]
        self.unique = {}
        self.apply_cache = {}

    def node(self, variable, low, high):
        if low == high:
            return low
        key = (variable, low, high)
        if key not in self.unique:
            self.unique[key] = len(self.nodes)
            self.nodes.append(key)
        return self.unique[key]

    def conjunction(self, first, second):
        if first == 0 or second == 0:
            return 0
        if first == 1:
            return second
        if second == 1 or first == second:
            return first
        key = tuple(sorted((first, second)))
        if key in self.apply_cache:
            return self.apply_cache[key]
        v1, lo1, hi1 = self.nodes[first]
        v2, lo2, hi2 = self.nodes[second]
        variable = min(v1, v2)
        low = self.conjunction(lo1 if v1 == variable else first,
                               lo2 if v2 == variable else second)
        high = self.conjunction(hi1 if v1 == variable else first,
                                hi2 if v2 == variable else second)
        value = self.node(variable, low, high)
        self.apply_cache[key] = value
        return value

    def required(self, variables):
        root = 1
        for variable in reversed(sorted(variables)):
            root = self.node(variable, 0, root)
        return root

    def at_most(self, variables, allowance):
        variables = tuple(sorted(variables))

        @lru_cache(None)
        def visit(position, remaining):
            if remaining < 0:
                return 0
            if len(variables) - position <= remaining:
                return 1
            return self.node(variables[position], visit(position + 1, remaining),
                             visit(position + 1, remaining - 1))

        return visit(0, allowance)

    def probability(self, root, reviews):
        @lru_cache(None)
        def visit(position, node, remaining):
            available = self.size - position
            if remaining < 0 or remaining > available:
                return 0.
            if node < 2:
                return float(node)
            variable, low, high = self.nodes[node]
            assert variable >= position
            selected_probability = remaining / available
            # Skipped variables still consume or preserve one selection slot.
            return ((1. - selected_probability) * visit(position + 1, low if variable == position else node, remaining)
                    + selected_probability * visit(position + 1, high if variable == position else node, remaining - 1))

        value = visit(0, root, int(reviews))
        return float(value), visit.cache_info().currsize

    def accepts(self, root, selected):
        while root > 1:
            variable, low, high = self.nodes[root]
            root = high if variable in selected else low
        return bool(root)


def phase_graph_kernel(N, targets, empty_referrals, reviews, search_limit):
    graph = OrderedBDD(N)
    coverage = graph.required(range(targets))
    capacity = graph.at_most(range(targets + empty_referrals), search_limit)
    root = graph.conjunction(coverage, capacity)
    probability, states = graph.probability(root, reviews)
    return probability, len(graph.nodes) - 2, states


def _base_copula(u, v, family, parameter):
    if u <= 0 or v <= 0:
        return 0.
    if u >= 1:
        return float(v)
    if v >= 1:
        return float(u)
    if abs(parameter) < 1e-10:
        return float(u * v)
    if family == "Clayton":
        if parameter < 0:
            raise ValueError("Positive Clayton parameters are used in this comparison")
        return float((u ** (-parameter) + v ** (-parameter) - 1.) ** (-1. / parameter))
    if family == "Frank":
        return float(-np.log1p(np.expm1(-parameter * u) * np.expm1(-parameter * v)
                              / np.expm1(-parameter)) / parameter)
    raise ValueError(f"Unsupported copula: {family}")


def copula_probability_matrix(row_weights, column_weights, family, parameter, rotation=0):
    """Rectangle probabilities: a bivariate special case of a vine law."""
    p, q = np.asarray(row_weights, float), np.asarray(column_weights, float)
    if np.any(p < 0) or np.any(q < 0) or not np.isclose(p.sum(), 1) or not np.isclose(q.sum(), 1):
        raise ValueError("Marginal weights must be probabilities")
    if rotation not in (0, 90, 180, 270):
        raise ValueError("Unsupported rotation")

    def cdf(u, v):
        if rotation == 0:
            return _base_copula(u, v, family, parameter)
        if rotation == 90:
            return v - _base_copula(1 - u, v, family, parameter)
        if rotation == 180:
            return u + v - 1 + _base_copula(1 - u, 1 - v, family, parameter)
        return u - _base_copula(u, 1 - v, family, parameter)

    up = np.r_[0., np.cumsum(p)]; vp = np.r_[0., np.cumsum(q)]
    up[-1] = vp[-1] = 1.
    c = np.array([[cdf(u, v) for v in vp] for u in up])
    result = np.diff(np.diff(c, axis=0), axis=1)
    if result.min() < -2e-10:
        raise ArithmeticError("Negative copula rectangle mass")
    result = np.maximum(result, 0.)
    if max(abs(result.sum(1) - p).max(), abs(result.sum(0) - q).max()) > 2e-9:
        raise ArithmeticError("Copula marginal check failed")
    return result


@dataclass(frozen=True)
class CopulaFamily:
    name: str
    family: str
    bounds: tuple[float, float]
    rotation: int = 0


FAMILIES = (
    CopulaFamily("Frank", "Frank", (-20., 20.)),
    CopulaFamily("Clayton", "Clayton", (0., 10.)),
    CopulaFamily("Clayton90", "Clayton", (0., 10.), 90),
    CopulaFamily("Clayton180", "Clayton", (0., 10.), 180),
    CopulaFamily("Clayton270", "Clayton", (0., 10.), 270),
)


def continuous_family_extrema(payoff, p, q, specification, scan_points=81):
    """One-dimensional sampled-function optimization with local refinement.

    All scan points and each apparent local extremum are evaluated. This is
    a specified numerical search over the stated copula interval, with no
    claim of an analytic global optimizer for arbitrary response kernels.
    """
    payoff = np.asarray(payoff, float)
    evaluations = 0

    def objective(parameter):
        nonlocal evaluations
        evaluations += 1
        return float(np.sum(payoff * copula_probability_matrix(p, q, specification.family,
                                                               parameter, specification.rotation)))

    points = np.linspace(*specification.bounds, scan_points)
    values = np.array([objective(value) for value in points])
    candidates = [(float(value), float(parameter)) for value, parameter in zip(values, points)]
    for sign in (1., -1.):
        for i in range(1, len(points) - 1):
            if sign * values[i] <= sign * values[i - 1] and sign * values[i] <= sign * values[i + 1]:
                fitted = minimize_scalar(lambda value: sign * objective(value),
                                         bounds=(points[i - 1], points[i + 1]), method="bounded",
                                         options={"xatol": 1e-8})
                candidates.append((objective(fitted.x), float(fitted.x)))
    lower = min(candidates); upper = max(candidates)
    return dict(lower=lower[0], upper=upper[0], lower_parameter=lower[1],
                upper_parameter=upper[1], law_evaluations=evaluations)


def sample_conditional_effect(rng, profiles, empty_referrals, target_complete,
                              N, targets, reviews, searches, detection):
    """Sample the common finite-batch partial-minus-full event, conditional on a type.

    All-target inclusion is Bernoulli with its exact finite-population
    probability. Conditional non-target referral demand is hypergeometric.
    Physical detection is integrated for both policies, giving a common
    multiplicative factor. There is no deadline simulation here.
    """
    profiles = np.asarray(profiles, int)
    k = np.asarray(empty_referrals, int)[profiles]
    complete = np.asarray(target_complete, bool)[profiles]
    if reviews < targets or searches < targets:
        partial = np.zeros(profiles.shape, bool)
    else:
        inclusion = comb(reviews, targets) / comb(N, targets)
        included = rng.random(profiles.shape) < inclusion
        extra = rng.hypergeometric(k, N - targets - k, reviews - targets)
        partial = complete & included & (targets + extra <= searches)
    full = complete & (targets + k <= searches)
    return detection * (partial.astype(float) - full.astype(float))
