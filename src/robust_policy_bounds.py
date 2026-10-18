"""Finite-policy reliability bounds with common unknown response coupling.

Transportation and minimax regret are established tools. This module makes their
assumptions, witnesses and numerical primal-dual checks explicit. Policy kernels
may come from exact finite-population calculations or simultaneous simulation
envelopes. Source-population uncertainty requires a separately specified set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linprog

LP_OPTIONS = {"primal_feasibility_tolerance": 1e-10,
              "dual_feasibility_tolerance": 1e-10,
              "ipm_optimality_tolerance": 1e-11}


@dataclass
class TransportCertificate:
    value: float
    coupling: np.ndarray
    primal_dual_gap: float
    constraint_error: float


class CouplingSet:
    """Arbitrary finite marginal weights and optional observed moment bands."""

    def __init__(self, row_weights, column_weights, moment_bands=()):
        self.p = np.asarray(row_weights, dtype=float)
        self.q = np.asarray(column_weights, dtype=float)
        for weights in (self.p, self.q):
            if weights.ndim != 1 or np.any(weights < 0) or not np.isfinite(weights).all():
                raise ValueError("Marginal weights must be finite nonnegative vectors")
            if not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0):
                raise ValueError("Each marginal must sum to one")
        self.shape = (len(self.p), len(self.q))
        size = int(np.prod(self.shape))
        eq = []
        for i in range(len(self.p)):
            row = np.zeros(self.shape); row[i, :] = 1.; eq.append(row.ravel())
        for j in range(len(self.q)):
            row = np.zeros(self.shape); row[:, j] = 1.; eq.append(row.ravel())
        self.a_eq = np.asarray(eq)
        self.b_eq = np.r_[self.p, self.q]
        a_ub, b_ub = [], []
        for matrix, lower, upper in moment_bands:
            matrix = self._matrix(matrix)
            if lower > upper:
                raise ValueError("Inconsistent moment band")
            if np.isfinite(upper):
                a_ub.append(matrix.ravel()); b_ub.append(float(upper))
            if np.isfinite(lower):
                a_ub.append(-matrix.ravel()); b_ub.append(float(-lower))
        self.a_ub = np.asarray(a_ub).reshape((-1, size)) if a_ub else None
        self.b_ub = np.asarray(b_ub) if b_ub else None

    def _matrix(self, value):
        value = np.asarray(value, dtype=float)
        if value.shape != self.shape or not np.isfinite(value).all():
            raise ValueError(f"Expected finite matrix with shape {self.shape}")
        return value

    def extreme(self, matrix, maximize=False):
        matrix = self._matrix(matrix)
        objective = matrix.ravel() * (-1. if maximize else 1.)
        result = linprog(objective, A_ub=self.a_ub, b_ub=self.b_ub,
                         A_eq=self.a_eq, b_eq=self.b_eq, bounds=(0., None),
                         method="highs", options=LP_OPTIONS)
        if not result.success:
            raise ValueError(f"Coupling problem failed: {result.message}")
        dual = float(self.b_eq @ result.eqlin.marginals)
        error = float(np.max(np.abs(self.a_eq @ result.x - self.b_eq)))
        if self.b_ub is not None:
            dual += float(self.b_ub @ result.ineqlin.marginals)
            error = max(error, float(np.max(self.a_ub @ result.x - self.b_ub)))
        error = max(error, float(max(0., -result.x.min())))
        gap = abs(float(result.fun) - dual)
        if error > 1e-7 or gap > 1e-7:
            raise ArithmeticError("Primal-dual numerical check failed")
        return TransportCertificate(float(matrix.ravel() @ result.x),
                                    result.x.reshape(self.shape), gap, error)

    def bounds(self, matrix):
        return self.extreme(matrix), self.extreme(matrix, maximize=True)

    def paired_effect(self, first, second):
        return self.bounds(self._matrix(first) - self._matrix(second))

    def interaction(self, matrix):
        """Remove additive row/column terms, whose expectation is identified."""
        matrix = self._matrix(matrix)
        mean = float(self.p @ matrix @ self.q)
        interaction = matrix - (matrix @ self.q)[:, None] - (self.p @ matrix)[None, :] + mean
        return mean, interaction


def _kernel_arrays(model, lower, upper=None):
    lower = np.asarray(lower, dtype=float)
    upper = lower.copy() if upper is None else np.asarray(upper, dtype=float)
    if lower.ndim != 3 or lower.shape[1:] != model.shape or upper.shape != lower.shape:
        raise ValueError("Kernels must have shape (policies, rows, columns)")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper + 1e-12):
        raise ValueError("Invalid kernel envelopes")
    return lower, upper


def regret_certificate(model, weights, lower, upper=None, comparator_indices=None):
    """Worst regret over one common coupling and optional rectangular envelope.

    The comparator is the best member of the supplied finite policy catalogue.
    The same policy cancels exactly, including when its kernel is uncertain.
    """
    lower, upper = _kernel_arrays(model, lower, upper)
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(lower) or np.any(weights < -1e-10) or abs(weights.sum()-1.) > 1e-8:
        raise ValueError("Invalid policy probabilities")
    indices = list(range(len(lower))) if comparator_indices is None else list(comparator_indices)
    if not indices or any(not 0 <= b < len(lower) for b in indices):
        raise ValueError("A nonempty valid comparator catalogue is required")
    best = None
    for b in indices:
        other = lower.copy(); other[b] = upper[b]
        certificate = model.extreme(upper[b] - np.tensordot(weights, other, axes=1), maximize=True)
        if best is None or certificate.value > best[0]:
            best = (certificate.value, b, certificate)
    return max(0., best[0]), best[1], best[2]


def minimax_regret(model, lower, upper=None, randomized=True,
                   metric_limits: Sequence[tuple[np.ndarray, float]] = (),
                   tolerance=1e-9, maximum_iterations=300, comparator_indices=None):
    """Cutting-plane solution, with finite-policy regret lower/upper certificate.

    Each metric constraint requires sup_Q E_Q[sum_a w_a metric_a] <= limit.
    Metrics represent costs/risks only when their kernels have been validated.
    Comparators are the pure policies satisfying the same metric limits.
    A mixture may include a costly policy if its aggregate expected use is
    feasible. The comparator scope remains the feasible deterministic catalogue.
    """
    lower, upper = _kernel_arrays(model, lower, upper)
    count = len(lower)
    metrics = []
    for matrix, limit in metric_limits:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != lower.shape or not np.isfinite(matrix).all():
            raise ValueError("Metric kernels must match policy kernels")
        metrics.append((matrix, float(limit)))
    feasible_pure = [a for a in range(count)
                     if all(model.extreme(matrix[a], True).value <= limit + tolerance
                            for matrix, limit in metrics)]
    comparators = feasible_pure if comparator_indices is None else list(comparator_indices)
    if not comparators or any(a not in feasible_pure for a in comparators):
        raise ValueError("Comparators must be feasible pure policies under the common limits")
    if not randomized:
        options = []
        for a in feasible_pure:
            weights = np.eye(count)[a]
            value, comparator, certificate = regret_certificate(model, weights, lower, upper, comparators)
            options.append((value, a, comparator, certificate))
        if not options:
            raise ValueError("No feasible deterministic policy in the catalogue")
        value, a, comparator, certificate = min(options, key=lambda item: item[:2])
        return dict(weights=np.eye(count)[a], lower_bound=value, upper_bound=value,
                    gap=0., iterations=count, comparator=comparator,
                    witness=certificate.coupling, largest_transport_gap=certificate.primal_dual_gap,
                    comparator_indices=comparators)

    constraints, limits = [], []
    seed = model.extreme(np.zeros(model.shape)).coupling
    largest_gap = 0.

    def add_regret_cut(comparator, coupling):
        other = lower.copy(); other[comparator] = upper[comparator]
        values = np.einsum("aij,ij->a", other, coupling)
        constraints.append(np.r_[-values, -1.])
        limits.append(-float(np.sum(upper[comparator] * coupling)))

    def add_metric_cut(matrix, limit, coupling):
        constraints.append(np.r_[np.einsum("aij,ij->a", matrix, coupling), 0.])
        limits.append(limit)

    for b in comparators:
        add_regret_cut(b, seed)
    for matrix, limit in metrics:
        add_metric_cut(matrix, limit, seed)
    best_upper, best_weights, best_witness, best_comparator = np.inf, None, None, None
    for iteration in range(1, maximum_iterations + 1):
        result = linprog(np.r_[np.zeros(count), 1.],
                         A_ub=np.asarray(constraints), b_ub=np.asarray(limits),
                         A_eq=np.array([np.r_[np.ones(count), 0.]]), b_eq=[1.],
                         bounds=[(0., 1.)] * count + [(0., None)], method="highs", options=LP_OPTIONS)
        if not result.success:
            raise ValueError(f"Policy master problem failed: {result.message}")
        weights = np.maximum(0., result.x[:count]); weights /= weights.sum()
        regret, comparator, certificate = regret_certificate(model, weights, lower, upper, comparators)
        largest_gap = max(largest_gap, certificate.primal_dual_gap)
        feasible = True
        for matrix, limit in metrics:
            metric = model.extreme(np.tensordot(weights, matrix, axes=1), True)
            largest_gap = max(largest_gap, metric.primal_dual_gap)
            if metric.value > limit + tolerance:
                feasible = False
                add_metric_cut(matrix, limit, metric.coupling)
        if feasible and regret < best_upper:
            best_upper, best_weights = regret, weights.copy()
            best_witness, best_comparator = certificate.coupling, comparator
        bound = max(0., float(result.fun))
        if best_weights is not None and bound > best_upper + tolerance:
            raise ArithmeticError("Master lower bound exceeds feasible regret upper bound")
        if best_weights is not None and best_upper - bound <= tolerance:
            return dict(weights=best_weights, lower_bound=bound, upper_bound=best_upper,
                        gap=max(0., best_upper-bound), iterations=iteration,
                        comparator=best_comparator, witness=best_witness,
                        largest_transport_gap=largest_gap, comparator_indices=comparators)
        add_regret_cut(comparator, certificate.coupling)
    raise ArithmeticError("Regret solver did not reach the requested certificate gap")
