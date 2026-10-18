"""Finite search allocation under partial information about group targets."""

from __future__ import annotations

from fractions import Fraction
from itertools import product, combinations
from math import comb
from pathlib import Path
import argparse
import json

import numpy as np
from scipy.optimize import linprog


def compositions(capacities: tuple[int, ...], total: int) -> tuple[tuple[int, ...], ...]:
    return tuple(x for x in product(*(range(p + 1) for p in capacities)) if sum(x) == total)


def quotas(capacities: tuple[int, ...], allowance: int) -> tuple[tuple[int, ...], ...]:
    return tuple(x for x in product(*(range(p + 1) for p in capacities)) if sum(x) <= allowance)


def coverage_fraction(capacities: tuple[int, ...], theta: tuple[int, ...], x: tuple[int, ...]) -> Fraction:
    out = Fraction(1, 1)
    for p, m, s in zip(capacities, theta, x):
        if s < m:
            return Fraction(0, 1)
        out *= Fraction(comb(s, m), comb(p, m))
    return out


def recovery_fraction(theta: tuple[int, ...], capacities: tuple[int, ...], x: tuple[int, ...]) -> Fraction:
    total_targets = sum(theta)
    if total_targets == 0:
        return Fraction(1, 1)
    return sum((Fraction(m * s, p) for p, m, s in zip(capacities, theta, x)), Fraction(0, 1)) / total_targets


def primal_dual_value(capacities: tuple[int, ...], theta_set: tuple[tuple[int, ...], ...], allowance: int, risk_limit: float | None = None) -> dict:
    xs = quotas(capacities, allowance)
    # Primal: maximize t over a lottery on search quotas.
    nf = len(xs)
    c = np.zeros(nf + 1)
    c[-1] = -1.0
    A_ub = []
    b_ub = []
    for theta in theta_set:
        A_ub.append([-float(coverage_fraction(capacities, theta, x)) for x in xs] + [1.0])
        b_ub.append(0.0)
    if risk_limit is not None:
        for theta in theta_set:
            A_ub.append([-float(recovery_fraction(theta, capacities, x)) for x in xs] + [0.0])
            b_ub.append(-(1.0 - risk_limit))
    primal = linprog(c, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub),
                     A_eq=[([1.0] * nf) + [0.0]], b_eq=[1.0],
                     bounds=[(0.0, None)] * nf + [(0.0, 1.0)], method="highs")
    if primal.status == 2:
        return {"feasible": False}
    if not primal.success:
        raise RuntimeError(f"primal LP failed: {primal.message}")

    # Dual game: minimize the least-favorable mixture's best quota payoff.
    nt = len(theta_set)
    cd = np.zeros(nt + 1)
    cd[-1] = 1.0
    Ad = []
    bd = []
    for x in xs:
        Ad.append([float(coverage_fraction(capacities, theta, x)) for theta in theta_set] + [-1.0])
        bd.append(0.0)
    dual = linprog(cd, A_ub=np.asarray(Ad), b_ub=np.asarray(bd),
                   A_eq=[([1.0] * nt) + [0.0]], b_eq=[1.0],
                   bounds=[(0.0, None)] * nt + [(0.0, 1.0)], method="highs")
    if not dual.success:
        raise RuntimeError(f"dual LP failed: {dual.message}")
    if risk_limit is not None:
        return {"feasible": True, "value": float(-primal.fun), "quota_count": nf,
                "support": [{"quota": list(x), "probability": float(p)}
                            for x, p in zip(xs, primal.x[:-1]) if p > 1e-9]}
    return {"feasible": True, "value": float(-primal.fun), "dual_value": float(dual.fun),
            "gap": abs(float(-primal.fun - dual.fun)), "quota_count": nf,
            "least_favorable_mix": dual.x[:-1].tolist()}


def greedy_known_composition(capacities: tuple[int, ...], theta: tuple[int, ...], allowance: int) -> tuple[int, ...] | None:
    if sum(theta) > allowance:
        return None
    x = list(theta)
    while sum(x) < min(allowance, sum(capacities)):
        candidates = []
        for g, (p, m, s) in enumerate(zip(capacities, theta, x)):
            if s < p and m > 0:
                candidates.append((Fraction(s + 1, s + 1 - m), g))
        if not candidates:
            break
        _, g = max(candidates)
        x[g] += 1
    return tuple(x)


def greedy_mean_recovery(capacities: tuple[int, ...], theta: tuple[int, ...], allowance: int) -> tuple[int, ...]:
    x = [0] * len(capacities)
    for _ in range(min(allowance, sum(capacities))):
        options = [(Fraction(m, p), g) for g, (p, m, s) in enumerate(zip(capacities, theta, x)) if s < p]
        if not options:
            break
        _, g = max(options)
        x[g] += 1
    return tuple(x)


def baseline(capacities: tuple[int, ...], total_targets: int, allowance: int) -> Fraction:
    p = sum(capacities)
    z = min(allowance, p)
    if total_targets > z:
        return Fraction(0, 1)
    return Fraction(comb(z, total_targets), comb(p, total_targets))


def smoke() -> dict:
    capacities = (2, 3)
    theta = (1, 1)
    z = 2
    theta_set = (theta,)
    lp = primal_dual_value(capacities, theta_set, z)
    base = baseline(capacities, sum(theta), z)
    cover_allocations = [x for x in quotas(capacities, z)
                         if coverage_fraction(capacities, theta, x) == Fraction(1, 6)]
    recovery_alloc = greedy_mean_recovery(capacities, theta, z)
    assert lp["gap"] < 1e-9
    assert abs(lp["value"] - 1 / 6) < 1e-9
    assert base == Fraction(1, 10)
    assert cover_allocations == [(1, 1)]
    assert recovery_alloc == (2, 0)
    assert recovery_fraction(theta, capacities, recovery_alloc) == Fraction(1, 2)
    assert recovery_fraction(theta, capacities, (1, 1)) == Fraction(5, 12)

    full = compositions(capacities, sum(theta))
    full_value = primal_dual_value(capacities, full, z)
    assert abs(full_value["value"] - float(base)) < 1e-9
    frontier_checks = []
    for risk_limit in (0.49, 0.5, 0.52, 0.55, 7 / 12, 0.6):
        observed = primal_dual_value(capacities, theta_set, z, risk_limit=risk_limit)
        if risk_limit < 0.5:
            assert not observed["feasible"]
            expected = None
        else:
            expected = min(1 / 6, 2 * risk_limit - 1)
            assert abs(observed["value"] - expected) < 1e-9
        frontier_checks.append({"risk_limit": risk_limit, "expected": expected, "observed": observed})
    return {"capacities": capacities, "known_composition": theta,
            "search_allowance": z, "group_informed_value": lp["value"],
            "group_free_value": float(base), "full_uncertainty_value": full_value["value"],
            "coverage_optimal_quota": [1, 1], "mean_recovery_optimal_quota": list(recovery_alloc),
            "mean_recovery_coverage_quota": float(recovery_fraction(theta, capacities, (1, 1))),
            "mean_recovery_risk_quota": float(recovery_fraction(theta, capacities, recovery_alloc)),
            "primal_dual_gap": lp["gap"], "joint_frontier_checks": frontier_checks}


def validate(max_group_capacity: int = 4) -> dict:
    checked_singletons = 0
    checked_full_sets = 0
    checked_pairs = 0
    max_lp_gap = 0.0
    max_known_greedy_error = 0.0
    max_mean_greedy_error = 0.0
    strict_information_cases = 0
    for p1 in range(1, max_group_capacity + 1):
        for p2 in range(1, max_group_capacity + 1):
            capacities = (p1, p2)
            p_total = p1 + p2
            for total_targets in range(1, p_total + 1):
                states = compositions(capacities, total_targets)
                for z in range(0, p_total + 1):
                    full_lp = primal_dual_value(capacities, states, z)
                    expected = float(baseline(capacities, total_targets, z))
                    max_lp_gap = max(max_lp_gap, full_lp["gap"], abs(full_lp["value"] - expected))
                    checked_full_sets += 1
                    if full_lp["value"] > expected + 1e-9:
                        raise AssertionError("full composition uncertainty exceeded global-uniform minimax value")
                    for theta in states:
                        singleton_lp = primal_dual_value(capacities, (theta,), z)
                        max_lp_gap = max(max_lp_gap, singleton_lp["gap"])
                        if sum(theta) <= z:
                            greedy = greedy_known_composition(capacities, theta, z)
                            candidates = [x for x in quotas(capacities, z) if all(a >= b for a, b in zip(x, theta))]
                            optimum = max((coverage_fraction(capacities, theta, x) for x in candidates), default=Fraction(0, 1))
                            err = abs(float(coverage_fraction(capacities, theta, greedy) - optimum))
                            max_known_greedy_error = max(max_known_greedy_error, err)
                            if err > 1e-12:
                                raise AssertionError("known-composition greedy did not maximize coverage")
                        candidate_allocations = [x for x in quotas(capacities, z)]
                        min_risk = min((1 - recovery_fraction(theta, capacities, x) for x in candidate_allocations), default=Fraction(1, 1))
                        risk_alloc = greedy_mean_recovery(capacities, theta, z)
                        risk_error = abs(float((1 - recovery_fraction(theta, capacities, risk_alloc)) - min_risk))
                        max_mean_greedy_error = max(max_mean_greedy_error, risk_error)
                        if risk_error > 1e-12:
                            raise AssertionError("mean-recovery greedy did not minimize mean risk")
                        checked_singletons += 1
                    for theta_pair in combinations(states, 2):
                        pair_lp = primal_dual_value(capacities, theta_pair, z)
                        max_lp_gap = max(max_lp_gap, pair_lp["gap"])
                        if pair_lp["value"] > expected + 1e-9:
                            strict_information_cases += 1
                        checked_pairs += 1
    return {"max_group_capacity": max_group_capacity,
            "checked_full_uncertainty_games": checked_full_sets,
            "checked_known_compositions": checked_singletons,
            "checked_two_composition_games": checked_pairs,
            "strict_improvement_cases": strict_information_cases,
            "max_primal_dual_or_full_set_gap": max_lp_gap,
            "max_known_composition_greedy_error": max_known_greedy_error,
            "max_mean_recovery_greedy_error": max_mean_greedy_error}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    out_dir = Path("results/stopping_information_20261018")
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke_result = smoke()
    (out_dir / "smoke.json").write_text(json.dumps(smoke_result, indent=2), encoding="utf-8")
    if args.full:
        full_result = validate(max_group_capacity=4)
        (out_dir / "full.json").write_text(json.dumps(full_result, indent=2), encoding="utf-8")
        print(json.dumps({"smoke": smoke_result, "full": full_result}, indent=2))
    else:
        print(json.dumps(smoke_result, indent=2))


if __name__ == "__main__":
    main()
