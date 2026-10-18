"""Independent validation by literal target placements and search subsets."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product, permutations
from math import comb
from pathlib import Path
import argparse
import importlib.util
import json
import sys

import numpy as np
from scipy.optimize import linprog


def load_candidate():
    path = Path(__file__).with_name("stopping_information_theory.py")
    spec = importlib.util.spec_from_file_location("stopping_information_candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def all_subsets(items, limit):
    return tuple(frozenset(s) for size in range(limit + 1) for s in combinations(items, size))


def literal_game(capacities, theta_set, allowance, risk_limit=None):
    offsets = [sum(capacities[:g]) for g in range(len(capacities))]
    group_items = [tuple(range(start, start + size)) for start, size in zip(offsets, capacities)]
    target_sets = []
    for theta in theta_set:
        for group_targets in product(*(tuple(combinations(items, m)) for items, m in zip(group_items, theta))):
            target_sets.append(frozenset(item for group in group_targets for item in group))
    selections = all_subsets(tuple(range(sum(capacities))), min(allowance, sum(capacities)))
    c = [0.0] * len(selections) + [-1.0]
    A = []
    b = []
    for targets in target_sets:
        A.append([-float(targets <= selection) for selection in selections] + [1.0])
        b.append(0.0)
        if risk_limit is not None:
            A.append([-len(targets & selection) / len(targets) for selection in selections] + [0.0])
            b.append(-(1.0 - risk_limit))
    result = linprog(c, A_ub=np.asarray(A), b_ub=b,
                     A_eq=[[1.0] * len(selections) + [0.0]], b_eq=[1.0],
                     bounds=[(0.0, None)] * len(selections) + [(0.0, 1.0)], method="highs")
    if result.status == 2:
        return None
    if not result.success:
        raise RuntimeError(result.message)
    return float(-result.fun)


def literal_local_stopping(N, P, m, z):
    """Fixed targets, fixed positive non-targets, all literal image orders."""
    targets = frozenset(range(m))
    positives = frozenset(range(P))
    hits = Fraction(0, 1)
    recovered = Fraction(0, 1)
    readings = Fraction(0, 1)
    orders = 0
    for order in permutations(range(N)):
        admitted = set()
        used_readings = 0
        if z:
            for position, item in enumerate(order, start=1):
                used_readings = position
                if item in positives:
                    admitted.add(item)
                    if len(admitted) >= z:
                        break
        hits += int(targets <= admitted)
        recovered += len(targets & admitted)
        readings += used_readings
        orders += 1
    return hits / orders, recovered / orders, readings / orders


def validate(smoke=False):
    candidate = load_candidate()
    max_gap = 0.0
    game_count = 0
    constrained_game_count = 0
    local_count = 0
    max_capacity = 2 if smoke else 3
    for p1 in range(1, max_capacity + 1):
        for p2 in range(1, max_capacity + 1):
            capacities = (p1, p2)
            for m in range(1, p1 + p2 + 1):
                theta_all = candidate.compositions(capacities, m)
                sets = [theta_all] + [(theta,) for theta in theta_all]
                if len(theta_all) >= 2:
                    sets.append((theta_all[0], theta_all[-1]))
                for theta_set in sets:
                    for z in range(p1 + p2 + 1):
                        literal = literal_game(capacities, theta_set, z)
                        compressed = candidate.primal_dual_value(capacities, theta_set, z)["value"]
                        gap = abs(literal - compressed)
                        max_gap = max(max_gap, gap)
                        assert gap < 1e-9
                        game_count += 1
                        if z == max(m, 1):
                            for eps in (0.25, 0.5, 0.75):
                                literal_risk = literal_game(capacities, theta_set, z, risk_limit=eps)
                                collapsed_risk = candidate.primal_dual_value(capacities, theta_set, z, risk_limit=eps)
                                assert (literal_risk is not None) == collapsed_risk["feasible"]
                                if literal_risk is not None:
                                    gap = abs(literal_risk - collapsed_risk["value"])
                                    max_gap = max(max_gap, gap)
                                    assert gap < 1e-9
                                constrained_game_count += 1

    for N in range(1, 4 if smoke else 5):
        for P in range(1, N + 1):
            for m in range(0, P + 1):
                for z in range(0, P + 1):
                    success, recovered, readings = literal_local_stopping(N, P, m, z)
                    formula_success = Fraction(comb(z, m), comb(P, m)) if z >= m else Fraction(0, 1)
                    formula_recovered = Fraction(m * z, P)
                    formula_readings = Fraction(z * (N + 1), P + 1)
                    assert success == formula_success
                    assert recovered == formula_recovered
                    assert readings == formula_readings
                    local_count += 1
    return {"literal_target_placement_games": game_count,
            "literal_joint_success_risk_games": constrained_game_count,
            "literal_within_group_stopping_cases": local_count,
            "max_value_gap": max_gap,
            "local_stopping_exact_arithmetic_error": 0,
            "local_expected_readings_exact_arithmetic_error": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    result = validate(smoke=not args.full)
    out = Path("results/stopping_information_20261018")
    out.mkdir(parents=True, exist_ok=True)
    name = "independent_full.json" if args.full else "independent_smoke.json"
    (out / name).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
