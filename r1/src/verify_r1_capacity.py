from __future__ import annotations

import argparse
import json
from fractions import Fraction
from functools import lru_cache
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from r1_image_unit_analysis import OUT, prepare_cases


@lru_cache(maxsize=None)
def enumerate_prefixes(
    target_mask: tuple[bool, ...], referrals: tuple[bool, ...], readings: int,
    slots: int, stop: bool, total_targets: int,
) -> tuple[float, float]:
    target_count = sum(target_mask)
    success_mass = Fraction(0)
    recovered_mass = Fraction(0)
    sample_count = 0
    for selected in combinations(range(len(referrals)), min(readings, len(referrals))):
        sample_count += 1
        positive = tuple(index for index in selected if referrals[index])
        if stop and len(positive) > slots:
            retained = combinations(positive, slots)
        else:
            retained = (positive,)
        successes = recovered = orders = 0
        for admitted in retained:
            found = sum(target_mask[index] for index in admitted)
            recovered += found
            successes += int(found == target_count and len(admitted) <= slots)
            orders += 1
        success_mass += Fraction(successes, orders)
        recovered_mass += Fraction(recovered, orders)
    success = success_mass / sample_count
    risk = (target_count - recovered_mass / sample_count) / total_targets
    return float(success), float(risk)


def extreme(matrix: np.ndarray, maximize: bool = False) -> float:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("An assignment check needs equal uniform group sizes")
    rows, columns = linear_sum_assignment(matrix, maximize=maximize)
    return float(matrix[rows, columns].mean())


def close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, atol=1e-12, rtol=0):
        raise AssertionError(f"{label}: independently got {actual}; expected {expected}")


def smoke() -> dict:
    mask = (True, False, False)
    responses = (True, True, False)
    fixed, fixed_risk = enumerate_prefixes(mask, responses, 2, 1, False, 1)
    stopped, stopped_risk = enumerate_prefixes(mask, responses, 2, 1, True, 1)
    close(fixed, 1 / 3, "Three-image fixed success")
    close(fixed_risk, 1 / 3, "Three-image fixed risk")
    close(stopped, 1 / 2, "Three-image stopping success")
    close(stopped_risk, 1 / 2, "Three-image stopping risk")
    sample = np.array([[0.1, 0.8], [0.9, 0.1]])
    close(extreme(sample), 0.1, "Minimum over uniform pairings")
    close(extreme(sample, True), 0.85, "Maximum over uniform pairings")
    return {"passed": True, "analytical_probability_checks": 6}


def verify() -> dict:
    report = smoke()
    saved = pd.read_csv(OUT / "risk_constrained_capacity.csv")
    thresholds = pd.read_csv(OUT / "risk_constrained_thresholds.csv")
    cases = prepare_cases()
    results = []
    for condition, case in cases.items():
        routes = [profile["route"] for profile in case["primary"]]
        secondary = [profile["referral"] for profile in case["secondary"]]
        target_units = int(case["targets"].sum())
        mandatory = [int(np.sum(route != "U")) for route in routes]
        pools = [route == "U" for route in routes]
        capacity_floor = max(mandatory)
        read_limit = max(int(pool.sum()) for pool in pools)
        shape = (len(routes), len(secondary))
        condition_rows = saved[saved["condition"].eq(condition)].sort_values("search_capacity")
        assert condition_rows["search_capacity"].tolist() == list(
            range(capacity_floor, case["image_count"] + 1))
        checks = 0
        decisions = []
        for row in condition_rows.itertuples(index=False):
            matrices = np.empty((read_limit + 2, 2, *shape))
            for first, pool in enumerate(pools):
                target_mask = tuple(map(bool, case["targets"][pool]))
                slots = row.search_capacity - mandatory[first]
                for second, profile in enumerate(secondary):
                    referrals = tuple(map(bool, profile[pool]))
                    for readings in range(read_limit + 1):
                        matrices[readings, :, first, second] = enumerate_prefixes(
                            target_mask, referrals, readings, slots, False, target_units)
                    matrices[-1, :, first, second] = enumerate_prefixes(
                        target_mask, referrals, read_limit, slots, True, target_units)
                    checks += read_limit + 2
            fixed_full, fixed_full_risk = matrices[-2]
            stopped, stopped_risk = matrices[-1]
            candidates = []
            for readings, (fixed, risk) in enumerate(matrices[:-1]):
                if extreme(risk, True) <= 0.001 + 1e-12:
                    candidates.append((extreme(fixed), readings))
            best, best_count = max(candidates, key=lambda pair: (round(pair[0], 12), -pair[1]))
            effects = stopped - fixed_full
            stop_lower = extreme(stopped)
            stop_risk = extreme(stopped_risk, True)
            for key, value in {
                "stopping_success_lower": stop_lower,
                "stopping_mean_miss_upper": stop_risk,
                "risk_constrained_best_fixed_success_lower": best,
                "risk_constrained_best_fixed_readings": best_count,
                "same_reading_allowance_fixed_success_lower": extreme(fixed_full),
                "same_reading_allowance_fixed_mean_miss_upper": extreme(fixed_full_risk, True),
                "matched_stop_minus_fixed_effect_lower": extreme(effects),
                "matched_stop_minus_fixed_effect_upper": extreme(effects, True),
            }.items():
                close(value, getattr(row, key), f"{condition}, {row.search_capacity}, {key}")
            decisions.append((row.search_capacity, stop_lower, stop_risk, best))
        for row in thresholds[thresholds["condition"].eq(condition)].itertuples(index=False):
            requirement = row.joint_referral_and_capacity_requirement - 1e-12
            stop_capacity = min(capacity for capacity, success, risk, _ in decisions
                                if success >= requirement and risk <= row.mean_miss_limit + 1e-12)
            fixed_capacity = min(capacity for capacity, _, _, success in decisions
                                 if success >= requirement)
            close(stop_capacity, row.stopping_minimum_capacity, f"{condition} stopping threshold")
            close(fixed_capacity, row.risk_constrained_best_fixed_minimum_capacity,
                  f"{condition} fixed threshold")
        results.append({"condition": condition, "mandatory_capacity_floor": capacity_floor,
                        "reading_limit": read_limit, "conditional_checks": checks})
    return {**report, "scope": "Finite public image batches and reserved mandatory workload",
            "method": "Exact subset enumeration with assignment extrema for uniform margins",
            "conditions": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()
    result = smoke() if args.smoke else verify()
    if args.full:
        (OUT / "independent_capacity_audit.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
