"""Target-definition robustness for the finite-batch screening kernel.

The source records provide three pre-specified target definitions.  This
analysis applies the same policy kernel, physical-detection inputs, and
10-by-10 marginal transport bounds to each definition.  It is deliberately
separate from plotting and from the manuscript build.

Run ``--smoke`` before ``--full``.  The full run only evaluates exact finite
population formulas on the saved source records; it does not rerun service
simulation or alter the source data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from full_counterfactual_model import build_records, complete_batch_matrix, expected_batch, transport_bounds

BASE = Path(__file__).resolve().parents[1]
OUT_NAME = "endpoint_definition_robustness"
ENDPOINTS = ("gun", "gun_knife", "all_labels")
CAPACITIES = tuple(sorted(set([796, 827, 887, 907, 916, 930, 949, 969, 1040])))
ALL_CAPACITIES = tuple(range(796, 1041))
RISK_LIMIT = 0.001
SUCCESS_LEVELS = (0.80, 0.90)
TOL = 1e-12


def source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def endpoint_counts(state, endpoint: str) -> dict[str, int]:
    y = state["targets"][endpoint]
    route = state["route"]
    return {
        "endpoint": endpoint,
        "positive_tasks": int(y.sum()),
        "mandatory_target_count": int((y & (route != 2)).sum()),
        "original_target_count": int((y & (route == 0)).sum()),
        "alarm_target_count": int((y & (route == 1)).sum()),
        "eligible_target_count": int((y & (route == 2)).sum()),
        "eligible_pool": int((route == 2).sum()),
    }


def smoke_test() -> dict:
    state = build_records(persist=False)
    counts = [endpoint_counts(state, endpoint) for endpoint in ENDPOINTS]
    assert [row["positive_tasks"] for row in counts] == [180, 290, 680]
    assert [row["eligible_target_count"] for row in counts] == [0, 2, 9]

    # The endpoint-specific matrix must be a probability in every compatible
    # response pairing, including the no-target eligible endpoint.
    checked = 0
    for endpoint in ENDPOINTS:
        for n, capacity in ((0, 907), (111, 907), (244, 969)):
            matrix = complete_batch_matrix(state, endpoint, n, capacity)
            assert matrix.shape == (10, 10)
            assert np.isfinite(matrix).all()
            assert (matrix >= -TOL).all() and (matrix <= 1 + TOL).all()
            lower, mean, upper, _, _ = transport_bounds(matrix)
            assert lower - TOL <= mean <= upper + TOL
            checked += 1

    # Preserve the previously reported primary endpoint value exactly.
    primary = complete_batch_matrix(state, "gun_knife", 218, 907)
    lower, mean, upper, _, _ = transport_bounds(primary)
    assert np.isclose(lower, 0.5014770831936326, atol=1e-12)
    assert np.isclose(mean, 0.6409981834752017, atol=1e-12)
    assert np.isclose(upper, 0.7447996478629147, atol=1e-12)
    return {
        "passed": True,
        "source_sha256": source_hash(),
        "endpoint_counts": counts,
        "matrix_cases_checked": checked,
        "primary_endpoint_reproduced": True,
    }


def full_run(out: Path) -> dict:
    state = build_records(persist=False)
    review_counts = np.arange(245, dtype=int)
    rows: list[dict] = []
    summary: list[dict] = []
    thresholds: list[dict] = []

    for endpoint in ENDPOINTS:
        counts = endpoint_counts(state, endpoint)
        for capacity in CAPACITIES:
            for review_count in review_counts:
                matrix = complete_batch_matrix(state, endpoint, int(review_count), int(capacity))
                lower, mean, upper, _, _ = transport_bounds(matrix)
                expected = expected_batch(state, endpoint, int(review_count))
                rows.append({
                    **counts,
                    "capacity": int(capacity),
                    "review_count": int(review_count),
                    "joint_lower": lower,
                    "joint_independent": mean,
                    "joint_upper": upper,
                    "mean_missed_target_fraction": float(expected["conditional_miss"]),
                    "mean_inspection_demand": float(expected["expected_physical"]),
                    "risk_feasible": bool(expected["conditional_miss"] <= RISK_LIMIT + TOL),
                })

        # Summary at the reference capacity and full review.  The first
        # feasible review count is selected by the stated risk constraint;
        # the best lower success is selected among the same feasible choices.
        ref = pd.DataFrame([row for row in rows if row["endpoint"] == endpoint and row["capacity"] == 907])
        feasible = ref[ref.risk_feasible]
        best = feasible.sort_values(["joint_lower", "review_count"], ascending=[False, True]).iloc[0] if len(feasible) else None
        full = ref[ref.review_count.eq(244)].iloc[0]
        count_row = counts.copy()
        ceiling_kernel = complete_batch_matrix(state, endpoint, 244, 1040)
        ceiling_lower, _, _, _, _ = transport_bounds(ceiling_kernel)
        count_row.update({
            "reference_capacity": 907,
            "minimum_review_for_risk": int(feasible.review_count.min()) if len(feasible) else None,
            "best_lower_success_review": int(best.review_count) if best is not None else None,
            "best_lower_success": float(best.joint_lower) if best is not None else None,
            "full_review_lower_success_at_907": float(full.joint_lower),
            "full_review_mean_success_at_907": float(full.joint_independent),
            "full_review_mean_miss_at_907": float(full.mean_missed_target_fraction),
            "unlimited_success_ceiling": ceiling_lower,
        })
        # Compute every integer-capacity point for full review so the reported
        # thresholds are exact within the finite source population.
        full_rows = []
        for capacity in ALL_CAPACITIES:
            matrix = complete_batch_matrix(state, endpoint, 244, capacity)
            lower, mean, upper, _, _ = transport_bounds(matrix)
            expected = expected_batch(state, endpoint, 244)
            full_rows.append({
                **counts,
                "capacity": capacity,
                "review_count": 244,
                "joint_lower": lower,
                "joint_independent": mean,
                "joint_upper": upper,
                "mean_missed_target_fraction": float(expected["conditional_miss"]),
                "risk_feasible": bool(expected["conditional_miss"] <= RISK_LIMIT + TOL),
            })
        full_rows = pd.DataFrame(full_rows)
        full_rows.to_csv(out / f"full_review_capacity_frontier_{endpoint}.csv", index=False)
        for level in SUCCESS_LEVELS:
            eligible = full_rows[full_rows.joint_lower >= level - TOL]
            threshold = int(eligible.capacity.min()) if len(eligible) else None
            thresholds.append({
                "endpoint": endpoint,
                "review_count": 244,
                "success_requirement": level,
                "minimum_capacity": threshold,
                "joint_lower_at_threshold": float(eligible.loc[eligible.capacity.idxmin(), "joint_lower"]) if len(eligible) else None,
                "risk_feasible_at_threshold": bool(eligible.loc[eligible.capacity.idxmin(), "risk_feasible"]) if len(eligible) else None,
            })
        summary.append(count_row)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "endpoint_policy_grid.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "endpoint_summary.csv", index=False)
    pd.DataFrame(thresholds).to_csv(out / "full_review_capacity_thresholds.csv", index=False)

    # Endpoint ordering checks: with the same policy and capacity, the exact
    # kernels must reproduce the expected endpoint-specific target counts.
    checks = []
    for endpoint in ENDPOINTS:
        subset = frame[frame.endpoint.eq(endpoint)]
        checks.append({
            "endpoint": endpoint,
            "rows": int(len(subset)),
            "risk_feasible_rows": int(subset.risk_feasible.sum()),
            "lower_success_min": float(subset.joint_lower.min()),
            "lower_success_max": float(subset.joint_lower.max()),
            "mean_miss_min": float(subset.mean_missed_target_fraction.min()),
            "mean_miss_max": float(subset.mean_missed_target_fraction.max()),
        })
    (out / "endpoint_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    report = {
        "passed": True,
        "source_sha256": source_hash(),
        "endpoints": list(ENDPOINTS),
        "capacities": list(CAPACITIES),
        "review_counts": 245,
        "grid_rows": int(len(frame)),
        "risk_limit": RISK_LIMIT,
        "success_levels": list(SUCCESS_LEVELS),
        "no_service_simulation_rerun": True,
    }
    (out / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()
    out = BASE / "results" / (OUT_NAME + ("_smoke" if args.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        report = smoke_test()
        (out / "smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    smoke = json.loads((BASE / "results" / (OUT_NAME + "_smoke") / "smoke.json").read_text(encoding="utf-8"))
    if not smoke.get("passed") or smoke.get("source_sha256") != source_hash():
        raise RuntimeError("The current endpoint analysis has no matching passing smoke test")
    started = time.perf_counter()
    report = full_run(out)
    report["seconds"] = time.perf_counter() - started
    (out / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
