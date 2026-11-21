from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from recent_method_comparison import (
    MAISANO_RATES, majority_errors, majority_errors_by_subsets,
    panel_batch_success, panel_expected_miss, phase_graph_kernel,
)
from r1_image_unit_analysis import prepare_cases, response_components
from verify_r1_capacity import close, enumerate_prefixes


OUT = Path(__file__).resolve().parents[1] / "results" / "R1_benchmarks"


def smoke() -> dict:
    rates = MAISANO_RATES["L"]
    for inspectors in (1, 3, 5, 7):
        positive = rates["false_positive"][:inspectors]
        negative = rates["false_negative"][:inspectors]
        actual = majority_errors(positive, negative)
        expected = majority_errors_by_subsets(positive, negative)
        for error, reference in zip(actual, expected):
            close(error, reference, "Majority-vote aggregation")
    close(phase_graph_kernel(3, 1, 1, 2, 1)[0], 1 / 3, "Three-image diagram")
    return {"passed": True, "independent_vote_and_graph_checks": 9}


def run() -> dict:
    report = smoke()
    rates = MAISANO_RATES["L"]
    rows = []
    for capacity in (2, 7):
        for inspectors in (1, 3, 5, 7):
            false_positive, false_negative = majority_errors(
                rates["false_positive"][:inspectors], rates["false_negative"][:inspectors])
            for readings in range(min(60, 180 // inspectors) + 1):
                risk = panel_expected_miss(60, 1, readings, false_negative)
                rows.append({"capacity": capacity, "inspectors": inspectors,
                             "reading_count": readings, "judgments": inspectors * readings,
                             "referral_and_capacity": panel_batch_success(
                                 60, 1, readings, capacity, false_positive, false_negative),
                             "mean_unreferred_fraction": risk, "meets_risk_limit": risk <= 0.1})
    frame = pd.DataFrame(rows)
    chosen = frame[frame["meets_risk_limit"]].sort_values(
        ["referral_and_capacity", "judgments"], ascending=[False, True], kind="stable",
    ).groupby(["capacity", "inspectors"], sort=True).head(1).sort_values(["capacity", "inspectors"])
    displayed = chosen[chosen["inspectors"].isin([1, 3])]
    assert displayed["reading_count"].tolist() == [56, 60, 60, 60]
    assert displayed["referral_and_capacity"].round(4).tolist() == [0.1855, 0.9103, 0.9390, 0.9950]
    assert displayed["mean_unreferred_fraction"].round(4).tolist() == [0.0909, 0.0050, 0.0260, 0.0050]
    tested = set()
    largest_error = 0.0
    for case in prepare_cases().values():
        for first in range(len(case["primary"])):
            for second in range(len(case["secondary"])):
                part = response_components(case, first, second)
                population, targets = part["population"], part["required_targets"]
                if not part["all_targets_referred"]:
                    continue
                non_targets = part["non_target_referrals"]
                for readings in range(population + 1):
                    for slots in range(population + 1):
                        key = population, targets, non_targets, readings, slots
                        if key in tested:
                            continue
                        tested.add(key)
                        graph, *_ = phase_graph_kernel(*key)
                        literal, _ = enumerate_prefixes(
                            tuple(map(bool, part["eligible_target_mask"])),
                            tuple(map(bool, part["secondary_referrals"])),
                            readings, slots, False, case["target_count"],
                        )
                        close(graph, literal, f"Binary decision diagram for {key}")
                        largest_error = max(largest_error, abs(graph - literal))
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "panel_comparison.csv", index=False)
    chosen.to_csv(OUT / "panel_selected_allocations.csv", index=False)
    return {**report, "diagram_count_configurations": len(tested),
            "maximum_diagram_probability_error": largest_error,
            "panel_allocations": len(rows), "panel_rows_verified": len(displayed)}


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()
    report = smoke() if args.smoke else run()
    if args.full:
        (OUT / "benchmark_verification.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
