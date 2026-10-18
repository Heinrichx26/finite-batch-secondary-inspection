"""Screen response-support sensitivity by the already specified mean-risk limit.

This reads exact probability results; it does not rerun the study or plot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from full_counterfactual_model import build_records, expected_batch

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results/target_support_upgrade_20261018"


def main():
    parser = argparse.ArgumentParser(); mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true"); mode.add_argument("--full", action="store_true")
    args = parser.parse_args()
    state = build_records(False)
    target = state["targets"]["gun_knife"]; eligible = state["route"] == 2
    N = int(eligible.sum()); m = int((eligible & target).sum()); total = int(target.sum())
    mandatory0 = int(((state["route"] == 0) & target).sum())
    mandatory1 = int(((state["route"] == 1) & target).sum())
    def risk(n, epsilon, d0, d1):
        expected_mandatory_failures = mandatory0 * (1 - d0) + mandatory1 * (1 - d1)
        return (expected_mandatory_failures + m - (n / N) * (1 - epsilon) * m * d1) / total
    maximum = 0.
    for n in range(N + 1):
        actual = expected_batch(state, "gun_knife", n)["conditional_miss"]
        maximum = max(maximum, abs(actual - risk(n, 0., .9999, .99)))
        # Direct sum over target selection, original response failure, and search detection.
        for epsilon in (0., .05, .10, .20):
            optional_miss = m * ((1 - n / N) + (n / N) * epsilon + (n / N) * (1 - epsilon) * .01)
            direct = (mandatory0 * .0001 + mandatory1 * .01 + optional_miss) / total
            maximum = max(maximum, abs(direct - risk(n, epsilon, .9999, .99)))
    assert maximum < 1e-12
    if args.smoke:
        report = dict(passed=True, maximum_mean_risk_error=maximum, tested_review_counts=N + 1,
                      source_counts=[N, m, total, mandatory0, mandatory1])
        (OUT / "decision_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        assert json.loads((OUT / "decision_smoke.json").read_text())["passed"]
        cells = pd.read_csv(OUT / "target_failure_all_counts.csv")
        rows = []
        for case, d0, d1 in (("perfect_detection", 1., 1.), ("specified_detection", .9999, .99)):
            cells["mean_risk_upper"] = risk(cells.reviews, cells.epsilon, d0, d1)
            for risk_limit in (.0005, .001, .002):
                for (capacity, epsilon), group in cells.groupby(["capacity", "epsilon"]):
                    allowed = group[group.mean_risk_upper <= risk_limit + 1e-12]
                    selected = allowed.sort_values(["S_lower", "reviews"], ascending=[False, True])
                    rows.append(dict(case=case, capacity=int(capacity), epsilon=float(epsilon),
                        mean_risk_limit=risk_limit, feasible=bool(len(allowed)),
                        minimum_feasible_reviews=int(allowed.reviews.min()) if len(allowed) else None,
                        best_reviews=int(selected.reviews.iloc[0]) if len(selected) else None,
                        best_S_lower=float(selected.S_lower.iloc[0]) if len(selected) else None,
                        mean_risk_upper_at_best=float(selected.mean_risk_upper.iloc[0]) if len(selected) else None))
        pd.DataFrame(rows).to_csv(OUT / "mean_risk_screened_decisions.csv", index=False)
        report = dict(passed=True, maximum_mean_risk_error=maximum, decision_rows=len(rows),
                      epsilon_upper_bound_includes_complete_omission_of_all_eligible_targets=True,
                      specified_detection_is_the_preexisting_supplement_scenario=True,
                      feasible_means_the_mean_risk_limit_only=True)
        (OUT / "decision_full.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
