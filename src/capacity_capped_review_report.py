"""Additional capability/mean-risk reports from the validated capped kernel.

No plotting and no source-record changes. Independent gate checks precede full.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pandas as pd

from capacity_capped_review_study import (OUT, affine_extreme, capped_exact,
    metric_matrices, profile_kernel, source_metrics, supported_bounds)
from full_counterfactual_model import build_records, transport_bounds


def main():
    parser = argparse.ArgumentParser(); mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true"); mode.add_argument("--full", action="store_true")
    args = parser.parse_args(); state = build_records(False)
    u = state["route"] == 2; y = state["targets"]["gun_knife"]
    N = int(u.sum()); m = int((u & y).sum()); B = int((~u).sum())
    loads = state["referrals"][:, u & ~y].sum(axis=1)
    dstar = .9999 ** 286 * .99 ** 4
    if args.smoke:
        assert json.loads((OUT / "full.json").read_text())["passed"]
        exact_error = 0.
        for C in (907, 916, 930, 949, 969):
            for k in np.unique(loads):
                result = source_metrics(N, m, int(k), N, C - B)
                exact = capped_exact(N, m, m, int(k), N, C - B)
                exact_error = max(exact_error, abs(result["success"] - float(exact[0])),
                                  abs(result["captured"] - float(exact[1])))
        assert exact_error < 1e-10
        (OUT / "report_smoke.json").write_text(json.dumps(dict(passed=True,
            maximum_full_allowance_error=exact_error), indent=2))
        print(json.dumps(dict(passed=True, maximum_full_allowance_error=exact_error)))
        return
    assert json.loads((OUT / "report_smoke.json").read_text())["passed"]
    frontier = []; minimum = []; table = []; paired = []
    for C in range(B, B + N + 1):
        positive, deleted, one_missing = metric_matrices(N, m, loads, N, C - B)
        for zeta in (0., .01, .025, .05, .10):
            lo, nominal, hi = supported_bounds(positive["success"], zeta)
            captured_lower = supported_bounds(positive["captured"], zeta)[0]
            risk_perfect = (m - captured_lower) / 290
            risk_specified = (.0486 + m - .99 * captured_lower) / 290
            frontier.append(dict(capacity=C, zeta=zeta, lower_S=lo, upper_S=hi,
                captured_lower=captured_lower, risk_upper_perfect=risk_perfect,
                risk_upper_specified=risk_specified, lower_J_specified=dstar * lo))
            if C in (907, 916, 930, 949, 969) and zeta in (0., .01, .05):
                reads_lo = affine_extreme(positive["readings"], one_missing["readings"], zeta)
                reads_hi = affine_extreme(positive["readings"], deleted["readings"], zeta, True)
                searches_lo = affine_extreme(positive["searches"], deleted["searches"], zeta)
                searches_hi = affine_extreme(positive["searches"], one_missing["searches"], zeta, True)
                table.append(dict(capacity=C, zeta=zeta, lower_S=lo, upper_S=hi,
                    lower_J_specified=dstar * lo, mean_risk_upper_specified=risk_specified,
                    mean_risk_upper_perfect=risk_perfect,
                    mean_reads_lower=reads_lo, mean_reads_upper=reads_hi,
                    mean_searches_lower=B + searches_lo, mean_searches_upper=B + searches_hi,
                    nominal_independent_reads=float(positive["readings"].mean()),
                    nominal_independent_searches=B + float(positive["searches"].mean())))
    frame = pd.DataFrame(frontier)
    for zeta in (0., .01, .025, .05, .10):
        group = frame[frame.zeta.eq(zeta)]
        for tau in (.8, .9):
            for case, success, risk in (("perfect_detection", "lower_S", "risk_upper_perfect"),
                                        ("specified_detection", "lower_J_specified", "risk_upper_specified")):
                meet_success = group[group[success] >= tau - 1e-12]
                meet_both = meet_success[meet_success[risk] <= .001 + 1e-12]
                minimum.append(dict(zeta=zeta, success_requirement=tau, detection_case=case,
                    mean_risk_limit=.001,
                    minimum_capacity_success_only=int(meet_success.capacity.min()) if len(meet_success) else None,
                    minimum_capacity_success_and_risk=int(meet_both.capacity.min()) if len(meet_both) else None))
    for C in (907, 916, 930, 949, 969):
        fixed = profile_kernel(state, y, C)
        best_fixed = int(np.argmax([transport_bounds(k)[0] for k in fixed]))
        for zeta in (0., .01, .05):
            for n in sorted(set((218, best_fixed, N))):
                for nmax in sorted(set((n, N))):
                    D = metric_matrices(N, m, loads, nmax, C - B)[0]["success"] - fixed[n]
                    lo, nominal, hi = supported_bounds(D, zeta)
                    paired.append(dict(capacity=C, zeta=zeta, fixed_reviews=n, capped_nmax=nmax,
                        lower_effect=max(0., lo) if lo > -1e-10 else lo,
                        nominal_independent_effect=nominal,
                        upper_effect=max(0., hi) if hi > -1e-10 else hi))
    pd.DataFrame(table).to_csv(OUT / "nmax244_key_table.csv", index=False)
    frame.to_csv(OUT / "expanded_capacity_frontier.csv", index=False)
    pd.DataFrame(minimum).to_csv(OUT / "joint_success_risk_minimum_capacity.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "expanded_same_distribution_effects.csv", index=False)
    report = dict(passed=True, key_rows=len(table), frontier_rows=len(frame), minimum_capacity_rows=len(minimum),
                  physical_detection_factor=dstar, same_distribution_effect_rows=len(paired),
                  zeta_values=[0., .01, .025, .05, .10],
                  nmax244_is_optimal_within_capped_family_for_both_outcomes=True)
    (OUT / "report_full.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
