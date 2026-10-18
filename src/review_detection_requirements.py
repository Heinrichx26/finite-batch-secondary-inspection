"""Separate response/capacity success from the physical-detection requirement.

S is the probability that all targets receive a physical-inspection referral
and all mandatory plus optional inspections fit the count capacity. The
record-based calculation uses unit physical-detection probability. A common
joint physical-detection factor d_star then gives J = d_star * S.

Run --smoke before --full. No service simulation or plotting is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from full_counterfactual_model import build_records, complete_batch_matrix, transport_bounds
from marginal_weight_sensitivity import marginal_extreme

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results/review_detection_requirements"
ENDPOINT = "gun_knife"
CAPACITIES = (916, 930, 949, 969)
RADII = (0.0, 0.01, 0.025, 0.05)
TAUS = (0.8, 0.9)
REFERENCE_D0, REFERENCE_D1 = 0.9999, 0.99
TOL = 1e-11


def fingerprints(state):
    scripts = ("review_detection_requirements.py", "full_counterfactual_model.py",
               "marginal_weight_sensitivity.py")
    return {
        "scripts": {name: hashlib.sha256((BASE / "src" / name).read_bytes()).hexdigest()
                    for name in scripts},
        "data": state["provenance"]["source_sha256"],
    }


def physical_factor(state, d0=REFERENCE_D0, d1=REFERENCE_D1):
    y, route = state["targets"][ENDPOINT], state["route"]
    return float(d0 ** int((y & (route == 0)).sum()) * d1 ** int((y & (route != 0)).sum()))


def response_matrix(state, n, capacity):
    return complete_batch_matrix(state, ENDPOINT, n, capacity, db=1.0, d=1.0, eta=1.0)


def literal_probability(N, m, n, k, z):
    """Direct integer-binomial sum, independent of the scipy probability code."""
    if n < m or z < m:
        return 0.0
    denominator = math.comb(N, n)
    maximum = min(k, n - m, z - m)
    total = sum(math.comb(k, h) * math.comb(N - m - k, n - m - h)
                for h in range(max(0, n - m - (N - m - k)), maximum + 1))
    return total / denominator


def smoke_test(state):
    y, route = state["targets"][ENDPOINT], state["route"]
    eligible = route == 2
    N, m, B = int(eligible.sum()), int((eligible & y).sum()), int((~eligible).sum())
    assert (N, m, B) == (244, 2, 796)
    target_referrals = state["referrals"][:, eligible & y].sum(axis=1)
    assert np.all(target_referrals == m)
    loads = state["referrals"][:, eligible & ~y].sum(axis=1)
    literal_error = scaling_error = assignment_error = 0.0
    checked = 0
    for n, capacity in ((0, 907), (2, 907), (217, 907), (218, 907),
                        (244, 907), (244, 916), (244, 930), (244, 949), (244, 969)):
        S = response_matrix(state, n, capacity)
        literal = np.asarray([literal_probability(N, m, n, int(k), capacity - B)
                              for k in loads]).reshape(10, 10)
        literal_error = max(literal_error, float(np.abs(S - literal).max()))
        p = q = np.repeat(0.1, 10)
        llo, _ = marginal_extreme(S, p, q)
        lhi, _ = marginal_extreme(S, p, q, maximize=True)
        alo, _, ahi, _, _ = transport_bounds(S)
        assignment_error = max(assignment_error, abs(llo - alo), abs(lhi - ahi))
        for d0, d1 in ((REFERENCE_D0, REFERENCE_D1), (0.999, 0.97)):
            J = complete_batch_matrix(state, ENDPOINT, n, capacity, db=d0, d=d1)
            scaling_error = max(scaling_error, float(np.abs(J - physical_factor(state, d0, d1) * S).max()))
        checked += S.size
    assert max(literal_error, scaling_error, assignment_error) < TOL
    return dict(passed=True, fingerprints=fingerprints(state), literal_profile_cases=checked,
                maximum_literal_sum_error=literal_error, maximum_scaling_error=scaling_error,
                maximum_LP_assignment_error=assignment_error,
                source_counts=dict(eligible_population=N, eligible_targets=m, mandatory_inspections=B),
                no_detection_risk_constraint_applied=True)


def full_run(state):
    p = q = np.repeat(0.1, 10)
    dstar = physical_factor(state)
    rows, comparisons, requirements = [], [], []
    scaling_error = 0.0
    for n in range(245):
        S = response_matrix(state, n, 907)
        lower, independent, upper, _, _ = transport_bounds(S)
        J = complete_batch_matrix(state, ENDPOINT, n, 907)
        scaling_error = max(scaling_error, float(np.abs(J - dstar * S).max()))
        rows.append(dict(capacity=907, reviews=n, S_lower=lower, S_independent=independent,
                         S_upper=upper, J_lower_reference=dstar * lower))
    frame = pd.DataFrame(rows)
    maximum = float(frame.S_lower.max())
    optimum = frame[np.isclose(frame.S_lower, maximum, rtol=0.0, atol=1e-12)]
    assert optimum.reviews.tolist() == [218], optimum.to_dict("records")
    frame.to_csv(OUT / "C907_all_review_counts.csv", index=False, float_format="%.17g")

    for capacity in (907, 930):
        S218, SN = response_matrix(state, 218, capacity), response_matrix(state, 244, capacity)
        for quantity, matrix in (("S_218", S218), ("S_244", SN), ("S_218_minus_S_244", S218 - SN)):
            lower, independent, upper, _, _ = transport_bounds(matrix)
            comparisons.append(dict(capacity=capacity, quantity=quantity, lower=lower,
                                    independent=independent, upper=upper,
                                    reference_scaled_lower=dstar * lower,
                                    reference_scaled_upper=dstar * upper))
    pd.DataFrame(comparisons).to_csv(OUT / "normalized_policy_comparisons.csv", index=False, float_format="%.17g")

    for capacity in CAPACITIES:
        S = response_matrix(state, 244, capacity)
        J = complete_batch_matrix(state, ENDPOINT, 244, capacity)
        for radius in RADII:
            S_lower, _ = marginal_extreme(S, p, q, radius, radius)
            J_lower, _ = marginal_extreme(J, p, q, radius, radius)
            assert -TOL <= S_lower <= 1.0 + TOL
            S_lower = min(1.0, max(0.0, S_lower))
            scaling_error = max(scaling_error, abs(J_lower - dstar * S_lower))
            for tau in TAUS:
                required = tau / S_lower if S_lower > 0 else math.inf
                requirements.append(dict(capacity=capacity, common_TV_radius=radius,
                    S_full_lower=S_lower, final_success_requirement=tau,
                    required_joint_detection_probability=required,
                    feasible_with_joint_detection_at_most_one=bool(required <= 1.0 + TOL),
                    reference_joint_detection_probability=dstar,
                    reference_final_success_lower=J_lower,
                    reference_meets_requirement=bool(J_lower >= tau - TOL)))
    pd.DataFrame(requirements).to_csv(OUT / "joint_detection_requirements.csv", index=False, float_format="%.17g")
    assert scaling_error < TOL
    report = dict(passed=True, fingerprints=fingerprints(state), reference_d0=REFERENCE_D0,
        reference_d1=REFERENCE_D1, reference_joint_detection_probability=dstar,
        C907_global_maximum_S_lower=maximum, C907_global_optimal_reviews=optimum.reviews.tolist(),
        no_detection_risk_constraint_applied=True, normalized_policy_comparisons=comparisons,
        review_count_rows=len(frame), requirement_rows=len(requirements),
        maximum_reference_scaling_error=scaling_error,
        common_physical_detection_factor_required=True,
        target_referral_retention_fixed_at_one=True, service_simulation_run=False)
    (OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    state = build_records(persist=False)
    started = time.perf_counter()
    if args.smoke:
        report = smoke_test(state)
        (OUT / "smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        smoke = json.loads((OUT / "smoke.json").read_text(encoding="utf-8"))
        if not smoke.get("passed") or smoke.get("fingerprints") != fingerprints(state):
            raise RuntimeError("Run a matching passing --smoke before --full")
        report = full_run(state)
    report["seconds"] = time.perf_counter() - started
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
