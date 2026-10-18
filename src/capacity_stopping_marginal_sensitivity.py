"""Marginal-profile sensitivity for the validated capacity-limited stop.

This report reweights observed response profiles under total-variation bounds.
It leaves the target-referral support unchanged (zeta=0). Plotting is separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from capacity_capped_review_study import OUT, metric_matrices
from full_counterfactual_model import build_records, transport_bounds
from marginal_weight_sensitivity import marginal_extreme


def fingerprints(state):
    paths = [Path(__file__), Path(__file__).with_name("capacity_capped_review_study.py"),
             Path(__file__).with_name("marginal_weight_sensitivity.py")]
    return {"scripts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "source": state["provenance"]["source_sha256"]}


def inputs(state):
    eligible = state["route"] == 2
    targets = state["targets"]["gun_knife"]
    return (int(eligible.sum()), int((eligible & targets).sum()), int((~eligible).sum()),
            state["referrals"][:, eligible & ~targets].sum(axis=1))


def smoke(state):
    N, m, B, loads = inputs(state)
    p = q = np.full(10, .1)
    maximum_error = 0.
    for capacity in (907, 916, 930, 949):
        matrices = metric_matrices(N, m, loads, N, capacity - B)[0]
        for name in ("success", "captured"):
            matrix = matrices[name]
            lo, _, hi, _, _ = transport_bounds(matrix)
            lo0, _ = marginal_extreme(matrix, p, q)
            hi0, _ = marginal_extreme(matrix, p, q, maximize=True)
            lounrestricted, _ = marginal_extreme(matrix, p, q, 1., 1.)
            hiunrestricted, _ = marginal_extreme(matrix, p, q, 1., 1., maximize=True)
            error = max(abs(lo - lo0), abs(hi - hi0),
                        abs(float(matrix.min()) - lounrestricted),
                        abs(float(matrix.max()) - hiunrestricted))
            maximum_error = max(maximum_error, error)
    assert maximum_error < 1e-9
    return dict(passed=True, fingerprints=fingerprints(state),
                maximum_transport_and_unrestricted_error=maximum_error)


def full(state):
    N, m, B, loads = inputs(state)
    p = q = np.full(10, .1)
    dstar = .9999 ** 286 * .99 ** 4
    rows = []
    for capacity in (907, 916, 930, 949):
        matrices = metric_matrices(N, m, loads, N, capacity - B)[0]
        for delta in (0., .01, .05):
            lower_s, _ = marginal_extreme(matrices["success"], p, q, delta, delta)
            upper_s, _ = marginal_extreme(matrices["success"], p, q, delta, delta, maximize=True)
            lower_s = float(np.clip(lower_s, 0., 1.))
            upper_s = float(np.clip(upper_s, 0., 1.))
            lower_targets, _ = marginal_extreme(matrices["captured"], p, q, delta, delta)
            upper_risk = (.0486 + m - .99 * lower_targets) / 290
            rows.append(dict(capacity=capacity, maximum_readings=N, zeta=0.,
                marginal_tv_radius=delta, lower_S=lower_s, upper_S=upper_s,
                lower_J_specified=dstar * lower_s, upper_mean_risk_specified=upper_risk,
                required_dstar_for_80pct=.8 / lower_s,
                required_dstar_for_90pct=.9 / lower_s,
                passes_joint_specified_80pct=bool(dstar * lower_s >= .8 - 1e-12 and upper_risk <= .001 + 1e-12)))
    frame = pd.DataFrame(rows)
    for _, group in frame.groupby("capacity"):
        assert np.all(np.diff(group.lower_S) <= 1e-10)
        assert np.all(np.diff(group.upper_mean_risk_specified) >= -1e-10)
    frame.to_csv(OUT / "stopping_marginal_tv_sensitivity.csv", index=False)
    report = dict(passed=True, fingerprints=fingerprints(state), rows=len(frame),
        maximum_readings=N, zeta=0., marginal_tv_radii=[0., .01, .05],
        capacities=[907, 916, 930, 949], physical_detection_factor=dstar,
        target_referral_support_unchanged=True, monotonicity_checks_passed=True)
    (OUT / "stopping_marginal_full.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()
    state = build_records(False)
    if args.smoke:
        report = smoke(state)
        (OUT / "stopping_marginal_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        gate = json.loads((OUT / "stopping_marginal_smoke.json").read_text(encoding="utf-8"))
        if not gate["passed"] or gate["fingerprints"] != fingerprints(state):
            raise RuntimeError("A matching passing smoke test is required")
        report = full(state)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
