"""Smoke-gated general policy bounds; reuses saved deadline simulations.

No queue simulation or training is run here. Plotting is a separate program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import beta

from robust_policy_bounds import CouplingSet, minimax_regret, regret_certificate

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results/general_method_20261017"


def source_hashes():
    return {name: hashlib.sha256((BASE / "src" / name).read_bytes()).hexdigest()
            for name in ["robust_policy_bounds.py", "run_general_policy_bounds.py"]}


def smoke():
    checks = 0
    model = CouplingSet([.3, .7], [.4, .6])
    rng = np.random.default_rng(2026101701)
    for _ in range(12):
        matrix = rng.normal(size=(2, 2))
        endpoints = [np.array([[t, .3-t], [.4-t, .3+t]]) for t in [0., .3]]
        expected = [float(np.sum(matrix*q)) for q in endpoints]
        lo, hi = model.bounds(matrix)
        assert abs(lo.value-min(expected)) < 1e-10
        assert abs(hi.value-max(expected)) < 1e-10
        checks += 2
    additive = np.array([-.2, .7])[:, None] + np.array([.1, .8])[None, :]
    lo, hi = model.bounds(additive)
    mean, interaction = model.interaction(additive)
    assert hi.value-lo.value < 1e-12 and abs(mean-lo.value) < 1e-12
    assert np.max(np.abs(interaction)) < 1e-12
    checks += 3

    model = CouplingSet([.5, .5], [.5, .5])
    diagonal = np.eye(2)
    kernels = np.array([diagonal, 1.-diagonal])
    deterministic = minimax_regret(model, kernels, randomized=False)
    mixture = minimax_regret(model, kernels)
    assert abs(deterministic["upper_bound"]-1.) < 1e-10
    assert abs(mixture["upper_bound"]-.5) < 1e-10
    assert np.max(np.abs(mixture["weights"]-.5)) < 1e-10
    checks += 3

    restricted = CouplingSet([.5, .5], [.5, .5], [(diagonal, .6, .8)])
    informed = minimax_regret(restricted, kernels)
    assert informed["upper_bound"] < 1e-10 and informed["weights"][0] > 1.-1e-9
    checks += 1
    uncertain = minimax_regret(model, .1+.8*kernels-.05, .1+.8*kernels+.05)
    assert abs(uncertain["upper_bound"]-.45) < 1e-10
    self_regret = regret_certificate(model, [1.], [.5*diagonal], [.5+.5*diagonal])[0]
    assert self_regret < 1e-10
    checks += 2
    budget_kernels = np.array([diagonal, 1.-diagonal, .8*np.ones((2, 2))])
    metrics = np.array([np.ones((2, 2)), np.ones((2, 2)), 3.*np.ones((2, 2))])
    budgeted = minimax_regret(model, budget_kernels, metric_limits=[(metrics, 1.5)])
    assert np.max(np.abs(budgeted["weights"]-[.375, .375, .25])) < 1e-10
    assert abs(budgeted["upper_bound"]-.425) < 1e-10
    assert budgeted["comparator_indices"] == [0, 1]
    checks += 3
    infeasible_detected = False
    try:
        CouplingSet([.5, .5], [.5, .5], [(diagonal, 1.1, 1.2)]).extreme(diagonal)
    except ValueError:
        infeasible_detected = True
    assert infeasible_detected
    checks += 1
    return dict(passed=True, checks=checks,
                weighted_transport_matches_analytic_endpoints=True,
                additive_interaction_identified=True,
                regret_matches_hand_calculation=True,
                common_policy_uncertainty_cancels=True,
                robust_budget_constraint_verified=True,
                moment_information_example_is_hypothetical=True)


def run_existing_deadlines():
    from full_counterfactual_model import build_records
    state = build_records(False)
    y, route = state["targets"]["gun_knife"], state["route"]
    detection = .9999**int((y & (route == 0)).sum())*.99**int((y & (route != 0)).sum())
    folder = BASE / "results/capacity_online_refined"
    original = json.loads((folder / "verification.json").read_text(encoding="utf-8-sig"))
    assert original["passed"] and original["simultaneous_confidence_covers_simulation_only"]
    cells = pd.read_csv(folder / "service_profile_probabilities.csv")
    reliability = pd.read_csv(folder / "service_reliability.csv")
    # Exactly the multiplicity correction used in the saved simulation source.
    configurations = int(original["service_configurations"])
    original_alpha = .05/(configurations*3*3*100)
    # Allocate one overall 5% error budget between success and risk families.
    success_family_alpha = risk_family_alpha = .025
    alpha = success_family_alpha/(configurations*3*3*100)
    keys = ["arrival_pattern", "readers", "inspectors", "mean_search_seconds",
            "positive_retention", "replicates", "deadline_minutes"]
    policies = ["fixed_218", "full_244", "stop_111"]
    rows, policy_rows, witnesses = [], [], []
    largest_gap = 0.
    for group_key, group in cells.groupby(keys, sort=True):
        key = dict(zip(keys, group_key))
        summary = reliability
        for name, value in key.items():
            summary = summary[summary[name].eq(value)]
        summary = summary.copy()
        reps = int(key["replicates"])
        risk_scale = int((y & (route == 2)).sum())*.99/int(y.sum())
        risk_extra = risk_scale*(np.sqrt(np.log(1100/risk_family_alpha)/(2*reps))
                                 - np.sqrt(np.log(1100/.05)/(2*reps)))
        summary.loc[summary.policy.eq("stop_111"), "miss_upper_confidence"] += risk_extra
        # All candidates satisfy the same saved upper bound on target miss risk.
        feasible = summary.loc[summary.miss_upper_confidence.le(.001+1e-12), "policy"]
        names = [name for name in policies if name in set(feasible)]
        if not names:
            continue
        point = np.array([group[group.policy.eq(name)].sort_values("profile").probability.to_numpy()
                          for name in names]).reshape((-1, 10, 10))
        raw_hits = point/detection*reps
        hits = np.rint(raw_hits).astype(int)
        assert np.max(np.abs(raw_hits-hits)) < 1e-8
        lower = np.where(hits > 0, beta.ppf(alpha/2, np.maximum(hits, 1), reps-hits+1), 0.)*detection
        upper = np.where(hits < reps, beta.ppf(1-alpha/2, hits+1, np.maximum(reps-hits, 1)), 1.)*detection
        model = CouplingSet(np.repeat(.1, 10), np.repeat(.1, 10))
        for a, name in enumerate(names):
            low, high = model.bounds(point[a])
            saved = summary[summary.policy.eq(name)].iloc[0]
            assert abs(low.value-saved.joint_lower) < 1e-10
            assert abs(high.value-saved.joint_upper) < 1e-10
            old_lower = np.where(hits[a] > 0,
                                 beta.ppf(original_alpha/2, np.maximum(hits[a], 1), reps-hits[a]+1), 0.)*detection
            assert abs(model.extreme(old_lower).value-saved.joint_lower_confidence) < 1e-10
            policy_rows.append({**key, "policy": name, "joint_lower": low.value,
                                "joint_upper": high.value,
                                "mean_readings_saved": saved.mean_readings,
                                "mean_searches_saved": saved.mean_searches})
        for label, low, high in [("point", point, point), ("simulation_envelope", lower, upper)]:
            pure = minimax_regret(model, low, high, randomized=False)
            mixture = minimax_regret(model, low, high)
            largest_gap = max(largest_gap, mixture["gap"], mixture["largest_transport_gap"])
            for design, result in [("deterministic", pure), ("randomized", mixture)]:
                row = {**key, "kernel_uncertainty": label, "selection": design,
                       "optimization_lower_bound": result["lower_bound"],
                       "worst_regret_upper_bound": result["upper_bound"],
                       "certificate_gap": result["gap"], "iterations": result["iterations"]}
                row.update({"weight_"+name: float(result["weights"][a]) for a, name in enumerate(names)})
                rows.append(row)
                for i in range(10):
                    for j in range(10):
                        witnesses.append({**key, "kernel_uncertainty": label, "selection": design,
                                          "row": i, "column": j, "probability": result["witness"][i, j]})
    pd.DataFrame(rows).to_csv(OUT / "deadline_regret_design.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(OUT / "deadline_policy_bounds.csv", index=False)
    pd.DataFrame(witnesses).to_csv(OUT / "deadline_regret_witnesses.csv", index=False)
    return dict(saved_simulations_reused=True, new_queue_simulations=0,
                deadline_scenarios=len(rows)//4, design_rows=len(rows),
                largest_numerical_certificate_gap=largest_gap,
                comparator="same prespecified three-policy catalogue",
                same_staff_service_arrivals_and_deadline=True,
                actual_mean_resource_use_is_policy_dependent=True,
                overall_simulation_error_budget=.05,
                success_family_alpha=success_family_alpha,
                risk_family_alpha=risk_family_alpha,
                lower_certificate_bounds_optimization_value_only=True,
                simulation_envelope_does_not_cover_source_population=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.smoke:
        verification = smoke()
        destination = OUT / "bounds_smoke_verification.json"
    else:
        previous = json.loads((OUT / "bounds_smoke_verification.json").read_text(encoding="utf-8"))
        assert previous["passed"] and previous["source_hashes"] == source_hashes()
        verification = {"passed": True, **run_existing_deadlines()}
        destination = OUT / "bounds_verification.json"
    verification.update(smoke=args.smoke, seconds=time.perf_counter()-started,
                        source_hashes=source_hashes())
    destination.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
