"""Response-support and source-label sensitivity for count-limited review.

The added target-incomplete responses have the same non-target referrals as
their recorded parent response. Their total probability is bounded by a stated
design allowance. The allowance is not estimated from the online experiment.
All response states are drawn before uniform review sampling. No plotting.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import hypergeom

from finite_population_policy_kernel import fixed_sample_success_exact
from full_counterfactual_model import build_records, transport_bounds

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results/target_support_upgrade_20261018"
TOL = 1e-8


class TargetFailureSet:
    """Recorded response margins plus at-most-epsilon target failure states."""

    def __init__(self, shape=(10, 10), epsilon=0.):
        self.shape = shape
        self.size = int(np.prod(shape))
        self.epsilon = epsilon
        marginal = []
        for i in range(shape[0]):
            row = np.zeros(shape); row[i, :] = 1.; marginal.append(row.ravel())
        for j in range(shape[1]):
            row = np.zeros(shape); row[:, j] = 1.; marginal.append(row.ravel())
        marginal = np.asarray(marginal)
        self.eq = np.concatenate((marginal, marginal), axis=1)
        self.rhs = np.r_[np.repeat(1 / shape[0], shape[0]), np.repeat(1 / shape[1], shape[1])]
        self.ub = np.r_[np.zeros(self.size), np.ones(self.size)][None, :]
        self.ub_rhs = [epsilon]

    def extreme(self, matrix, maximize=False):
        matrix = np.asarray(matrix, float)
        objective = np.r_[matrix.ravel(), np.zeros(self.size)]
        result = linprog(-objective if maximize else objective, A_eq=self.eq, b_eq=self.rhs,
                         A_ub=self.ub, b_ub=self.ub_rhs, bounds=(0., None), method="highs")
        if not result.success:
            raise RuntimeError(result.message)
        error = max(float(np.abs(self.eq @ result.x - self.rhs).max()),
                    max(0., float((self.ub @ result.x).item() - self.epsilon)),
                    max(0., float(-result.x.min())))
        dual = float(self.rhs @ result.eqlin.marginals + self.epsilon * result.ineqlin.marginals[0])
        gap = abs(float(result.fun) - dual)
        if error > TOL or gap > TOL:
            raise ArithmeticError((error, gap))
        return float(objective @ result.x), error, gap


def exact_two_by_two(matrix, epsilon, maximize=False):
    """Independent rational enumeration of all 2x2 coupling breakpoints."""
    values = [Fraction(x) for row in matrix for x in row]
    epsilon = Fraction(epsilon)
    order = sorted((j for j, v in enumerate(values) if (v < 0 if maximize else v > 0)),
                   key=lambda j: values[j], reverse=not maximize)
    base = [Fraction(0), Fraction(1, 2), Fraction(1, 2), Fraction(0)]
    slope = [1, -1, -1, 1]
    candidates = {Fraction(0), Fraction(1, 2)}
    a = Fraction(0); b = 0
    for j in order:
        a += base[j]; b += slope[j]
        if b:
            point = (epsilon - a) / b
            if 0 <= point <= Fraction(1, 2):
                candidates.add(point)
    results = []
    for x in candidates:
        weights = [a + b * x for a, b in zip(base, slope)]
        result = sum(w * v for w, v in zip(weights, values))
        remaining = epsilon
        for j in order:
            removed = min(remaining, weights[j])
            result -= removed * values[j]
            remaining -= removed
        results.append(result)
    return max(results) if maximize else min(results)


def fingerprints(state):
    return {"script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "kernel": hashlib.sha256((BASE / "src/finite_population_policy_kernel.py").read_bytes()).hexdigest(),
            "source": state["provenance"]["source_sha256"]}


def profile_kernel(state, target, capacity):
    """All counts at once, retaining each actual source response pair."""
    eligible = state["route"] == 2
    N = int(eligible.sum()); m = int((eligible & target).sum())
    B = int((~eligible).sum()); z = capacity - B
    responses = state["referrals"]
    complete = responses[:, eligible & target].sum(axis=1) == m
    loads = responses[:, eligible & ~target].sum(axis=1)
    all_n = np.arange(N + 1)
    result = np.zeros((N + 1, 100))
    selected = all_n >= m
    if z >= m:
        inclusion = hypergeom.pmf(m, N, m, all_n[selected])
        result[selected] = (inclusion[:, None] *
            hypergeom.cdf(z - m, N - m, loads[None, :], (all_n[selected] - m)[:, None]) * complete[None, :])
    return result.reshape(N + 1, 10, 10)


def source_label_definitions(state):
    labels = sorted(set().union(*state["first"].truth_set))
    rows = []
    for size in range(1, len(labels) + 1):
        for selected in itertools.combinations(labels, size):
            mask = state["first"].truth_set.map(lambda x: bool(x.intersection(selected))).to_numpy(bool)
            rows.append((" + ".join(selected), mask))
    return rows


def smoke(state):
    checks = 0; maximum = 0.
    for N in range(1, 6):
        for m in range(1, N + 1):
            ks = [0, (N - m) // 2, N - m, max(0, N - m - 1)]
            for n in range(N + 1):
                for z in range(N + 1):
                    kernel = [fixed_sample_success_exact(N, m, k, n, z) for k in ks]
                    full = [fixed_sample_success_exact(N, m, k, N, z) for k in ks]
                    for matrix in (np.asarray(kernel, object).reshape(2, 2),
                                   np.asarray([a - b for a, b in zip(kernel, full)], object).reshape(2, 2)):
                        for eps in (Fraction(0), Fraction(1, 4), Fraction(1, 2)):
                            model = TargetFailureSet((2, 2), float(eps))
                            for maximize in (False, True):
                                actual, error, gap = model.extreme(matrix, maximize)
                                expected = float(exact_two_by_two(matrix, eps, maximize))
                                maximum = max(maximum, abs(actual - expected), error, gap)
                                checks += 1
    assert maximum < TOL
    current = profile_kernel(state, state["targets"]["gun_knife"], 907)
    lo, ind, hi, _, _ = transport_bounds(current[218])
    assert abs(lo - .537194745507407) < 1e-10
    literal_error = 0.
    u = state["route"] == 2; target = state["targets"]["gun_knife"]
    for n in (0, 2, 217, 218, 244):
        for i, k in enumerate(state["referrals"][:, u & ~target].sum(axis=1)):
            expected = float(fixed_sample_success_exact(244, 2, int(k), n, 111))
            literal_error = max(literal_error, abs(expected - current[n].ravel()[i]))
    assert literal_error < TOL
    return dict(passed=True, fingerprints=fingerprints(state), exact_breakpoint_checks=checks,
                largest_independent_error=maximum, largest_source_kernel_error=literal_error,
                target_failure_chosen_before_uniform_sample=True)


def pure_summary(kernels, expected_load, capacity, model=None):
    lowers = np.asarray([transport_bounds(k)[0] if model is None else model.extreme(k)[0] for k in kernels])
    independent = kernels.mean(axis=(1, 2))
    robust = int(np.argmax(lowers)); indep = int(np.argmax(independent))
    valid = np.flatnonzero(expected_load <= capacity + 1e-12)
    mean_rule = int(valid.max()) if len(valid) else 0
    return lowers, dict(robust=robust, independent=indep, mean_load=mean_rule, full=len(kernels) - 1)


def rule_rows(kernels, lower, rules, capacity, epsilon, endpoint, regret=True):
    rows = []
    model = TargetFailureSet(epsilon=epsilon)
    for rule, n in rules.items():
        worst_regret = 0.; comparator = n
        if regret:
            for other in range(len(kernels)):
                diff = kernels[other] - kernels[n]
                if epsilon == 0:
                    loss = transport_bounds(diff)[2]
                else:
                    loss = model.extreme(diff, True)[0]
                if loss > worst_regret + 1e-12:
                    worst_regret, comparator = loss, other
        rows.append(dict(endpoint=endpoint, capacity=capacity, epsilon=epsilon, rule=rule,
                         reviews=n, S_lower=float(lower[n]), best_guarantee=float(lower.max()),
                         guarantee_loss=float(lower.max() - lower[n]),
                         worst_common_distribution_regret=worst_regret if regret else None,
                         regret_comparator= comparator if regret else None))
    return rows


def full(state):
    target = state["targets"]["gun_knife"]
    eligible = state["route"] == 2
    N = int(eligible.sum()); B = int((~eligible).sum())
    load = B + np.arange(N + 1) / N * state["referrals"][:, eligible].sum(axis=1).mean()
    policy_rows = []; rules_rows = []; requirements = []; pairing_rows = []
    largest_error = 0.; largest_gap = 0.
    epsilons = (0., .01, .025, .05, .10, .15, .20)
    for capacity in (907, 916, 930, 949, 969):
        kernels = profile_kernel(state, target, capacity)
        base_lower, base_rules = pure_summary(kernels, load, capacity)
        for eps in epsilons:
            model = TargetFailureSet(epsilon=eps)
            if eps == 0:
                lower = base_lower.copy()
            else:
                lower = []
                for k in kernels:
                    value, error, gap = model.extreme(k)
                    lower.append(value); largest_error = max(largest_error, error); largest_gap = max(largest_gap, gap)
                lower = np.asarray(lower)
            optimum = int(np.argmax(lower))
            assert abs(lower[-1] - max(0., base_lower[-1] - eps)) < TOL
            # Removing probability from a zero-success type can never improve the guarantee.
            assert np.all(lower <= base_lower + TOL)
            assert np.all(lower >= np.maximum(0., base_lower - eps) - TOL)
            for n, value in enumerate(lower):
                policy_rows.append(dict(capacity=capacity, epsilon=eps, reviews=n, S_lower=value))
            current_rules = dict(base_rules, robust=optimum, original_robust=base_rules["robust"])
            # Complete 245-policy common-distribution comparison for every main setting.
            rules_rows.extend(rule_rows(kernels, lower, current_rules, capacity, eps, "gun + knife"))
            for n in sorted(set((218, 244, optimum))):
                for tau in (.8, .9):
                    dreq = tau / lower[n] if lower[n] > 0 else None
                    requirements.append(dict(capacity=capacity, epsilon=eps, reviews=n,
                        S_lower=lower[n], tau=tau, required_dstar=dreq,
                        feasible=bool(dreq is not None and dreq <= 1 + TOL)))
            diff = kernels[218] - kernels[244]
            elo = model.extreme(diff)[0]; ehi = model.extreme(diff, True)[0]
            pairing_rows.append(dict(capacity=capacity, epsilon=eps, lower=elo, upper=ehi))
    pd.DataFrame(policy_rows).to_csv(OUT / "target_failure_all_counts.csv", index=False)
    pd.DataFrame(rules_rows).to_csv(OUT / "simple_rule_losses.csv", index=False)
    pd.DataFrame(requirements).to_csv(OUT / "target_failure_detection_requirements.csv", index=False)
    pd.DataFrame(pairing_rows).to_csv(OUT / "target_failure_paired_effects.csv", index=False)

    endpoint_rows = []; definition_rows = []; cache = {}
    for name, mask in source_label_definitions(state):
        m = int((mask & eligible).sum())
        definition_rows.append(dict(endpoint=name, targets=int(mask.sum()), eligible_targets=m,
            eligible_target_images=int(state["first"].loc[mask & eligible, "image"].nunique()),
            target_complete_profiles=int(np.all(state["referrals"][:, mask & eligible], axis=1).sum())))
        for capacity in (907, 916, 930, 949, 969):
            cache_key = (m, capacity)
            if cache_key not in cache:
                kernels = profile_kernel(state, mask, capacity)
                lower, rules = pure_summary(kernels, load, capacity)
                cache[cache_key] = (kernels, lower, rules)
            kernels, lower, rules = cache[cache_key]
            endpoint_rows.extend(rule_rows(kernels, lower, rules, capacity, 0., name))
    pd.DataFrame(definition_rows).to_csv(OUT / "all_source_label_definitions.csv", index=False)
    pd.DataFrame(endpoint_rows).to_csv(OUT / "all_source_label_rule_losses.csv", index=False)
    # Additional out-of-library non-target loads: arbitrary joint contamination.
    additional = []
    for row in rules_rows:
        for beta in (0., .01, .05, .10):
            additional.append(dict(**row, unseen_load_probability=beta,
                                   supported_success_lower=(1 - beta) * row["S_lower"]))
    pd.DataFrame(additional).to_csv(OUT / "joint_unseen_load_sensitivity.csv", index=False)
    report = dict(passed=True, fingerprints=fingerprints(state), capacities=5,
        target_failure_probability_allowances=list(epsilons), all_review_counts=N + 1,
        target_failure_policy_rows=len(policy_rows), rule_comparison_rows=len(rules_rows),
        target_definition_rows=len(definition_rows), target_definition_policy_rows=len(endpoint_rows),
        target_definitions_are_overlapping_sensitivities=True, target_failure_is_a_design_allowance=True,
        target_failure_can_depend_on_recorded_nontarget_workload=True,
        target_failure_drawn_before_review_sample=True, source_labels_unchanged=True,
        mean_risk_feasibility_not_assessed=True, observed_operational_deployment=False,
        largest_constraint_error=largest_error, largest_primal_dual_gap=largest_gap)
    (OUT / "full.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(); mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true"); mode.add_argument("--full", action="store_true")
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    state = build_records(False); start = time.perf_counter()
    if args.smoke:
        report = smoke(state)
        (OUT / "smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        gate = json.loads((OUT / "smoke.json").read_text(encoding="utf-8"))
        if not gate["passed"] or gate["fingerprints"] != fingerprints(state):
            raise RuntimeError("Run a matching passing --smoke before --full")
        report = full(state)
    report["elapsed_seconds"] = time.perf_counter() - start
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
