"""Exact capacity-capped random review and fixed-count comparisons.

The capped policy reads a uniform random order until n_max readings or z
positive referrals. It uses no target labels. Source image records, response
profiles and whole-batch target-omission allowances retain their prior meaning.
Experiment and plotting are separate. Run --smoke before --full.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from functools import lru_cache
import hashlib
import itertools
import json
from math import comb
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from capacity_upgrade_model import stopped_rule
from finite_population_policy_kernel import fixed_sample_success_exact
from full_counterfactual_model import build_records, transport_bounds
from target_support_upgrade import TargetFailureSet, profile_kernel

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results/capacity_capped_review_20261018"


def choose(n, k):
    return comb(n, k) if 0 <= k <= n else 0


def count_pmf(N, L, n, h):
    return Fraction(choose(L, h) * choose(N - L, n - h), comb(N, n))


@lru_cache(None)
def capped_exact(N, m, a, k, n, z):
    """Independent rational count sums, including target-incomplete patterns."""
    assert 0 <= a <= m <= N and 0 <= k <= N - m
    assert 0 <= n <= N and z >= 0
    if z == 0:
        return (Fraction(m == 0), Fraction(0), Fraction(0), Fraction(0))
    L = a + k
    if L == 0:
        return (Fraction(m == 0), Fraction(0), Fraction(0), Fraction(n))
    searches = sum((count_pmf(N, L, n, h) * min(h, z) for h in range(L + 1)), Fraction())
    captured = Fraction(a, L) * searches
    success = sum((count_pmf(N, L, n, h) * Fraction(choose(min(h, z), m), comb(L, m))
                   for h in range(L + 1)), Fraction()) if a == m else Fraction(0)
    readings = sum((sum((count_pmf(N, L, t, h) for h in range(min(z, L + 1))), Fraction())
                    for t in range(n)), Fraction())
    return success, captured, searches, readings


def enumerate_orders(N, m, a, k, n, z):
    values = [Fraction(0)] * 4
    target = set(range(m)); positive = set(range(a)) | set(range(m, m + k))
    count = 0; domination = 0
    for order in itertools.permutations(range(N)):
        reads = searches = captured = 0
        for i in order:
            if reads >= n or searches >= z:
                break
            reads += 1
            if i in positive:
                searches += 1; captured += int(i in target)
        fixed_selected = set(order[:n])
        fixed_success = target <= fixed_selected and target <= positive and len(positive & fixed_selected) <= z
        capped_success = captured == m
        assert not fixed_success or capped_success
        domination += int(capped_success and not fixed_success)
        for j, v in enumerate((capped_success, captured, searches, reads)):
            values[j] += int(v)
        count += 1
    return tuple(v / count for v in values), count, domination


def fingerprints(state):
    return {"scripts": {p: hashlib.sha256((BASE / "src" / p).read_bytes()).hexdigest()
                         for p in ("capacity_capped_review_study.py", "capacity_upgrade_model.py",
                                   "target_support_upgrade.py")}, "source": state["provenance"]["source_sha256"]}


def smoke(state):
    cases = orders = dominated = 0; maximum = 0.
    for N in range(1, 6):
        for m in range(N + 1):
            for a in range(m + 1):
                for k in range(N - m + 1):
                    for n in range(N + 1):
                        for z in range(N + 1):
                            exact = capped_exact(N, m, a, k, n, z)
                            direct, checked, gained = enumerate_orders(N, m, a, k, n, z)
                            assert exact == direct, (N, m, a, k, n, z, exact, direct)
                            prior = stopped_rule(N, a, k, z, n)
                            prior_values = (prior["success"] if a == m else 0., prior["captured"],
                                            prior["searches"], prior["readings"])
                            maximum = max(maximum, max(abs(float(x) - y) for x, y in zip(exact, prior_values)))
                            cases += 1; orders += checked; dominated += gained
    assert maximum < 2e-13
    # Full source kernels independently checked by integer-binomial arithmetic.
    source_error = 0.
    eligible = state["route"] == 2; y = state["targets"]["gun_knife"]
    loads = state["referrals"][:, eligible & ~y].sum(axis=1)
    for z in (111, 134, 153, 173):
        for n in (218, 244):
            for k in np.unique(loads):
                exact = capped_exact(244, 2, 2, int(k), n, z)
                prior = stopped_rule(244, 2, int(k), z, n)
                source_error = max(source_error, max(abs(float(x) - prior[key])
                    for x, key in zip(exact, ("success", "captured", "searches", "readings"))))
    assert source_error < 1e-10
    return dict(passed=True, fingerprints=fingerprints(state), exact_configuration_checks=cases,
                individual_random_orders_checked=orders, additional_success_orders=dominated,
                maximum_old_kernel_error=maximum, maximum_source_kernel_error=source_error,
                every_fixed_success_path_retained=True, target_incomplete_states_included=True)


def affine_extreme(positive, failure, epsilon, maximize=False):
    """One common Q+,Q- distribution, with a failure-state metric supplied."""
    model = TargetFailureSet(epsilon=epsilon)
    objective = np.r_[np.asarray(positive).ravel(), np.asarray(failure).ravel()]
    result = linprog(-objective if maximize else objective, A_eq=model.eq, b_eq=model.rhs,
                     A_ub=model.ub, b_ub=model.ub_rhs, bounds=(0., None), method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    error = max(float(np.abs(model.eq @ result.x - model.rhs).max()),
                max(0., float((model.ub @ result.x).item()) - epsilon))
    dual = float(model.rhs @ result.eqlin.marginals + epsilon * result.ineqlin.marginals[0])
    gap = abs(float(result.fun) - dual)
    assert max(error, gap) < 1e-8
    return float(objective @ result.x)


def supported_bounds(matrix, epsilon):
    if epsilon == 0:
        lo, mean, hi, _, _ = transport_bounds(matrix)
        return lo, mean, hi
    model = TargetFailureSet(epsilon=epsilon)
    return model.extreme(matrix)[0], float(np.asarray(matrix).mean()), model.extreme(matrix, True)[0]


@lru_cache(None)
def source_metrics(N, a, k, n, z):
    return stopped_rule(N, a, k, z, n)


def metric_matrices(N, m, loads, n, z):
    complete = {k: np.asarray([source_metrics(N, m, int(v), n, z)[k] for v in loads]).reshape(10, 10)
                for k in ("success", "captured", "searches", "readings")}
    deleted = {}
    one_missing = {}
    for name in ("captured", "searches", "readings"):
        deleted[name] = np.asarray([source_metrics(N, 0, int(v), n, z)[name] for v in loads]).reshape(10, 10)
        one_missing[name] = np.asarray([source_metrics(N, max(0, m - 1), int(v), n, z)[name] for v in loads]).reshape(10, 10)
    return complete, deleted, one_missing


def full(state):
    eligible = state["route"] == 2; y = state["targets"]["gun_knife"]
    N = int(eligible.sum()); m = int((eligible & y).sum()); B = int((~eligible).sum())
    loads = state["referrals"][:, eligible & ~y].sum(axis=1)
    rows = []; paired = []; all_counts = []; metrics = []; frontier = []
    epsilons = (0., .025, .05, .10)
    for capacity in (907, 916, 930, 949, 969):
        z = capacity - B
        fixed = profile_kernel(state, y, capacity)
        lower_fixed = np.asarray([transport_bounds(k)[0] for k in fixed])
        independent_fixed = np.asarray([k.mean() for k in fixed])
        fixedbest = int(np.argmax(lower_fixed)); independent = int(np.argmax(independent_fixed))
        stopped = np.asarray([metric_matrices(N, m, loads, n, z)[0]["success"] for n in range(N + 1)])
        assert np.all(stopped >= fixed - 1e-12)
        assert np.all(np.diff(stopped, axis=0) >= -1e-12)
        for n in range(N + 1):
            lo, ind, hi = supported_bounds(stopped[n], 0.)
            all_counts.append(dict(capacity=capacity, nmax=n, lower=lo, independent=ind, upper=hi))
        for eps in epsilons:
            model = TargetFailureSet(epsilon=eps)
            fxlow = np.asarray([supported_bounds(k, eps)[0] for k in fixed])
            robust_fixed = int(np.argmax(fxlow))
            catalogue = dict(fixed_largest_guarantee=("fixed", robust_fixed),
                             fixed_independence=("fixed", independent), fixed_full=("fixed", N),
                             capped_218=("capped", 218), capped_full_allowance=("capped", N))
            for label, (family, n) in catalogue.items():
                matrix = fixed[n] if family == "fixed" else stopped[n]
                lo, nominal_ind, hi = supported_bounds(matrix, eps)
                rows.append(dict(capacity=capacity, z=z, zeta=eps, policy=label, nmax=n,
                    lower_S=lo, nominal_independent_S=nominal_ind, upper_S=hi,
                    lower_J_specified=lo * (.9999 ** 286) * (.99 ** 4)))
            for n in sorted(set((218, fixedbest, independent, N))):
                for capmax in sorted(set((n, N))):
                    D = stopped[capmax] - fixed[n]
                    elo, ind, ehi = supported_bounds(D, eps)
                    assert elo >= -1e-10
                    paired.append(dict(capacity=capacity, zeta=eps, capped_nmax=capmax,
                                       fixed_reviews=n, lower=elo, nominal_independent=ind, upper=ehi))
            for n in (218, N):
                positive, deleted, one_missing = metric_matrices(N, m, loads, n, z)
                target_min = supported_bounds(positive["captured"], eps)[0]
                riskmax = (.0486 + m - .99 * target_min) / 290
                riskmin = (.0486 + m - .99 * supported_bounds(positive["captured"], eps)[2]) / 290
                read_lo = affine_extreme(positive["readings"], one_missing["readings"], eps)
                read_hi = affine_extreme(positive["readings"], deleted["readings"], eps, True)
                search_lo = affine_extreme(positive["searches"], deleted["searches"], eps)
                search_hi = affine_extreme(positive["searches"], one_missing["searches"], eps, True)
                metrics.append(dict(capacity=capacity, zeta=eps, nmax=n,
                    mean_reads_lower=read_lo, mean_reads_nominal=float(positive["readings"].mean()), mean_reads_upper=read_hi,
                    mean_total_searches_lower=B + search_lo, mean_total_searches_nominal=B + float(positive["searches"].mean()),
                    mean_total_searches_upper=B + search_hi, target_referrals_lower=target_min,
                    mean_target_risk_lower=riskmin, mean_target_risk_upper=riskmax,
                    passes_mean_risk_0p1pct=bool(riskmax <= .001 + 1e-12)))
        # Minimum capacity under the capped full-reading allowance, across all integer capacities.
    for capacity in range(B, B + N + 1):
        matrices = metric_matrices(N, m, loads, N, capacity - B)[0]
        for eps in epsilons:
            lo, ind, hi = supported_bounds(matrices["success"], eps)
            target_min = supported_bounds(matrices["captured"], eps)[0]
            frontier.append(dict(capacity=capacity, zeta=eps, lower_S=lo, upper_S=hi,
                mean_risk_upper=(.0486 + m - .99 * target_min) / 290,
                final_success_specified=lo * (.9999 ** 286) * (.99 ** 4)))
    pd.DataFrame(rows).to_csv(OUT / "policy_comparison.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "same_distribution_effects.csv", index=False)
    pd.DataFrame(all_counts).to_csv(OUT / "all_capped_allowances.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT / "capped_workload_and_mean_risk.csv", index=False)
    pd.DataFrame(frontier).to_csv(OUT / "capped_capacity_frontier.csv", index=False)
    report = dict(passed=True, fingerprints=fingerprints(state), policy_comparison_rows=len(rows),
        paired_comparison_rows=len(paired), all_allowance_rows=len(all_counts), metric_rows=len(metrics),
        capacity_frontier_rows=len(frontier), same_random_order_pathwise_dominance=True,
        monotone_in_nmax=True, mean_risk_accounts_for_target_truncation=True,
        physical_detection_is_preexisting_specified_scenario=True, new_observational_data=False)
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
        gate = json.loads((OUT / "smoke.json").read_text())
        if not gate["passed"] or gate["fingerprints"] != fingerprints(state):
            raise RuntimeError("A matching passing smoke test is required")
        report = full(state)
    print(json.dumps(dict(report, elapsed_seconds=time.perf_counter() - start), indent=2))


if __name__ == "__main__":
    main()
