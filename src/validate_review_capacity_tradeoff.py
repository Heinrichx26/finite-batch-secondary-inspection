"""Smoke-gated checks of capacity savings allowed by finite target coverage.

No training, new participant data, service simulation, or plotting is used.
Direct subset enumeration checks the inequalities independently of the
hypergeometric implementation used for the saved-record calculation.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
from itertools import combinations
import json
from math import comb
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import hypergeom

from resource_capacity_bounds import minimum_robust_capacity
from robust_policy_bounds import CouplingSet

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results/review_capacity_tradeoff_20261018"


def source_hashes():
    paths = [Path(__file__), BASE / "src/resource_capacity_bounds.py",
             BASE / "src/robust_policy_bounds.py"]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def direct_success(N, m, k, n, z):
    targets = set(range(m))
    referrals = targets | set(range(m, m+k))
    selected = list(combinations(range(N), n))
    hits = sum(targets.issubset(s) and len(set(s) & referrals) <= z
               for s in map(set, selected))
    return Fraction(hits, len(selected))


def exact_checks(maximum_N):
    checks = strict_lower = strict_upper = 0
    equality_lower = equality_upper = 0
    for N in range(1, maximum_N+1):
        for m in range(1, N+1):
            for k in range(N-m+1):
                for n in range(N+1):
                    inclusion = Fraction(comb(n, m), comb(N, m)) if n >= m else Fraction(0)
                    for z in range(N+1):
                        value = direct_success(N, m, k, n, z)
                        lower = inclusion * int(m+k <= z)
                        upper = inclusion * int(m+k <= z+N-n)
                        assert lower <= value <= upper, (N, m, k, n, z)
                        checks += 2
                        strict_lower += value > lower
                        strict_upper += value < upper
                        equality_lower += value == lower
                        equality_upper += value == upper
    return dict(maximum_N=maximum_N, exact_inequality_checks=checks,
                strict_lower_examples=strict_lower, strict_upper_examples=strict_upper,
                attained_lower_examples=equality_lower,
                attained_upper_examples=equality_upper,
                largest_error=0, enumeration_includes_omitted_targets=True)


def source_calculation():
    source = BASE / "results/general_method_20261017/full_review_marginal_loads.csv"
    table = pd.read_csv(source)
    selected = table[table.endpoint.eq("gun_knife")]
    x = selected[selected.response_group.eq(1)].encoded_load.to_numpy(float)
    y = selected[selected.response_group.eq(2)].encoded_load.to_numpy(float)
    p = selected[selected.response_group.eq(1)].marginal_weight.to_numpy(float)
    q = selected[selected.response_group.eq(2)].marginal_weight.to_numpy(float)
    assert len(x) == len(y) == 10 and np.allclose(p, .1) and np.allclose(q, .1)
    N, m, B = 244, 2, 796
    K = x[:, None]+y[None, :]
    complete = np.isfinite(K)
    finite_max = int(K[complete].max())
    ceiling = max(0., p[np.isfinite(x)].sum()+q[np.isfinite(y)].sum()-1.)
    Cmax = B+m+finite_max
    model = CouplingSet(p, q)
    largest_LP_error = 0.
    kernel_calls = 0

    def lower(n, capacity, detection):
        nonlocal kernel_calls
        if n < m or capacity < B+m:
            return 0.
        inclusion = comb(n, m)/comb(N, m)
        # scipy hypergeometric sampling is used only for these record results.
        kernel = detection*inclusion*np.where(complete,
            hypergeom.cdf(capacity-B-m, N-m, np.where(complete, K, 0), n-m), 0.)
        rows, cols = linear_sum_assignment(kernel)
        kernel_calls += 1
        return float(kernel[rows, cols].sum()/len(x))

    rows, summary = [], []
    scenarios = [(1., .5), (1., .8), (1., .9), (1., .95), (1., .99),
                 (.93351077498, .8), (.93351077498, .9)]
    for detection, target in scenarios:
        feasible_counts = [n for n in range(N+1)
                           if detection*comb(n, m)/comb(N, m)*ceiling >= target-1e-12]
        nmin = min(feasible_counts)
        full_requirement = minimum_robust_capacity(x, p, y, q, target, B+m, detection)
        full_capacity = full_requirement.minimum_capacity
        assert full_capacity is not None
        candidates = []
        count_lower_bounds = []
        for n in feasible_counts:
            inclusion = comb(n, m)/comb(N, m)
            shifted_full = minimum_robust_capacity(x, p, y, q,
                target/inclusion, B+m, detection)
            assert shifted_full.feasible
            necessary = max(B+m, shifted_full.minimum_capacity-(N-n))
            lo, hi = B+m-1, Cmax
            while hi-lo > 1:
                mid = (lo+hi)//2
                if lower(n, mid, detection) >= target-1e-10:
                    hi = mid
                else:
                    lo = mid
            attained = lower(n, hi, detection)
            previous = lower(n, hi-1, detection)
            assert attained >= target-1e-10 and previous < target-1e-10
            assert hi >= necessary
            candidates.append((hi, n, attained, previous))
            count_lower_bounds.append(necessary)
            rows.append(dict(detection=detection, required_success=target,
                review_count=n, target_inclusion=inclusion,
                count_specific_capacity_lower_bound=necessary,
                exact_minimum_capacity=hi, lower_success=attained,
                lower_success_one_less_capacity=previous))
        minimum = min(row[0] for row in candidates)
        optimal = [row[1] for row in candidates if row[0] == minimum]
        coarse = max(B+m, full_capacity-(N-nmin))
        sharper = min(count_lower_bounds)
        assert coarse <= sharper <= minimum <= full_capacity
        for n in optimal + [N]:
            capacity = next(item[0] for item in candidates if item[1] == n)
            inclusion = comb(n, m)/comb(N, m)
            kernel = detection*inclusion*np.where(complete,
                hypergeom.cdf(capacity-B-m, N-m, np.where(complete, K, 0), n-m), 0.)
            LP = model.extreme(kernel).value
            err = abs(LP-lower(n, capacity, detection))
            largest_LP_error = max(largest_LP_error, err)
            assert err < 1e-10
        summary.append(dict(detection=detection, required_success=target,
            minimum_review_count_from_coverage=nmin,
            maximum_omitted_reviews=N-nmin,
            full_review_minimum_capacity=full_capacity,
            simple_capacity_lower_bound=coarse,
            count_specific_global_lower_bound=sharper,
            exact_global_minimum_capacity=minimum,
            capacity_saving=full_capacity-minimum,
            optimal_review_counts=optimal,
            simple_bound_gap=minimum-coarse,
            sharper_bound_gap=minimum-sharper))
    # A separate transparent coverage-only table varies target count, not data.
    coverage = []
    for target_count in [1, 2, 5, 10]:
        for relative_requirement in [.8, .9, .95, .99]:
            first = next(n for n in range(N+1)
                         if comb(n, target_count)/comb(N, target_count)
                         >= relative_requirement-1e-12)
            coverage.append(dict(N=N, target_count=target_count,
                requirement_relative_to_referral_detection_ceiling=relative_requirement,
                minimum_review_count=first, maximum_omitted_reviews=N-first,
                interpretation="mathematical coverage calculation; target count varied"))
    pd.DataFrame(rows).to_csv(OUT / "source_count_capacity_requirements.csv", index=False)
    pd.DataFrame(summary).to_json(OUT / "source_capacity_savings.json", orient="records", indent=2)
    pd.DataFrame(coverage).to_csv(OUT / "coverage_limits.csv", index=False)
    curve = []
    for target_count in [1, 2, 5, 10]:
        inclusion = np.asarray([comb(n, target_count)/comb(N, target_count)
                                for n in range(N+1)])
        for relative_requirement in np.linspace(.5, 1., 501):
            first = int(np.flatnonzero(inclusion >= relative_requirement-1e-12)[0])
            curve.append(dict(N=N, target_count=target_count,
                requirement_relative_to_referral_detection_ceiling=float(relative_requirement),
                minimum_review_count=first, maximum_omitted_reviews=N-first))
    pd.DataFrame(curve).to_csv(OUT / "coverage_savings_curve.csv", index=False)
    return dict(source_input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_scenarios=summary, eligible_count=N, eligible_targets=m,
                mandatory_searches=B, minimum_complete_target_referral=ceiling,
                saturated_capacity=Cmax, kernel_calls=kernel_calls,
                largest_assignment_vs_LP_error=largest_LP_error,
                new_participant_data=0, new_service_simulations=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    hashes = source_hashes()
    if not args.smoke:
        gate = json.loads((OUT / "smoke.json").read_text(encoding="utf-8"))
        if not gate["passed"] or gate["source_sha256"] != hashes:
            raise RuntimeError("A successful matching-source smoke test is required")
    record = dict(passed=True, smoke=args.smoke,
                  **exact_checks(4 if args.smoke else 8))
    if not args.smoke:
        record.update(source_calculation())
    record.update(source_sha256=hashes, seconds=time.perf_counter()-started)
    destination = OUT / ("smoke.json" if args.smoke else "full.json")
    destination.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
