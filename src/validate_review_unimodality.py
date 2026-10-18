"""Exact checks of sufficient conditions for a single review-count optimum."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from math import comb
from pathlib import Path
import time

from finite_population_policy_kernel import fixed_sample_success_exact

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results/review_unimodality_20261018"


def signs_once(values):
    signs = [1 if b > a else -1 for a, b in zip(values, values[1:]) if a != b]
    return all(signs[t] >= signs[t+1] for t in range(len(signs)-1))


def check(maximum_N):
    fixed_cases = independent_cases = optimality_checks = 0
    fixed_decreasing = independent_decreasing = 0
    largest_error = Fraction(0)
    for N in range(1, maximum_N+1):
        for m in range(1, N+1):
            u = N-m
            for z in range(m, N+1):
                for k in range(u+1):
                    curve = [fixed_sample_success_exact(N, m, k, n, z)
                             for n in range(N+1)]
                    assert signs_once(curve), (N, m, z, k, curve)
                    first_negative = next((n for n in range(m-1, N)
                                           if curve[n+1] < curve[n]), N)
                    assert curve[first_negative] == max(curve)
                    fixed_cases += 1
                    fixed_decreasing += first_negative < N
                    optimality_checks += 1
                for theta in map(Fraction, [0, Fraction(1, 4), Fraction(1, 2),
                                            Fraction(3, 4), 1]):
                    curve = []
                    for n in range(N+1):
                        if n < m:
                            curve.append(Fraction(0))
                            continue
                        v, c = n-m, z-m
                        include = Fraction(comb(n, m), comb(N, m))
                        probability = sum((Fraction(comb(v, t))*theta**t*(1-theta)**(v-t)
                                           for t in range(min(v, c)+1)), Fraction(0))
                        value = include*probability
                        mixture = sum((Fraction(comb(u, k))*theta**k*(1-theta)**(u-k)
                            *fixed_sample_success_exact(N, m, k, n, z)
                            for k in range(u+1)), Fraction(0))
                        assert value == mixture
                        largest_error = max(largest_error, abs(value-mixture))
                        curve.append(value)
                    assert signs_once(curve), (N, m, z, theta, curve)
                    first_negative = next((n for n in range(m-1, N)
                                           if curve[n+1] < curve[n]), N)
                    assert curve[first_negative] == max(curve)
                    independent_cases += 1
                    independent_decreasing += first_negative < N
                    optimality_checks += 1
    return dict(maximum_N=maximum_N, fixed_load_cases=fixed_cases,
                independent_referral_cases=independent_cases,
                exact_first_negative_increment_checks=optimality_checks,
                fixed_cases_with_an_interior_optimum=fixed_decreasing,
                independent_cases_with_an_interior_optimum=independent_decreasing,
                binomial_mixture_matches_direct_sampling=True,
                largest_absolute_error=str(largest_error))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    files = [Path(__file__), BASE / "src/finite_population_policy_kernel.py"]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if not args.smoke:
        gate = json.loads((OUT / "smoke.json").read_text(encoding="utf-8"))
        if not gate["passed"] or gate["source_sha256"] != hashes:
            raise RuntimeError("A successful matching-source smoke test is required")
    record = dict(passed=True, smoke=args.smoke, **check(4 if args.smoke else 12),
                  source_sha256=hashes, seconds=time.perf_counter()-started)
    (OUT / ("smoke.json" if args.smoke else "full.json")).write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
