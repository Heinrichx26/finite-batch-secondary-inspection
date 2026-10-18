"""Exact checks for referral stopping with a hard reading and search allowance.

No participant-data calculation or service simulation is executed here. Literal
permutations independently verify the two closed-form success calculations,
the union of successful fixed-prefix events, and resource monotonicity.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
from itertools import combinations, permutations
import json
from math import comb, factorial
from pathlib import Path
import time

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results/capacity_capped_theory_20261018"


def choose(n, k):
    return comb(n, k) if 0 <= k <= n else 0


def hypergeometric_cdf(N, referrals, readings, limit):
    return sum((Fraction(choose(referrals, t)*choose(N-referrals, readings-t),
                         comb(N, readings))
                for t in range(max(0, readings-(N-referrals)),
                               min(referrals, readings, limit)+1)), Fraction(0))


def capped_by_positive_count(N, m, k, nmax, z):
    """Return exact success, captured targets, searches and image readings."""
    if not (0 <= m <= N and 0 <= k <= N-m and 0 <= nmax <= N and z >= 0):
        raise ValueError("Invalid finite-population or resource counts")
    r = m+k
    if z == 0:
        return Fraction(m == 0), Fraction(0), Fraction(0), Fraction(0)
    if r == 0:
        return Fraction(1), Fraction(0), Fraction(0), Fraction(nmax)
    success = searched = Fraction(0)
    for v in range(max(0, nmax-(N-r)), min(r, nmax)+1):
        weight = Fraction(choose(r, v)*choose(N-r, nmax-v), comb(N, nmax))
        accepted = min(v, z)
        success += weight*Fraction(choose(accepted, m), comb(r, m))
        searched += weight*accepted
    captured = Fraction(m, r)*searched
    readings = sum((hypergeometric_cdf(N, r, v, z-1)
                    for v in range(nmax)), Fraction(0))
    return success, captured, searched, readings


def capped_by_final_target(N, m, k, nmax, z):
    if m == 0:
        return Fraction(1)
    if nmax < m or z < m:
        return Fraction(0)
    return sum((Fraction(comb(j-1, m-1), comb(N, m))
                *hypergeometric_cdf(N-m, k, j-m, z-m)
                for j in range(m, nmax+1)), Fraction(0))


def direct_permutation_checks(maximum_N):
    path_checks = success_cases = moments_checked = strict_dominance_paths = 0
    largest_error = Fraction(0)
    for N in range(1, maximum_N+1):
        for m in range(N+1):
            for k in range(N-m+1):
                r = m+k
                sums = {(nmax, z): [0, 0, 0, 0]
                        for nmax in range(N+1) for z in range(N+1)}
                for order in permutations(range(N)):
                    fixed_target = [0]
                    fixed_positive = [0]
                    for item in order:
                        fixed_target.append(fixed_target[-1]+int(item < m))
                        fixed_positive.append(fixed_positive[-1]+int(item < r))
                    outcomes = {}
                    for nmax in range(N+1):
                        for z in range(N+1):
                            # Literal causal controller receives binary referrals only.
                            read = positives = captured = 0
                            while read < nmax and positives < z:
                                item = order[read]
                                read += 1
                                if item < r:
                                    positives += 1
                                    captured += int(item < m)
                            success = int(captured == m)
                            successful_fixed = [int(fixed_target[n] == m
                                                    and fixed_positive[n] <= z)
                                                for n in range(nmax+1)]
                            assert success == max(successful_fixed)
                            assert positives <= z and read <= nmax
                            strict_dominance_paths += success > successful_fixed[-1]
                            outcomes[nmax, z] = (success, captured, positives, read)
                            sums[nmax, z] = [a+b for a, b in
                                            zip(sums[nmax, z], outcomes[nmax, z])]
                            path_checks += 2
                    for nmax in range(N+1):
                        for z in range(N+1):
                            current = outcomes[nmax, z]
                            if nmax < N:
                                later = outcomes[nmax+1, z]
                                assert current[0] <= later[0] and current[1] <= later[1]
                                path_checks += 2
                            if z < N:
                                larger = outcomes[nmax, z+1]
                                assert current[0] <= larger[0] and current[1] <= larger[1]
                                path_checks += 2
                for (nmax, z), total in sums.items():
                    direct = tuple(Fraction(t, factorial(N)) for t in total)
                    predicted = capped_by_positive_count(N, m, k, nmax, z)
                    final_target = capped_by_final_target(N, m, k, nmax, z)
                    assert direct == predicted, (N, m, k, nmax, z, direct, predicted)
                    assert predicted[0] == final_target
                    largest_error = max(largest_error,
                                        max(abs(a-b) for a, b in zip(direct, predicted)))
                    success_cases += 1
                    moments_checked += 4
    return dict(maximum_N=maximum_N, literal_path_checks=path_checks,
                exact_kernel_cases=success_cases, exact_moment_comparisons=moments_checked,
                strict_improvement_over_same_fixed_prefix_paths=strict_dominance_paths,
                largest_absolute_error=str(largest_error))


def increment_and_minimax_checks(maximum_N):
    increment_cases = full_formula_cases = minimax_cases = 0
    for N in range(1, maximum_N+1):
        for m in range(N+1):
            for k in range(N-m+1):
                r = m+k
                for z in range(N+1):
                    full = capped_by_positive_count(N, m, k, N, z)[0]
                    expected = (Fraction(1) if m == 0 else
                                Fraction(choose(min(z, r), m), comb(r, m)))
                    assert full == expected
                    full_formula_cases += 1
                    for n in range(N):
                        old = capped_by_final_target(N, m, k, n, z)
                        new = capped_by_final_target(N, m, k, n+1, z)
                        gain = (Fraction(0) if m == 0 or n+1 < m or z < m else
                                Fraction(comb(n, m-1), comb(N, m))
                                *hypergeometric_cdf(N-m, k, n+1-m, z-m))
                        assert new-old == gain >= 0
                        increment_cases += 1
    # Every terminal accepted set of a binary-response-only policy satisfies
    # the average-target-placement upper bound; randomized rules are mixtures.
    for r in range(1, min(maximum_N, 8)+1):
        for m in range(1, r+1):
            targets = list(map(set, combinations(range(r), m)))
            for z in range(r+1):
                bound = Fraction(choose(z, m), comb(r, m))
                for size in range(z+1):
                    for accepted in map(set, combinations(range(r), size)):
                        average = Fraction(sum(t.issubset(accepted) for t in targets),
                                           len(targets))
                        assert average <= bound
                        minimax_cases += 1
                accepted_sets = list(map(set, combinations(range(r), z)))
                for target in targets:
                    attained = Fraction(sum(target.issubset(a) for a in accepted_sets),
                                        len(accepted_sets))
                    assert attained == bound
                    minimax_cases += 1
    return dict(maximum_N_for_increment=maximum_N,
                nonnegative_exact_increment_checks=increment_cases,
                full_scan_closed_form_checks=full_formula_cases,
                target_placement_minimax_checks=minimax_cases)


def mean_risk_counterexample():
    # One target, one positive non-target and one negative image.
    # The fixed n=2 rule includes the target with probability 2/3, even when
    # both assigned searches eventually finish after exceeding count capacity.
    fixed_success = Fraction(1, 3)
    fixed_risk = Fraction(1, 3)
    capped_success, captured, _, _ = capped_by_positive_count(3, 1, 1, 2, 1)
    capped_risk = 1-captured
    assert capped_success == Fraction(1, 2) > fixed_success
    assert capped_risk == Fraction(1, 2) > fixed_risk
    return dict(N=3, m=1, k=1, maximum_readings=2, search_allowance=1,
                fixed_success=str(fixed_success), capped_success=str(capped_success),
                fixed_mean_target_risk=str(fixed_risk),
                capped_mean_target_risk=str(capped_risk),
                physical_detection=1,
                demonstrates_no_mean_risk_dominance=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if not args.smoke:
        gate = json.loads((OUT / "smoke.json").read_text(encoding="utf-8"))
        if not gate["passed"] or gate["source_sha256"] != hashes:
            raise RuntimeError("A matching-source successful smoke test is required")
    started = time.perf_counter()
    results = dict(passed=True, smoke=args.smoke,
        **direct_permutation_checks(3 if args.smoke else 7),
        **increment_and_minimax_checks(4 if args.smoke else 10),
        mean_risk_counterexample=mean_risk_counterexample(),
        source_sha256=hashes, seconds=time.perf_counter()-started)
    (OUT / ("smoke.json" if args.smoke else "full.json")).write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
