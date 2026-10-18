"""Exact finite-reading minimax check with exhaustive adaptive decisions.

The comparison controller may choose the next unread identity after each binary
referral result and may defer its final search subset until all readings finish.
This enlarges the causal stopping policy class. Hidden target labels are never
revealed. Every positive/target assignment and every adaptive reading action is
enumerated for the stated small populations; all arithmetic is rational.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, permutations
import hashlib
import json
from math import comb, factorial
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "capacity_capped_theory_20261018"


def masks_from(items, size):
    return [sum(1 << item for item in selected)
            for selected in combinations(items, size)]


def choose(n, k):
    return comb(n, k) if 0 <= k <= n else 0


def exact_kernel(N, m, k, n, z):
    total_positive = m+k
    return sum((Fraction(choose(total_positive, v)*choose(N-total_positive, n-v),
                         comb(N, n))
                *Fraction(choose(min(v, z), m), comb(total_positive, m))
                for v in range(max(0, n-(N-total_positive)), min(n, total_positive)+1)),
               Fraction(0))


def bayes_optimum(N, m, k, n, z, worlds, counters):
    """Exhaust identities, binary branches, and terminal accepted subsets."""
    @lru_cache(None)
    def consistent(positive_seen, negative_seen):
        return tuple((positive, target) for positive, target in worlds
                     if positive & positive_seen == positive_seen
                     and positive & negative_seen == 0)

    @lru_cache(None)
    def final_value(positive_seen, negative_seen):
        candidates = [i for i in range(N) if positive_seen & (1 << i)]
        accepted_sets = masks_from(candidates, min(z, len(candidates)))
        counters["terminal_search_subsets"] += len(accepted_sets)
        return max((sum(target & accepted == target
                        for _, target in consistent(positive_seen, negative_seen))
                    for accepted in accepted_sets), default=0)

    @lru_cache(None)
    def solve(positive_seen, negative_seen):
        if not consistent(positive_seen, negative_seen):
            return 0
        best = final_value(positive_seen, negative_seen)
        seen = positive_seen | negative_seen
        if seen.bit_count() == n:
            return best
        for item in range(N):
            bit = 1 << item
            if seen & bit:
                continue
            counters["adaptive_reading_actions"] += 1
            value = solve(positive_seen | bit, negative_seen) + solve(
                positive_seen, negative_seen | bit)
            best = max(best, value)
        return best

    best = Fraction(solve(0, 0), len(worlds))
    counters["belief_states"] += solve.cache_info().currsize
    return best


def literal_stopping_values(N, m, k, counters):
    """Every world: count successful literal permutations for all resources."""
    worlds = []
    for positive in masks_from(range(N), m+k):
        positive_items = [i for i in range(N) if positive & (1 << i)]
        worlds.extend((positive, target)
                      for target in masks_from(positive_items, m))
    sums = {world: {(n, z): 0 for n in range(N+1) for z in range(N+1)}
            for world in worlds}
    for positive, target in worlds:
        for order in permutations(range(N)):
            counters["world_permutation_pairs"] += 1
            for n in range(N+1):
                for z in range(N+1):
                    accepted = count_positive = 0
                    for position, item in enumerate(order):
                        if position >= n or count_positive >= z:
                            break
                        bit = 1 << item
                        if positive & bit:
                            count_positive += 1
                            accepted |= bit
                    sums[(positive, target)][(n, z)] += int(
                        target & accepted == target)
                    counters["stopping_path_checks"] += 1
    return worlds, sums


def run(maximum_N):
    start = time.perf_counter()
    counters = {name: 0 for name in ["configurations", "belief_states",
                "adaptive_reading_actions", "terminal_search_subsets",
                "world_permutation_pairs", "stopping_path_checks",
                "pointwise_attainment_checks"]}
    for N in range(1, maximum_N+1):
        for m in range(N+1):
            for k in range(N-m+1):
                worlds, sums = literal_stopping_values(N, m, k, counters)
                for n in range(N+1):
                    for z in range(N+1):
                        kernel = exact_kernel(N, m, k, n, z)
                        optimum = bayes_optimum(N, m, k, n, z, worlds, counters)
                        assert optimum == kernel, (N, m, k, n, z, optimum, kernel)
                        for world in worlds:
                            attained = Fraction(sums[world][(n, z)], factorial(N))
                            assert attained == optimum, (N, m, k, n, z, world,
                                                         attained, optimum)
                            counters["pointwise_attainment_checks"] += 1
                        counters["configurations"] += 1
    return {"maximum_population": maximum_N, "exact_arithmetic": True,
            "largest_error": 0, **counters,
            "runtime_seconds": round(time.perf_counter()-start, 4),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "controller_information": "completed binary referrals only; no target feedback",
            "comparison_controller": "adaptive distinct-image readings, then arbitrary observed-positive subset",
            "hidden_worlds": "all positive and target location assignments with fixed counts"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-n", type=int, default=3)
    parser.add_argument("--label", default="finite_minimax_smoke")
    args = parser.parse_args()
    result = run(args.max_n)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (args.label+".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
