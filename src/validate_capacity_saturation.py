"""Check the capacity-saturation identity by exhaustive item-set enumeration.

The reference calculation explicitly tests every target's selection and referral
and counts all selected referrals.  It does not use a hypergeometric formula.
All probability comparisons use exact rational arithmetic.  These checks assess
the stated identity and decision implications, not their scientific novelty.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from itertools import combinations
import json
from math import comb
from pathlib import Path
import random
import time


def inclusion(population: int, targets: int, reviews: int) -> Fraction:
    return (Fraction(comb(reviews, targets), comb(population, targets))
            if reviews >= targets else Fraction(0))


def direct_table(population: int, target_items: tuple[int, ...],
                 referral_mask: int, mandatory: int) -> list[list[Fraction]]:
    """Enumerate every sampled item set and directly apply the success event."""
    table: list[list[Fraction]] = []
    for reviews in range(population + 1):
        success_by_load: Counter[int] = Counter()
        for sample_tuple in combinations(range(population), reviews):
            selected = set(sample_tuple)
            if not all(item in selected and referral_mask & (1 << item)
                       for item in target_items):
                continue
            searches = mandatory + sum(bool(referral_mask & (1 << item))
                                       for item in selected)
            success_by_load[searches] += 1
        table.append([
            Fraction(sum(count for load, count in success_by_load.items()
                         if load <= capacity), comb(population, reviews))
            for capacity in range(mandatory, mandatory + population + 2)
        ])
    return table


class Checks:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def equal(self, left: Fraction, right: Fraction, label: str,
              context: object) -> None:
        if left != right:
            raise AssertionError((label, context, str(left), str(right)))
        self.counts[label] += 1

    def at_least(self, left: Fraction, right: Fraction, label: str,
                 context: object) -> None:
        if left < right:
            raise AssertionError((label, context, str(left), str(right)))
        self.counts[label] += 1


def check_distribution(population: int, targets: int, masks: list[int],
                       weights: list[Fraction], checks: Checks,
                       features: Counter[str]) -> None:
    """Allow zero-weight patterns in the list used for the saturation bound."""
    assert sum(weights) == 1 and all(weight >= 0 for weight in weights)
    target_items = tuple(range(targets))
    mandatory = 2
    tables = [direct_table(population, target_items, mask, mandatory)
              for mask in masks]
    complete = [all(mask & (1 << item) for item in target_items)
                for mask in masks]
    probability_a = sum((weight for weight, event in zip(weights, complete)
                         if event), Fraction(0))
    supported_loads = [sum(bool(mask & (1 << item))
                           for item in range(targets, population))
                       for mask, event in zip(masks, complete) if event]
    saturation = (mandatory + targets + max(supported_loads)
                  if supported_loads else None)
    mixed = [[sum((weight * table[n][c]
                   for weight, table in zip(weights, tables)), Fraction(0))
              for c in range(population + 2)]
             for n in range(population + 1)]
    for n in range(population + 1):
        for c in range(population + 1):
            checks.at_least(mixed[n][c + 1], mixed[n][c],
                            "capacity_monotonicity", (population, targets, n, c))
    features["distributions"] += 1
    features["zero_target_distributions"] += targets == 0
    features["multiple_target_distributions"] += targets >= 2
    features["zero_weight_distributions"] += any(weight == 0 for weight in weights)
    features["nonuniform_mixtures"] += len(set(weights)) > 1
    features["no_complete_response_distributions"] += not supported_loads
    if saturation is not None:
        features["failed_response_load_above_saturation"] += any(
            not event and mandatory + mask.bit_count() > saturation
            for mask, event in zip(masks, complete))
    for detection in (Fraction(1), Fraction(2, 3), Fraction(7, 11)):
        for n in range(population + 1):
            for c, value in enumerate(mixed[n]):
                context = (population, targets, n, mandatory + c,
                           str(detection), masks, [str(w) for w in weights])
                if saturation is None:
                    checks.equal(detection * value, Fraction(0),
                                 "no_complete_response_zero_success", context)
                elif mandatory + c >= saturation:
                    expected = detection * inclusion(population, targets, n) * probability_a
                    checks.equal(detection * value, expected,
                                 "saturation_identity", context)
                    checks.at_least(detection * mixed[population][c], detection * value,
                                    "full_review_weak_dominance", context)
                    if targets == 0:
                        checks.equal(detection * value,
                                     detection * mixed[population][c],
                                     "zero_target_tie", context)
                    elif n < population and probability_a > 0:
                        assert detection * mixed[population][c] > detection * value
                        checks.counts["strict_full_review_dominance"] += 1
                    if n < population:
                        checks.at_least(mixed[n + 1][c], value,
                                        "review_monotonicity_at_saturation", context)


def check_common_pairing(population: int, targets: int, rng: random.Random,
                         checks: Checks, features: Counter[str]) -> None:
    """For 2x2 marginals, linear-program extrema occur at the two endpoints."""
    group_size = max(1, population // 2)
    first_mask = (1 << group_size) - 1
    second_mask = ((1 << population) - 1) ^ first_mask
    target_mask = (1 << targets) - 1
    row_masks = [target_mask & first_mask, rng.randrange(1 << population) & first_mask]
    col_masks = [second_mask, rng.randrange(1 << population) & second_mask]
    masks = [row | col for row in row_masks for col in col_masks]
    target_items = tuple(range(targets))
    tables = [direct_table(population, target_items, mask, 2) for mask in masks]
    complete = [all(mask & (1 << item) for item in target_items) for mask in masks]
    finite_loads = [sum(bool(mask & (1 << item)) for item in range(targets, population))
                    for mask, event in zip(masks, complete) if event]
    saturation = 2 + targets + max(finite_loads) if finite_loads else 2
    for p0, q0 in ((Fraction(1, 3), Fraction(2, 5)),
                   (Fraction(1), Fraction(1, 2))):
        lo = max(Fraction(0), p0 + q0 - 1)
        hi = min(p0, q0)
        xs = sorted({lo, hi, (lo + hi) / 2})
        laws = [[x, p0 - x, q0 - x, 1 - p0 - q0 + x] for x in xs]
        endpoint_indices = [xs.index(lo), xs.index(hi)]
        complete_probs = [sum((w for w, event in zip(law, complete) if event), Fraction(0))
                          for law in laws]
        g = min(complete_probs[index] for index in endpoint_indices)
        features["common_pairing_configurations"] += 1
        for c in range(max(0, saturation - 2), population + 2):
            values = [[sum((w * table[n][c] for w, table in zip(law, tables)), Fraction(0))
                       for n in range(population + 1)] for law in laws]
            for detection in (Fraction(1), Fraction(3, 4)):
                for n in range(population + 1):
                    lower = min(values[index][n] for index in endpoint_indices)
                    checks.equal(detection * lower,
                                 detection * inclusion(population, targets, n) * g,
                                 "common_pairing_lower_ceiling", (population, targets, n, c))
                    for v in range(n + 1, population + 1):
                        effects = [law_values[n] - law_values[v] for law_values in values]
                        for extreme in (min, max):
                            direct = extreme(detection * effects[index]
                                             for index in endpoint_indices)
                            scaled = detection * extreme(effects[index]
                                                          for index in endpoint_indices)
                            checks.equal(direct, scaled, "common_pairing_effect_scaling",
                                         (population, targets, n, v, c))
                            checks.equal(extreme(effects),
                                         extreme(effects[index] for index in endpoint_indices),
                                         "two_by_two_endpoint_sharpness",
                                         (population, targets, n, v, c))


def run(mode: str) -> dict:
    max_population = 3 if mode == "smoke" else 7
    checks, features = Checks(), Counter()
    rng = random.Random(20261018)
    started = time.perf_counter()
    for population in range(1, max_population + 1):
        all_mask = (1 << population) - 1
        for targets in range(population + 1):
            target_mask = (1 << targets) - 1
            for mask in range(1 << population):
                check_distribution(population, targets, [mask], [Fraction(1)],
                                   checks, features)
            # A nonuniform mixture with an allowed zero-weight full-load pattern.
            masks = [target_mask, all_mask,
                     all_mask ^ 1 if targets else 0, rng.randrange(1 << population)]
            check_distribution(population, targets, masks,
                               [Fraction(2, 7), Fraction(3, 7), Fraction(2, 7), Fraction(0)],
                               checks, features)
            # A target-incomplete type can create more work than the sufficient
            # saturation capacity computed only from target-complete types.
            if targets > 0:
                check_distribution(population, targets,
                                   [target_mask, all_mask ^ 1, 0],
                                   [Fraction(1, 3), Fraction(2, 3), Fraction(0)],
                                   checks, features)
            check_common_pairing(population, targets, rng, checks, features)
    assert features["zero_target_distributions"] > 0
    assert features["multiple_target_distributions"] > 0
    assert features["no_complete_response_distributions"] > 0
    assert features["failed_response_load_above_saturation"] > 0
    assert features["zero_weight_distributions"] > 0
    return {"mode": mode, "max_population": max_population,
            "method": "direct exhaustive sample subsets; itemwise target checks; exact fractions",
            "features": dict(features), "checks": dict(checks.counts),
            "total_checks": sum(checks.counts.values()),
            "largest_exact_absolute_error": "0", "passed": True,
            "elapsed_seconds": round(time.perf_counter() - started, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    result = run(args.mode)
    output = Path(__file__).resolve().parents[1] / "results" / "capacity_saturation" / f"{args.mode}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
