"""Independently check the exact marginal value of one additional review.

This verification enumerates selected item sets and then compares consecutive
success probabilities with the gain/loss identity.  It also checks mixtures of
two response groups and the existing all-but-one special case.  All comparisons
use rational arithmetic.  No screening observations or experiment outputs are
changed by this program.
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


def choose(population: int, selected: int) -> int:
    return comb(population, selected) if 0 <= selected <= population else 0


def enumerate_success_counts(population: int, targets: int, referrals: int,
                             reviews: int) -> Counter[int]:
    """Count samples containing every target by their non-target referral count."""
    counts: Counter[int] = Counter()
    target_items = set(range(targets))
    referral_items = set(range(targets, targets + referrals))
    for sample in combinations(range(population), reviews):
        selected = set(sample)
        if target_items <= selected:
            counts[len(selected & referral_items)] += 1
    return counts


def success_from_counts(counts: Counter[int], population: int, reviews: int,
                        residual_non_target_capacity: int) -> Fraction:
    numerator = sum(value for load, value in counts.items()
                    if load <= residual_non_target_capacity)
    return Fraction(numerator, choose(population, reviews))


def hypergeometric_probability(population: int, referrals: int, sample: int,
                               count: int) -> Fraction:
    return Fraction(choose(referrals, count)
                    * choose(population - referrals, sample - count),
                    choose(population, sample))


def gain_loss(population: int, targets: int, referrals: int, reviews: int,
              residual_non_target_capacity: int) -> tuple[Fraction, Fraction]:
    """Return exact gain and loss, before the common physical-detection factor."""
    if reviews + 1 < targets or residual_non_target_capacity < 0:
        return Fraction(0), Fraction(0)
    selected_non_targets = reviews + 1 - targets
    non_targets = population - targets
    inclusion = Fraction(choose(reviews + 1, targets), choose(population, targets))
    cdf = sum((hypergeometric_probability(non_targets, referrals,
                                        selected_non_targets, count)
               for count in range(residual_non_target_capacity + 1)), Fraction(0))
    boundary = hypergeometric_probability(non_targets, referrals,
                                          selected_non_targets,
                                          residual_non_target_capacity + 1)
    return (inclusion * targets * cdf / (reviews + 1),
            inclusion * (residual_non_target_capacity + 1) * boundary / (reviews + 1))


def check_finite_populations(max_population: int) -> dict:
    checked = 0
    boundary_counts: Counter[str] = Counter()
    for population in range(1, max_population + 1):
        for targets in range(population + 1):
            for referrals in range(population - targets + 1):
                samples = [enumerate_success_counts(population, targets, referrals, n)
                           for n in range(population + 1)]
                for reviews in range(population):
                    for capacity in range(-2, population - targets + 2):
                        difference = (success_from_counts(samples[reviews + 1], population,
                                                         reviews + 1, capacity)
                                      - success_from_counts(samples[reviews], population,
                                                            reviews, capacity))
                        gain, loss = gain_loss(population, targets, referrals,
                                               reviews, capacity)
                        if difference != gain - loss:
                            raise AssertionError((population, targets, referrals,
                                                  reviews, capacity, difference,
                                                  gain, loss))
                        if targets == 0:
                            assert gain == 0 and difference <= 0
                            boundary_counts["zero_optional_targets"] += 1
                        if reviews + 1 < targets:
                            assert difference == 0
                            boundary_counts["insufficient_reviews_for_target_coverage"] += 1
                        if capacity < 0:
                            assert difference == 0
                            boundary_counts["capacity_below_target_count"] += 1
                        if referrals == 0:
                            assert loss == 0
                            boundary_counts["zero_non_target_referrals"] += 1
                        checked += 1
    return {"comparisons": checked, "boundary_comparisons": dict(boundary_counts),
            "largest_exact_absolute_difference": "0"}


def check_response_mixtures(cases: int, seed: int) -> dict:
    generator = random.Random(seed)
    checked = 0
    zero_target_cases = 0
    incomplete_referral_cases = 0
    for _ in range(cases):
        population = generator.randint(2, 10)
        group_one_size = generator.randint(1, population - 1)
        group_two_size = population - group_one_size
        first_targets = generator.randint(0, group_one_size)
        second_targets = generator.randint(0, group_two_size)
        targets = first_targets + second_targets
        first_profiles = generator.randint(1, 4)
        second_profiles = generator.randint(1, 4)
        first_loads = [generator.randint(0, group_one_size - first_targets)
                       for _ in range(first_profiles)]
        second_loads = [generator.randint(0, group_two_size - second_targets)
                        for _ in range(second_profiles)]
        first_complete = [first_targets == 0 or bool(generator.getrandbits(1))
                          for _ in range(first_profiles)]
        second_complete = [second_targets == 0 or bool(generator.getrandbits(1))
                           for _ in range(second_profiles)]
        weights = [[generator.randint(0, 5) for _ in range(second_profiles)]
                   for _ in range(first_profiles)]
        if not sum(map(sum, weights)):
            weights[0][0] = 1
        total_weight = sum(map(sum, weights))
        reviews = generator.randrange(population)
        capacity = generator.randint(-2, population - targets + 1)
        detection = generator.choice([Fraction(1), Fraction(19, 20), Fraction(1, 3)])
        expected_direct = Fraction(0)
        expected_gain = Fraction(0)
        expected_loss = Fraction(0)
        for i, first in enumerate(first_loads):
            for j, second in enumerate(second_loads):
                weight = Fraction(weights[i][j], total_weight)
                if not first_complete[i] or not second_complete[j]:
                    incomplete_referral_cases += 1
                    continue
                load = first + second
                before = success_from_counts(enumerate_success_counts(
                    population, targets, load, reviews), population, reviews, capacity)
                after = success_from_counts(enumerate_success_counts(
                    population, targets, load, reviews + 1), population, reviews + 1, capacity)
                gain, loss = gain_loss(population, targets, load, reviews, capacity)
                expected_direct += weight * detection * (after - before)
                expected_gain += weight * detection * gain
                expected_loss += weight * detection * loss
        if expected_direct != expected_gain - expected_loss:
            raise AssertionError((population, targets, reviews, capacity,
                                  expected_direct, expected_gain, expected_loss))
        if targets == 0:
            assert expected_gain == 0 and expected_direct <= 0
            zero_target_cases += 1
        checked += 1
    return {"comparisons": checked, "seed": seed,
            "zero_optional_target_mixtures": zero_target_cases,
            "excluded_target_incomplete_profile_pairs": incomplete_referral_cases,
            "physical_detection_factors": ["1", "19/20", "1/3"],
            "largest_exact_absolute_difference": "0"}


def check_near_full(max_population: int) -> dict:
    checked = 0
    for population in range(1, max_population + 1):
        for targets in range(population + 1):
            for referrals in range(population - targets + 1):
                for capacity in range(-2, population - targets + 2):
                    gain, loss = gain_loss(population, targets, referrals,
                                           population - 1, capacity)
                    if capacity < 0:
                        expected = Fraction(0)
                    else:
                        expected = Fraction(
                            targets * int(referrals <= capacity)
                            - (capacity + 1) * int(referrals == capacity + 1),
                            population)
                    if gain - loss != expected:
                        raise AssertionError((population, targets, referrals,
                                              capacity, gain, loss, expected))
                    checked += 1
    return {"comparisons": checked, "largest_exact_absolute_difference": "0"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    arguments = parser.parse_args()
    started = time.perf_counter()
    max_population = 4 if arguments.mode == "smoke" else 9
    mixture_cases = 25 if arguments.mode == "smoke" else 1000
    report = {
        "mode": arguments.mode,
        "arithmetic": "exact rational",
        "finite_population_method": "enumeration of selected item sets",
        "maximum_population": max_population,
        "finite_population_checks": check_finite_populations(max_population),
        "response_mixture_checks": check_response_mixtures(mixture_cases, 20261018),
        "near_full_special_case": check_near_full(max_population),
        "status": "passed",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    destination = Path(__file__).resolve().parents[1] / "results" / "one_step_review"
    destination.mkdir(parents=True, exist_ok=True)
    result_file = destination / f"{arguments.mode}.json"
    result_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
