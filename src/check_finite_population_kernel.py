"""Smoke-gated independent enumeration of finite-population policy kernels."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import itertools
import json
from math import comb
from pathlib import Path
import time

from finite_population_policy_kernel import (
    disconnected_review_examples,
    feasible_review_counts,
    fixed_effect_decomposition_exact,
    fixed_sample_success_exact,
    has_internal_gap,
    high_reliability_disconnected_example,
    high_reliability_family_miss_exact,
    high_reliability_family_success_exact,
    mixture_success_exact,
    near_full_criterion_exact,
    near_full_effect_exact,
    repeated_reversal_rows,
    single_crossing_condition,
)


BASE = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = BASE / "results" / "general_method_20261017"


def source_hashes() -> dict[str, str]:
    files = (Path(__file__).resolve(), Path(__file__).with_name("finite_population_policy_kernel.py"))
    return {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in files}


def enumerated_successes(N: int, m: int, k: int, n: int) -> list[Fraction]:
    """Directly inspect every sample, with no hypergeometric expressions."""
    targets = set(range(m))
    referrals = set(range(m + k))
    histogram = [0] * (N + 1)
    for subset in itertools.combinations(range(N), n):
        selected = set(subset)
        if targets.issubset(selected):
            histogram[len(selected.intersection(referrals))] += 1
    denominator = comb(N, n)
    cumulative = list(itertools.accumulate(histogram))
    return [Fraction(value, denominator) for value in cumulative]


def enumeration_checks(max_population: int) -> dict[str, int]:
    probabilities = decompositions = samples = 0
    for N in range(max_population + 1):
        for m in range(N + 1):
            for k in range(N - m + 1):
                for n in range(N + 1):
                    direct = enumerated_successes(N, m, k, n)
                    samples += comb(N, n)
                    for z in range(N + 2):
                        expected = direct[min(z, N)]
                        value = fixed_sample_success_exact(N, m, k, n, z)
                        assert value == expected, (N, m, k, n, z, value, expected)
                        probabilities += 1
                        decomposition = fixed_effect_decomposition_exact(N, m, k, n, z)
                        assert decomposition.effect == decomposition.capacity_relief - decomposition.coverage_loss
                        assert decomposition.partial_success == expected
                        assert decomposition.full_success == direct_full_success(N, m, k, z)
                        assert decomposition.capacity_relief >= 0 and decomposition.coverage_loss >= 0
                        decompositions += 1
    return dict(exact_probability_checks=probabilities,
                exact_decomposition_checks=decompositions, directly_enumerated_samples=samples)


def direct_full_success(N: int, m: int, k: int, z: int) -> Fraction:
    # A complete review refers every target and each of the k non-targets.
    return Fraction(int(len(set(range(m + k))) <= z))


def hand_checked_cases() -> dict[str, object]:
    # N=4, m=1, n=3, k=1/3 equally likely: omit exactly one item.
    rows = repeated_reversal_rows(2)
    expected_partial = (Fraction(1, 8), Fraction(3, 8), Fraction(3, 4), Fraction(3, 4))
    expected_full = (Fraction(0), Fraction(1, 2), Fraction(1, 2), Fraction(1))
    # At z=1 only the k=1 type can succeed, with probability 1/4.
    # Its mixture contribution is 1/8.
    expected_effect = (Fraction(1, 8), Fraction(-1, 8), Fraction(1, 4), Fraction(-1, 4))
    assert tuple(row["partial_success"] for row in rows) == expected_partial
    assert tuple(row["full_success"] for row in rows) == expected_full
    assert tuple(row["effect"] for row in rows) == expected_effect
    # N=5, m=1, z=2, k=0/4 equally likely.
    five = tuple(mixture_success_exact(5, 1, (0, 4), (Fraction(1, 2),) * 2, n, 2)
                 for n in range(6))
    assert five == (Fraction(0), Fraction(1, 5), Fraction(2, 5),
                    Fraction(3, 10), Fraction(2, 5), Fraction(1, 2))
    return dict(four_item_effects=list(map(str, expected_effect)),
                five_item_review_curve=list(map(str, five)))


def counterexample_rows() -> list[dict[str, object]]:
    rows = []
    for example in disconnected_review_examples():
        arguments = {key: example[key] for key in ("N", "m", "empty_referrals", "weights", "z")}
        feasible = feasible_review_counts(**arguments, threshold=example["threshold"])
        assert feasible == example["expected_feasible"] and has_internal_gap(feasible)
        for n in range(example["N"] + 1):
            success = mixture_success_exact(**arguments, n=n)
            rows.append(dict(name=example["name"], N=example["N"], m=example["m"],
                             z=example["z"], n=n, empty_referrals=" ".join(map(str, example["empty_referrals"])),
                             weights=" ".join(map(str, example["weights"])), success_exact=str(success),
                             success=float(success), threshold_exact=str(example["threshold"]),
                             feasible=n in feasible))
    return rows


def equal_weight_minimality_check() -> int:
    checked = 0
    for N in range(1, 4):
        for m in range(1, N + 1):
            for k1, k2 in itertools.combinations_with_replacement(range(N - m + 1), 2):
                for z in range(N + 1):
                    curve = tuple(mixture_success_exact(N, m, (k1, k2), (Fraction(1, 2),) * 2, n, z)
                                  for n in range(N + 1))
                    # The feasible set only changes at values attained by the curve.
                    for threshold in set(curve):
                        assert not has_internal_gap(n for n, value in enumerate(curve) if value >= threshold)
                        checked += 1
    return checked


def invalid_input_checks() -> int:
    cases = ((-1, 0, 0, 0, 0), (2, 3, 0, 0, 0), (2, 1, 2, 1, 1),
             (2, 1, 0, 3, 1), (2, 1, 0, 1, -1), (2.5, 1, 0, 1, 1),
             (True, 0, 0, 0, 0))
    for case in cases:
        try:
            fixed_sample_success_exact(*case)
        except ValueError:
            continue
        raise AssertionError(f"Invalid parameters accepted: {case}")
    return len(cases)


def near_full_checks(smoke: bool, max_population: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Compare the boundary formula with the general kernel, then test structure."""
    rows = []
    criterion_checks = binomial_distributions = binomial_sign_checks = 0
    maximum = 4 if smoke else max_population
    # Single support points independently test every coefficient in the formula.
    for N in range(1, maximum + 1):
        for m in range(N + 1):
            for k in range(N - m + 1):
                for mass in (Fraction(1), Fraction(2, 3)):
                    for z in range(N + 2):
                        closed = near_full_effect_exact(N, m, (k,), (mass,), z)
                        direct = mass * (fixed_sample_success_exact(N, m, k, N - 1, z)
                                         - fixed_sample_success_exact(N, m, k, N, z))
                        assert closed == direct, (N, m, k, mass, z, closed, direct)
                        criterion_checks += 1
    theta_values = (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)) if smoke else (
        Fraction(1, 10), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(9, 10))
    for N in range(1, maximum + 1):
        for m in range(1, N + 1):
            for M in range(N - m + 1):
                for theta in theta_values:
                    types = tuple(range(M + 1))
                    probabilities = tuple(Fraction(comb(M, k)) * theta ** k * (1 - theta) ** (M - k)
                                          for k in types)
                    assert sum(probabilities) == 1
                    assert single_crossing_condition(types, probabilities)
                    binomial_distributions += 1
                    negative_seen = False
                    for z in range(N + 2):
                        criterion = near_full_criterion_exact(N, m, types, probabilities, z)
                        direct = (mixture_success_exact(N, m, types, probabilities, N - 1, z)
                                  - mixture_success_exact(N, m, types, probabilities, N, z))
                        assert criterion.effect == direct
                        assert not (negative_seen and criterion.effect > 0)
                        negative_seen |= criterion.effect < 0
                        criterion_checks += 1
                        binomial_sign_checks += 1
                        rows.append(dict(distribution="binomial", N=N, m=m, M=M, theta_exact=str(theta),
                                         z=z, x=criterion.x, boundary_exact=str(criterion.boundary_probability),
                                         cumulative_exact=str(criterion.cumulative_probability),
                                         capacity_relief_exact=str(criterion.capacity_relief),
                                         coverage_loss_exact=str(criterion.coverage_loss),
                                         effect_exact=str(criterion.effect), effect=float(criterion.effect),
                                         sufficient_condition=True))
    logconcave = tuple(Fraction(value, 209) for value in (25, 40, 60, 84))
    assert all(logconcave[k] ** 2 >= logconcave[k - 1] * logconcave[k + 1] for k in (1, 2))
    assert not single_crossing_condition((0, 1, 2, 3), logconcave)
    counterexample_effects = []
    for z in range(7):
        criterion = near_full_criterion_exact(5, 2, (0, 1, 2, 3), logconcave, z)
        direct = (mixture_success_exact(5, 2, (0, 1, 2, 3), logconcave, 4, z)
                  - mixture_success_exact(5, 2, (0, 1, 2, 3), logconcave, 5, z))
        assert criterion.effect == direct
        counterexample_effects.append(criterion.effect)
        rows.append(dict(distribution="ordinary_logconcave_counterexample", N=5, m=2, M=3, theta_exact="",
                         z=z, x=criterion.x, boundary_exact=str(criterion.boundary_probability),
                         cumulative_exact=str(criterion.cumulative_probability),
                         capacity_relief_exact=str(criterion.capacity_relief),
                         coverage_loss_exact=str(criterion.coverage_loss),
                         effect_exact=str(criterion.effect), effect=float(criterion.effect),
                         sufficient_condition=False))
    assert counterexample_effects[2:6] == [Fraction(-10, 1045), Fraction(-10, 1045),
                                         Fraction(2, 1045), Fraction(-2, 5)]
    assert single_crossing_condition((0, 1, 1, 2), (Fraction(1, 5), Fraction(1, 10),
                                                              Fraction(1, 10), Fraction(1, 10)))
    assert not single_crossing_condition((0, 2), (Fraction(1, 2), Fraction(1, 2)))
    assert not single_crossing_condition((1,), (Fraction(1),))
    assert not single_crossing_condition((0,), (Fraction(0),))
    return rows, dict(near_full_formula_checks=criterion_checks,
                      binomial_distributions_checked=binomial_distributions,
                      binomial_capacity_sign_checks=binomial_sign_checks,
                      subprobability_masses_checked=["1", "2/3"],
                      logconcavity_is_insufficient=True,
                      logconcave_counterexample_effects=list(map(str, counterexample_effects)),
                      sufficient_condition_is_claimed_necessary=False)


def joint_feasibility_family_checks(smoke: bool) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Check high-reliability examples without enumerating large populations."""
    rows = []
    formula_checks = risk_checks = 0
    sizes = (3, 10) if smoke else (3, 10, 100, 2000)
    largest_checked = max(sizes)
    for N in sizes:
        example = high_reliability_disconnected_example(N)
        selected_n = sorted({0, 1, N // 2, N - 3, N - 2, N - 1, N})
        for n in selected_n:
            closed = high_reliability_family_success_exact(N, n)
            direct = mixture_success_exact(N, 1, example.empty_referrals,
                                           example.weights, n, example.z)
            assert closed == direct
            mean_miss = high_reliability_family_miss_exact(N, n)
            assert mean_miss == 1 - Fraction(n, N)
            meets_reliability = closed >= example.reliability_target
            meets_mean_risk = mean_miss <= example.miss_limit
            assert (meets_reliability and meets_mean_risk) == (n in example.expected_feasible)
            formula_checks += 1
            risk_checks += 1
            rows.append(dict(N=N, m=1, z=example.z, n=n,
                             response_weight_no_empty_referrals_exact=str(example.weights[0]),
                             reliability_target_exact=str(example.reliability_target),
                             reliability_target=float(example.reliability_target),
                             miss_limit_exact=str(example.miss_limit), miss_limit=float(example.miss_limit),
                             joint_success_exact=str(closed), joint_success=float(closed),
                             mean_target_miss_exact=str(mean_miss), mean_target_miss=float(mean_miss),
                             meets_reliability=meets_reliability, meets_mean_risk=meets_mean_risk,
                             jointly_feasible=meets_reliability and meets_mean_risk))
        if N <= 10:
            feasible = tuple(n for n in range(N + 1)
                             if high_reliability_family_success_exact(N, n) >= example.reliability_target
                             and high_reliability_family_miss_exact(N, n) <= example.miss_limit)
            assert feasible == example.expected_feasible
    # The exact inequalities prove the omitted counts: n<=N-3 fails both
    # requirements; N-1 meets mean risk but has success p*(N-1)/N<p.
    for eps in (Fraction(1, 2), Fraction(1, 10), Fraction(1, 100), Fraction(1, 1000)):
        N = max(3, (2 * eps.denominator + eps.numerator - 1) // eps.numerator)
        example = high_reliability_disconnected_example(N)
        assert example.reliability_target >= 1 - eps and example.miss_limit <= eps
    return rows, dict(population_sizes_checked=list(sizes), exact_family_formula_checks=formula_checks,
                      exact_mean_risk_checks=risk_checks, full_domain_checks_only_through_N=10,
                      large_population_checks_use_selected_counts_and_algebraic_formula=True,
                      joint_feasible_counts_formula="{N-2,N}",
                      largest_population=largest_checked,
                      largest_population_reliability_target=str(Fraction(largest_checked - 2, largest_checked)),
                      largest_population_miss_limit=str(Fraction(2, largest_checked)),
                      arbitrary_epsilon_construction_verified=True,
                      target_detection_in_this_family_is_perfect=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--max-population", type=int, default=8)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    if not 1 <= args.max_population <= 9:
        parser.error("--max-population must lie between 1 and 9")
    started = time.perf_counter()
    hashes = source_hashes()
    if args.full:
        gate = args.results_root / "kernel_smoke" / "verification.json"
        if not gate.exists():
            raise RuntimeError("Run --smoke before --full")
        previous = json.loads(gate.read_text(encoding="utf-8"))
        if not previous.get("passed") or previous.get("source_sha256") != hashes:
            raise RuntimeError("A passing smoke check of the current source files is required")
    output = args.results_root / ("kernel_smoke" if args.smoke else "kernel_validation")
    output.mkdir(parents=True, exist_ok=True)
    verification = dict(passed=False, smoke=args.smoke, source_sha256=hashes,
                        no_source_data_required=True, max_population=4 if args.smoke else args.max_population)
    try:
        verification["hand_calculations"] = hand_checked_cases()
        verification["invalid_input_checks"] = invalid_input_checks()
        verification.update(enumeration_checks(verification["max_population"]))
        verification["equal_two_type_minimality_checks"] = equal_weight_minimality_check()
        near_full_rows, condition_checks = near_full_checks(args.smoke, verification["max_population"])
        verification["condition_checks"] = condition_checks
        write_csv(output / "near_full_criterion.csv", near_full_rows)
        joint_family_rows, joint_family_checks = joint_feasibility_family_checks(args.smoke)
        verification["joint_feasibility_family"] = joint_family_checks
        write_csv(output / "joint_feasibility_family.csv", joint_family_rows)
        examples = counterexample_rows()
        write_csv(output / "disconnected_review_sets.csv", examples)
        reversal_rows = []
        reversal_sizes = (1, 2, 3) if args.smoke else range(1, 13)
        for L in reversal_sizes:
            rows = repeated_reversal_rows(L)
            assert all(row["effect"] == row["expected_effect"] for row in rows)
            assert all(rows[i]["effect"] * rows[i + 1]["effect"] < 0 for i in range(len(rows) - 1))
            assert all(rows[i]["partial_success"] <= rows[i + 1]["partial_success"]
                       and rows[i]["full_success"] <= rows[i + 1]["full_success"]
                       for i in range(len(rows) - 1))
            for row in rows:
                reversal_rows.append({**{key: value for key, value in row.items() if not isinstance(value, Fraction)},
                                      **{key + "_exact": str(value) for key, value in row.items() if isinstance(value, Fraction)},
                                      **{key: float(value) for key, value in row.items() if isinstance(value, Fraction)}})
        write_csv(output / "repeated_capacity_reversals.csv", reversal_rows)
        verification.update(passed=True, disconnected_review_examples=3,
                            maximum_reversal_population=2 * max(reversal_sizes),
                            maximum_strict_reversals=2 * max(reversal_sizes) - 1,
                            nonconnected_minimum_population=3,
                            nonconnected_equal_two_type_minimum_population=4,
                            finite_population_checks_exact_rational=True,
                            counterexamples_prove_general_interval_assumption_invalid=True)
    except Exception as exc:
        verification["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        verification["seconds"] = time.perf_counter() - started
        (output / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
