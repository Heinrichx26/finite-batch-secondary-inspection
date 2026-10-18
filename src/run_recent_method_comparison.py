"""Smoke-gated recent-method core reproductions and common-resource studies."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
from math import comb
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import binom

from finite_population_policy_kernel import fixed_sample_success_exact
from recent_method_comparison import (
    FAMILIES, MAISANO_RATES, OrderedBDD, continuous_family_extrema,
    copula_probability_matrix, full_lot_acceptance, majority_errors,
    majority_errors_by_subsets, panel_batch_success, panel_expected_miss,
    phase_graph_kernel, sample_conditional_effect, type_b_acceptance,
)


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results/manuscript_20261017/comparison"


def file_hashes():
    paths = [BASE / "src/recent_method_comparison.py", Path(__file__).resolve(),
             BASE / "src/finite_population_policy_kernel.py"]
    paths += [BASE / "data/literature/recent_method_sources.json"]
    return {str(path.relative_to(BASE)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def smoke_checks():
    majority_checks = graph_checks = batch_checks = copula_checks = 0
    for rates in MAISANO_RATES.values():
        for h in (1, 3, 5, 7):
            a, b = rates["false_positive"][:h], rates["false_negative"][:h]
            direct = majority_errors_by_subsets(a, b)
            actual = majority_errors(a, b)
            assert max(abs(np.array(actual) - direct)) < 1e-14
            majority_checks += 1
    # Independently reproduced source-reported producer/consumer risks.
    producer = 1 - type_b_acceptance(14, 1, .04, 0., 0.)
    consumer = type_b_acceptance(14, 1, .245, 0., 0.)
    assert abs(producer - .106) < .001 and abs(consumer - .108) < .001
    for D in range(6):
        assert full_lot_acceptance(5, D, 2, 0., 0.) == float(D <= 2)
    # Literal Boolean selection and inspector-vote enumeration on a tiny batch.
    N, m, n, z, h = 3, 1, 2, 2, 3
    rates = MAISANO_RATES["H"]
    a, b = rates["false_positive"][:h], rates["false_negative"][:h]
    agg_a, agg_b = majority_errors(a, b)
    literal = 0.
    for selected in itertools.combinations(range(N), n):
        if 0 not in selected:
            continue
        for votes in itertools.product((False, True), repeat=n * h):
            probability = 1.
            referred = []
            for position, item in enumerate(selected):
                item_votes = votes[position * h:(position + 1) * h]
                positive = 0
                for inspector, vote in enumerate(item_votes):
                    positive_probability = 1 - b[inspector] if item == 0 else a[inspector]
                    probability *= positive_probability if vote else 1 - positive_probability
                    positive += vote
                referred.append(positive >= 2)
            if referred[selected.index(0)] and sum(referred) <= z:
                literal += probability / comb(N, n)
    assert abs(literal - panel_batch_success(N, m, n, z, agg_a, agg_b)) < 1e-12
    batch_checks += 1
    for N in range(1, 7):
        for m in range(min(N, 2) + 1):
            for k in range(N - m + 1):
                graph = OrderedBDD(N)
                for z in (0, N // 2, N):
                    coverage = graph.required(range(m))
                    capacity = graph.at_most(range(m + k), z)
                    root = graph.conjunction(coverage, capacity)
                    for n in range(N + 1):
                        literal = []
                        for sample in itertools.combinations(range(N), n):
                            selected = set(sample)
                            truth = set(range(m)).issubset(selected) and len(selected & set(range(m + k))) <= z
                            assert graph.accepts(root, selected) == truth
                            literal.append(truth)
                        actual, _ = graph.probability(root, n)
                        exact = float(fixed_sample_success_exact(N, m, k, n, z))
                        assert abs(actual - np.mean(literal)) < 1e-12 and abs(actual - exact) < 1e-12
                        graph_checks += 1
    p, q = np.array([.2, .3, .5]), np.array([.4, .6])
    for family in FAMILIES:
        for parameter in (0., sum(family.bounds) / 2, family.bounds[-1]):
            mass = copula_probability_matrix(p, q, family.family, parameter, family.rotation)
            assert abs(mass.sum() - 1.) < 1e-10
            if parameter == 0:
                assert np.max(abs(mass - p[:, None] * q[None, :])) < 1e-12
            copula_checks += 1
    # Importance identity verified by finite summation, independent of MC.
    payoff = np.array([[.2, -.1], [-.4, .7], [.3, -.6]])
    proposal = p[:, None] * q[None, :]
    for family in FAMILIES:
        law = copula_probability_matrix(p, q, family.family, family.bounds[-1], family.rotation)
        direct = np.sum(payoff * law)
        weighted = np.sum(proposal * payoff * law / proposal)
        assert abs(direct - weighted) < 1e-14
    return dict(passed=True, majority_subset_checks=majority_checks,
                source_reported_risk_checks=2, original_OC_boundary_checks=6,
                independent_panel_event_checks=batch_checks, exhaustive_graph_checks=graph_checks,
                copula_marginal_checks=copula_checks, importance_identity_checks=len(FAMILIES),
                source_producer_risk=producer, source_consumer_risk=consumer)


def reproduce_maisano():
    aggregated, oc = [], []
    for name, rates in MAISANO_RATES.items():
        for h in (1, 3, 5, 7):
            a, b = majority_errors(rates["false_positive"][:h], rates["false_negative"][:h])
            aggregated.append(dict(scenario=name, inspectors=h, aggregated_false_positive=a,
                                   aggregated_false_negative=b, sample_units=14, total_judgments=14 * h))
            for prevalence in np.linspace(0, .5, 101):
                oc.append(dict(scenario=name, inspectors=h, prevalence=prevalence,
                               acceptance=type_b_acceptance(14, 1, prevalence, a, b),
                               error_free_acceptance=type_b_acceptance(14, 1, prevalence, 0., 0.)))
    full = []
    for name, rates in MAISANO_RATES.items():
        for D in range(141):
            full.append(dict(scenario=name, N=140, defectives=D, acceptance_number=20,
                             acceptance=full_lot_acceptance(140, D, 20, rates["false_positive"][0],
                                                            rates["false_negative"][0])))
    pd.DataFrame(aggregated).to_csv(OUT / "maisano_original_aggregate_errors.csv", index=False)
    pd.DataFrame(oc).to_csv(OUT / "maisano_original_typeB_OC.csv", index=False)
    pd.DataFrame(full).to_csv(OUT / "maisano_original_full_OC.csv", index=False)

    # Same total number of manual judgments and same search capacity for all
    # policies. Rates are synthetic source-example scenarios, never estimates
    # of repeated-reader behavior in the airport response records.
    candidates, choices = [], []
    N = 60
    for name, rates in MAISANO_RATES.items():
        for m in (1, 3):
            for budget in (60, 120, 180):
                for z in (m + 1, m + 6, 20):
                    group = []
                    for h in (1, 3, 5, 7):
                        if h * m > budget:
                            continue
                        a, b = majority_errors(rates["false_positive"][:h], rates["false_negative"][:h])
                        for n in range(m, min(N, budget // h) + 1):
                            row = dict(scenario=name, N=N, targets=m, judgment_budget=budget, search_limit=z,
                                       inspectors=h, reviews=n, total_judgments=h * n,
                                       unused_judgments=budget - h * n, false_positive=a, false_negative=b,
                                       joint_success=panel_batch_success(N, m, n, z, a, b),
                                       mean_target_miss=panel_expected_miss(N, m, n, b),
                                       expected_searches=n * ((m / N) * (1 - b) + (1 - m / N) * a))
                            candidates.append(row); group.append(row)
                    for risk_limit in (None, .1, .01):
                        safe = group if risk_limit is None else [row for row in group
                                                               if row["mean_target_miss"] <= risk_limit + 1e-12]
                        for label, allowed in (("single_inspector", [row for row in safe if row["inspectors"] == 1]),
                                               ("multiple_inspectors", [row for row in safe if row["inspectors"] > 1]),
                                               ("complete_catalogue", safe)):
                            if not allowed:
                                choices.append(dict(scenario=name, N=N, targets=m, judgment_budget=budget,
                                                    search_limit=z, mean_risk_limit=risk_limit,
                                                    catalogue=label, feasible=False))
                                continue
                            best = max(allowed, key=lambda row: (row["joint_success"], -row["total_judgments"]))
                            choices.append({**best, "catalogue": label, "mean_risk_limit": risk_limit, "feasible": True})
    candidate_frame = pd.DataFrame(candidates)
    choice_frame = pd.DataFrame(choices)
    candidate_frame.to_csv(OUT / "maisano_common_budget_candidates.csv", index=False)
    choice_frame.to_csv(OUT / "maisano_common_budget_choices.csv", index=False)
    return dict(original_rate_conditions=len(aggregated), adapted_candidate_count=len(candidate_frame),
                common_resource_scenarios=36, real_response_rates_estimated=False,
                original_and_adapted_endpoints_separate=True)


def compare_graphs(smoke):
    from full_counterfactual_model import build_records, complete_batch_matrix
    state = build_records(False)
    route, y, referral = state["route"], state["targets"]["gun_knife"], state["referrals"]
    u = route == 2; m = int((y & u).sum()); N = int(u.sum())
    mandatory = int((~u).sum())
    detection = .9999 ** int((y & (route == 0)).sum()) * .99 ** int((y & (route != 0)).sum())
    cases = [(218, 907), (244, 907)] if smoke else [(n, c) for n in (0, 2, 122, 217, 218, 228, 239, 244)
                                                for c in (827, 887, 907, 947, 1040)]
    rows = []
    for n, capacity in cases:
        start = time.perf_counter()
        reference = complete_batch_matrix(state, "gun_knife", n, capacity)
        exact_seconds = time.perf_counter() - start
        start = time.perf_counter(); values = []; cache = {}; nodes = states = 0
        for response in referral:
            if not response[y & u].all():
                values.append(0.); continue
            k = int(response[u & ~y].sum())
            if k not in cache:
                probability, graph_nodes, visited = phase_graph_kernel(N, m, k, n, capacity - mandatory)
                cache[k] = detection * probability; nodes += graph_nodes; states += visited
            values.append(cache[k])
        graph_seconds = time.perf_counter() - start
        error = float(abs(np.asarray(values).reshape(reference.shape) - reference).max())
        assert error < 2e-11
        rows.append(dict(review_count=n, inspection_capacity=capacity, endpoint="retain_all_mandatory",
                         maximum_absolute_error=error, exact_kernel_seconds=exact_seconds,
                         phase_graph_seconds=graph_seconds, constructed_nodes=nodes,
                         probability_recursion_states=states, unique_valid_loads=len(cache)))
    pd.DataFrame(rows).to_csv(OUT / (("smoke_" if smoke else "") + "li_phase_BDD_comparison.csv"), index=False)
    return dict(graph_conditions=len(rows), maximum_graph_error=max(row["maximum_absolute_error"] for row in rows),
                policies_and_resource_values_identical=True)


def source_dependence_comparison(smoke):
    from full_counterfactual_model import build_records, complete_batch_matrix, transport_bounds
    state = build_records(False)
    route, y, responses = state["route"], state["targets"]["gun_knife"], state["referrals"]
    u = route == 2; N = int(u.sum()); m = int((u & y).sum()); mandatory = int((~u).sum())
    target_complete = responses[:, u & y].all(axis=1)
    empty = responses[:, u & ~y].sum(axis=1)
    detection = .9999 ** int((y & (route == 0)).sum()) * .99 ** int((y & (route != 0)).sum())
    loads = responses[:, u].sum(axis=1).reshape(10, 10)
    order = (np.argsort(loads.mean(1), kind="stable"), np.argsort(loads.mean(0), kind="stable"))
    p = q = np.repeat(.1, 10)
    interval_rows = []
    count_cases = [(218, 907)] if smoke else [(n, capacity) for n in (218, 228, 239)
                                            for capacity in (887, 907, 947)]
    for n, capacity in count_cases:
        payoff = complete_batch_matrix(state, "gun_knife", n, capacity) - complete_batch_matrix(state, "gun_knife", N, capacity)
        ordered = payoff[np.ix_(*order)]
        sharp_lower, independent, sharp_upper, *_ = transport_bounds(payoff)
        for family in FAMILIES:
            bound = continuous_family_extrema(ordered, p, q, family, 25 if smoke else 81)
            assert bound["lower"] >= sharp_lower - 1e-9 and bound["upper"] <= sharp_upper + 1e-9
            interval_rows.append(dict(condition="count", review_count=n, inspection_capacity=capacity,
                                      family=family.name, **bound, unrestricted_lower=sharp_lower,
                                      independent_effect=independent, unrestricted_upper=sharp_upper,
                                      lower_uncovered=bound["lower"] - sharp_lower,
                                      upper_uncovered=sharp_upper - bound["upper"]))
    interval_frame = pd.DataFrame(interval_rows)
    interval_frame.to_csv(OUT / (("smoke_" if smoke else "") + "yang_copula_sensitivity.csv"), index=False)

    # Strictly equal number of simulated conditional policy-effect payoffs.
    # DLMC assigns samples evenly across a prespecified parameter catalogue;
    # ISMC draws once from independence and reweights over that same catalogue.
    n, capacity = 218, 907
    exact_payoff = complete_batch_matrix(state, "gun_knife", n, capacity) - complete_batch_matrix(state, "gun_knife", N, capacity)
    ordered_payoff = exact_payoff[np.ix_(*order)].ravel()
    ordered_empty = empty.reshape(10, 10)[np.ix_(*order)].ravel()
    ordered_complete = target_complete.reshape(10, 10)[np.ix_(*order)].ravel()
    grid_size = 7 if smoke else 21
    repetitions = 8 if smoke else 128
    budgets = (210,) if smoke else (210, 2100, 21000)
    rng = np.random.default_rng(2026101707)
    sample_rows = []
    for family in FAMILIES:
        parameters = np.linspace(*family.bounds, grid_size)
        laws = np.array([copula_probability_matrix(p, q, family.family, parameter, family.rotation).ravel()
                         for parameter in parameters])
        truth = laws @ ordered_payoff
        for budget in budgets:
            per_law = max(1, budget // len(parameters)); total = per_law * len(parameters)
            start = time.perf_counter()
            direct = np.zeros((len(parameters), repetitions))
            for index, law in enumerate(laws):
                profiles = rng.choice(100, (repetitions, per_law), p=law)
                direct[index] = sample_conditional_effect(rng, profiles, ordered_empty, ordered_complete,
                                                          N, m, n, capacity - mandatory, detection).mean(1)
            direct_seconds = time.perf_counter() - start
            start = time.perf_counter()
            profiles = rng.integers(0, 100, (repetitions, total))
            sampled = sample_conditional_effect(rng, profiles, ordered_empty, ordered_complete,
                                                N, m, n, capacity - mandatory, detection)
            counts = np.zeros((repetitions, 100))
            for repetition in range(repetitions):
                counts[repetition] = np.bincount(profiles[repetition], weights=sampled[repetition], minlength=100) / total
            weighted = 100 * laws @ counts.T
            importance_seconds = time.perf_counter() - start
            for method, estimates, seconds in (("DLMC", direct, direct_seconds), ("ISMC", weighted, importance_seconds)):
                for boundary, optimum in (("lower", np.min), ("upper", np.max)):
                    actual = float(optimum(truth)); estimated = optimum(estimates, axis=0)
                    sample_rows.append(dict(family=family.name, method=method, boundary=boundary,
                                            conditional_event_evaluations=total, parameter_catalogue=len(parameters),
                                            repetitions=repetitions, exact_catalogue_bound=actual,
                                            mean_estimate=float(estimated.mean()), bias=float(estimated.mean() - actual),
                                            rmse=float(np.sqrt(np.mean((estimated - actual) ** 2))),
                                            wrong_sign_fraction=float(np.mean((estimated >= 0) != (actual >= 0))),
                                            milliseconds_per_repetition=1000 * seconds / repetitions))
    sample_frame = pd.DataFrame(sample_rows)
    assert np.isfinite(sample_frame.select_dtypes(include=[np.number])).all().all()
    sample_frame.to_csv(OUT / (("smoke_" if smoke else "") + "yang_equal_budget_sampling.csv"), index=False)
    return dict(copula_conditions=len(interval_frame), sampling_rows=len(sample_frame),
                numerical_family_extrema_are_not_unrestricted_sharp_bounds=True,
                copula_parameters_fitted_to_source=False,
                source_task_ordering=[value.tolist() for value in order],
                simulated_inner_event_not_replaced_by_exact_payoff=True,
                equal_conditional_event_evaluation_budgets=True,
                sampling_repetitions=repetitions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    hashes = file_hashes()
    if not args.smoke:
        gate = json.loads((OUT / "smoke_verification.json").read_text(encoding="utf-8"))
        if not gate.get("passed") or gate.get("source_sha256") != hashes:
            raise RuntimeError("A passing smoke check of the current source files is required")
    verification = dict(passed=False, smoke=args.smoke, source_sha256=hashes,
                        source_data_not_expanded=True, new_queue_simulations=0)
    try:
        verification["smoke_checks"] = smoke_checks()
        verification["graph_comparison"] = compare_graphs(args.smoke)
        verification["dependence_comparison"] = source_dependence_comparison(args.smoke)
        if not args.smoke:
            verification["maisano_reproduction"] = reproduce_maisano()
        verification.update(passed=True,
                            usable_for_algorithm_and_assumption_comparison=True,
                            usable_for_new_field_validation=False,
                            no_claim_all_comparators_are_policy_optimizers=True)
    except Exception as exc:
        verification["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        verification["seconds"] = time.perf_counter() - started
        (OUT / (("smoke_" if args.smoke else "") + "verification.json")).write_text(
            json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
