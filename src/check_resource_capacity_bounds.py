"""Smoke-gated checks of exact resource bounds; no queue simulations or figures."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from resource_capacity_bounds import (compressed_load, full_review_resource_bounds,
    minimum_robust_capacity, sum_cdf_bounds)
from robust_policy_bounds import CouplingSet

BASE = Path(__file__).resolve().parents[1]
DESTINATION = BASE / "results/general_method_20261017"
HASH_FILES = [Path(__file__), Path(__file__).with_name("resource_capacity_bounds.py"),
              Path(__file__).with_name("robust_policy_bounds.py")]


def hashes():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in HASH_FILES}


def matrix(values, other, capacity):
    x, y = np.asarray(values, float), np.asarray(other, float)
    finite = np.isfinite(x[:, None]) & np.isfinite(y[None, :])
    return (finite & (x[:, None]+y[None, :] <= capacity)).astype(float)


def hand_smoke():
    # Hand calculation 1: unequal two-point laws and the inclusive integer sum.
    x, p, y, q = [0, 2], [.25, .75], [0, 1], [.6, .4]
    wanted = {0: (0., .15, .25), 1: (.25, .25, .25),
              2: (.6, .7, .85), 3: (1., 1., 1.)}
    for capacity, expected in wanted.items():
        bound = sum_cdf_bounds(x, p, y, q, capacity)
        assert np.allclose([bound.lower, bound.independent, bound.upper], expected)
    # Hand calculation 2: +inf means failed full-target referral, never success.
    for capacity in [0, 1, math.inf]:
        bound = sum_cdf_bounds([0, math.inf], [.8, .2],
                               [0, math.inf], [.7, .3], capacity)
        assert np.allclose([bound.lower, bound.independent, bound.upper], [.5, .56, .7])
    need = minimum_robust_capacity(x, p, y, q, .6, offset=8)
    assert need.feasible and need.minimum_capacity == 10
    impossible = minimum_robust_capacity([0, math.inf], [.8, .2],
        [0, math.inf], [.7, .3], .5001)
    assert not impossible.feasible
    return 9


def random_checks(cases, seed):
    rng = np.random.default_rng(seed)
    largest_error = 0.
    thresholds_checked = compression_checked = requirement_checked = 0
    infinite_cases = zero_weight_cases = unequal_weight_cases = 0
    for case in range(cases):
        nr, nc = map(int, rng.integers(1, 6, size=2))
        x, y = rng.integers(0, 8, nr).astype(float), rng.integers(0, 8, nc).astype(float)
        if case % 3 == 0:
            x[int(rng.integers(nr))] = math.inf
        if case % 4 == 0:
            y[int(rng.integers(nc))] = math.inf
        wr, wc = rng.integers(0, 7, nr), rng.integers(0, 7, nc)
        if wr.sum() == 0: wr[0] = 1
        if wc.sum() == 0: wc[0] = 1
        p, q = wr/wr.sum(), wc/wc.sum()
        infinite_cases += int(np.isinf(x).any() or np.isinf(y).any())
        zero_weight_cases += int((p == 0).any() or (q == 0).any())
        unequal_weight_cases += int(np.ptp(p) > 0 or np.ptp(q) > 0)
        model = CouplingSet(p, q)
        cx, cy = compressed_load(x, p), compressed_load(y, q)
        compressed = CouplingSet(cx.weights, cy.weights)
        max_sum = int(max(x[np.isfinite(x)], default=0)+max(y[np.isfinite(y)], default=0))
        grid = list(range(-1, max_sum+2)) + [math.inf]
        previous = (0., 0.)
        for capacity in grid:
            bound = sum_cdf_bounds(x, p, y, q, capacity)
            event = matrix(x, y, capacity)
            lower, upper = model.bounds(event)
            error = max(abs(bound.lower-lower.value), abs(bound.upper-upper.value),
                        abs(bound.independent-float(p@event@q)))
            largest_error = max(largest_error, error)
            assert error < 1e-9, (case, capacity, bound, lower.value, upper.value)
            assert bound.lower >= previous[0]-1e-10 and bound.upper >= previous[1]-1e-10
            previous = (bound.lower, bound.upper)
            cl, cu = compressed.bounds(matrix(cx.values, cy.values, capacity))
            assert abs(cl.value-lower.value) < 1e-9 and abs(cu.value-upper.value) < 1e-9
            thresholds_checked += 1
            compression_checked += 2
        offset, detection = int(rng.integers(0, 5)), float(rng.uniform(.5, 1.))
        ceiling = sum_cdf_bounds(x, p, y, q, math.inf).lower_ceiling*detection
        for target in [0., .2, .7, ceiling, min(1., ceiling+1e-4)]:
            need = minimum_robust_capacity(x, p, y, q, target, offset, detection)
            possible = []
            for capacity in range(max_sum+offset+2):
                bound = full_review_resource_bounds(x, p, y, q, capacity, offset, detection)
                if bound.lower >= target-1e-12:
                    possible.append(capacity)
            assert need.feasible == bool(possible), (case, target, need, possible)
            if possible:
                assert need.minimum_capacity == min(possible), (case, target, need, possible)
            requirement_checked += 1
    return dict(random_cases=cases, threshold_LP_checks=thresholds_checked,
        compression_extrema_checks=compression_checked,
        capacity_necessary_sufficient_checks=requirement_checked,
        includes_infinite_load_cases=infinite_cases, includes_zero_weight_cases=zero_weight_cases,
        includes_unequal_weight_cases=unequal_weight_cases, largest_absolute_error=largest_error)


def source_frontiers():
    """Read existing source records; all conclusions use their fixed marginals."""
    from full_counterfactual_model import build_records, complete_batch_matrix, moment
    state = build_records(False)
    first, second = state["first"], state["second"]
    groups = sorted(set(second.groupby("image").person.apply(frozenset)), key=lambda s: sorted(s))
    owner = {image: groups.index(frozenset(frame.person))
             for image, frame in second.groupby("image")}
    record_owner = first.image.map(owner).to_numpy(int)
    responses = state["referrals"].reshape(10, 10, -1)
    route = state["route"]
    u = route == 2
    mandatory = int((~u).sum())
    model = CouplingSet(np.repeat(.1, 10), np.repeat(.1, 10))
    rows, ceilings, loads = [], [], []
    largest_error = 0.
    comparisons = 0
    for endpoint, target in state["targets"].items():
        m = int((target & u).sum())
        values = []
        for group in range(2):
            mask = u & (record_owner == group)
            block = responses[:, 0, :] if group == 0 else responses[0, :, :]
            positive = target & mask
            all_referable = np.all(block[:, positive], axis=1)
            non_target_load = block[:, mask & ~target].sum(axis=1).astype(float)
            encoded = np.where(all_referable, non_target_load, math.inf)
            values.append(encoded)
            for person, (load, available, code) in enumerate(zip(non_target_load, all_referable, encoded)):
                loads.append(dict(endpoint=endpoint, response_group=group+1,
                    response_index=person+1, non_target_load=int(load),
                    all_group_targets_referable=bool(available),
                    encoded_load=float(code), marginal_weight=.1))
        x, y = values
        p = q = np.repeat(.1, 10)
        mb, mk = int((target & (route == 0)).sum()), int((target & (route == 1)).sum())
        detection = moment(.9999, 0., mb)*moment(.99, 0., mk+m)
        offset = mandatory+m
        finite_sums = (x[:, None]+y[None, :])[np.isfinite(x[:, None]+y[None, :])]
        if len(finite_sums):
            begin, end = offset+int(finite_sums.min()), offset+int(finite_sums.max())
        else:
            begin = end = offset
        capacities = sorted(set([max(0, begin-1), 907, 947]+list(range(begin, end+1))))
        for capacity in capacities:
            bound = full_review_resource_bounds(x, p, y, q, capacity, offset, detection)
            original = complete_batch_matrix(state, endpoint, int(u.sum()), capacity,
                                              policy="retain", db=.9999, d=.99, eta=1.)
            encoded = detection*matrix(x, y, capacity-offset)
            assert np.allclose(original, encoded, atol=1e-12)
            lower, upper = model.bounds(original)
            error = max(abs(bound.lower-lower.value), abs(bound.upper-upper.value),
                        abs(bound.independent-float(original.mean())))
            assert error < 1e-9, (endpoint, capacity, bound, lower.value, upper.value)
            largest_error = max(largest_error, error)
            comparisons += 1
            rows.append(dict(endpoint=endpoint, policy="retain_full_review",
                review_count=int(u.sum()), mandatory_searches=mandatory, reviewed_targets=m,
                complete_search_capacity=capacity, detection_factor=detection,
                joint_lower=bound.lower, joint_independent=bound.independent,
                joint_upper=bound.upper, LP_lower=lower.value, LP_upper=upper.value,
                absolute_LP_error=error, fixed_marginal_condition=True))
        fx, fy = compressed_load(x, p).finite_mass, compressed_load(y, q).finite_mass
        for target_level in [.5, .8, .9, .95]:
            requirement = minimum_robust_capacity(x, p, y, q, target_level, offset, detection)
            ceilings.append(dict(endpoint=endpoint, first_target_referability=fx,
                second_target_referability=fy, fixed_marginal_condition=True,
                **asdict(requirement)))
    pd.DataFrame(rows).to_csv(DESTINATION/"full_review_resource_frontier.csv", index=False)
    pd.DataFrame(ceilings).to_csv(DESTINATION/"feasible_ceiling.csv", index=False)
    pd.DataFrame(loads).to_csv(DESTINATION/"full_review_marginal_loads.csv", index=False)
    return dict(source_kernel_and_transport_comparisons=comparisons,
        largest_source_absolute_error=largest_error,
        source_hashes=state["provenance"]["source_sha256"],
        source_results_condition_on_fixed_marginals=True,
        population_uncertainty_included=False, queue_simulations_run=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started = time.time()
    destination = DESTINATION/("resource_smoke" if args.smoke else "resource_validation")
    destination.mkdir(parents=True, exist_ok=True)
    source_hashes = hashes()
    if not args.smoke:
        gate = json.loads((DESTINATION/"resource_smoke/verification.json").read_text())
        if not gate.get("passed") or gate["source_sha256"] != source_hashes:
            raise RuntimeError("Current source requires a successful matching-hash smoke test")
    checks = dict(hand_checks=hand_smoke(), **random_checks(36 if args.smoke else 320,
                  20261017031 if args.smoke else 20261017032))
    if not args.smoke:
        checks.update(source_frontiers())
    checks.update(passed=True, smoke=args.smoke, seconds=time.time()-started,
        source_sha256=source_hashes, fixed_marginal_bounds_only=True)
    (destination/"verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
