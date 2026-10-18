"""Complete screening cascade and paired conditional interventions."""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import betaln
from scipy.optimize import linear_sum_assignment
from scipy.stats import hypergeom

from conditional_theory_experiment import BASE, prepare, target_mask
from conditional_airport_simulation import fcfs, lognormal_from_z
try:
    from numba import njit
except ImportError:
    njit = None


def _fcfs_heap(arrivals, durations, capacity):
    servers = np.zeros(capacity)
    finishes = np.empty(len(arrivals)); waits = np.empty(len(arrivals))
    for i in range(len(arrivals)):
        start = max(servers[0], arrivals[i])
        end = start+durations[i]
        finishes[i] = end; waits[i] = start-arrivals[i]
        node = 0
        while 2*node+1 < capacity:
            child = 2*node+1
            if child+1 < capacity and servers[child+1] < servers[child]:
                child += 1
            if end <= servers[child]:
                break
            servers[node] = servers[child]; node = child
        servers[node] = end
    return finishes, waits


fast_fcfs = njit(cache=True)(_fcfs_heap) if njit else fcfs


def build_records(persist=True):
    first, second, parameters, _, source = prepare()
    first = first.reset_index(drop=True)
    first_time = pd.to_numeric(first.TaskTime, errors="raise").to_numpy(float)
    second = second.copy()
    second["seconds"] = pd.to_numeric(second.TaskTime, errors="raise")
    assert np.isfinite(first_time).all() and (first_time > 0).all()
    groups = sorted(set(second.groupby("image").person.apply(frozenset)), key=lambda s: sorted(s))
    referrals, seconds = [], []
    for people in itertools.product(*[sorted(g) for g in groups]):
        person = second[second.person.isin(people)].set_index("image")
        referrals.append(first.image.map(person.refer).to_numpy(bool))
        seconds.append(first.image.map(person.seconds).to_numpy(float))
    refer = np.asarray(referrals)
    read_seconds = np.asarray(seconds)
    assert refer.shape == read_seconds.shape == (100, len(first))
    assert np.isfinite(read_seconds).all() and (read_seconds > 0).all()
    route = first.route.map({"B": 0, "K": 1, "U": 2}).to_numpy(int)
    targets = {name: target_mask(first, name).to_numpy(bool) for name in ["gun_knife", "gun", "all_labels"]}
    frame = pd.DataFrame({"image": first.image, "route": first.route,
                          "primary_task_seconds": first_time,
                          "gun_knife_target": targets["gun_knife"], "empty_annotation": ~first.truth})
    destination = BASE / "data/curated/full_counterfactual"
    if persist:
        destination.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination / "source_record_structure.csv", index=False)
        np.savez_compressed(destination / "paired_responses.npz", route=route, primary_seconds=first_time,
                            referrals=refer, secondary_seconds=read_seconds,
                            **{key: value for key, value in targets.items()})
    provenance = {"source_doi": source["data_DOI"], "source_sha256": source["source_sha256"],
        "license": source["license"], "primary_records": len(first), "secondary_people": 20,
        "response_combinations": 100, "combinations_are_not_independent_people": True,
        "seconds_unit_from_source_description": "seconds",
        "TaskTime_is_total_online_task_duration_including_interface_actions": True,
        "physical_inspection_duration_separate_scenario": True}
    return dict(first=first, second=second, parameters=parameters.set_index("endpoint"), route=route,
                primary_seconds=first_time, referrals=refer, secondary_seconds=read_seconds,
                targets=targets, empty=(~first.truth).to_numpy(bool), provenance=provenance)


def moment(mean, rho, exponent):
    if exponent == 0 or mean == 1:
        return 1.
    if mean == 0:
        return 0.
    if rho == 0:
        return float(mean ** exponent)
    concentration = 1/rho-1
    return float(np.exp(betaln(mean*concentration+exponent, (1-mean)*concentration)
                        - betaln(mean*concentration, (1-mean)*concentration)))


def transport_bounds(matrix):
    matrix = np.asarray(matrix).reshape(10, 10)
    i, j = linear_sum_assignment(matrix)
    lower = float(matrix[i, j].mean())
    lower_permutation = j[np.argsort(i)]
    i, j = linear_sum_assignment(-matrix)
    upper = float(matrix[i, j].mean())
    upper_permutation = j[np.argsort(i)]
    return lower, float(matrix.mean()), upper, lower_permutation, upper_permutation


def complete_batch_matrix(state, endpoint, n, physical_limit, policy="retain",
                          db=.9999, d=.99, eta=1., rho_b=0., rho_d=0., rho_read=0.):
    """Exact zero-miss/count probability including every original B record."""
    y = state["targets"][endpoint]; r = state["route"]; e = state["referrals"]
    baseline = int((r == 0).sum()); mb = int((y & (r == 0)).sum())
    u = r == 2; k = r == 1
    pool = int(u.sum()); mu = int((u & y).sum()); mk = int((k & y).sum())
    ku = e[:, u & y].sum(axis=1)
    ou = e[:, u & ~y].sum(axis=1)
    if policy == "retain":
        physical_k = np.repeat(int(k.sum()), 100)
        k_complete = np.ones(100, bool)
        read_exponent = mu
    else:
        physical_k = e[:, k].sum(axis=1)
        k_complete = e[:, k & y].sum(axis=1) == mk
        read_exponent = mu+mk
    usable = k_complete & (ku == mu)
    if n < mu:
        return np.zeros((10, 10))
    all_selected = float(hypergeom.pmf(mu, pool, mu, n))
    available = physical_limit-baseline-physical_k-mu
    count_probability = hypergeom.cdf(available, pool-mu, ou, n-mu)
    detection = (moment(db, rho_b, mb) * moment(d, rho_d, mk+mu)
                 * moment(eta, rho_read, read_exponent))
    return (usable * all_selected * count_probability * detection).reshape(10, 10)


def expected_batch(state, endpoint, n, policy="retain", db=.9999, d=.99, eta=1.):
    y = state["targets"][endpoint]; r = state["route"]; e = state["referrals"]
    pool = int((r == 2).sum()); q = n/pool
    counts = {name: int((y & (r == value)).sum()) for name, value in [("B", 0), ("K", 1), ("U", 2)]}
    refer_k_positive = float(e[:, (r == 1) & y].sum(axis=1).mean())
    refer_u_positive = float(e[:, (r == 2) & y].sum(axis=1).mean())
    miss = counts["B"]*(1-db)
    miss += counts["K"]*(1-d) if policy == "retain" else counts["K"]-eta*d*refer_k_positive
    miss += counts["U"]-q*eta*d*refer_u_positive
    cost_k = float((r == 1).sum()) if policy == "retain" else float(e[:, r == 1].sum(axis=1).mean())
    if policy != "retain":
        cost_k -= (1-eta)*refer_k_positive
    cost_u = float(e[:, r == 2].sum(axis=1).mean())-(1-eta)*refer_u_positive
    cost = int((r == 0).sum())+cost_k+q*cost_u
    return dict(conditional_miss=miss/y.sum(), expected_misses=miss, expected_physical=cost)


def coupled_permutation(state, permutation):
    return np.asarray([10*i+int(permutation[i]) for i in range(10)], dtype=int)


def schedule(arrival, route, referral, selected, primary_duration, secondary_duration,
             physical_duration, capacities, policy="retain", primary_cache=None):
    """Three FCFS stages; all mandatory checks retain tasks until completion."""
    primary_finish, primary_wait = (fast_fcfs(arrival, primary_duration, capacities[0])
                                   if primary_cache is None else primary_cache)
    read_needed = (route == 2) & selected
    if policy == "clear":
        read_needed |= route == 1
    physical_needed = (route == 0) | ((route == 1) & (policy == "retain")) | (read_needed & referral)
    finish = primary_finish.copy()
    read_idx = np.flatnonzero(read_needed)
    order = np.argsort(primary_finish[read_idx], kind="stable")
    read_idx = read_idx[order]
    read_finish, read_wait = fast_fcfs(primary_finish[read_idx], secondary_duration[read_idx], capacities[1])
    finish[read_idx] = read_finish
    physical_idx = np.flatnonzero(physical_needed)
    order = np.argsort(finish[physical_idx], kind="stable")
    physical_idx = physical_idx[order]
    physical_finish, physical_wait = fast_fcfs(finish[physical_idx], physical_duration[physical_idx], capacities[2])
    finish[physical_idx] = physical_finish
    assert np.all(finish >= arrival) and np.all(primary_finish >= arrival)
    return dict(finish=finish, primary_finish=primary_finish, primary_wait=primary_wait,
                read_needed=read_needed, physical_needed=physical_needed,
                read_wait=read_wait, physical_wait=physical_wait)


def zero_miss_probability(y, physical, route, db=.9999, d=.99, rho_b=0., rho_d=0.):
    if np.any(y & ~physical):
        return 0.
    mb = int((y & (route == 0)).sum()); md = int((y & (route != 0)).sum())
    return moment(db, rho_b, mb)*moment(d, rho_d, md)
