"""Exact full-review capacity under TV reweighting of finite response supports.

The extremal marginal moves probability from the smallest loads to the largest
allowed load. It is simultaneously worst for every finite capacity. Generic
transportation LPs independently verify the resulting resource calculation.
"""
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

from marginal_weight_sensitivity import marginal_extreme
from resource_capacity_bounds import full_review_resource_bounds, minimum_robust_capacity

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results/robust_capacity_reweighting"
FILES = [Path(__file__), Path(__file__).with_name("resource_capacity_bounds.py"),
         Path(__file__).with_name("marginal_weight_sensitivity.py")]


def hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in FILES}


def worst_load_weights(values, weights, radius):
    """Keep every allowed support value, including initially zero-weight types."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if values.ndim != 1 or values.shape != weights.shape or not len(values):
        raise ValueError("Loads and weights must be equal-length nonempty vectors")
    if np.any(np.isnan(values)) or np.any(values < 0):
        raise ValueError("Loads must be nonnegative integers or positive infinity")
    finite = np.isfinite(values)
    if np.any(values[finite] != np.floor(values[finite])):
        raise ValueError("Finite loads must be integers")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Weights must be finite and nonnegative")
    if not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0):
        raise ValueError("Weights must sum to one")
    if not math.isfinite(radius) or not 0 <= radius <= 1:
        raise ValueError("TV radius must lie in [0,1]")
    support, index = np.unique(values, return_inverse=True)
    original = np.bincount(index, weights=weights, minlength=len(support))
    shifted = original.copy()
    moved = min(radius, float(original[:-1].sum()))
    remaining = moved
    for k in range(len(support)-1):
        take = min(remaining, float(shifted[k]))
        shifted[k] -= take
        remaining -= take
    shifted[-1] += moved
    assert remaining < 1e-12
    assert np.isclose(shifted.sum(), 1.)
    assert .5*np.abs(shifted-original).sum() <= radius+1e-12
    return support, shifted


def event_matrix(x, y, capacity):
    x, y = np.asarray(x, float), np.asarray(y, float)
    finite = np.isfinite(x[:, None]) & np.isfinite(y[None, :])
    return (finite & (x[:, None]+y[None, :] <= capacity)).astype(float)


def check_cases(cases, seed):
    rng = np.random.default_rng(seed)
    count = capacities = requirements = 0
    error = 0.
    fixtures = [([0, 2], [.25, .75], [0, 1], [.6, .4], .1, .2),
                ([0, math.inf], [1., 0.], [0], [1.], .2, 0.),
                ([0, 0, 3], [.2, .8, 0.], [math.inf], [1.], 1., 1.),
                ([math.inf], [1.], [0], [1.], 0., 0.),
                ([1, 4], [.5, .5], [2, 6], [.4, .6], 1., 0.)]
    for case in range(cases):
        nr, nc = rng.integers(1, 6, size=2)
        x, y = rng.integers(0, 9, nr).astype(float), rng.integers(0, 9, nc).astype(float)
        if case % 3 == 0: x[-1] = math.inf
        if case % 4 == 0: y[-1] = math.inf
        p, q = rng.integers(0, 8, nr), rng.integers(0, 8, nc)
        if not p.sum(): p[0] = 1
        if not q.sum(): q[0] = 1
        p, q = p/p.sum(), q/q.sum()
        dp, dq = rng.choice([0., .01, .05, .2, .5, 1.], 2)
        fixtures.append((x, p, y, q, dp, dq))
    for x, p, y, q, dp, dq in fixtures:
        count += 1
        x, y, p, q = map(lambda z: np.asarray(z, float), (x, y, p, q))
        wx, wp = worst_load_weights(x, p, float(dp))
        wy, wq = worst_load_weights(y, q, float(dq))
        # Independent pointwise CDF identity, including the finite-maximum edge.
        for original, weights, shifted, new_weights, radius in [(x,p,wx,wp,dp),(y,q,wy,wq,dq)]:
            for t in range(-1, 11):
                expected = 1. if t >= max(original) else max(0., float(weights[original <= t].sum())-radius)
                assert np.isclose(new_weights[shifted <= t].sum(), expected, atol=1e-12)
        max_sum = int(max(x[np.isfinite(x)], default=0)+max(y[np.isfinite(y)], default=0))
        grid = sorted(set([-1, 0, max_sum//2, max_sum-1, max_sum, max_sum+1]))+[math.inf]
        for c in grid:
            direct, _ = marginal_extreme(event_matrix(x,y,c), p,q,dp,dq)
            calculated = full_review_resource_bounds(wx,wp,wy,wq,c).lower
            diff = abs(calculated-direct)
            error = max(error,diff)
            assert diff < 1e-9, (count,c,direct,calculated)
            capacities += 1
        for target in [.2,.7,.9]:
            need = minimum_robust_capacity(wx,wp,wy,wq,target)
            if need.feasible:
                c = need.minimum_capacity
                at, _ = marginal_extreme(event_matrix(x,y,c),p,q,dp,dq)
                before, _ = marginal_extreme(event_matrix(x,y,c-1),p,q,dp,dq)
                assert at >= target-1e-10 and before < target-1e-10
            else:
                ceiling, _ = marginal_extreme(event_matrix(x,y,math.inf),p,q,dp,dq)
                assert ceiling < target-1e-10
            requirements += 1
    return dict(cases=count, capacity_LP_comparisons=capacities,
                minimum_capacity_checks=requirements, largest_absolute_error=error)


def application():
    from full_counterfactual_model import build_records, complete_batch_matrix, moment
    state = build_records(persist=False)
    first, second, route = state["first"], state["second"], state["route"]
    groups = sorted(set(second.groupby("image").person.apply(frozenset)), key=lambda s: sorted(s))
    owner = {image: groups.index(frozenset(frame.person)) for image,frame in second.groupby("image")}
    record_owner = first.image.map(owner).to_numpy(int)
    u, target = route == 2, state["targets"]["gun_knife"]
    responses = state["referrals"].reshape(10,10,-1)
    values = []
    for group,block in [(0,responses[:,0,:]), (1,responses[0,:,:])]:
        mask = u & (record_owner == group)
        all_referred = np.all(block[:, mask & target],axis=1)
        values.append(np.where(all_referred, block[:, mask & ~target].sum(axis=1),math.inf))
    x,y = values
    p = q = np.full(10,.1)
    m = int((u & target).sum())
    offset = int((~u).sum())+m
    detection = moment(.9999,0.,int(((route==0)&target).sum()))*moment(.99,0.,int(((route==1)&target).sum())+m)
    rows, details = [], []
    for radius in [0., .01, .025, .05, .1, .2, 1.]:
        wx,wp = worst_load_weights(x,p,radius)
        wy,wq = worst_load_weights(y,q,radius)
        base = full_review_resource_bounds(wx,wp,wy,wq,949,offset,detection)
        row = dict(radius=radius, lower_at_949=base.lower, ceiling=base.lower_ceiling)
        for requirement in [.8,.9]:
            need = minimum_robust_capacity(wx,wp,wy,wq,requirement,offset,detection)
            capacity = need.minimum_capacity
            at = full_review_resource_bounds(wx,wp,wy,wq,capacity,offset,detection).lower
            before = full_review_resource_bounds(wx,wp,wy,wq,capacity-1,offset,detection).lower
            # Compare to the original 10-by-10 response kernel, independently.
            for c,expected in [(capacity,at),(capacity-1,before)]:
                kernel = complete_batch_matrix(state,"gun_knife",int(u.sum()),c)
                lp,_ = marginal_extreme(kernel,p,q,radius,radius)
                assert np.isclose(lp,expected,atol=1e-10)
            assert at >= requirement-1e-10 and before < requirement-1e-10
            suffix = str(round(100*requirement))
            row["minimum_capacity_"+suffix] = capacity
            row["guarantee_"+suffix] = at
            details.append(dict(radius=radius, guarantee=at, previous_guarantee=before,**asdict(need)))
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/"capacity_by_radius.csv",index=False)
    pd.DataFrame(details).to_csv(OUT/"capacity_certificates.csv",index=False)
    return dict(application_rows=rows, source_hashes=state["provenance"]["source_sha256"],
                detection_factor=detection, mandatory_plus_targets=offset,
                first_loads=x.tolist(),second_loads=y.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    started = time.time()
    current = hashes()
    if not args.smoke:
        gate = json.loads((OUT/"smoke.json").read_text())
        if not gate.get("passed") or gate["source_sha256"] != current:
            raise RuntimeError("Run a passing smoke test for the current source first")
    result = check_cases(10 if args.smoke else 120,2026101801 if args.smoke else 2026101802)
    if not args.smoke: result.update(application())
    result.update(passed=True,smoke=args.smoke,source_sha256=current,seconds=time.time()-started)
    (OUT/("smoke.json" if args.smoke else "verification.json")).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
