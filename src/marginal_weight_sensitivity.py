"""Sensitivity of finite-batch guarantees to reweighting observed response profiles.

The optimization perturbs the empirical marginal weights over the observed
participant profiles. It does not introduce profiles absent from the data.
The numerical routines are separate from the full policy study and plotting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results/manuscript_20261017/marginal_weight_sensitivity"
SOURCE_FILES = [Path(__file__), Path(__file__).with_name("full_counterfactual_model.py")]


def source_hashes():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in SOURCE_FILES}


def _problem(matrix, p, q, delta_p, delta_q, *, maximize=False, tipping=None):
    """Build a transportation LP with total-variation marginal budgets."""
    matrix = np.asarray(matrix, dtype=float)
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    rows, cols = matrix.shape
    if p.shape != (rows,) or q.shape != (cols,):
        raise ValueError("Matrix dimensions must match the two marginal vectors")
    if not np.isclose(p.sum(), 1) or not np.isclose(q.sum(), 1):
        raise ValueError("Marginals must each sum to one")
    if min(delta_p, delta_q) < 0:
        raise ValueError("Total-variation radii must be nonnegative")

    n_q = rows * cols
    # Variables: Q, positive/negative row shifts, positive/negative column shifts.
    row_pos = n_q
    row_neg = row_pos + rows
    col_pos = row_neg + rows
    col_neg = col_pos + cols
    n_var = col_neg + cols
    q_index = lambda i, j: i * cols + j

    a_eq = np.zeros((1 + rows + cols, n_var))
    b_eq = np.zeros(1 + rows + cols)
    a_eq[0, :n_q] = 1
    b_eq[0] = 1
    for i in range(rows):
        a_eq[1 + i, [q_index(i, j) for j in range(cols)]] = 1
        a_eq[1 + i, row_pos + i] = -1
        a_eq[1 + i, row_neg + i] = 1
        b_eq[1 + i] = p[i]
    for j in range(cols):
        r = 1 + rows + j
        a_eq[r, [q_index(i, j) for i in range(rows)]] = 1
        a_eq[r, col_pos + j] = -1
        a_eq[r, col_neg + j] = 1
        b_eq[r] = q[j]

    a_ub = np.zeros((2, n_var))
    b_ub = np.array([2 * delta_p, 2 * delta_q], dtype=float)
    a_ub[0, row_pos:row_neg + rows] = 1
    a_ub[1, col_pos:col_neg + cols] = 1
    if tipping is not None:
        # One common radius bounds both empirical marginal perturbations.
        radius_index = n_var
        expanded_eq = np.pad(a_eq, ((0, 0), (0, 1)))
        expanded_ub = np.pad(a_ub, ((0, 0), (0, 1)))
        expanded_ub[0, radius_index] = -2
        expanded_ub[1, radius_index] = -2
        values = np.asarray(tipping["kernel"], dtype=float).reshape(-1)
        if values.size != n_q:
            raise ValueError("Tipping kernel shape does not match the marginal support")
        direction = tipping["direction"]
        threshold = float(tipping["threshold"])
        condition = np.zeros(n_var + 1)
        condition[:n_q] = values if direction == "below" else -values
        bound = threshold if direction == "below" else -threshold
        expanded_ub = np.vstack([expanded_ub, condition])
        b_ub = np.append(b_ub, bound)
        objective = np.zeros(n_var + 1)
        objective[radius_index] = 1
        bounds = [(0, None)] * n_var + [(0, 1)]
        return objective, expanded_ub, b_ub, expanded_eq, b_eq, bounds

    objective = np.zeros(n_var)
    coefficients = matrix.reshape(-1)
    objective[:n_q] = -coefficients if maximize else coefficients
    bounds = [(0, None)] * n_var
    return objective, a_ub, b_ub, a_eq, b_eq, bounds


def marginal_extreme(matrix, p, q, delta_p=0.0, delta_q=0.0, *, maximize=False):
    """Return the sharp expectation bound over the declared marginal set."""
    objective, a_ub, b_ub, a_eq, b_eq, bounds = _problem(
        matrix, p, q, delta_p, delta_q, maximize=maximize
    )
    result = linprog(objective, A_ub=a_ub, b_ub=b_ub,
                     A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Marginal-bound LP failed: {result.message}")
    value = float(np.sum(np.asarray(matrix, dtype=float).reshape(-1) * result.x[:np.asarray(matrix).size]))
    return value, result.x[:np.asarray(matrix).size].reshape(np.asarray(matrix).shape)


def tipping_radius(matrix, p, q, threshold, *, direction):
    """Smallest common TV radius permitting an expectation to cross a threshold.

    direction='above' asks whether some admissible joint law reaches at least
    the threshold; direction='below' asks whether some law reaches at most it.
    """
    if direction not in {"above", "below"}:
        raise ValueError("direction must be 'above' or 'below'")
    objective, a_ub, b_ub, a_eq, b_eq, bounds = _problem(
        matrix, p, q, 0.0, 0.0,
        tipping={"kernel": matrix, "threshold": threshold, "direction": direction},
    )
    result = linprog(objective, A_ub=a_ub, b_ub=b_ub,
                     A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if result.status == 2:
        return None
    if not result.success:
        raise RuntimeError(f"Tipping-radius LP failed: {result.message}")
    return float(result.x[-1])


def smoke_test():
    p = q = np.array([0.5, 0.5])
    row_kernel = np.array([[1.0, 1.0], [0.0, 0.0]])
    lo0, _ = marginal_extreme(row_kernel, p, q)
    hi0, _ = marginal_extreme(row_kernel, p, q, maximize=True)
    lo1, _ = marginal_extreme(row_kernel, p, q, 0.1, 0.0)
    hi1, _ = marginal_extreme(row_kernel, p, q, 0.1, 0.0, maximize=True)
    tip = tipping_radius(row_kernel, p, q, 0.6, direction="above")
    assert np.isclose(lo0, 0.5) and np.isclose(hi0, 0.5)
    assert np.isclose(lo1, 0.4) and np.isclose(hi1, 0.6)
    assert np.isclose(tip, 0.1)
    # The full-support radius must contain the unrestricted 2-by-2 coupling set.
    check_kernel = np.array([[0.0, 1.0], [1.0, 0.0]])
    lo_full, _ = marginal_extreme(check_kernel, p, q, 1.0, 1.0)
    hi_full, _ = marginal_extreme(check_kernel, p, q, 1.0, 1.0, maximize=True)
    assert np.isclose(lo_full, 0.0) and np.isclose(hi_full, 1.0)
    return dict(passed=True, empirical_bound=(lo0, hi0), perturbed_bound=(lo1, hi1),
                tipping_radius=tip, unrestricted_bound=(lo_full, hi_full),
                source_sha256=source_hashes())


def miss_kernel(state, reviews, *, db=0.9999, d=0.99):
    """Conditional mean missed-target fraction for each recorded profile pair."""
    route = state["route"]
    target = state["targets"]["gun_knife"]
    referrals = state["referrals"]
    eligible = route == 2
    eligible_targets = eligible & target
    population = int(eligible.sum())
    all_targets = int(target.sum())
    baseline_misses = int((target & (route == 0)).sum()) * (1 - db)
    baseline_misses += int((target & (route == 1)).sum()) * (1 - d)
    review_probability = reviews / population
    result = np.empty((10, 10), dtype=float)
    for first in range(10):
        for second in range(10):
            profile = first * 10 + second
            referred_targets = int(referrals[profile, eligible_targets].sum())
            expected_misses = baseline_misses + int(eligible_targets.sum()) - review_probability * d * referred_targets
            result[first, second] = expected_misses / all_targets
    return result


def run_full():
    from full_counterfactual_model import build_records, complete_batch_matrix, expected_batch

    state = build_records(persist=False)
    p = np.full(10, 0.1)
    q = np.full(10, 0.1)
    delta_grid = [0.0, 0.005, 0.01, 0.025, 0.05, 1.0]
    matrices = {
        "partial_218_C907": complete_batch_matrix(state, "gun_knife", 218, 907),
        "full_244_C907": complete_batch_matrix(state, "gun_knife", 244, 907),
        "partial_218_C930": complete_batch_matrix(state, "gun_knife", 218, 930),
        "full_244_C930": complete_batch_matrix(state, "gun_knife", 244, 930),
        "full_244_C949": complete_batch_matrix(state, "gun_knife", 244, 949),
    }
    effects = {
        "effect_218_minus_full_C907": matrices["partial_218_C907"] - matrices["full_244_C907"],
        "effect_218_minus_full_C930": matrices["partial_218_C930"] - matrices["full_244_C930"],
    }
    risks = {"risk_218": miss_kernel(state, 218), "risk_244": miss_kernel(state, 244)}

    rows = []
    for radius in delta_grid:
        for name, kernel in {**effects, **{"success_full_C949": matrices["full_244_C949"]}, **risks}.items():
            lo, _ = marginal_extreme(kernel, p, q, radius, radius)
            hi, _ = marginal_extreme(kernel, p, q, radius, radius, maximize=True)
            rows.append(dict(radius=radius, quantity=name, lower=lo, upper=hi))

    tipping = [
        dict(claim="218-review superiority remains possible at C=930", threshold=0.0,
             direction="above", radius=tipping_radius(effects["effect_218_minus_full_C930"], p, q,
                                                        0.0, direction="above")),
        dict(claim="full-review success can fall below 80% at C=949", threshold=0.8,
             direction="below", radius=tipping_radius(matrices["full_244_C949"], p, q,
                                                        0.8, direction="below")),
        dict(claim="218-review mean miss risk can exceed 0.1%", threshold=0.001,
             direction="above", radius=tipping_radius(risks["risk_218"], p, q,
                                                        0.001, direction="above")),
        dict(claim="full-review mean miss risk can exceed 0.1%", threshold=0.001,
             direction="above", radius=tipping_radius(risks["risk_244"], p, q,
                                                        0.001, direction="above")),
    ]
    # Nominal risk kernels must reproduce the existing independent-profile result.
    for reviews, key in [(218, "risk_218"), (244, "risk_244")]:
        nominal = float(np.mean(risks[key]))
        target = expected_batch(state, "gun_knife", reviews)["conditional_miss"]
        assert np.isclose(nominal, target, atol=1e-12), (reviews, nominal, target)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "marginal_sensitivity.csv", index=False)
    pd.DataFrame(tipping).to_csv(OUT / "tipping_radii.csv", index=False)
    report = dict(passed=True, source_sha256=source_hashes(),
                  profile_support="10 observed participant profiles in each response group",
                  empirical_weights="uniform within each group",
                  includes_unobserved_profiles=False,
                  delta_grid=delta_grid,
                  sensitivity_rows=len(rows), tipping_points=tipping)
    (OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        report = smoke_test()
        (OUT / "smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    smoke = json.loads((OUT / "smoke.json").read_text(encoding="utf-8"))
    if not smoke.get("passed") or smoke.get("source_sha256") != source_hashes():
        raise RuntimeError("The current implementation has no matching passing smoke test")
    run_full()


if __name__ == "__main__":
    main()
