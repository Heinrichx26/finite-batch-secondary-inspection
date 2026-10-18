"""Participant-level bootstrap of empirical response-profile weights.

Each of the two disjoint ten-person groups is resampled independently. For
each resample, sharp paired policy-effect bounds are recomputed over every
joint distribution compatible with the resampled group marginals. Bootstrap
frequencies are descriptive stability diagnostics, not probabilities that a
policy is truly better and not confidence levels.

Run ``--smoke`` before ``--full``. No manuscript text or source data is edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from full_counterfactual_model import build_records, complete_batch_matrix, transport_bounds
from marginal_weight_sensitivity import marginal_extreme

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results" / "participant_weight_bootstrap"
ENDPOINT = "gun_knife"
POLICIES = (218, 244)
CAPACITIES = (907, 930)
SEED = 20261017
REPLICATES = 2000
TOL = 1e-11


def source_hashes() -> dict[str, str]:
    paths = [Path(__file__), Path(__file__).with_name("full_counterfactual_model.py"),
             Path(__file__).with_name("marginal_weight_sensitivity.py")]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def response_groups(state) -> list[list[str]]:
    second = state["second"]
    groups = sorted(set(second.groupby("image").person.apply(frozenset)), key=lambda s: sorted(s))
    result = [sorted(group) for group in groups]
    if len(result) != 2 or [len(group) for group in result] != [10, 10]:
        raise ValueError("Expected two disjoint response-profile groups with ten people each")
    if set(result[0]) & set(result[1]):
        raise ValueError("Response-profile groups must contain disjoint participants")
    return result


def profile_matrices(state) -> dict[int, dict[str, np.ndarray]]:
    matrices = {}
    for capacity in CAPACITIES:
        matrices[capacity] = {
            str(n): complete_batch_matrix(state, ENDPOINT, n, capacity)
            for n in POLICIES
        }
        matrices[capacity]["effect"] = matrices[capacity]["218"] - matrices[capacity]["244"]
    return matrices


def paired_bounds(matrix: np.ndarray, p: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    lower, _ = marginal_extreme(matrix, p, q)
    upper, _ = marginal_extreme(matrix, p, q, maximize=True)
    return lower, upper


def smoke_test() -> dict:
    state = build_records(persist=False)
    groups = response_groups(state)
    assert state["referrals"].shape[0] == len(groups[0]) * len(groups[1])
    matrices = profile_matrices(state)
    p = q = np.repeat(0.1, 10)
    rng = np.random.default_rng(SEED)
    checked = 0
    for _ in range(8):
        pc = rng.multinomial(10, p)
        qc = rng.multinomial(10, q)
        assert pc.sum() == qc.sum() == 10
        pp, qq = pc / 10, qc / 10
        for capacity in CAPACITIES:
            low, high = paired_bounds(matrices[capacity]["effect"], pp, qq)
            assert -1 - TOL <= low <= high + TOL <= 1 + TOL
            checked += 1
    for capacity in CAPACITIES:
        exact = transport_bounds(matrices[capacity]["effect"])
        lp = paired_bounds(matrices[capacity]["effect"], p, q)
        assert np.allclose(lp, (exact[0], exact[2]), atol=1e-10)
    nominal_930 = paired_bounds(matrices[930]["effect"], p, q)
    assert np.isclose(nominal_930[1], -0.0020089728725697586, atol=1e-10)
    return {
        "passed": True,
        "source_sha256": source_hashes(),
        "participants_per_group": [len(group) for group in groups],
        "bootstrap_draws_smoke": 8,
        "paired_bounds_checked": checked,
        "nominal_effect_bounds_C907": paired_bounds(matrices[907]["effect"], p, q),
        "nominal_effect_bounds_C930": nominal_930,
    }


def run_full(repetitions: int = REPLICATES) -> dict:
    state = build_records(persist=False)
    groups = response_groups(state)
    matrices = profile_matrices(state)
    p0 = q0 = np.repeat(0.1, 10)
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    for replicate in range(repetitions):
        p_counts = rng.multinomial(10, p0)
        q_counts = rng.multinomial(10, q0)
        p, q = p_counts / 10, q_counts / 10
        tv_p = float(np.abs(p - p0).sum() / 2)
        tv_q = float(np.abs(q - q0).sum() / 2)
        for capacity in CAPACITIES:
            lower, upper = paired_bounds(matrices[capacity]["effect"], p, q)
            rows.append({
                "replicate": replicate + 1,
                "capacity": capacity,
                "effect_lower": lower,
                "effect_upper": upper,
                "effect_interval_contains_zero": bool(lower <= TOL and upper >= -TOL),
                "partial_superior_for_all_compatible_pairings": bool(lower > TOL),
                "full_superior_for_all_compatible_pairings": bool(upper < -TOL),
                "tv_group_1": tv_p,
                "tv_group_2": tv_q,
                "group_1_resampled_counts": ",".join(map(str, p_counts.tolist())),
                "group_2_resampled_counts": ",".join(map(str, q_counts.tolist())),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "bootstrap_effect_bounds.csv", index=False)

    summaries = []
    for capacity, group in frame.groupby("capacity", sort=True):
        summaries.append({
            "capacity": int(capacity),
            "replicates": int(len(group)),
            "nominal_lower": paired_bounds(matrices[int(capacity)]["effect"], p0, q0)[0],
            "nominal_upper": paired_bounds(matrices[int(capacity)]["effect"], p0, q0)[1],
            "lower_q025": float(group.effect_lower.quantile(.025)),
            "lower_median": float(group.effect_lower.median()),
            "lower_q975": float(group.effect_lower.quantile(.975)),
            "upper_q025": float(group.effect_upper.quantile(.025)),
            "upper_median": float(group.effect_upper.median()),
            "upper_q975": float(group.effect_upper.quantile(.975)),
            "resamples_allowing_partial_superiority": float((group.effect_upper >= -TOL).mean()),
            "resamples_certifying_full_superiority": float((group.effect_upper < -TOL).mean()),
            "resamples_certifying_partial_superiority": float((group.effect_lower > TOL).mean()),
            "median_tv_group_1": float(group.tv_group_1.median()),
            "median_tv_group_2": float(group.tv_group_2.median()),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "bootstrap_summary.csv", index=False)

    # Compare the empirical bootstrap movement to the independently computed
    # total-variation tipping radius for the C=930 policy-effect upper bound.
    from marginal_weight_sensitivity import tipping_radius
    nominal_tip = tipping_radius(matrices[930]["effect"], p0, q0, 0.0, direction="above")
    verification = {
        "passed": True,
        "source_sha256": source_hashes(),
        "seed": SEED,
        "replicates": repetitions,
        "participants_per_group": [len(group) for group in groups],
        "capacities": list(CAPACITIES),
        "primary_endpoint": ENDPOINT,
        "effect_sign": "partial-review success minus full-review success under the same joint law",
        "common_law_bounds_recomputed_for_each_resample": True,
        "bootstrap_frequency_is_confidence_level": False,
        "nominal_C930_tipping_radius": nominal_tip,
        "median_minimum_group_TV": float(np.median(np.minimum(
            frame.drop_duplicates("replicate").tv_group_1,
            frame.drop_duplicates("replicate").tv_group_2))),
        "rows": int(len(frame)),
    }
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        result = smoke_test()
        (OUT / "smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    smoke = json.loads((OUT / "smoke.json").read_text(encoding="utf-8"))
    if not smoke.get("passed") or smoke.get("source_sha256") != source_hashes():
        raise RuntimeError("No matching passing smoke test exists for the current implementation")
    started = time.perf_counter()
    result = run_full(args.replicates)
    result["seconds"] = time.perf_counter() - started
    (OUT / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
