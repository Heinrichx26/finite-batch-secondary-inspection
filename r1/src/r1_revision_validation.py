from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import r1_image_unit_analysis as base


BOOTSTRAP_REPLICATES = 2000
RISK_LIMIT = 0.001
RESPONSE_CELLS = [
    "type_of_responsive", "type_of_tv", "type_of_vis", "beforeOrAfterTV",
]


def leave_one_image_out_case(case: dict, dropped_index: int) -> dict:
    reduced = {key: value for key, value in case.items()}
    retained = np.arange(case["image_count"]) != dropped_index
    reduced["image_ids"] = [image for index, image in enumerate(case["image_ids"])
                            if retained[index]]
    reduced["targets"] = case["targets"][retained]
    reduced["machine"] = case["machine"][retained]
    reduced["primary"] = [
        {"route": person["route"][retained]} for person in case["primary"]
    ]
    reduced["secondary"] = [
        {"referral": person["referral"][retained]}
        for person in case["secondary"]
    ]
    reduced["image_count"] = int(retained.sum())
    reduced["target_count"] = int(reduced["targets"].sum())
    if reduced["target_count"] == 0:
        raise ValueError("An image-removal case cannot remove every target")
    return reduced


def transport_extreme_with_omissions(
        complete: np.ndarray, omitted: np.ndarray, left: np.ndarray,
        right: np.ndarray, allowance: float, maximize: bool = False,
        eligible: np.ndarray | None = None) -> float:
    rows, columns = complete.shape
    cells = rows * columns
    equality = np.zeros((rows + columns, 2 * cells), dtype=float)
    for row in range(rows):
        indices = np.arange(row * columns, (row + 1) * columns)
        equality[row, indices] = 1
        equality[row, cells + indices] = 1
    for column in range(columns):
        indices = np.arange(column, cells, columns)
        equality[rows + column, indices] = 1
        equality[rows + column, cells + indices] = 1
    marginal = np.concatenate((left, right))
    objective = np.concatenate((complete.ravel(), omitted.ravel()))
    if maximize:
        objective = -objective
    removal = np.zeros(2 * cells, dtype=float)
    removal[cells:] = 1
    if eligible is None:
        eligible = np.ones_like(complete, dtype=bool)
    bounds = [(0, None)] * cells + [
        (0, None if possible else 0) for possible in eligible.ravel()
    ]
    result = base.linprog(
        objective,
        A_ub=removal[None, :], b_ub=[allowance],
        A_eq=equality[:-1], b_eq=marginal[:-1],
        bounds=bounds, method="highs",
    )
    if not result.success:
        raise ArithmeticError(f"Target-omission transport LP failed: {result.message}")
    return float(np.dot(np.concatenate((complete.ravel(), omitted.ravel())), result.x))


def order_lower(matrix: np.ndarray, radius: float) -> np.ndarray:
    stable = (matrix <= 1e-12) | (matrix >= 1 - 1e-12)
    return np.where(stable, matrix, np.maximum(0, matrix - radius))


def order_upper(matrix: np.ndarray, radius: float) -> np.ndarray:
    stable = (matrix <= 1e-12) | (matrix >= 1 - 1e-12)
    return np.where(stable, matrix, np.minimum(1, matrix + radius))


def joint_risk_thresholds(case: dict, policy: dict,
                         include_target_omissions: bool = True) -> list[dict]:
    components = policy["components"]
    left, right = policy["left_weights"], policy["right_weights"]
    maximum_readings = int(policy["capacity_rows"][0]["common_reading_limit"])
    targets = int(case["target_count"])
    rows = []

    for capacity_row in policy["capacity_rows"]:
        capacity = int(capacity_row["total_search_capacity"])
        stopped = np.asarray([[
            base.stop_success(components[i][j], maximum_readings, capacity)
            for j in range(len(right))
        ] for i in range(len(left))])
        stopped_risk = np.asarray([[
            base.referral_miss(components[i][j], maximum_readings, targets,
                               stopping=True, total_capacity=capacity)
            for j in range(len(right))
        ] for i in range(len(left))])
        stop_lower = base.transport_extreme(stopped, left, right)
        stop_risk_upper = base.transport_extreme(stopped_risk, left, right, maximize=True)
        fixed_candidates = []
        common_fixed = np.empty_like(stopped)
        common_fixed_risk = np.empty_like(stopped_risk)

        for readings in range(maximum_readings + 1):
            fixed = np.asarray([[
                base.fixed_success(components[i][j], readings, capacity)
                for j in range(len(right))
            ] for i in range(len(left))])
            fixed_risk = np.asarray([[
                base.referral_miss(components[i][j], readings, targets)
                for j in range(len(right))
            ] for i in range(len(left))])
            risk_upper = base.transport_extreme(fixed_risk, left, right, maximize=True)
            if readings == maximum_readings:
                common_fixed[:] = fixed
                common_fixed_risk[:] = fixed_risk
            if risk_upper <= RISK_LIMIT + 1e-12:
                fixed_candidates.append((
                    base.transport_extreme(fixed, left, right), readings, risk_upper,
                ))

        best_fixed = max(fixed_candidates, key=lambda row: (round(row[0], 12), -row[1]),
                         default=(float("nan"), -1, float("nan")))
        common_fixed_lower = base.transport_extreme(common_fixed, left, right)
        common_fixed_risk_upper = base.transport_extreme(
            common_fixed_risk, left, right, maximize=True,
        )
        effect_lower, effect_upper = base.transport_bounds(
            stopped - common_fixed, left, right,
        )
        optional_target_counts = np.asarray([
            [part["required_targets"] for part in row] for row in components
        ])
        omission_eligible = optional_target_counts > 0
        omission_risk = optional_target_counts / targets
        omission_values = []
        omission_allowances = (0.0, 0.005, 0.009, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20) \
            if include_target_omissions else ()
        for allowance in omission_allowances:
            stop_omission_success = transport_extreme_with_omissions(
                stopped, np.zeros_like(stopped), left, right, allowance,
                eligible=omission_eligible,
            )
            fixed_omission_success = transport_extreme_with_omissions(
                common_fixed, np.zeros_like(common_fixed), left, right, allowance,
                eligible=omission_eligible,
            )
            stop_omission_risk = transport_extreme_with_omissions(
                stopped_risk, omission_risk, left, right, allowance,
                maximize=True, eligible=omission_eligible,
            )
            fixed_omission_risk = transport_extreme_with_omissions(
                common_fixed_risk, omission_risk, left, right, allowance,
                maximize=True, eligible=omission_eligible,
            )
            omission_values.append({
                "batch_level_target_omission_allowance": allowance,
                "stopping_success_lower": stop_omission_success,
                "stopping_mean_miss_upper": stop_omission_risk,
                "stopping_jointly_feasible":
                    stop_omission_success >= 0.8 - 1e-12 and
                    stop_omission_risk <= RISK_LIMIT + 1e-12,
                "same_reading_allowance_fixed_success_lower": fixed_omission_success,
                "same_reading_allowance_fixed_mean_miss_upper": fixed_omission_risk,
                "same_reading_allowance_fixed_jointly_feasible":
                    fixed_omission_success >= 0.8 - 1e-12 and
                    fixed_omission_risk <= RISK_LIMIT + 1e-12,
            })

        order_values = []
        if include_target_omissions:
            if not np.all((common_fixed <= 1e-12) | (common_fixed >= 1 - 1e-12)):
                raise AssertionError("The matched fixed rule must read every eligible image")
            for radius in (0.0, 0.001, 0.01, 0.05, 0.10, 1.0):
                lower_matrix = order_lower(stopped, radius)
                upper_matrix = order_upper(stopped, radius)
                order_values.append({
                    "permutation_total_variation_radius": radius,
                    "stopping_success_lower_under_order_perturbation":
                        base.transport_extreme(lower_matrix, left, right),
                    "fixed_success_lower_under_order_perturbation": common_fixed_lower,
                    "matched_effect_lower_under_order_perturbation":
                        base.transport_extreme(lower_matrix - common_fixed, left, right),
                    "matched_effect_upper_under_order_perturbation":
                        base.transport_extreme(upper_matrix - common_fixed, left, right, True),
                    "stopping_risk_upper_under_order_perturbation": base.transport_extreme(
                        order_upper(stopped_risk, radius), left, right, True,
                    ),
                    "fixed_risk_upper_under_order_perturbation": common_fixed_risk_upper,
                })
        rows.append({
            "condition": case["condition"],
            "unique_images": case["image_count"],
            "target_images": targets,
            "maximum_reading_allowance": maximum_readings,
            "search_capacity": capacity,
            "stopping_success_lower": stop_lower,
            "stopping_mean_miss_upper": stop_risk_upper,
            "stopping_jointly_feasible": stop_risk_upper <= RISK_LIMIT + 1e-12,
            "risk_constrained_best_fixed_success_lower": best_fixed[0],
            "risk_constrained_best_fixed_readings": best_fixed[1],
            "risk_constrained_best_fixed_mean_miss_upper": best_fixed[2],
            "risk_constrained_fixed_feasible": best_fixed[1] >= 0,
            "same_reading_allowance_fixed_success_lower": common_fixed_lower,
            "same_reading_allowance_fixed_mean_miss_upper": common_fixed_risk_upper,
            "same_reading_allowance_fixed_jointly_feasible":
                common_fixed_risk_upper <= RISK_LIMIT + 1e-12,
            "matched_stop_minus_fixed_effect_lower": effect_lower,
            "matched_stop_minus_fixed_effect_upper": effect_upper,
            "target_omission_sensitivity": omission_values,
            "order_sensitivity": order_values,
        })
    return rows


def capacity_at_threshold(rows: list[dict], policy: str,
                          requirement: float) -> float | int:
    if policy == "stopping":
        qualifying = [row for row in rows
                      if row["stopping_jointly_feasible"] and
                      row["stopping_success_lower"] >= requirement - 1e-12]
    else:
        qualifying = [row for row in rows
                      if row["risk_constrained_fixed_feasible"] and
                      row["risk_constrained_best_fixed_success_lower"]
                      >= requirement - 1e-12]
    return min((int(row["search_capacity"]) for row in qualifying), default=np.nan)


def analyse_external_responses(replicates: int) -> pd.DataFrame:
    data = pd.read_csv(base.EXTERNAL_FILE, keep_default_na=False)
    data["correct"] = pd.to_numeric(data["userCorrect"], errors="raise")
    data["truth_request"] = pd.to_numeric(data["openBaggage"], errors="raise")
    identifiers = data[["userID", "type_of_responsive", "type_of_vis"]].drop_duplicates()
    if identifiers["userID"].duplicated().any():
        raise ValueError("Participant-level visualization strata are not stable")
    participants = data["userID"].drop_duplicates().tolist()
    participant_index = {person: index for index, person in enumerate(participants)}
    data["person_index"] = data["userID"].map(participant_index)
    strata = list(data.groupby(RESPONSE_CELLS, sort=True))
    identities = identifiers.groupby(["type_of_responsive", "type_of_vis"])["userID"] \
        .apply(list).to_dict()
    point_rows = []
    for key, subset in strata:
        row = dict(zip(RESPONSE_CELLS, key))
        row["participants"] = int(subset["userID"].nunique())
        row["tasks"] = int(len(subset))
        row["task_accuracy"] = float(subset["correct"].mean())
        row["truth_label_request"] = float(subset["truth_request"].mean())
        point_rows.append(row)

    accuracy_draws = np.empty((replicates, len(strata)))
    request_draws = np.empty_like(accuracy_draws)
    rng = np.random.default_rng(base.SEED + 7)
    for repetition in range(replicates):
        participant_weights = np.zeros(len(participants), dtype=int)
        for candidates in identities.values():
            sampled = rng.choice(candidates, size=len(candidates), replace=True)
            sampled_indices = [participant_index[person] for person in sampled]
            participant_weights += np.bincount(
                sampled_indices, minlength=len(participants),
            )
        for column, (_, subset) in enumerate(strata):
            indices = subset["person_index"].to_numpy(dtype=int)
            weights = participant_weights[indices].astype(float)
            accuracy_draws[repetition, column] = np.average(
                subset["correct"], weights=weights,
            )
            request_draws[repetition, column] = np.average(
                subset["truth_request"], weights=weights,
            )

    for column, row in enumerate(point_rows):
        for name, draws in (("task_accuracy", accuracy_draws),
                            ("truth_label_request", request_draws)):
            row[f"{name}_bootstrap_2_5"] = float(np.quantile(draws[:, column], 0.025))
            row[f"{name}_bootstrap_median"] = float(np.quantile(draws[:, column], 0.5))
            row[f"{name}_bootstrap_97_5"] = float(np.quantile(draws[:, column], 0.975))
        row["bootstrap_replicates"] = replicates
        row["bootstrap_unit"] = "Whole participants, stratified by response and visualization"
        row["scope"] = "Task correctness and requests to reveal the true label"
    return pd.DataFrame(point_rows)


def analyse_pilot_task_times(replicates: int) -> pd.DataFrame:
    data = pd.read_excel(base.PRIMARY_FILE, keep_default_na=False)
    data["task_time_seconds"] = pd.to_numeric(data["TaskTime"], errors="coerce")
    data = data[data["num_of_tv"] == "tv3"].copy()
    if not np.isfinite(data["task_time_seconds"]).all() or \
            (data["task_time_seconds"] <= 0).any():
        raise ValueError("The prespecified pilot timing records contain invalid values")

    groups = (
        ("AI-assisted", "False Alarm", "AI", "False Alarm"),
        ("AI-assisted", "False Miss", "AI", "False Miss"),
        ("Unaided", "False Alarm", "NoAI", "falselyAlarmNoAI"),
        ("Unaided", "False Miss", "NoAI", "falselyMissNoAI"),
    )
    rng = np.random.default_rng(base.SEED + 19)
    records = []
    for assistance, condition, group, label in groups:
        subset = data[(data["GROUP"] == group) & (data["type_of_tv"] == label)].copy()
        if subset["user_identifier"].nunique() != 10 or \
                subset["ImageNumber"].nunique() != 27 or len(subset) != 270:
            raise ValueError(f"Unexpected task-time structure for {assistance}, {condition}")
        participant_medians = subset.groupby("user_identifier")[
            "task_time_seconds"].median().to_numpy(dtype=float)
        if len(participant_medians) != 10 or not np.isfinite(participant_medians).all():
            raise ValueError("Every participant must contribute a complete timing profile")
        resampled_means = np.empty(replicates)
        for repetition in range(replicates):
            sample = rng.integers(0, len(participant_medians), len(participant_medians))
            resampled_means[repetition] = participant_medians[sample].mean()
        records.append({
            "assistance_condition": assistance,
            "AI_error_condition": condition,
            "participants": len(participant_medians),
            "image_judgments": int(len(subset)),
            "unique_images": int(subset["ImageNumber"].nunique()),
            "task_time_unit": "Seconds",
            "pooled_task_time_median": float(subset["task_time_seconds"].median()),
            "mean_of_participant_median_task_times": float(participant_medians.mean()),
            "participant_median_minimum": float(participant_medians.min()),
            "participant_median_maximum": float(participant_medians.max()),
            "bootstrap_2_5": float(np.quantile(resampled_means, 0.025)),
            "bootstrap_median": float(np.quantile(resampled_means, 0.5)),
            "bootstrap_97_5": float(np.quantile(resampled_means, 0.975)),
            "bootstrap_replicates": replicates,
            "bootstrap_unit": "Whole participants within the four recorded conditions",
            "scope": "Elapsed time for each online image task",
        })
    return pd.DataFrame(records)


def _assert_close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=3e-12):
        raise AssertionError(f"{name}: expected {expected:.16g}, obtained {actual:.16g}")


def smoke_check(cases: dict[str, dict]) -> dict:
    source_verification = json.loads((base.OUT / "verification.json").read_text(
        encoding="utf-8"))
    if source_verification.get("status") != "completed":
        raise ValueError("The source-image analysis has no completed verification")
    if set(source_verification["source_sha256"]) != {
        "pilotResult_combinedAIandNoAI.xlsx", "task_selected_for_study.csv",
        "behavior_XAI_presented_beforeAndAfter_trustViolation.csv",
        "SIXRay_annotations_positive_all.csv",
    }:
        raise ValueError("The source hash manifest is incomplete")

    toy = np.asarray([[0.0, 0.25], [0.75, 1.0]])
    uniform = np.asarray([0.5, 0.5])
    lower, upper = base.transport_bounds(toy, uniform, uniform)
    _assert_close(lower, 0.5, "Exact two-by-two transportation lower endpoint")
    _assert_close(upper, 0.5, "Exact two-by-two transportation upper endpoint")
    omission_free = transport_extreme_with_omissions(
        toy, np.zeros_like(toy), uniform, uniform, 0.0,
    )
    _assert_close(omission_free, lower, "Zero-omission LP lower endpoint")
    omitted_at_one = transport_extreme_with_omissions(
        toy, np.zeros_like(toy), uniform, uniform, 1.0,
    )
    _assert_close(omitted_at_one, 0.0, "Complete-omission LP endpoint")
    no_eligible_targets = transport_extreme_with_omissions(
        toy, np.zeros_like(toy), uniform, uniform, 1.0,
        eligible=np.zeros_like(toy, dtype=bool),
    )
    _assert_close(no_eligible_targets, lower, "No target can be removed in a target-free pool")
    one_row_eligible = transport_extreme_with_omissions(
        np.ones_like(toy), np.zeros_like(toy), uniform, uniform, 1.0,
        eligible=np.array([[True, True], [False, False]]),
    )
    _assert_close(one_row_eligible, 0.5, "Omission is limited to positive eligible-target counts")
    endpoints = np.array([0.0, 0.4, 1.0])
    np.testing.assert_allclose(order_lower(endpoints, 0.2), [0.0, 0.2, 1.0])
    np.testing.assert_allclose(order_upper(endpoints, 0.2), [0.0, 0.6, 1.0])

    permutation_errors = 0
    test_component = {
        "population": 4, "mandatory": 0, "required_targets": 1,
        "all_targets_referred": True, "non_target_referrals": 2,
    }
    from itertools import permutations
    orders = list(permutations(range(4)))
    referrals = {0, 2, 3}
    targets = {2}
    for capacity in range(4):
        for reads in range(5):
            direct_success = 0
            for order in orders:
                selected = order[:min(reads, 4)]
                if capacity == 0:
                    continue
                referrals_seen = [image for image in selected if image in referrals]
                admitted = referrals_seen[:capacity]
                if targets.issubset(admitted) and len(admitted) <= capacity:
                    direct_success += 1
            exact = direct_success / len(orders)
            calculated = base.stop_success(test_component, reads, capacity)
            _assert_close(calculated, exact,
                          f"Stopping success at capacity={capacity}, reads={reads}")
            permutation_errors += 1

    for name, case in cases.items():
        if case["image_count"] != 27:
            raise ValueError(f"Unexpected image count in {name}")
        if len(set(case["image_ids"])) != case["image_count"]:
            raise ValueError(f"Duplicate image-unit key in {name}")
        leave_one_image_out_case(case, 0)

    pilot = pd.read_excel(base.PRIMARY_FILE, keep_default_na=False)
    trial = pilot[pilot["num_of_tv"] == "tv3"].copy()
    trial["time_seconds"] = pd.to_numeric(trial["TaskTime"], errors="coerce")
    if len(trial) != 1080 or not np.isfinite(trial["time_seconds"]).all() or \
            (trial["time_seconds"] <= 0).any():
        raise ValueError("Task times in the matched analysis conditions failed the smoke audit")
    if trial.groupby("user_identifier").size().ne(27).any():
        raise ValueError("Each matched-condition participant must have 27 image tasks")

    return {
        "passed": True,
        "matched_unique_image_conditions": sorted(cases),
        "brute_force_stop_policy_checks": permutation_errors,
        "transportation_bounds_checks": 2,
        "complete_positive_pilot_task_times": int(len(trial)),
        "claims": [
            "Response-pair uncertainty is kept separate from participant resampling.",
            "Physical detection and service completion have no pilot estimates.",
            "The 232-person file measures task correctness and truth requests.",
        ],
    }


def run_full(cases: dict, bootstrap_replicates: int) -> dict:
    base.OUT.mkdir(parents=True, exist_ok=True)
    detail_rows = []
    threshold_rows = []
    leave_one_out_rows = []
    leave_one_risk_rows = []

    for name, original_case in cases.items():
        case = dict(original_case)
        case["condition"] = name
        policy = base.scenario_matrices(case)
        detailed = joint_risk_thresholds(case, policy)
        detail_rows.extend(detailed)
        for requirement in (0.5, 0.8, 0.9):
            threshold_rows.append({
                "condition": name,
                "mean_miss_limit": RISK_LIMIT,
                "joint_referral_and_capacity_requirement": requirement,
                "stopping_minimum_capacity": capacity_at_threshold(
                    detailed, "stopping", requirement),
                "risk_constrained_best_fixed_minimum_capacity": capacity_at_threshold(
                    detailed, "fixed", requirement),
            })

        for dropped_index, dropped_image in enumerate(case["image_ids"]):
            sensitivity_case = leave_one_image_out_case(case, dropped_index)
            sensitivity_policy = base.scenario_matrices(sensitivity_case)
            loo_thresholds = sensitivity_policy["capacity_thresholds"]
            for result in loo_thresholds:
                leave_one_out_rows.append({
                    "condition": name,
                    "dropped_image_id": dropped_image,
                    "dropped_image_was_target": bool(case["targets"][dropped_index]),
                    "images_after_removal": sensitivity_case["image_count"],
                    "targets_after_removal": sensitivity_case["target_count"],
                    **result,
                })
            loo_risk = joint_risk_thresholds(
                sensitivity_case, sensitivity_policy,
                include_target_omissions=False,
            )
            for requirement in (0.5, 0.8, 0.9):
                for rule in ("stopping", "fixed"):
                    leave_one_risk_rows.append({
                        "condition": name,
                        "dropped_image_id": dropped_image,
                        "dropped_image_was_target": bool(case["targets"][dropped_index]),
                        "referral_and_capacity_requirement": requirement,
                        "risk_constrained_policy": rule,
                        "mean_miss_limit": RISK_LIMIT,
                        "minimum_capacity": capacity_at_threshold(
                            loo_risk, rule, requirement,
                        ),
                    })

    detail_path = base.OUT / "risk_constrained_capacity.csv"
    threshold_path = base.OUT / "risk_constrained_thresholds.csv"
    omission_path = base.OUT / "target_omission_sensitivity.csv"
    order_path = base.OUT / "random_order_sensitivity.csv"
    loo_path = base.OUT / "leave_one_image_out.csv"
    external_path = base.OUT / "external_behavior_strata.csv"
    time_path = base.OUT / "pilot_task_time.csv"
    leave_one_risk_path = base.OUT / "leave_one_image_out_risk_constrained.csv"
    pd.DataFrame([
        {key: value for key, value in row.items()
         if key not in {"target_omission_sensitivity", "order_sensitivity"}}
        for row in detail_rows
    ]).to_csv(detail_path, index=False)
    pd.DataFrame(threshold_rows).to_csv(threshold_path, index=False)
    omission_rows = []
    order_rows = []
    for row in detail_rows:
        for result in row["target_omission_sensitivity"]:
            omission_rows.append({
                key: row[key] for key in (
                    "condition", "search_capacity", "maximum_reading_allowance",
                )
            } | result)
        for result in row["order_sensitivity"]:
            order_rows.append({
                "condition": row["condition"],
                "search_capacity": row["search_capacity"],
                "uniform_order_stopping_lower": row["stopping_success_lower"],
                "uniform_order_fixed_lower":
                    row["same_reading_allowance_fixed_success_lower"],
                "uniform_order_matched_effect_lower":
                    row["matched_stop_minus_fixed_effect_lower"],
                "uniform_order_matched_effect_upper":
                    row["matched_stop_minus_fixed_effect_upper"],
                "uniform_order_stopping_risk_upper": row["stopping_mean_miss_upper"],
                "uniform_order_fixed_risk_upper":
                    row["same_reading_allowance_fixed_mean_miss_upper"],
            } | result)
    omission_frame = pd.DataFrame(omission_rows)
    order_frame = pd.DataFrame(order_rows)
    omission_frame.to_csv(omission_path, index=False)
    order_frame.to_csv(order_path, index=False)

    robust_threshold_rows = []
    for condition, subset in order_frame.groupby("condition", sort=True):
        for delta, perturbed in subset.groupby(
                "permutation_total_variation_radius", sort=True):
            for rule, success_column, risk_column in (
                    ("stopping", "stopping_success_lower_under_order_perturbation",
                     "stopping_risk_upper_under_order_perturbation"),
                    ("fixed at the same maximum reading allowance",
                     "fixed_success_lower_under_order_perturbation",
                     "fixed_risk_upper_under_order_perturbation")):
                qualifying = perturbed[
                    (perturbed[success_column] >= 0.8 - 1e-12) &
                    (perturbed[risk_column] <= RISK_LIMIT + 1e-12)
                ]
                robust_threshold_rows.append({
                    "condition": condition,
                    "mean_miss_limit": RISK_LIMIT,
                    "uniform_permutation_total_variation_radius": float(delta),
                    "rule": rule,
                    "joint_referral_and_capacity_requirement": 0.8,
                    "minimum_search_capacity": int(qualifying["search_capacity"].min())
                    if not qualifying.empty else None,
                })
    robust_threshold_path = base.OUT / "random_order_thresholds.csv"
    pd.DataFrame(robust_threshold_rows).to_csv(robust_threshold_path, index=False)
    pd.DataFrame(leave_one_out_rows).to_csv(loo_path, index=False)
    pd.DataFrame(leave_one_risk_rows).to_csv(leave_one_risk_path, index=False)
    external = analyse_external_responses(bootstrap_replicates)
    external.to_csv(external_path, index=False)
    task_times = analyse_pilot_task_times(bootstrap_replicates)
    task_times.to_csv(time_path, index=False)

    loo_frame = pd.DataFrame(leave_one_out_rows)
    loo_summary = []
    for (condition, requirement), subset in loo_frame.groupby(
            ["condition", "referral_and_capacity_target"], sort=True):
        for policy, column in (("stopping", "stop_capacity"),
                               ("risk-constrained fixed", "fixed_capacity")):
            values = subset[column].dropna().astype(int)
            loo_summary.append({
                "condition": condition,
                "requirement": float(requirement),
                "policy": policy,
                "finite_image_deletions": int(len(values)),
                "minimum_capacity": int(values.min()) if len(values) else None,
                "median_capacity": float(values.median()) if len(values) else None,
                "maximum_capacity": int(values.max()) if len(values) else None,
            })
    loo_risk_frame = pd.DataFrame(leave_one_risk_rows)
    for (condition, requirement, rule), subset in loo_risk_frame.groupby(
            ["condition", "referral_and_capacity_requirement",
             "risk_constrained_policy"], sort=True):
        values = subset["minimum_capacity"].dropna().astype(int)
        loo_summary.append({
            "condition": condition,
            "requirement": float(requirement),
            "policy": f"{rule}, with the mean-miss limit",
            "finite_image_deletions": int(len(values)),
            "minimum_capacity": int(values.min()) if len(values) else None,
            "median_capacity": float(values.median()) if len(values) else None,
            "maximum_capacity": int(values.max()) if len(values) else None,
        })

    summary = {
        "status": "completed",
        "source_verification": str((base.OUT / "verification.json").name),
        "participants_resampled": bootstrap_replicates,
        "pilot_leave_one_image_out_cases": int(len(loo_frame)),
        "mean_target_miss_design_limit": RISK_LIMIT,
        "joint_resource_thresholds": json.loads(threshold_path.read_text(
            encoding="utf-8")) if threshold_path.suffix == ".json" else
            pd.DataFrame(threshold_rows).to_dict(orient="records"),
        "leave_one_image_out_threshold_ranges": loo_summary,
        "additional_public_sample": {
            "participants": 232,
            "behavioral_outcomes": ["task correctness", "truth-label request"],
            "not_used_for": ["secondary-referral response", "physical search detection",
                             "service completion"],
            "bootstrap_unit": "Whole participants, stratified by response and visualization",
        },
        "pilot_task_time": {
            "complete_trials": 1080,
            "condition_groups": 4,
            "bootstrap_replicates": bootstrap_replicates,
            "scope": "Elapsed time for each online image task",
        },
        "random_order_sensitivity": pd.DataFrame(robust_threshold_rows).to_dict(
            orient="records"),
        "target_omission_sensitivity_at_minimum_capacity": omission_frame[
            (omission_frame["batch_level_target_omission_allowance"].isin(
                [0.0, 0.05, 0.10, 0.20])) &
            (omission_frame["stopping_jointly_feasible"] |
             omission_frame["same_reading_allowance_fixed_jointly_feasible"])
        ].to_dict(orient="records"),
        "deliverables": [detail_path.name, threshold_path.name, loo_path.name,
                         leave_one_risk_path.name, external_path.name,
                         time_path.name, omission_path.name, order_path.name,
                         robust_threshold_path.name],
        "interpretation": "The ranges show how the capacity requirements vary when each image is removed in turn.",
    }
    report_path = base.OUT / "revision_validation.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--full", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    cases = base.prepare_cases()
    if args.smoke:
        report = smoke_check(cases)
        base.OUT.mkdir(parents=True, exist_ok=True)
        (base.OUT / "revision_validation_smoke.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    smoke_path = base.OUT / "revision_validation_smoke.json"
    if not smoke_path.exists():
        raise FileNotFoundError("Run and inspect --smoke before running --full")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if not smoke.get("passed"):
        raise ArithmeticError("The saved revision-validation smoke check failed")
    if args.bootstrap < 1:
        raise ValueError("At least one participant bootstrap replicate is required")
    result = run_full(cases, args.bootstrap)
    print(json.dumps({
        "status": result["status"],
        "joint_resource_thresholds": result["joint_resource_thresholds"],
        "leave_one_image_out_threshold_ranges": result[
            "leave_one_image_out_threshold_ranges"],
        "deliverables": result["deliverables"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
