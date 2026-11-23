from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from r1_followup_analysis import (
    EXTERNAL_FILE,
    FIELD_FILE,
    OUT,
    STIMULUS_FILE,
    TARGET_LABELS,
    choose,
    consecutive_service_times,
    finish_probability,
    object_set,
    service_records,
)
from r1_image_unit_analysis import PRIMARY_FILE, prepare_cases, scenario_matrices
from r1_revision_validation import capacity_at_threshold, joint_risk_thresholds
from r1_followup_analysis import audit_trial, image_difficulty, quota_analysis


SEED = 20261121
BOOTSTRAP_REPLICATES = 2000
QUOTA_OUTSIDE_FRACTIONS = (0.0, 0.1, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.4, 1.0)
OUTPUT = OUT / "external_sensitivity.csv"
VERIFICATION = OUT / "external_sensitivity.json"
MATCHED_TARGET_OUTPUT = OUT / "external_matched_target_sensitivity.csv"
MATCHED_TARGET_DELETION_OUTPUT = OUT / "external_matched_target_deletion.csv"
IMAGE_GENERALISATION_OUTPUT = OUT / "external_image_generalisation.csv"
CONSECUTIVE_SERVICE_OUTPUT = OUT / "consecutive_service_session_bootstrap.csv"
LINKED_COMPLETION_OUTPUT = OUT / "linked_completion_session_bootstrap.csv"


def quota_composition_value(positive_counts, search_slots, target_compositions, missed_limit):
    quotas = [
        quota
        for quota in product(*(range(min(positive_count, search_slots) + 1)
                               for positive_count in positive_counts))
        if sum(quota) <= search_slots
    ]
    coverage = []
    recovery = []
    for targets in target_compositions:
        if sum(targets) == 0 or any(targets[group] > positive_counts[group]
                                    for group in range(len(positive_counts))):
            raise ValueError("Each target composition must fit the group referral counts")
        coverage.append([
            math.prod(choose(quota[group], targets[group]) / choose(positive_counts[group], targets[group])
                      for group in range(len(positive_counts)))
            for quota in quotas
        ])
        recovery.append([
            sum(targets[group] * quota[group] / positive_counts[group]
                for group in range(len(positive_counts))) / sum(targets)
            for quota in quotas
        ])

    coverage = np.asarray(coverage, dtype=float)
    recovery = np.asarray(recovery, dtype=float)
    objective = np.r_[np.zeros(len(quotas)), -1.0]
    coverage_constraints = [np.r_[-coverage[row], 1.0] for row in range(len(target_compositions))]
    recovery_constraints = [np.r_[-recovery[row], 0.0] for row in range(len(target_compositions))]
    result = linprog(
        objective,
        A_ub=np.asarray([*coverage_constraints, *recovery_constraints]),
        b_ub=np.r_[np.zeros(len(target_compositions)),
                   np.full(len(target_compositions), -(1.0 - missed_limit))],
        A_eq=[np.r_[np.ones(len(quotas)), 0.0]],
        b_eq=[1.0],
        bounds=[(0.0, 1.0)] * len(quotas) + [(0.0, 1.0)],
        method="highs",
    )
    if not result.success:
        return {"feasible": False, "guarantee": None, "quotas": quotas, "weights": []}
    weights = result.x[:-1]
    return {
        "feasible": True,
        "guarantee": float(result.x[-1]),
        "quotas": quotas,
        "weights": weights.tolist(),
        "minimum_mean_recovery": float(np.min(recovery @ weights)),
    }


def quota_recovery_ceiling(positive_counts, search_slots, target_compositions):
    quotas = [
        quota
        for quota in product(*(range(min(positive_count, search_slots) + 1)
                               for positive_count in positive_counts))
        if sum(quota) <= search_slots
    ]
    recovery = np.asarray([
        [sum(targets[group] * quota[group] / positive_counts[group]
             for group in range(len(positive_counts))) / sum(targets)
         for quota in quotas]
        for targets in target_compositions
    ], dtype=float)
    objective = np.r_[np.zeros(len(quotas)), -1.0]
    recovery_constraints = [np.r_[-recovery[row], 1.0]
                            for row in range(len(target_compositions))]
    result = linprog(
        objective,
        A_ub=np.asarray(recovery_constraints),
        b_ub=np.zeros(len(target_compositions)),
        A_eq=[np.r_[np.ones(len(quotas)), 0.0]],
        b_eq=[1.0],
        bounds=[(0.0, 1.0)] * len(quotas) + [(0.0, 1.0)],
        method="highs",
    )
    if not result.success:
        raise ArithmeticError("The target-recovery ceiling could not be calculated")
    return float(result.x[-1])


def quota_capacity_frontier():
    target_cases = {
        "composition_11": [(1, 1)],
        "all_compositions": [(2, 0), (1, 1), (0, 2)],
    }
    rows = []
    for search_slots in range(2, 6):
        for case, target_compositions in target_cases.items():
            maximum_recovery = quota_recovery_ceiling(
                (2, 3), search_slots, target_compositions
            )
            rows.append({
                "analysis": "quota recovery frontier across search capacities",
                "case": case,
                "positive_referrals_by_group": "2;3",
                "target_compositions": ";".join(
                    "/".join(map(str, item)) for item in target_compositions
                ),
                "search_slots": search_slots,
                "maximum_mean_target_recovery": maximum_recovery,
                "minimum_feasible_mean_miss": 1.0 - maximum_recovery,
                "feasible_at_empirical_0_001_limit": maximum_recovery >= 0.999 - 1e-12,
            })
    return rows


def quota_composition_analysis():
    positive_counts = (2, 3)
    target_cases = {
        "composition_11": [(1, 1)],
        "all_compositions": [(2, 0), (1, 1), (0, 2)],
    }
    specifications = (
        ("composition_11", 0.60),
        ("all_compositions", 0.60),
    )
    rows = []
    for composition_name, missed_limit in specifications:
        target_compositions = target_cases[composition_name]
        outcome = quota_composition_value(positive_counts, 2, target_compositions, missed_limit)
        if not outcome["feasible"]:
            raise ArithmeticError("The selected target-recovery requirement is infeasible")
        recovery_ceiling = quota_recovery_ceiling(positive_counts, 2, target_compositions)
        rows.append({
            "analysis": "quota allocation by target composition",
            "case": composition_name,
            "positive_referrals_by_group": "2;3",
            "target_compositions": ";".join("/".join(map(str, item)) for item in target_compositions),
            "search_slots": 2,
            "mean_unreferred_target_limit": missed_limit,
            "complete_referral_guarantee": outcome["guarantee"],
            "mean_target_recovery_lower": outcome["minimum_mean_recovery"],
            "maximum_mean_target_recovery": recovery_ceiling,
            "minimum_feasible_mean_unreferred_limit": 1.0 - recovery_ceiling,
            "feasible_at_empirical_0_001_limit": 1.0 - recovery_ceiling <= 0.001,
        })
    return rows


def target_category_sensitivity(smoke_only=False):
    pilot = pd.read_excel(PRIMARY_FILE, keep_default_na=False)
    pilot["image"] = pilot["ImageNumber"].str.casefold().str.replace(r"\.jpg$", "", regex=True)
    pilot_labels = pilot.groupby("image")["ground truth"].agg(
        lambda values: set(map(object_set, values)).pop()
        if len(set(map(object_set, values))) == 1 else None
    )
    if pilot_labels.isna().any():
        raise ValueError("Pilot images contain inconsistent target labels")
    pilot_categories = pilot.groupby("image")["difficulty level"].agg(
        lambda values: set(values).pop() if len(set(values)) == 1 else None
    )
    if pilot_categories.isna().any():
        raise ValueError("Pilot images contain inconsistent difficulty categories")
    rows = []
    cases = list(prepare_cases().items())
    if smoke_only:
        cases = cases[:1]
    for condition, original in cases:
        annotations = [pilot_labels[image] for image in original["image_ids"]]
        difficulties = [str(pilot_categories[image]) for image in original["image_ids"]]
        primary_labels = np.asarray([bool(objects & TARGET_LABELS) for objects in annotations])
        if not np.array_equal(primary_labels, original["targets"]):
            raise ValueError("Image labels differ between the pilot and stimulus files")
        broader_categories = set().union(*annotations)
        for name, categories in (("gun_or_knife", TARGET_LABELS),
                                 ("all_labeled_object_classes", broader_categories)):
            targets = np.asarray([bool(objects & categories) for objects in annotations])
            sensitivity_case = {**original, "condition": condition, "targets": targets,
                                "target_count": int(targets.sum())}
            policy = scenario_matrices(sensitivity_case)
            thresholds = joint_risk_thresholds(sensitivity_case, policy, include_target_omissions=False)
            stop_threshold = capacity_at_threshold(thresholds, "stopping", 0.8)
            fixed_threshold = capacity_at_threshold(thresholds, "fixed", 0.8)
            for capacity in thresholds:
                rows.append({
                    "target_definition": name,
                    "categories": ";".join(sorted(categories)),
                    "distinct_target_images": int(targets.sum()),
                    "target_images_by_difficulty": json.dumps({
                        level: int(sum(bool(objects & categories)
                                       for objects, difficulty in zip(
                                           annotations, difficulties, strict=True)
                                       if difficulty == level))
                        for level in sorted(set(difficulties))
                    }, sort_keys=True),
                    "minimum_capacity_stopping_80_percent_and_0_1_percent": stop_threshold,
                    "minimum_capacity_fixed_80_percent_and_0_1_percent": fixed_threshold,
                    **{key: value for key, value in capacity.items()
                       if key not in {"target_omission_sensitivity", "order_sensitivity"}},
                })
    return rows


def quota_composition_robustness(outside_fractions=QUOTA_OUTSIDE_FRACTIONS, *, protect_single_target=False):
    quotas = [quota for quota in product(range(3), repeat=2)
              if sum(quota) <= 2]
    compositions = ((1, 1), (2, 0), (0, 2))
    payoffs = np.asarray([
        [math.prod(choose(quota[group], target_count)
                   / choose(positive_counts, target_count)
                   for group, (positive_counts, target_count) in enumerate(
                       zip((2, 3), target_composition, strict=True)))
         for quota in quotas]
        for target_composition in compositions
    ])
    recoveries = np.asarray([
        [sum(target_composition[group] * quota[group] / (2, 3)[group]
             for group in range(2)) / sum(target_composition)
         for quota in quotas]
        for target_composition in compositions
    ])
    rows = []
    for outside_fraction in outside_fractions:
        coverage_vertices = np.asarray([
            payoffs[0],
            (1 - outside_fraction) * payoffs[0]
            + outside_fraction * payoffs[1],
            (1 - outside_fraction) * payoffs[0]
            + outside_fraction * payoffs[2],
        ])
        recovery_vertices = np.asarray([
            recoveries[0],
            (1 - outside_fraction) * recoveries[0]
            + outside_fraction * recoveries[1],
            (1 - outside_fraction) * recoveries[0]
            + outside_fraction * recoveries[2],
        ])
        if protect_single_target:
            single_target_recoveries = np.asarray(quotas, dtype=float).T / np.asarray([[2], [3]])
            recovery_vertices = np.vstack((recovery_vertices, single_target_recoveries))
        objective = np.r_[np.zeros(len(quotas)), -1.0]
        result = linprog(
            objective,
            A_ub=np.vstack([
                np.c_[-coverage_vertices, np.ones(len(coverage_vertices))],
                np.c_[-recovery_vertices, np.zeros(len(recovery_vertices))],
            ]),
            b_ub=np.r_[np.zeros(len(coverage_vertices)),
                       np.full(len(recovery_vertices), -0.40)],
            A_eq=[np.r_[np.ones(len(quotas)), 0.0]],
            b_eq=[1.0],
            bounds=[(0.0, 1.0)] * len(quotas) + [(0.0, 1.0)],
            method="highs",
        )
        if not result.success:
            raise ArithmeticError("The composition-contamination game did not solve")
        rows.append({
            "mean_miss_limit": 0.60,
            "composition_outside_one_target_per_group": outside_fraction,
            "protect_single_target_in_either_group": protect_single_target,
            "worst_case_quota_referral_probability": float(result.x[-1]),
            "worst_case_quota_mean_recovery": float(np.min(recovery_vertices @ result.x[:-1])),
            "pooled_uniform_referral_probability": 0.10,
            "quota_strictly_improves_on_pooled": bool(result.x[-1] > 0.10 + 1e-10),
            "probability_of_one_search_per_group": float(result.x[quotas.index((1, 1))]),
        })
    return rows


def target_category_summary(rows):
    summary = {}
    for row in rows:
        identity = (row["condition"], row["target_definition"])
        summary.setdefault(identity, {key: value for key, value in row.items()
                                      if key != "search_capacity" and
                                      not key.endswith("_lower") and
                                      not key.endswith("_upper") and
                                      key not in {"stopping_jointly_feasible",
                                                  "risk_constrained_fixed_feasible",
                                                  "same_reading_allowance_fixed_jointly_feasible"}})
    return list(summary.values())


def panel_participant_bootstrap(replicates=BOOTSTRAP_REPLICATES, seed=SEED):
    responses = pd.read_csv(EXTERNAL_FILE, keep_default_na=False)
    images = pd.read_csv(STIMULUS_FILE, keep_default_na=False)
    images["image"] = images["filename"].str.casefold().str.removesuffix(".jpg")
    target_labels = images.set_index("image")["ground truth"].map(
        lambda value: bool(object_set(value) & TARGET_LABELS)
    )
    responses["image"] = responses["imageID"].str.casefold().str.removesuffix(".jpg")
    panel = responses.loc[responses["beforeOrAfterTV"].eq("afterTV")].copy()
    panel["target"] = panel["image"].map(target_labels)
    if panel["target"].isna().any() or not panel["userCorrect"].isin([0, 1]).all():
        raise ValueError("The external participant records need complete target labels and binary outcomes")

    generator = np.random.default_rng(seed)
    rows = []
    for condition, group in panel.groupby("type_of_tv", sort=True):
        participant_scores = []
        target_records = group.loc[group["target"]].groupby("userID")["userCorrect"]
        for _participant, judgments in target_records:
            total = len(judgments)
            correct = int(judgments.sum())
            if total < 3:
                raise ValueError("Each participant must have at least three target judgments")
            participant_scores.append(choose(correct, 3) / choose(total, 3))
        participant_scores = np.asarray(participant_scores, dtype=float)
        draws = generator.integers(0, len(participant_scores), size=(replicates, len(participant_scores)))
        means = participant_scores[draws].mean(axis=1)
        rows.append({
            "analysis": "external participant-profile bootstrap",
            "case": condition,
            "participants": len(participant_scores),
            "bootstrap_replicates": replicates,
            "standardized_target_count": 3,
            "all_target_accuracy": float(participant_scores.mean()),
            "bootstrap_2_5": float(np.quantile(means, 0.025)),
            "bootstrap_median": float(np.quantile(means, 0.5)),
            "bootstrap_97_5": float(np.quantile(means, 0.975)),
        })
    if sum(row["participants"] for row in rows) != 232:
        raise ValueError("The public response file should contain 232 unique participants")
    return rows


def panel_target_image_deletion():
    responses = pd.read_csv(EXTERNAL_FILE, keep_default_na=False)
    images = pd.read_csv(STIMULUS_FILE, keep_default_na=False)
    images["image"] = images["filename"].str.casefold().str.removesuffix(".jpg")
    target_labels = images.set_index("image")["ground truth"].map(
        lambda value: bool(object_set(value) & TARGET_LABELS)
    )
    responses["image"] = responses["imageID"].str.casefold().str.removesuffix(".jpg")
    panel = responses.loc[responses["beforeOrAfterTV"].eq("afterTV")].copy()
    panel["target"] = panel["image"].map(target_labels)
    if panel["target"].isna().any() or not panel["userCorrect"].isin([0, 1]).all():
        raise ValueError("The public response records need complete target labels and binary outcomes")

    rows = []
    for condition, group in panel.groupby("type_of_tv", sort=True):
        target_images = sorted(group.loc[group["target"], "image"].unique())
        if len(target_images) != 10:
            raise ValueError("Each response condition must contain ten distinct target images")
        for removed_image in target_images:
            scores = []
            for _participant, judgments in group.loc[
                group["target"] & group["image"].ne(removed_image)
            ].groupby("userID"):
                total = len(judgments)
                if total >= 3:
                    correct = int(judgments["userCorrect"].sum())
                    scores.append(choose(correct, 3) / choose(total, 3))
            if not scores:
                raise ValueError("Image deletion left too few target judgments for standardization")
            rows.append({
                "analysis": "external target-image deletion sensitivity",
                "case": condition,
                "target_images": len(target_images),
                "deleted_image": removed_image,
                "participants": len(scores),
                "standardized_target_count": 3,
                "all_correct_fraction": float(np.mean(scores)),
            })
    if len(rows) != 20:
        raise ValueError("Image deletion should produce ten records per error condition")
    return rows


def panel_matched_target_sensitivity(replicates=BOOTSTRAP_REPLICATES, seed=SEED):
    responses = pd.read_csv(EXTERNAL_FILE, keep_default_na=False)
    images = pd.read_csv(STIMULUS_FILE, keep_default_na=False)
    images["image"] = images["filename"].str.casefold().str.removesuffix(".jpg")
    target_labels = images.set_index("image")["ground truth"].map(
        lambda value: bool(object_set(value) & TARGET_LABELS)
    )
    responses["image"] = responses["imageID"].str.casefold().str.removesuffix(".jpg")
    panel = responses.loc[responses["beforeOrAfterTV"].eq("afterTV")].copy()
    panel["target"] = panel["image"].map(target_labels)
    if panel["target"].isna().any() or not panel["userCorrect"].isin([0, 1]).all():
        raise ValueError("The public response records need complete target labels and binary outcomes")

    cases = prepare_cases()
    case_mapping = {"falselyAlarm": "false_alarm", "falselyMiss": "false_miss"}
    generator = np.random.default_rng(seed)
    summaries = []
    deletion_rows = []
    for condition, case in case_mapping.items():
        group = panel.loc[panel["type_of_tv"].eq(condition) & panel["target"]]
        target_images = sorted(set(cases[case]["image_ids"]) & set(group["image"]))
        if not target_images:
            raise ValueError(f"The response panel has no target images matching {case}")
        matched = group.loc[group["image"].isin(target_images)]
        if matched.duplicated(["userID", "image"]).any():
            raise ValueError(f"Repeated judgments on a shared image within a participant profile in {case}")

        def standardized_scores(records):
            counts = records.groupby("userID")["userCorrect"].agg(["sum", "size"])
            counts = counts.loc[counts["size"].ge(3)]
            return pd.Series(
                [choose(int(correct), 3) / choose(total, 3)
                 for correct, total in counts.itertuples(index=False, name=None)],
                index=counts.index, dtype=float,
            )

        participant_scores = standardized_scores(matched)
        if not participant_scores.size:
            raise ValueError(f"Fewer than three matched target images are available in {case}")
        draws = generator.integers(
            0, len(participant_scores), size=(replicates, len(participant_scores)),
        )
        bootstrap_means = participant_scores.to_numpy()[draws].mean(axis=1)

        for deleted_image in target_images:
            reduced = matched.loc[matched["image"].ne(deleted_image)]
            deletion_scores = standardized_scores(reduced)
            if not deletion_scores.size:
                raise ValueError("Image deletion left too few matched target judgments")
            exposed = matched.loc[matched["image"].eq(deleted_image), "userID"]
            paired_scores = deletion_scores.loc[deletion_scores.index.intersection(exposed)]
            if paired_scores.empty:
                raise ValueError(f"Image deletion left no paired participants for {deleted_image}")
            paired_baseline = participant_scores.loc[paired_scores.index]
            deletion_rows.append({
                "analysis": "shared-target image deletion sensitivity",
                "case": condition,
                "matched_target_images": len(target_images),
                "deleted_image": deleted_image,
                "participants": len(deletion_scores),
                "standardized_target_count": 3,
                "all_correct_fraction": float(deletion_scores.mean()),
                "paired_participants": len(paired_scores),
                "paired_baseline_all_correct_fraction": float(paired_baseline.mean()),
                "paired_deletion_all_correct_fraction": float(paired_scores.mean()),
                "paired_change_percentage_points": float(100 * (paired_scores - paired_baseline).mean()),
            })

        condition_deletions = [row for row in deletion_rows if row["case"] == condition]
        deletion_values = [row["all_correct_fraction"] for row in condition_deletions]
        paired_changes = [row["paired_change_percentage_points"] for row in condition_deletions]
        summaries.append({
            "analysis": "participant sensitivity on shared target images",
            "case": condition,
            "primary_schedule_target_images": int(cases[case]["target_count"]),
            "matched_target_images": len(target_images),
            "target_image_ids": ";".join(target_images),
            "matched_target_judgments": len(matched),
            "participants": len(participant_scores),
            "standardized_target_count": 3,
            "all_correct_fraction": float(participant_scores.mean()),
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
            "bootstrap_2_5": float(np.quantile(bootstrap_means, 0.025)),
            "bootstrap_97_5": float(np.quantile(bootstrap_means, 0.975)),
            "image_deletion_min": float(min(deletion_values)),
            "image_deletion_max": float(max(deletion_values)),
            "paired_deletion_change_min_percentage_points": min(paired_changes),
            "paired_deletion_change_max_percentage_points": max(paired_changes),
        })
    if len(summaries) != 2:
        raise ValueError("Shared-target analysis should include both error conditions")
    return summaries, deletion_rows


def standardized_judgment_scores(judgments, target_count):
    scores = {}
    for participant, profile in judgments.groupby("userID"):
        total = len(profile)
        if total >= target_count:
            scores[participant] = (choose(int(profile["userCorrect"].sum()), target_count)
                                   / choose(total, target_count))
    return pd.Series(scores, dtype=float)


def panel_shared_other_target_sensitivity(replicates=20000, seed=SEED + 1):
    responses = pd.read_csv(EXTERNAL_FILE, keep_default_na=False)
    images = pd.read_csv(STIMULUS_FILE, keep_default_na=False)
    images["image"] = images["filename"].str.casefold().str.removesuffix(".jpg")
    target_labels = images.set_index("image")["ground truth"].map(
        lambda value: bool(object_set(value) & TARGET_LABELS)
    )
    responses["image"] = responses["imageID"].str.casefold().str.removesuffix(".jpg")
    panel = responses.loc[responses["beforeOrAfterTV"].eq("afterTV")].copy()
    panel["target"] = panel["image"].map(target_labels)
    if panel["target"].isna().any() or not panel["userCorrect"].isin([0, 1]).all():
        raise ValueError("The public response records need complete target labels and binary outcomes")

    case_mapping = {"falselyMiss": "false_miss", "falselyAlarm": "false_alarm"}
    cases = prepare_cases()
    rows = []
    for condition, case in case_mapping.items():
        group = panel.loc[panel["type_of_tv"].eq(condition) & panel["target"]].copy()
        shared_images = sorted(set(cases[case]["image_ids"]) & set(group["image"]))
        other_images = sorted(set(group["image"]) - set(shared_images))
        if group.duplicated(["userID", "image"]).any():
            raise ValueError(f"Repeated target judgments occur within a participant record for {case}")

        image_sets = {label: group.loc[group["image"].isin(selected_images)]
                      for label, selected_images in (("shared", shared_images), ("other", other_images))}
        participant_scores = {label: standardized_judgment_scores(judgments, 3)
                              for label, judgments in image_sets.items()}
        paired = pd.DataFrame(participant_scores).dropna()
        if paired.empty:
            raise ValueError(f"No participant judged at least three shared and other targets in {case}")
        difference = paired["other"] - paired["shared"]
        condition_seed = seed + (0 if case == "false_miss" else 1)
        generator = np.random.default_rng(condition_seed)
        indices = generator.integers(0, len(paired), size=(replicates, len(paired)))
        bootstrap = difference.to_numpy()[indices].mean(axis=1) * 100
        deletion_estimates, deletion_changes, deletion_counts = [], [], []
        two_image_scores = {label: standardized_judgment_scores(judgments, 2)
                            for label, judgments in image_sets.items()}
        pair_scores = {label: scores.loc[paired.index] for label, scores in two_image_scores.items()}
        larger_panel = pd.DataFrame(two_image_scores).dropna()
        larger_difference = (larger_panel["other"] - larger_panel["shared"]).to_numpy() * 100
        pair_difference = float((pair_scores["other"] - pair_scores["shared"]).mean() * 100)
        pair_generator = np.random.default_rng(condition_seed + 2)
        pair_indices = pair_generator.integers(0, len(paired), size=(replicates, len(paired)))
        pair_bootstrap = (pair_scores["other"] - pair_scores["shared"]).to_numpy()[
            pair_indices
        ].mean(axis=1) * 100
        larger_seed = condition_seed + 4
        larger_indices = np.random.default_rng(larger_seed).integers(
            0, len(larger_panel), size=(replicates, len(larger_panel)),
        )
        larger_bootstrap = larger_difference[larger_indices].mean(axis=1)
        pair_deletion_estimates = []
        for label, judgments in image_sets.items():
            for image in sorted(judgments["image"].unique()):
                retained = judgments.loc[judgments["image"].ne(image)]
                reduced = standardized_judgment_scores(retained, 3)
                paired_reduced = pd.DataFrame({**participant_scores, label: reduced}).dropna()
                after = paired_reduced["other"] - paired_reduced["shared"]
                deletion_estimates.append(float(after.mean() * 100))
                deletion_changes.append(float((after - difference.loc[after.index]).mean() * 100))
                deletion_counts.append(len(after))
                pair_after = {**pair_scores, label: standardized_judgment_scores(retained, 2).loc[paired.index]}
                pair_deletion_estimates.append(float((pair_after["other"] - pair_after["shared"]).mean() * 100))
        complete_people = set.intersection(*(
            set(judgments.groupby("userID").size().loc[lambda counts: counts >= 4].index)
            for judgments in image_sets.values()
        ))
        complete_profiles = {
            label: standardized_judgment_scores(judgments.loc[
                judgments["userID"].isin(complete_people)
            ], 3).reindex(sorted(complete_people))
            for label, judgments in image_sets.items()
        }
        complete_difference = complete_profiles["other"] - complete_profiles["shared"]
        complete_deletion_estimates = []
        complete_deletion_changes = []
        for label, judgments in image_sets.items():
            for image in sorted(judgments["image"].unique()):
                remaining = standardized_judgment_scores(judgments.loc[
                    judgments["userID"].isin(complete_people)
                    & judgments["image"].ne(image)
                ], 3).reindex(complete_difference.index)
                profiles_after = {**complete_profiles, label: remaining}
                after = profiles_after["other"] - profiles_after["shared"]
                complete_deletion_estimates.append(float(after.mean() * 100))
                complete_deletion_changes.append(float(
                    (after - complete_difference).mean() * 100
                ))
        rows.append({
            "analysis": "paired participant sensitivity across shared and other target images",
            "condition": condition,
            "matched_target_images": len(shared_images),
            "other_target_images": len(other_images),
            "participants_with_three_or_more_in_both_sets": len(paired),
            "standardized_target_count": 3,
            "shared_all_correct_fraction": float(paired["shared"].mean()),
            "other_all_correct_fraction": float(paired["other"].mean()),
            "other_minus_shared_percentage_points": float(difference.mean() * 100),
            "paired_bootstrap_replicates": replicates,
            "paired_bootstrap_seed": condition_seed,
            "paired_bootstrap_2_5": float(np.quantile(bootstrap, 0.025)),
            "paired_bootstrap_97_5": float(np.quantile(bootstrap, 0.975)),
            "target_image_deletions": len(deletion_estimates),
            "participants_after_deletion_min": min(deletion_counts),
            "participants_after_deletion_max": max(deletion_counts),
            "other_minus_shared_after_deletion_min_pp": min(deletion_estimates),
            "other_minus_shared_after_deletion_max_pp": max(deletion_estimates),
            "within_participant_deletion_change_min_pp": min(deletion_changes),
            "within_participant_deletion_change_max_pp": max(deletion_changes),
            "two_judgment_other_minus_shared_pp": pair_difference,
            "two_judgment_bootstrap_replicates": replicates,
            "two_judgment_bootstrap_2_5": float(np.quantile(pair_bootstrap, 0.025)),
            "two_judgment_bootstrap_97_5": float(np.quantile(pair_bootstrap, 0.975)),
            "two_judgment_deletion_min_pp": min(pair_deletion_estimates),
            "two_judgment_deletion_max_pp": max(pair_deletion_estimates),
            "two_judgment_deletions_with_negative_difference": sum(
                estimate < 0 for estimate in pair_deletion_estimates
            ),
            "two_judgment_deletions_with_positive_difference": sum(
                estimate > 0 for estimate in pair_deletion_estimates
            ),
            "participants_with_two_or_more_in_both_sets": len(larger_panel),
            "larger_panel_two_judgment_difference_pp": float(larger_difference.mean()),
            "larger_panel_two_judgment_bootstrap_seed": larger_seed,
            "larger_panel_two_judgment_bootstrap_2_5": float(np.quantile(larger_bootstrap, 0.025)),
            "larger_panel_two_judgment_bootstrap_97_5": float(np.quantile(larger_bootstrap, 0.975)),
            "complete_image_deletion_panel_participants": len(complete_people),
            "complete_image_deletion_panel_other_minus_shared_pp":
                float(complete_difference.mean() * 100),
            "image_deletion_effect_min_percentage_points": min(complete_deletion_estimates),
            "image_deletion_effect_max_percentage_points": max(complete_deletion_estimates),
            "complete_panel_deletion_change_min_pp": min(complete_deletion_changes),
            "complete_panel_deletion_change_max_pp": max(complete_deletion_changes),
        })
    return rows


def service_session_bootstrap(replicates=BOOTSTRAP_REPLICATES, seed=SEED):
    counts, sessions, _linked_times = service_records()
    durations = [duration for session in sessions for duration in session]
    generator = np.random.default_rng(seed + 1)
    specifications = ((3, 240), (3, 300), (5, 240), (5, 300), (5, 420))
    bootstrap_values = {specification: [] for specification in specifications}
    for _ in range(replicates):
        selected = generator.integers(0, len(sessions), size=len(sessions))
        sample = [duration for index in selected for duration in sessions[index]]
        for checks, available_seconds in specifications:
            station_loads = [(checks + station) // 2 for station in range(2)]
            probability = math.prod(
                finish_probability(sample, jobs, available_seconds) for jobs in station_loads
            )
            bootstrap_values[checks, available_seconds].append(probability)

    rows = []
    for checks, available_seconds in specifications:
        station_loads = [(checks + station) // 2 for station in range(2)]
        point = math.prod(finish_probability(durations, jobs, available_seconds) for jobs in station_loads)
        samples = np.asarray(bootstrap_values[checks, available_seconds], dtype=float)
        rows.append({
            "analysis": "field-session cluster bootstrap for service completion",
            "case": f"{checks}_checks_{available_seconds}_seconds",
            "lane_sessions": len(sessions),
            "observed_checks": len(durations),
            "bootstrap_replicates": replicates,
            "independent_duration_model_estimate": point,
            "bootstrap_2_5": float(np.quantile(samples, 0.025)),
            "bootstrap_median": float(np.quantile(samples, 0.5)),
            "bootstrap_97_5": float(np.quantile(samples, 0.975)),
        })
    if counts["usable_intervals"] != len(durations):
        raise ValueError("Field timing counts differ from the session records")
    return rows


def consecutive_service_session_bootstrap(replicates=BOOTSTRAP_REPLICATES, seed=SEED):
    _counts, sessions, _linked_times = service_records()
    specifications = ((3, 240), (3, 300), (5, 240), (5, 300), (5, 420))
    observed = {}
    for checks, available_seconds in specifications:
        session_times = [
            consecutive_service_times([session], checks, 2)
            for session in sessions
        ]
        flattened = [duration for session in session_times for duration in session]
        successes = sum(duration <= available_seconds for duration in flattened)
        observed[checks, available_seconds] = (session_times, flattened, successes)

    generator = np.random.default_rng(seed + 2)
    bootstrap = {specification: [] for specification in specifications}
    for _ in range(replicates):
        selected = generator.integers(0, len(sessions), size=len(sessions))
        for checks, available_seconds in specifications:
            session_times = observed[checks, available_seconds][0]
            sample = [duration for index in selected for duration in session_times[index]]
            bootstrap[checks, available_seconds].append(
                sum(duration <= available_seconds for duration in sample) / len(sample)
            )

    rows = []
    for checks, available_seconds in specifications:
        session_times, flattened, successes = observed[checks, available_seconds]
        samples = np.asarray(bootstrap[checks, available_seconds], dtype=float)
        window_counts = [len(times) for times in session_times if times]
        success_counts = [sum(duration <= available_seconds for duration in times)
                          for times in session_times if times]
        deletions = [(successes - session_successes) / (len(flattened) - windows)
                     for session_successes, windows in zip(success_counts, window_counts, strict=True)]
        rows.append({
            "analysis": "session-cluster bootstrap of consecutive baggage-check windows",
            "checks": checks,
            "available_seconds": available_seconds,
            "stations": 2,
            "lane_sessions": len(sessions),
            "sessions_with_complete_windows": len(window_counts),
            "overlapping_windows": len(flattened),
            "successful_windows": successes,
            "observed_completion_fraction": successes / len(flattened),
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed + 2,
            "bootstrap_2_5": float(np.quantile(samples, 0.025)),
            "bootstrap_97_5": float(np.quantile(samples, 0.975)),
            "equal_session_weight_completion_fraction": float(np.mean(
                np.asarray(success_counts) / np.asarray(window_counts))),
            "leave_one_session_out_min": min(deletions),
            "leave_one_session_out_max": max(deletions),
        })
    return rows


def linked_completion_sensitivity(replicates=BOOTSTRAP_REPLICATES, seed=SEED + 3):
    _counts, _sessions, linked_sessions = service_records(group_linked_times=True)
    deadline = 180
    four_minute_deadline = 240
    session_counts = []
    for session in linked_sessions:
        search_completions = sum(search <= deadline for _, search, _ in session)
        elapsed_completions = sum(elapsed <= deadline for _, _, elapsed in session)
        four_minute_search = sum(search <= four_minute_deadline for _, search, _ in session)
        four_minute_elapsed = sum(elapsed <= four_minute_deadline for _, _, elapsed in session)
        session_counts.append((len(session), search_completions, elapsed_completions,
                               four_minute_search, four_minute_elapsed))
    session_counts = np.asarray(session_counts)
    totals = session_counts.sum(axis=0)
    checks, search_completions, elapsed_completions, four_minute_search, four_minute_elapsed = totals
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(session_counts), size=(replicates, len(session_counts)))
    sampled_totals = session_counts[indices].sum(axis=1)
    differences = (sampled_totals[:, 2] - sampled_totals[:, 1]) / sampled_totals[:, 0]
    lower, upper = np.quantile(differences * 100, (0.025, 0.975))
    session_differences = (session_counts[:, 2] - session_counts[:, 1]) / session_counts[:, 0]
    equal_session_lower, equal_session_upper = np.quantile(
        session_differences[indices].mean(axis=1) * 100, (0.025, 0.975),
    )
    four_minute_bounds = np.quantile(sampled_totals[:, 3:] / sampled_totals[:, :1],
                                     (0.025, 0.975), axis=0)
    deleted_totals = totals - session_counts
    deleted_differences = 100 * (deleted_totals[:, 2] - deleted_totals[:, 1]) / deleted_totals[:, 0]
    paired_times = [np.asarray(session, dtype=int)[:, 1:] for session in linked_sessions]
    search_seconds, elapsed_seconds = np.quantile(
        np.concatenate(paired_times), 0.95, axis=0, method="inverted_cdf",
    )
    shifts = []
    for draw in indices:
        thresholds = np.quantile(np.concatenate([paired_times[index] for index in draw]),
                                 0.95, axis=0, method="inverted_cdf")
        shifts.append(int(thresholds[1] - thresholds[0]))
    shift_lower, shift_upper = np.quantile(shifts, (0.025, 0.975))
    return {
        "analysis": "paired check completion from drop-off end and check start",
        "deadline_seconds": deadline,
        "linked_intervals": int(checks),
        "contributing_lane_sessions": len(linked_sessions),
        "completions_from_check_start": int(search_completions),
        "completions_from_dropoff_end": int(elapsed_completions),
        "dropoff_end_minus_check_start_percentage_points":
            float((elapsed_completions - search_completions) * 100 / checks),
        "paired_session_bootstrap_2_5": float(lower),
        "paired_session_bootstrap_97_5": float(upper),
        "equal_session_weight_difference_pp": float(session_differences.mean() * 100),
        "equal_session_weight_bootstrap_2_5": float(equal_session_lower),
        "equal_session_weight_bootstrap_97_5": float(equal_session_upper),
        "three_minute_difference_leave_one_session_out_min_pp": float(min(deleted_differences)),
        "three_minute_difference_leave_one_session_out_max_pp": float(max(deleted_differences)),
        "four_minute_completions_from_check_start": int(four_minute_search),
        "four_minute_completions_from_dropoff_end": int(four_minute_elapsed),
        "four_minute_search_start_bootstrap_2_5": float(four_minute_bounds[0, 0]),
        "four_minute_search_start_bootstrap_97_5": float(four_minute_bounds[1, 0]),
        "four_minute_dropoff_end_bootstrap_2_5": float(four_minute_bounds[0, 1]),
        "four_minute_dropoff_end_bootstrap_97_5": float(four_minute_bounds[1, 1]),
        "seconds_to_complete_95_percent_from_search_start": int(search_seconds),
        "seconds_to_complete_95_percent_from_dropoff_end": int(elapsed_seconds),
        "difference_in_95_percent_completion_seconds": int(elapsed_seconds - search_seconds),
        "seconds_difference_session_bootstrap_2_5": float(shift_lower),
        "seconds_difference_session_bootstrap_97_5": float(shift_upper),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def smoke():
    known_mix = quota_composition_value((2, 3), 2, [(1, 1)], 0.60)
    unrestricted_mix = quota_composition_value((2, 3), 2, [(2, 0), (1, 1), (0, 2)], 0.60)
    no_empirical_margin = quota_composition_value((2, 3), 2, [(1, 1)], 0.5)
    if not known_mix["feasible"] or not unrestricted_mix["feasible"] or not no_empirical_margin["feasible"]:
        raise ArithmeticError("Quota sensitivity smoke test failed")
    if not math.isclose(known_mix["guarantee"], 1 / 6, abs_tol=1e-9):
        raise ArithmeticError("The known-composition quota result differs from its exact value")
    if not math.isclose(unrestricted_mix["guarantee"], 1 / 10, abs_tol=1e-9):
        raise ArithmeticError("The unrestricted quota result differs from global uniform sampling")
    fully_unrestricted_mix = quota_composition_value(
        (2, 3), 2, [(2, 0), (1, 1), (0, 2)], 0.60,
    )
    if not math.isclose(fully_unrestricted_mix["guarantee"], 1 / 10, abs_tol=1e-9):
        raise ArithmeticError("Unrestricted placement must retain pooled sampling's guarantee")
    if not math.isclose(no_empirical_margin["guarantee"], 0.0, abs_tol=1e-9):
        raise ArithmeticError("The target-recovery constraint should give zero complete coverage at this limit")
    if not math.isclose(max(0, 1 - 2 * 0.01), 0.98, abs_tol=1e-12):
        raise ArithmeticError("The dependent physical-detection bound failed")
    if not math.isclose(quota_recovery_ceiling((2, 3), 2, [(1, 1)]), 0.5, abs_tol=1e-9):
        raise ArithmeticError("The known-composition recovery ceiling differs from its exact value")
    if not math.isclose(quota_recovery_ceiling((2, 3), 2, [(2, 0), (1, 1), (0, 2)]),
                        0.4, abs_tol=1e-9):
        raise ArithmeticError("The unrestricted recovery ceiling differs from uniform sampling")
    linked_smoke = linked_completion_sensitivity(replicates=40)
    assert linked_smoke["linked_intervals"] == 159
    assert linked_smoke["completions_from_check_start"] == 149
    assert linked_smoke["completions_from_dropoff_end"] == 133
    assert linked_smoke["paired_session_bootstrap_97_5"] <= 0
    assert linked_smoke["three_minute_difference_leave_one_session_out_max_pp"] < 0
    assert (133 <= linked_smoke["four_minute_completions_from_dropoff_end"]
            <= linked_smoke["four_minute_completions_from_check_start"] <= 159)
    assert linked_smoke["seconds_to_complete_95_percent_from_search_start"] > 180
    assert linked_smoke["difference_in_95_percent_completion_seconds"] > 0
    sequential_smoke = consecutive_service_session_bootstrap(replicates=40)
    expected_windows = {
        (3, 240): 124, (3, 300): 131, (5, 240): 78,
        (5, 300): 96, (5, 420): 110,
    }
    for result in sequential_smoke:
        key = (result["checks"], result["available_seconds"])
        if result["successful_windows"] != expected_windows[key]:
            raise ArithmeticError("A consecutive field-service window count changed")
        if result["lane_sessions"] != 13 or not (
                0 <= result["bootstrap_2_5"] <= result["bootstrap_97_5"] <= 1):
            raise ArithmeticError("The field-session bootstrap result is invalid")
        if not (0 <= result["leave_one_session_out_min"]
                <= result["observed_completion_fraction"]
                <= result["leave_one_session_out_max"] <= 1):
            raise ArithmeticError("The field-session deletion sensitivity is invalid")
    capacity_frontier = quota_capacity_frontier()
    expected_frontier = {
        (2, "composition_11"): 0.50,
        (3, "composition_11"): 2 / 3,
        (4, "composition_11"): 5 / 6,
        (5, "composition_11"): 1.0,
        (2, "all_compositions"): 0.40,
        (3, "all_compositions"): 0.60,
        (4, "all_compositions"): 0.80,
        (5, "all_compositions"): 1.0,
    }
    observed_frontier = {
        (row["search_slots"], row["case"]): row["maximum_mean_target_recovery"]
        for row in capacity_frontier
    }
    if set(observed_frontier) != set(expected_frontier) or any(
        not math.isclose(observed_frontier[key], value, abs_tol=1e-9)
        for key, value in expected_frontier.items()
    ):
        raise ArithmeticError("The quota recovery frontier differs from its exact values")
    if any(row["feasible_at_empirical_0_001_limit"] != (row["search_slots"] == 5)
           for row in capacity_frontier):
        raise ArithmeticError("The strict mean-risk threshold has an unexpected feasible capacity")
    if quota_composition_value((2, 3), 2, [(1, 1)], 0.001)["feasible"]:
        raise ArithmeticError("The two-search quota should fail the strict empirical mean-risk limit")
    quota_robustness = quota_composition_robustness(QUOTA_OUTSIDE_FRACTIONS)
    expected_composition_values = {
        0.0: 1 / 6, 0.1: 0.15, 0.2: 2 / 15, 0.21: 79 / 750,
        0.22: 13 / 125, 0.23: 77 / 750, 0.24: 19 / 187.5,
        0.25: 0.1, 0.4: 0.1, 1.0: 0.1,
    }
    assert all(math.isclose(row["worst_case_quota_referral_probability"],
                            expected_composition_values[
                                row["composition_outside_one_target_per_group"]
                            ], abs_tol=1e-10) for row in quota_robustness)
    assert all(row["quota_strictly_improves_on_pooled"]
               == (row["composition_outside_one_target_per_group"] < 0.25)
               for row in quota_robustness)
    service_rows = service_session_bootstrap(replicates=100, seed=SEED)
    five_checks_seven_minutes = next(
        row for row in service_rows if row["case"] == "5_checks_420_seconds"
    )
    assert math.isclose(
        five_checks_seven_minutes["independent_duration_model_estimate"],
        0.9797295794205286, abs_tol=1e-12,
    )
    sequential_seven_minutes = next(
        row for row in sequential_smoke
        if row["checks"] == 5 and row["available_seconds"] == 420
    )
    assert sequential_seven_minutes["successful_windows"] == 110
    assert math.isclose(
        sequential_seven_minutes["observed_completion_fraction"], 110 / 111, abs_tol=1e-12,
    )
    assert 0 <= five_checks_seven_minutes["bootstrap_2_5"] <= \
        five_checks_seven_minutes["bootstrap_97_5"] <= 1
    category_rows = target_category_sensitivity(smoke_only=True)
    category_summary = target_category_summary(category_rows)
    assert len(category_rows) > 0 and {row["target_definition"] for row in category_summary} == {
        "gun_or_knife", "all_labeled_object_classes"
    }
    assert all(sum(json.loads(row["target_images_by_difficulty"]).values())
               == row["distinct_target_images"] for row in category_summary)
    image_deletions = panel_target_image_deletion()
    if len(image_deletions) != 20:
        raise ArithmeticError("The target-image deletion smoke test failed")
    matched_summaries, matched_deletions = panel_matched_target_sensitivity()
    matched_expected = {
        "falselyAlarm": (4, 68, 0.8455882352941176),
        "falselyMiss": (6, 109, 0.9738532110091743),
    }
    for row in matched_summaries:
        count, participants, accuracy = matched_expected[row["case"]]
        assert row["matched_target_images"] == count
        assert row["participants"] == participants
        assert math.isclose(row["all_correct_fraction"], accuracy, abs_tol=1e-12)
        lower, upper = {"falselyAlarm": (-50 / 7, 150 / 7),
                        "falselyMiss": (-90 / 61, 110 / 64)}[row["case"]]
        assert math.isclose(row["paired_deletion_change_min_percentage_points"], lower, abs_tol=1e-10)
        assert math.isclose(row["paired_deletion_change_max_percentage_points"], upper, abs_tol=1e-10)
    assert len(matched_deletions) == 10
    toy_judgments = pd.DataFrame({"userID": ["first"] * 3, "userCorrect": [1, 1, 0]})
    assert standardized_judgment_scores(toy_judgments, 2).loc["first"] == 1 / 3
    assert standardized_judgment_scores(toy_judgments, 3).loc["first"] == 0
    image_generalisation_rows = panel_shared_other_target_sensitivity()
    expected_image_generalisation = {
        "falselyMiss": (6, 4, 62, 0.9637096774193549, 0.8024193548387096,
                        -16.129032258064516),
        "falselyAlarm": (4, 6, 61, 0.8401639344262295, 0.9270491803278688,
                         8.688524590163934),
    }
    for row in image_generalisation_rows:
        expected = expected_image_generalisation[row["condition"]]
        observed = (
            row["matched_target_images"], row["other_target_images"],
            row["participants_with_three_or_more_in_both_sets"],
            row["shared_all_correct_fraction"], row["other_all_correct_fraction"],
            row["other_minus_shared_percentage_points"],
        )
        assert observed[:3] == expected[:3]
        assert all(math.isclose(actual, wanted, abs_tol=1e-12)
                   for actual, wanted in zip(observed[3:], expected[3:], strict=True))
        assert row["paired_bootstrap_2_5"] <= row["other_minus_shared_percentage_points"]
        assert row["other_minus_shared_percentage_points"] <= row["paired_bootstrap_97_5"]
        assert row["participants_with_two_or_more_in_both_sets"] >= observed[2]
        assert row["two_judgment_deletion_min_pp"] < 0 < row["two_judgment_deletion_max_pp"]
        assert (row["two_judgment_deletions_with_negative_difference"]
                + row["two_judgment_deletions_with_positive_difference"] <= row["target_image_deletions"])
        assert (row["larger_panel_two_judgment_bootstrap_2_5"]
                <= row["larger_panel_two_judgment_difference_pp"]
                <= row["larger_panel_two_judgment_bootstrap_97_5"])
    assert len(image_generalisation_rows) == 2
    expected_deletion_diagnostics = {
        "falselyMiss": (14, -23.214285714285715, 5.357142857142857),
        "falselyAlarm": (13, -5.769230769230769, 17.307692307692307),
    }
    for row in image_generalisation_rows:
        participants, lowest, highest = expected_deletion_diagnostics[row["condition"]]
        assert row["complete_image_deletion_panel_participants"] == participants
        assert math.isclose(row["image_deletion_effect_min_percentage_points"],
                            lowest, abs_tol=1e-10)
        assert math.isclose(row["image_deletion_effect_max_percentage_points"],
                            highest, abs_tol=1e-10)
    return {
        "passed": True,
        "known_composition_guarantee": known_mix["guarantee"],
        "unrestricted_composition_guarantee": unrestricted_mix["guarantee"],
        "fully_unrestricted_two_target_guarantee": fully_unrestricted_mix["guarantee"],
        "composition_11_maximum_mean_recovery": 0.5,
        "unrestricted_composition_maximum_mean_recovery": 0.4,
        "quota_capacity_frontier_rows": len(capacity_frontier),
        "empirical_0_001_risk_first_feasible_at_five_search_slots": True,
        "target_image_deletion_cases": len(image_deletions),
        "quota_composition_robustness_rows": len(quota_robustness),
        "service_session_bootstrap_rows": len(service_rows),
        "five_checks_seven_minutes_service_estimate":
            five_checks_seven_minutes["independent_duration_model_estimate"],
        "target_category_sensitivity_rows": len(category_rows),
        "matched_target_image_sensitivity": matched_summaries,
        "matched_target_image_deletion_rows": len(matched_deletions),
        "paired_shared_other_target_image_sensitivity": image_generalisation_rows,
        "linked_completion_sensitivity": linked_smoke,
        "consecutive_service_session_bootstrap_smoke_rows": len(sequential_smoke),
        "five_checks_420_seconds_observed_completion_fraction":
            sequential_seven_minutes["observed_completion_fraction"],
    }


def full():
    quota_rows = quota_composition_analysis()
    capacity_frontier_rows = quota_capacity_frontier()
    participant_rows = panel_participant_bootstrap()
    image_deletion_rows = panel_target_image_deletion()
    matched_target_rows, matched_target_deletion_rows = panel_matched_target_sensitivity()
    image_generalisation_rows = panel_shared_other_target_sensitivity()
    service_rows = service_session_bootstrap()
    consecutive_service_rows = consecutive_service_session_bootstrap()
    linked_completion_row = linked_completion_sensitivity()
    if (
        linked_completion_row["linked_intervals"] != 159
        or linked_completion_row["contributing_lane_sessions"] != 13
        or linked_completion_row["completions_from_check_start"] != 149
        or linked_completion_row["completions_from_dropoff_end"] != 133
        or not math.isclose(
            linked_completion_row["dropoff_end_minus_check_start_percentage_points"],
            -100 * 16 / 159, abs_tol=1e-12,
        )
    ):
        raise ArithmeticError("The paired completion origins differ from the public timing records")
    seven_minute = next(row for row in consecutive_service_rows
                        if row["checks"] == 5 and row["available_seconds"] == 420)
    if (
        seven_minute["successful_windows"] != 110
        or not math.isclose(seven_minute["bootstrap_2_5"], 0.9719560042, abs_tol=1e-10)
        or not math.isclose(seven_minute["leave_one_session_out_min"], 0.9891304348, abs_tol=1e-10)
    ):
        raise ArithmeticError("The seven-minute percentile and session-deletion ranges differ")
    category_rows = target_category_sensitivity()
    category_summaries = target_category_summary(category_rows)
    quota_robustness = quota_composition_robustness(QUOTA_OUTSIDE_FRACTIONS)
    rows = [*quota_rows, *participant_rows, *service_rows, linked_completion_row]
    pd.DataFrame(rows).to_csv(OUTPUT, index=False, float_format="%.10g")
    pd.DataFrame(capacity_frontier_rows).to_csv(
        OUT / "quota_capacity_frontier.csv", index=False, float_format="%.10g"
    )
    pd.DataFrame(image_deletion_rows).to_csv(
        OUT / "external_image_deletion.csv", index=False, float_format="%.10g"
    )
    pd.DataFrame(matched_target_rows).to_csv(
        MATCHED_TARGET_OUTPUT, index=False, float_format="%.10g"
    )
    pd.DataFrame(matched_target_deletion_rows).to_csv(
        MATCHED_TARGET_DELETION_OUTPUT, index=False, float_format="%.10g"
    )
    pd.DataFrame(image_generalisation_rows).to_csv(
        IMAGE_GENERALISATION_OUTPUT, index=False, float_format="%.10g"
    )
    pd.DataFrame(consecutive_service_rows).to_csv(
        CONSECUTIVE_SERVICE_OUTPUT, index=False, float_format="%.10g"
    )
    pd.DataFrame([linked_completion_row]).to_csv(
        LINKED_COMPLETION_OUTPUT, index=False, float_format="%.10g"
    )
    pd.DataFrame(category_rows).to_csv(
        OUT / "target_category_capacity_frontier.csv", index=False, float_format="%.10g"
    )
    pd.DataFrame(category_summaries).to_csv(
        OUT / "target_category_sensitivity.csv", index=False, float_format="%.10g"
    )
    pd.DataFrame(quota_robustness).to_csv(
        OUT / "quota_composition_robustness.csv", index=False, float_format="%.10g"
    )
    record = {
        "status": "completed",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "external_response_sha256": hashlib.sha256(EXTERNAL_FILE.read_bytes()).hexdigest(),
        "pilot_response_sha256": hashlib.sha256(PRIMARY_FILE.read_bytes()).hexdigest(),
        "stimulus_file_sha256": hashlib.sha256(STIMULUS_FILE.read_bytes()).hexdigest(),
        "checkpoint_file_sha256": hashlib.sha256(FIELD_FILE.read_bytes()).hexdigest(),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "random_seed": SEED,
        "rows": len(rows),
        "quota_capacity_frontier_rows": len(capacity_frontier_rows),
        "target_image_deletion_rows": len(image_deletion_rows),
        "matched_target_image_sensitivity": matched_target_rows,
        "matched_target_image_deletion_rows": len(matched_target_deletion_rows),
        "paired_shared_other_target_image_sensitivity": image_generalisation_rows,
        "consecutive_service_session_bootstrap_rows": len(consecutive_service_rows),
        "consecutive_service_session_bootstrap": consecutive_service_rows,
        "linked_completion_sensitivity": linked_completion_row,
        "target_category_rows": len(category_rows),
        "quota_composition_robustness_rows": len(quota_robustness),
        "passed": True,
    }
    VERIFICATION.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    return record


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    arguments = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    smoke_record = smoke()
    if arguments.smoke:
        print(json.dumps(smoke_record, indent=2))
        return
    result = full()
    result["smoke"] = smoke_record
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
