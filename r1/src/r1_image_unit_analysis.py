from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from collections import Counter
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize_scalar
from scipy.stats import hypergeom

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results" / "R1_image_unit"
PRIMARY_FILE = BASE / "data" / "pilotResult_combinedAIandNoAI.xlsx"
STIMULUS_FILE = BASE / "data" / "task_selected_for_study.csv"
EXTERNAL_FILE = BASE / "data" / "behavior_XAI_presented_beforeAndAfter_trustViolation.csv"
SIXRAY_FILE = BASE / "data" / "SIXRay_annotations_positive_all.csv"
SEED = 2026111901
BOOTSTRAP_REPLICATES = 2000
COPULA_GRID_POINTS = 40001
MEAN_MISS_REQUIREMENT = 0.001


def object_set(value: object) -> frozenset[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return frozenset()
    if isinstance(value, str):
        text = value.strip()
        if not text or text.casefold() in {"none", "null", "nan"}:
            return frozenset()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                value = text.split(",")
    if value is None:
        return frozenset()
    parts = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
    return frozenset(str(part).strip().casefold() for part in parts
                     if str(part).strip().casefold() not in {"", "none", "null", "nan"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transport_extreme(matrix: np.ndarray, left_weights: np.ndarray,
                      right_weights: np.ndarray, maximize: bool = False) -> float:
    rows, columns = matrix.shape
    equality = np.zeros((rows + columns, rows * columns), dtype=float)
    for row in range(rows):
        equality[row, row * columns:(row + 1) * columns] = 1
    for column in range(columns):
        equality[rows + column, column::columns] = 1
    marginals = np.concatenate([left_weights, right_weights])
    result = linprog((-1 if maximize else 1) * matrix.ravel(), A_eq=equality[:-1],
                     b_eq=marginals[:-1], bounds=(0, None), method="highs")
    if not result.success:
        raise ArithmeticError(f"Transportation LP failed: {result.message}")
    optimum = float(np.sum(matrix * result.x.reshape(matrix.shape)))
    return float(np.clip(optimum, matrix.min(), matrix.max()))


def transport_bounds(matrix: np.ndarray, left_weights: np.ndarray,
                     right_weights: np.ndarray) -> tuple[float, float]:
    return (transport_extreme(matrix, left_weights, right_weights),
            transport_extreme(matrix, left_weights, right_weights, True))


def prepare_cases() -> dict[str, dict]:
    trial = pd.read_excel(PRIMARY_FILE, keep_default_na=False)
    forbidden = {"gun", "knife"}
    cases = {}
    for condition in ("false_miss", "false_alarm"):
        if condition == "false_miss":
            first_stage = trial[(trial["GROUP"] == "AI") &
                                (trial["num_of_tv"] == "tv3") &
                                (trial["type_of_tv"] == "False Miss")].copy()
            second_stage = trial[(trial["GROUP"] == "NoAI") &
                                 (trial["num_of_tv"] == "tv3") &
                                 (trial["type_of_tv"] == "falselyMissNoAI")].copy()
        if condition == "false_alarm":
            first_stage = trial[(trial["GROUP"] == "AI") &
                                (trial["num_of_tv"] == "tv3") &
                                (trial["type_of_tv"] == "False Alarm")].copy()
            second_stage = trial[(trial["GROUP"] == "NoAI") &
                                 (trial["num_of_tv"] == "tv3") &
                                 (trial["type_of_tv"] == "falselyAlarmNoAI")].copy()
        first_stage["image"] = first_stage["ImageNumber"].str.casefold().str.replace(
            r"\.jpg$", "", regex=True)
        second_stage["image"] = second_stage["ImageNumber"].str.casefold().str.replace(
            r"\.jpg$", "", regex=True)
        image_ids = sorted(set(first_stage["image"]))
        if len(image_ids) != 27 or set(second_stage["image"]) != set(image_ids):
            raise ValueError(f"The {condition} task does not have 27 matched images")
        if first_stage["user_identifier"].nunique() != 10 or \
                second_stage["user_identifier"].nunique() != 10:
            raise ValueError(f"The {condition} task does not contain two groups of ten")
        if first_stage.duplicated(["user_identifier", "image"]).any() or \
                second_stage.duplicated(["user_identifier", "image"]).any():
            raise ValueError(f"The {condition} task repeats an image within a participant")
        for stage in (first_stage, second_stage):
            if not stage.groupby("user_identifier")["image"].size().eq(len(image_ids)).all():
                raise ValueError(f"The {condition} task has an incomplete response profile")
            if not stage["OpenBag"].astype(str).str.casefold().isin(("true", "false")).all():
                raise ValueError(f"The {condition} task has an unknown label-request value")
            if stage.groupby("image")["ground truth"].apply(
                    lambda labels: labels.map(object_set).nunique()).max() != 1:
                raise ValueError(f"The {condition} task has inconsistent target annotations")
        if first_stage.groupby("image")["model result"].apply(
                lambda labels: labels.map(object_set).nunique()).max() != 1:
            raise ValueError(f"The {condition} task has inconsistent machine annotations")

        primary_labels = first_stage.drop_duplicates("image").set_index("image")
        secondary_labels = second_stage.drop_duplicates("image").set_index("image")
        if not primary_labels.index.is_unique or not secondary_labels.index.is_unique:
            raise ValueError(f"The {condition} image labels are not unique")
        targets = np.array([bool(object_set(primary_labels.loc[image, "ground truth"]) &
                                      forbidden) for image in image_ids])
        machine = np.array([bool(object_set(primary_labels.loc[image, "model result"]))
                            for image in image_ids])
        for image in image_ids:
            if object_set(primary_labels.loc[image, "ground truth"]) != \
                    object_set(secondary_labels.loc[image, "ground truth"]):
                raise ValueError(f"The two study groups disagree about image {image}")
        primary_profiles = []
        for _, person in first_stage.groupby("user_identifier", sort=True):
            lookup = person.set_index("image")
            human = np.array([bool(object_set(lookup.loc[image, "UserDecision"]))
                              for image in image_ids])
            opened = np.array([str(lookup.loc[image, "OpenBag"]).casefold() == "true"
                               for image in image_ids])
            route = np.where(human | opened, "B", np.where(machine, "K", "U"))
            primary_profiles.append({
                "route": route,
            })
        secondary_profiles = []
        for _, person in second_stage.groupby("user_identifier", sort=True):
            lookup = person.set_index("image")
            referral = np.array([
                bool(object_set(lookup.loc[image, "UserDecision"])) or
                str(lookup.loc[image, "OpenBag"]).casefold() == "true"
                for image in image_ids
            ])
            secondary_profiles.append({"referral": referral})

        if targets.sum() == 0:
            raise ValueError(f"No gun-or-knife target images in {condition}")
        cases[condition] = {
            "image_ids": image_ids,
            "targets": targets,
            "machine": machine,
            "primary": primary_profiles,
            "secondary": secondary_profiles,
            "target_count": int(targets.sum()),
            "image_count": len(image_ids),
            "label": "False-miss trust-violation condition" if condition == "false_miss"
                     else "False-alarm trust-violation condition",
        }
    if len(trial["user_identifier"].unique()) != 60:
        raise ValueError("The pilot should contain 40 assisted and 20 unaided participants")
    return cases


def response_components(case: dict, primary_index: int,
                        secondary_index: int) -> dict:
    primary = case["primary"][primary_index]["route"]
    secondary = case["secondary"][secondary_index]["referral"]
    targets = case["targets"]
    eligible_indices = np.flatnonzero(primary == "U")
    eligible_targets = targets[eligible_indices]
    eligible_referrals = secondary[eligible_indices]
    required_targets = int(np.sum(eligible_targets))
    all_targets_referred = bool(np.all(eligible_referrals[eligible_targets]))
    non_target_referrals = int(np.sum(eligible_referrals & ~eligible_targets))
    return {
        "population": len(eligible_indices),
        "mandatory": int(np.sum(primary != "U")),
        "required_targets": required_targets,
        "all_targets_referred": all_targets_referred,
        "non_target_referrals": non_target_referrals,
        "response_referrals": eligible_referrals,
        "eligible_target_mask": eligible_targets,
        "secondary_referrals": eligible_referrals,
    }


def fixed_success(component: dict, reading_limit: int,
                  total_capacity: int) -> float:
    population = component["population"]
    mandatory = component["mandatory"]
    target_count = component["required_targets"]
    if mandatory > total_capacity or not component["all_targets_referred"]:
        return 0.0
    reads = min(reading_limit, population)
    if reads < target_count:
        return 0.0
    available = total_capacity - mandatory - target_count
    if population == 0:
        return float(target_count == 0)
    inclusion = math.comb(reads, target_count) / math.comb(population, target_count)
    if target_count == population:
        return inclusion * float(available >= 0)
    return inclusion * float(hypergeom.cdf(
        available, population - target_count,
        component["non_target_referrals"], reads - target_count,
    ))


def stop_success(component: dict, reading_limit: int,
                 total_capacity: int) -> float:
    population = component["population"]
    mandatory = component["mandatory"]
    target_count = component["required_targets"]
    if mandatory > total_capacity or not component["all_targets_referred"]:
        return 0.0
    reads = min(reading_limit, population)
    if population == 0:
        return float(target_count == 0)
    remaining = total_capacity - mandatory
    total_referrals = target_count + component["non_target_referrals"]
    if target_count == 0:
        return 1.0
    if remaining < target_count or reads < target_count:
        return 0.0
    probability = 0.0
    low = max(0, reads - (population - total_referrals))
    high = min(reads, total_referrals)
    denominator = math.comb(total_referrals, target_count)
    for observed in range(low, high + 1):
        admitted = min(observed, remaining)
        probability += float(hypergeom.pmf(observed, population, total_referrals, reads)) * (
            math.comb(admitted, target_count) / denominator
            if admitted >= target_count else 0.0
        )
    return probability


def expected_stopping_resources(component: dict, reading_limit: int,
                                total_capacity: int) -> tuple[float, float]:
    population = component["population"]
    reads = min(reading_limit, population)
    referrals = int(np.sum(component["secondary_referrals"]))
    slots = max(0, total_capacity - component["mandatory"])
    expected_searches = 0.0
    low = max(0, reads - (population - referrals))
    high = min(reads, referrals)
    for observed in range(low, high + 1):
        expected_searches += float(hypergeom.pmf(
            observed, population, referrals, reads)) * min(observed, slots)
    if slots == 0:
        expected_reads = 0.0
    else:
        expected_reads = sum(
            float(hypergeom.cdf(slots - 1, population, referrals, position))
            for position in range(reads)
        )
    return expected_reads, component["mandatory"] + expected_searches


def expected_fixed_resources(component: dict, reading_limit: int) -> tuple[float, float]:
    population = component["population"]
    reads = min(reading_limit, population)
    referrals = int(np.sum(component["secondary_referrals"]))
    searches = component["mandatory"] + (reads / population * referrals if population else 0)
    return float(reads), float(searches)


def referral_miss(component: dict, reading_limit: int, total_targets: int,
                  stopping: bool = False, total_capacity: int | None = None) -> float:
    population = component["population"]
    if total_targets == 0 or population == 0:
        return 0.0
    reads = min(reading_limit, population)
    target_mask = component["eligible_target_mask"]
    secondary = component["secondary_referrals"]
    if stopping:
        if total_capacity is None:
            raise ValueError("Stopping risk requires a search capacity")
        slots = total_capacity - component["mandatory"]
        referral_count = int(np.sum(secondary))
        recovery = 0.0
        if slots > 0 and reads > 0:
            if population == 1:
                recovery = float(slots > 0)
            else:
                for other_referrals in range(min(reads - 1, referral_count - 1) + 1):
                    probability = float(hypergeom.pmf(
                        other_referrals, population - 1, referral_count - 1, reads - 1))
                    recovery += probability * min(slots, other_referrals + 1) / (
                        other_referrals + 1)
            recovery *= reads / population
        target_referred = np.sum(target_mask & secondary) * recovery
    else:
        reads_fraction = reads / population
        target_referred = np.sum(target_mask & secondary) * reads_fraction
    return float((np.sum(target_mask) - target_referred) / total_targets)


def verify_literal_enumeration() -> dict:
    checked = 0
    for population in range(1, 7):
        for referral_bits in combinations(range(population), min(2, population)):
            referrals = np.zeros(population, dtype=bool)
            referrals[list(referral_bits)] = True
            for reads in range(population + 1):
                for slots in range(population + 1):
                    if not referrals.any():
                        continue
                    for target in range(population):
                        if not referrals[target]:
                            continue
                        failures = 0
                        fixed_successes = 0
                        stopping_successes = 0
                        total_orders = 0
                        for order in permutations(range(population)):
                            total_orders += 1
                            observed = order[:reads]
                            kept = []
                            fixed_referrals = sum(bool(referrals[item]) for item in observed)
                            fixed_successes += target in observed and fixed_referrals <= slots
                            for item in observed:
                                if referrals[item]:
                                    if len(kept) >= slots:
                                        break
                                    kept.append(item)
                            failures += target not in kept
                            stopping_successes += target in kept
                        component = {
                            "population": population,
                            "mandatory": 0,
                            "required_targets": 1,
                            "all_targets_referred": True,
                            "non_target_referrals": int(np.sum(referrals)) - 1,
                            "eligible_target_mask": np.ones(population, dtype=bool),
                            "secondary_referrals": referrals,
                        }
                        component["eligible_target_mask"][:] = False
                        component["eligible_target_mask"][target] = True
                        actual_miss = referral_miss(component, reads, 1, True, slots)
                        fixed_probability = fixed_success(component, reads, slots)
                        stopping_probability = stop_success(component, reads, slots)
                        if not np.isclose(actual_miss, failures / total_orders, atol=1e-12) or \
                                not np.isclose(fixed_probability, fixed_successes / total_orders,
                                               atol=1e-12) or \
                                not np.isclose(stopping_probability,
                                               stopping_successes / total_orders, atol=1e-12):
                            raise AssertionError("Exact kernels disagree with order enumeration")
                        checked += 1
    return {"literal_order_checks": checked, "passed": True}


def scenario_matrices(case: dict) -> dict:
    primary_count = len(case["primary"])
    secondary_count = len(case["secondary"])
    components = [[response_components(case, first, second)
                   for second in range(secondary_count)]
                  for first in range(primary_count)]
    mandatory = np.array([[part["mandatory"] for part in row]
                          for row in components], dtype=int)
    if not np.all(mandatory == mandatory[:, :1]):
        raise AssertionError("The first-stage mandatory work must not depend on review type")
    initial = mandatory[:, 0]
    eligible = np.array([[part["population"] for part in row]
                         for row in components], dtype=int)
    maximum_readings = int(eligible.max())
    capacity_min = int(initial.max())
    capacity_max = int(np.max(initial + eligible[:, 0]))
    capacities = list(range(capacity_min, capacity_max + 1))
    right = np.full(secondary_count, 1 / secondary_count)
    left = np.full(primary_count, 1 / primary_count)
    rows = []
    for capacity in capacities:
        stop = np.empty((primary_count, secondary_count))
        full = np.empty_like(stop)
        stop_risk = np.empty_like(stop)
        stop_reads = np.empty_like(stop)
        stop_searches = np.empty_like(stop)
        full_reads = np.empty_like(stop)
        full_searches = np.empty_like(stop)
        for first in range(primary_count):
            for second in range(secondary_count):
                part = components[first][second]
                stop[first, second] = stop_success(part, maximum_readings, capacity)
                full[first, second] = fixed_success(part, maximum_readings, capacity)
                stop_risk[first, second] = referral_miss(
                    part, maximum_readings, case["target_count"], True, capacity)
                stop_reads[first, second], stop_searches[first, second] = \
                    expected_stopping_resources(part, maximum_readings, capacity)
                full_reads[first, second], full_searches[first, second] = \
                    expected_fixed_resources(part, maximum_readings)
        stop_lower, stop_upper = transport_bounds(stop, left, right)
        full_lower, full_upper = transport_bounds(full, left, right)
        effect_lower, effect_upper = transport_bounds(stop - full, left, right)
        fixed_rows = []
        for count in range(maximum_readings + 1):
            fixed = np.empty_like(stop)
            fixed_risk = np.empty_like(stop)
            for first in range(primary_count):
                for second in range(secondary_count):
                    part = components[first][second]
                    fixed[first, second] = fixed_success(part, count, capacity)
                    fixed_risk[first, second] = referral_miss(
                        part, count, case["target_count"])
            lower, upper = transport_bounds(fixed, left, right)
            risk_upper = transport_extreme(fixed_risk, left, right, True)
            fixed_rows.append((lower, upper, risk_upper, count))
            if np.min(stop - fixed) < -1e-12:
                raise AssertionError("Matched-prefix stopping failed to dominate fixed review")
        best = max(fixed_rows, key=lambda row: (row[0], -row[3]))
        best_fixed_reads = np.empty_like(stop)
        best_fixed_searches = np.empty_like(stop)
        for first in range(primary_count):
            for second in range(secondary_count):
                part = components[first][second]
                reads = min(best[3], part["population"])
                referrals = int(np.sum(part["secondary_referrals"]))
                best_fixed_reads[first, second] = reads
                best_fixed_searches[first, second] = part["mandatory"] + (
                    reads / part["population"] * referrals if part["population"] else 0.0)
        independent_weights = np.outer(left, right)
        rows.append({
            "condition": case["condition"],
            "condition_label": case["label"],
            "unique_images": case["image_count"],
            "target_images": case["target_count"],
            "common_reading_limit": maximum_readings,
            "total_search_capacity": capacity,
            "capacity_above_largest_mandatory_work": capacity - capacity_min,
            "stop_lower": stop_lower,
            "stop_upper": stop_upper,
            "stop_product_coupling": float(np.sum(independent_weights * stop)),
            "stop_expected_readings_product_coupling": float(
                np.sum(independent_weights * stop_reads)),
            "stop_expected_searches_product_coupling": float(
                np.sum(independent_weights * stop_searches)),
            "stop_readings_lower": transport_extreme(stop_reads, left, right),
            "stop_readings_upper": transport_extreme(stop_reads, left, right, True),
            "stop_searches_lower": transport_extreme(stop_searches, left, right),
            "stop_searches_upper": transport_extreme(stop_searches, left, right, True),
            "matched_maximum_fixed_count_readings_product_coupling": float(
                np.sum(independent_weights * full_reads)),
            "matched_maximum_fixed_count_searches_product_coupling": float(
                np.sum(independent_weights * full_searches)),
            "stop_target_referral_miss_upper": transport_extreme(
                stop_risk, left, right, True),
            "fixed_full_lower": full_lower,
            "fixed_full_upper": full_upper,
            "matched_full_effect_lower": effect_lower,
            "matched_full_effect_upper": effect_upper,
            "best_fixed_count_lower": best[0],
            "best_fixed_count_upper": best[1],
            "best_fixed_count_miss_upper": best[2],
            "best_fixed_readings": best[3],
            "best_fixed_expected_readings_product_coupling": float(
                np.sum(independent_weights * best_fixed_reads)),
            "best_fixed_expected_searches_product_coupling": float(
                np.sum(independent_weights * best_fixed_searches)),
        })
    target_threshold_rows = []
    for threshold in (0.5, 0.8, 0.9):
        qualifying = [row for row in rows if row["stop_lower"] >= threshold - 1e-12]
        fixed_qualifying = [row for row in rows if row["best_fixed_count_lower"] >= threshold - 1e-12]
        stop_row = qualifying[0] if qualifying else None
        fixed_row = fixed_qualifying[0] if fixed_qualifying else None
        target_threshold_rows.append({
            "condition": case["condition"],
            "referral_and_capacity_target": threshold,
            "stop_capacity": None if stop_row is None else stop_row["total_search_capacity"],
            "stop_lower": None if stop_row is None else stop_row["stop_lower"],
            "stop_best_fixed_at_same_capacity": None if stop_row is None else
            stop_row["best_fixed_count_lower"],
            "fixed_capacity": None if fixed_row is None else
            fixed_row["total_search_capacity"],
            "fixed_best_readings": None if fixed_row is None else
            fixed_row["best_fixed_readings"],
            "required_conditional_capability_at_stop_capacity": None if stop_row is None
            else threshold / stop_row["stop_lower"],
        })
    return {
        "condition": case["condition"],
        "capacity_rows": rows,
        "capacity_thresholds": target_threshold_rows,
        "components": components,
        "left_weights": left,
        "right_weights": right,
        "primary_loads": initial,
        "secondary_loads": np.array([
            [int(np.sum(profile["referral"]))] for profile in case["secondary"]
        ]).reshape(-1),
        "kernel_validation": validate_all_policy_kernels(components, maximum_readings,
                                                         capacities),
    }


def build_external_response_summaries(replicates: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(EXTERNAL_FILE, keep_default_na=False)
    stimulus = pd.read_csv(STIMULUS_FILE, keep_default_na=False)
    stimulus_images = set(stimulus["filename"].str.casefold().str.replace(
        r"\.jpg$", "", regex=True))
    response_images = set(data["imageID"].str.casefold().str.replace(
        r"\.jpg$", "", regex=True))
    if data["userID"].nunique() != 232 or data["imageID"].nunique() != 42:
        raise ValueError("The public response sample does not match its 232-person, 42-image record")
    if stimulus_images != response_images:
        raise ValueError("The 232-person response sample and stimulus labels use different images")
    if not data.groupby("userID").size().eq(39).all():
        raise ValueError("The public response sample should contain 39 tasks per participant")
    person_features = data[["userID", "type_of_responsive", "type_of_vis"]].drop_duplicates()
    if person_features["userID"].duplicated().any():
        raise ValueError("Experimental-response strata vary within participants")
    data["correct"] = pd.to_numeric(data["userCorrect"], errors="raise")
    data["truth_request"] = pd.to_numeric(data["openBaggage"], errors="raise")
    identifiers = person_features.groupby(["type_of_responsive", "type_of_vis"])["userID"].apply(
        lambda values: values.to_list()).to_dict()
    people = data["userID"].drop_duplicates().to_list()
    person_index = {person: index for index, person in enumerate(people)}
    data["person_index"] = data["userID"].map(person_index)
    groups = list(data.groupby(
        ["type_of_responsive", "type_of_tv", "type_of_vis", "beforeOrAfterTV"],
        sort=True,
    ))
    point = data.groupby(
        ["type_of_responsive", "type_of_tv", "type_of_vis", "beforeOrAfterTV"],
        as_index=False,
    ).agg(
        participants=("userID", "nunique"), trials=("userID", "size"),
        task_accuracy=("correct", "mean"), true_label_requests=("truth_request", "mean"),
    )
    rng = np.random.default_rng(SEED)
    lower = np.empty((replicates, len(groups)))
    for repetition in range(replicates):
        sampled_indices = []
        for group, candidates in identifiers.items():
            sampled = rng.choice(candidates, len(candidates), replace=True)
            sampled_indices.extend(person_index[item] for item in sampled)
        weights = np.bincount(sampled_indices, minlength=len(people))
        for column, (keys, rows) in enumerate(groups):
            rows = rows[rows["person_index"].map(lambda index: weights[int(index)] > 0)]
            index = rows["person_index"].to_numpy(dtype=int)
            row_weights = weights[index]
            denominator = float(np.sum(row_weights))
            if denominator <= 0:
                raise ArithmeticError("Participant resampling produced an empty task cell")
            accuracy = float(np.average(rows["correct"], weights=row_weights))
            lower[repetition, column] = accuracy
    intervals = point.copy()
    intervals["bootstrap_2_5"] = np.quantile(lower, .025, axis=0)
    intervals["bootstrap_median"] = np.quantile(lower, .5, axis=0)
    intervals["bootstrap_97_5"] = np.quantile(lower, .975, axis=0)
    intervals["bootstrap_replicates"] = replicates
    intervals["interval_interpretation"] = "Participant-stratified percentile range; images held fixed"
    return point, intervals


def sixray_summary() -> tuple[pd.DataFrame, dict]:
    boxes = pd.read_csv(SIXRAY_FILE, keep_default_na=False)
    required = {"filename", "class", "xmin", "ymin", "xmax", "ymax"}
    if not required <= set(boxes):
        raise ValueError("The public SIXray label table lacks expected columns")
    boxes["image"] = boxes["filename"].str.casefold()
    boxes["category"] = boxes["class"].str.strip().str.casefold()
    image_classes = boxes.groupby("image")["category"].agg(lambda values: frozenset(values))
    target_classes = {"gun", "knife"}
    target_images = image_classes.map(lambda classes: bool(classes & target_classes))
    target_ids = target_images[target_images].index
    labels = []
    for image, categories in image_classes.items():
        target_box_count = int(boxes.loc[
            (boxes["image"] == image) & boxes["category"].isin(target_classes)
        ].shape[0])
        if target_box_count:
            labels.append({
                "image": image,
                "target_box_count": target_box_count,
                "target_class_count": len(categories & target_classes),
                "annotated_class_count": len(categories),
                "has_multiple_target_boxes": target_box_count > 1,
                "has_both_target_classes": target_classes <= categories,
                "has_other_prohibited_class": bool(categories - target_classes),
            })
    target_structure = pd.DataFrame(labels)
    class_summary = boxes.groupby("category").agg(
        annotated_boxes=("image", "size"), annotated_images=("image", "nunique"),
    ).reset_index()
    summary = {
        "annotated_boxes": int(len(boxes)),
        "unique_positive_images": int(boxes["image"].nunique()),
        "images_with_multiple_boxes": int(boxes.groupby("image").size().gt(1).sum()),
        "images_with_multiple_annotated_classes": int(image_classes.map(
            lambda categories: len(categories) > 1).sum()),
        "positive_images_with_gun_or_knife": int(len(target_ids)),
        "gun_or_knife_images_with_multiple_target_boxes": int(
            target_structure["has_multiple_target_boxes"].sum()),
        "gun_or_knife_images_with_both_classes": int(
            target_structure["has_both_target_classes"].sum()),
        "gun_or_knife_images_with_other_classes": int(
            target_structure["has_other_prohibited_class"].sum()),
        "data_role": "Descriptive annotation-positive image composition; not baggage prevalence or human-response data",
    }
    return class_summary, summary


def smoke_check(cases: dict) -> dict:
    parser_cases = {
        "empty_json_array": object_set("[]"),
        "empty_python_list": object_set("['None']"),
        "unread": object_set("None"),
        "comma_separated": object_set("Knife, Wrench"),
        "json_labels": object_set('["Gun", "Knife"]'),
    }
    if parser_cases != {
        "empty_json_array": frozenset(), "empty_python_list": frozenset(),
        "unread": frozenset(), "comma_separated": frozenset({"knife", "wrench"}),
        "json_labels": frozenset({"gun", "knife"}),
    }:
        raise AssertionError("Threat-label parsing did not preserve source encodings")
    route_counts = {
        condition: [Counter(profile["route"]) for profile in case["primary"]]
        for condition, case in cases.items()
    }
    if any(not any(profile.get("U", 0) for profile in profiles)
           for profiles in route_counts.values()):
        raise AssertionError("Each condition must contain an eligible image-review pool")
    subset = np.array([True, False, False, False])
    referrals = np.array([True, False, True, False])
    component = {
        "population": 4, "mandatory": 1, "required_targets": 1,
        "all_targets_referred": True, "non_target_referrals": 1,
        "eligible_target_mask": subset, "secondary_referrals": referrals,
    }
    full = fixed_success(component, 4, 2)
    stopped = stop_success(component, 4, 2)
    if not np.isclose(full, 0.0) or not np.isclose(stopped, 0.5):
        raise AssertionError("The analytical smoke case did not return its exact values")
    marginal = np.array([[1., 0.], [0., 1.]])
    bounds = transport_bounds(marginal, np.array([.5, .5]), np.array([.5, .5]))
    if not np.allclose(bounds, [0., 1.]):
        raise AssertionError("The transport solver did not reproduce the 2-by-2 endpoint")
    exact = verify_literal_enumeration()
    pilot = {key: {name: count for name, count in (
        ("participants", 10), ("matched_images", int(case["image_count"])),
        ("target_images", int(case["target_count"])),
        ("primary_profiles", len(case["primary"])),
        ("secondary_profiles", len(case["secondary"])),
    )} for key, case in cases.items()}
    return {
        "passed": True,
        "data_hashes": {path.name: sha256(path) for path in
                        (PRIMARY_FILE, STIMULUS_FILE, EXTERNAL_FILE, SIXRAY_FILE)},
        "analytical_case": {"fixed_full": full, "stopped_full": stopped,
                            "transport_bounds": bounds},
        "literal_order_verification": exact,
        "source_label_parser_cases": {name: sorted(value)
                                      for name, value in parser_cases.items()},
        "primary_route_pattern_checks": {
            condition: [{route: int(count) for route, count in profile.items()}
                        for profile in profiles]
            for condition, profiles in route_counts.items()
        },
        "unique_image_conditions": pilot,
        "participant_id_overlap": participant_overlap_audit(),
    }


def participant_overlap_audit() -> dict:
    dataset_files = {
        name: pd.read_csv(BASE / "data" / name, keep_default_na=False)
        for name in ("behavior_NRA_group.csv", "behavior_RA_group.csv",
                     "behavior_XAI_presented_beforeAndAfter_trustViolation.csv")
    }
    dataset_ids = {name: set(frame["userID"]) for name, frame in dataset_files.items()}
    nra, ra, response_232 = [dataset_ids[name] for name in dataset_files]
    pilot_ids = set(pd.read_excel(PRIMARY_FILE, keep_default_na=False)["user_identifier"])
    policy_overlaps = {
        name: len(pilot_ids & identifiers)
        for name, identifiers in dataset_ids.items()
    }
    if len(pilot_ids) != 60 or len(response_232) != 232 or any(policy_overlaps.values()):
        raise ValueError("The policy and 232-person response samples must have distinct participant IDs")
    return {
        "participant_counts": {name: len(ids) for name, ids in dataset_ids.items()},
        "pairwise_overlap_counts": {
            "NRA_with_232_person_file": len(nra & response_232),
            "RA_with_232_person_file": len(ra & response_232),
            "NRA_with_RA": len(nra & ra),
        },
        "union_across_three_behavior_csvs": len(nra | ra | response_232),
        "policy_participant_overlap_counts": policy_overlaps,
        "policy_sample_disjoint_from_232_person_file": not bool(pilot_ids & response_232),
    }


def validate_all_policy_kernels(components: list[list[dict]], reading_limit: int,
                                capacities: list[int]) -> dict:
    checks = 0
    largest_fixed_error = 0.0
    largest_stopping_error = 0.0
    for row in components:
        for component in row:
            population = component["population"]
            if population == 0:
                continue
            targets = set(np.flatnonzero(component["eligible_target_mask"]))
            referrals = set(np.flatnonzero(component["secondary_referrals"]))
            for capacity in capacities:
                for reads in range(min(reading_limit, population) + 1):
                    subsets = list(combinations(range(population), reads))
                    fixed_probability = 0.0
                    stop_probability = 0.0
                    for selection in subsets:
                        selected = set(selection)
                        positive = selected & referrals
                        if targets <= referrals and targets <= selected and \
                                component["mandatory"] + len(positive) <= capacity:
                            fixed_probability += 1 / len(subsets)
                        if targets - referrals or not targets <= selected:
                            continue
                        retained_count = min(
                            len(positive), max(0, capacity - component["mandatory"]))
                        if len(targets) > retained_count:
                            continue
                        total = math.comb(len(positive), retained_count)
                        if total == 0:
                            continue
                        favorable = sum(targets <= set(retained)
                                         for retained in combinations(positive, retained_count))
                        stop_probability += favorable / total / len(subsets)
                    largest_fixed_error = max(largest_fixed_error, abs(
                        fixed_probability - fixed_success(component, reads, capacity)))
                    largest_stopping_error = max(largest_stopping_error, abs(
                        stop_probability - stop_success(component, reads, capacity)))
                    checks += 1
    if max(largest_fixed_error, largest_stopping_error) > 1e-12:
        raise AssertionError("Image-unit kernels disagree with direct subset enumeration")
    return {
        "method": "Every selected subset; stop success enumerates retained-referral subsets",
        "conditional_kernel_checks": checks,
        "largest_fixed_error": largest_fixed_error,
        "largest_stopping_error": largest_stopping_error,
        "passed": True,
    }


def validate_sequence_stopping_resources() -> dict:
    checks = 0
    largest_read_error = 0.0
    largest_search_error = 0.0
    for population in range(1, 8):
        for referral_count in range(population + 1):
            referrals = np.zeros(population, dtype=bool)
            referrals[:referral_count] = True
            for slots in range(referral_count + 1):
                for reading_limit in range(population + 1):
                    component = {
                        "population": population,
                        "mandatory": 0,
                        "secondary_referrals": referrals,
                    }
                    analytic_reads, analytic_searches = expected_stopping_resources(
                        component, reading_limit, slots)
                    enumerated_reads = 0
                    enumerated_searches = 0
                    orders = 0
                    for order in permutations(range(population)):
                        used = 0
                        reads = 0
                        if slots > 0:
                            for image in order[:reading_limit]:
                                reads += 1
                                if referrals[image]:
                                    used += 1
                                    if used == slots:
                                        break
                        enumerated_reads += reads
                        enumerated_searches += used
                        orders += 1
                    read_error = abs(analytic_reads - enumerated_reads / orders)
                    search_error = abs(analytic_searches - enumerated_searches / orders)
                    if read_error > 1e-12 or search_error > 1e-12:
                        raise AssertionError(
                            "Stopping workload equations failed literal sequence "
                            f"enumeration: N={population}, referrals={referral_count}, "
                            f"slots={slots}, read_limit={reading_limit}, "
                            f"analytic=({analytic_reads:.16g},{analytic_searches:.16g}), "
                            f"enumerated=({enumerated_reads / orders:.16g},"
                            f"{enumerated_searches / orders:.16g})"
                        )
                    largest_read_error = max(largest_read_error, read_error)
                    largest_search_error = max(largest_search_error, search_error)
                    checks += 1
    if max(largest_read_error, largest_search_error) > 1e-12:
        raise AssertionError(
            "Stopping workload equations failed literal sequence enumeration: "
            f"mean-reading error={largest_read_error:.16g}, "
            f"mean-search error={largest_search_error:.16g}"
        )
    return {
        "exhaustive_sequence_cases": checks,
        "largest_mean_reading_error": largest_read_error,
        "largest_mean_search_error": largest_search_error,
        "passed": True,
    }


def copula_cdf(u: float, v: float, family: str,
               parameter: float, rotation: int = 0) -> float:
    if u <= 0 or v <= 0:
        return 0.0
    if u >= 1:
        return v
    if v >= 1:
        return u
    if abs(parameter) < 1e-10:
        return u * v
    def base(x: float, y: float) -> float:
        if family == "Frank":
            return -math.log1p(math.expm1(-parameter * x) *
                               math.expm1(-parameter * y) /
                               math.expm1(-parameter)) / parameter
        if family == "Clayton":
            departure = (math.expm1(-parameter * math.log(x)) +
                         math.expm1(-parameter * math.log(y)))
            return math.exp(-math.log1p(departure) / parameter)
        raise ValueError(f"Unknown copula family: {family}")
    if rotation == 0:
        return base(u, v)
    if rotation == 90:
        return v - base(1 - u, v)
    if rotation == 180:
        return u + v - 1 + base(1 - u, 1 - v)
    if rotation == 270:
        return u - base(u, 1 - v)
    raise ValueError(f"Unknown rotation: {rotation}")


def copula_matrix(left: np.ndarray, right: np.ndarray, family: str,
                  parameter: float, rotation: int = 0) -> np.ndarray:
    left_cdf = np.r_[0.0, np.cumsum(left)]
    right_cdf = np.r_[0.0, np.cumsum(right)]
    left_cdf[-1] = right_cdf[-1] = 1.0
    corners = np.array([
        [copula_cdf(float(u), float(v), family, parameter, rotation)
         for v in right_cdf] for u in left_cdf
    ])
    result = np.diff(np.diff(corners, axis=0), axis=1)
    if result.min() < -2e-9:
        raise ArithmeticError("Copula rectangle masses became negative")
    result = np.maximum(result, 0.0)
    if not np.allclose(result.sum(axis=1), left, atol=2e-9) or \
            not np.allclose(result.sum(axis=0), right, atol=2e-9):
        raise ArithmeticError("Copula mass does not preserve the response margins")
    return result


def joint_policy_risk_upper(policy: dict, case: dict,
                            reading_limit: int, capacity: int,
                            stopping: bool) -> float:
    components = policy["components"]
    risk = np.asarray([[
        referral_miss(
            components[first][second], reading_limit, int(case["target_count"]),
            stopping=stopping, total_capacity=capacity if stopping else None,
        )
        for second in range(len(components[first]))
    ] for first in range(len(components))])
    return transport_extreme(risk, policy["left_weights"], policy["right_weights"],
                             maximize=True)


def copula_extremes(case: dict, policy: dict, grid_points: int,
                    refinements: int) -> pd.DataFrame:
    condition = policy["condition"]
    capacity_row = next((row for row in policy["capacity_rows"]
                         if row["stop_lower"] >= 0.8 - 1e-12 and
                         joint_policy_risk_upper(
                             policy, case, row["common_reading_limit"],
                             row["total_search_capacity"], True,
                         ) <= MEAN_MISS_REQUIREMENT + 1e-12), None)
    if capacity_row is None:
        capacity_row = policy["capacity_rows"][-1]
    capacity = capacity_row["total_search_capacity"]
    components = policy["components"]
    limit = int(max(part["population"] for row in components for part in row))
    effect = np.empty((len(components), len(components[0])))
    for first in range(effect.shape[0]):
        for second in range(effect.shape[1]):
            part = components[first][second]
            effect[first, second] = stop_success(part, limit, capacity) - fixed_success(
                part, limit, capacity)
    primary_loads = policy["primary_loads"]
    secondary_loads = policy["secondary_loads"]
    first_order = sorted(range(len(primary_loads)), key=lambda index: (
        int(primary_loads[index]),
        tuple(policy["components"][index][0]["eligible_target_mask"].astype(int)),
    ))
    second_order = sorted(range(len(secondary_loads)), key=lambda index: (
        int(secondary_loads[index]),
        tuple(policy["components"][0][index]["secondary_referrals"].astype(int)),
    ))
    effect = effect[np.ix_(first_order, second_order)]
    left = policy["left_weights"][first_order]
    right = policy["right_weights"][second_order]
    settings = (
        ("Frank", -20.0, 20.0, 0), ("Clayton", 0.0, 10.0, 0),
        ("Clayton", 0.0, 10.0, 90), ("Clayton", 0.0, 10.0, 180),
        ("Clayton", 0.0, 10.0, 270),
    )
    rows = []
    for family, low, high, rotation in settings:
        parameters = np.linspace(low, high, grid_points)
        values = np.empty(grid_points)
        for index, parameter in enumerate(parameters):
            values[index] = np.sum(effect * copula_matrix(
                left, right, family, float(parameter), rotation))
        candidates_min = [(float(values.min()), float(parameters[values.argmin()]))]
        candidates_max = [(float(values.max()), float(parameters[values.argmax()]))]
        start_points = np.linspace(low, high, refinements + 1)
        for left_edge, right_edge in zip(start_points[:-1], start_points[1:]):
            if left_edge == low and right_edge == high:
                continue
            result_min = minimize_scalar(
                lambda parameter: float(np.sum(effect * copula_matrix(
                    left, right, family, float(parameter), rotation))),
                bounds=(float(left_edge), float(right_edge)), method="bounded",
                options={"xatol": 1e-9},
            )
            result_max = minimize_scalar(
                lambda parameter: -float(np.sum(effect * copula_matrix(
                    left, right, family, float(parameter), rotation))),
                bounds=(float(left_edge), float(right_edge)), method="bounded",
                options={"xatol": 1e-9},
            )
            if not (result_min.success and result_max.success):
                raise ArithmeticError(f"Copula refinement failed for {condition}, {family}")
            candidates_min.append((float(result_min.fun), float(result_min.x)))
            candidates_max.append((float(-result_max.fun), float(result_max.x)))
        min_value, min_parameter = min(candidates_min)
        max_value, max_parameter = max(candidates_max)
        rows.append({
            "condition": condition, "search_capacity": capacity,
            "common_maximum_readings": limit,
            "mean_target_miss_requirement": MEAN_MISS_REQUIREMENT,
            "stopping_mean_miss_upper_at_capacity": joint_policy_risk_upper(
                policy, case, limit, capacity, True,
            ),
            "dependence_family": family if rotation == 0 else f"Clayton {rotation} degree",
            "parameter_lower": low, "parameter_upper": high,
            "dense_grid_points": grid_points,
            "bounded_refinement_starts": refinements,
            "numerical_minimum_stop_minus_fixed": min_value,
            "minimum_parameter": min_parameter,
            "numerical_maximum_stop_minus_fixed": max_value,
            "maximum_parameter": max_parameter,
            "search_classification": "Numerically searched extrema; no analytic global certificate",
        })
    return pd.DataFrame(rows)


def participant_bootstrap(case: dict, policy: dict, repetitions: int) -> pd.DataFrame:
    matching = next((row for row in policy["capacity_rows"]
                     if row["stop_lower"] >= 0.8 - 1e-12 and
                     joint_policy_risk_upper(
                         policy, case, row["common_reading_limit"],
                         row["total_search_capacity"], True,
                     ) <= MEAN_MISS_REQUIREMENT + 1e-12),
                    policy["capacity_rows"][-1])
    capacity = int(matching["total_search_capacity"])
    maximum_readings = max(part["population"]
                           for row in policy["components"] for part in row)
    stop = np.empty((10, 10))
    fixed = np.empty((10, 10))
    for first in range(10):
        for second in range(10):
            part = policy["components"][first][second]
            stop[first, second] = stop_success(part, maximum_readings, capacity)
            fixed[first, second] = fixed_success(part, maximum_readings, capacity)
    rng = np.random.default_rng(SEED + (1 if case["condition"] == "false_miss" else 2))
    samples = np.empty((repetitions, 4), dtype=float)
    uniform = np.full(10, .1)
    for repetition in range(repetitions):
        left_counts = np.bincount(rng.integers(0, 10, 10), minlength=10) / 10
        right_counts = np.bincount(rng.integers(0, 10, 10), minlength=10) / 10
        samples[repetition, 0], _ = transport_bounds(stop, left_counts, right_counts)
        samples[repetition, 1], _ = transport_bounds(fixed, left_counts, right_counts)
        samples[repetition, 2], samples[repetition, 3] = transport_bounds(
            stop - fixed, left_counts, right_counts)
    nominal = {
        "stop_lower": transport_bounds(stop, uniform, uniform)[0],
        "fixed_lower": transport_bounds(fixed, uniform, uniform)[0],
        "effect_lower": transport_bounds(stop - fixed, uniform, uniform)[0],
        "effect_upper": transport_bounds(stop - fixed, uniform, uniform)[1],
    }
    rows = []
    labels = ["stopping lower bound", "full fixed-count lower bound",
              "paired effect lower bound", "paired effect upper bound"]
    for column, label in enumerate(labels):
        rows.append({
            "condition": case["condition"], "resampling_unit": "Whole participant profiles",
            "bootstrap_replicates": repetitions,
            "comparison_capacity": capacity,
            "maximum_reading_allowance": maximum_readings,
            "mean_target_miss_limit": MEAN_MISS_REQUIREMENT,
            "metric": label,
            "observed_value": nominal[
                "stop_lower" if column == 0 else "fixed_lower" if column == 1
                else "effect_lower" if column == 2 else "effect_upper"],
            "bootstrap_2_5": float(np.quantile(samples[:, column], .025)),
            "bootstrap_median": float(np.quantile(samples[:, column], .5)),
            "bootstrap_97_5": float(np.quantile(samples[:, column], .975)),
            "interpretation": "Participant-margin stability range; response-pair dependence remains separately optimized",
        })
    return pd.DataFrame(rows)


def write_full(cases: dict, bootstrap_replicates: int,
               copula_grid_points: int, copula_refinements: int) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, thresholds, boots, copulas, initial_copulas, images = [], [], [], [], [], []
    kernel_checks = {}
    for name, case in cases.items():
        case["condition"] = name
        result = scenario_matrices(case)
        kernel_checks[name] = result["kernel_validation"]
        summaries.extend(result["capacity_rows"])
        thresholds.extend(result["capacity_thresholds"])
        boots.append(participant_bootstrap(case, result, bootstrap_replicates))
        initial_copulas.append(copula_extremes(case, result, 20001, 20))
        copulas.append(copula_extremes(
            case, result, copula_grid_points, copula_refinements))
        for image, target, machine in zip(case["image_ids"], case["targets"], case["machine"]):
            images.append({"condition": name, "image": image, "target_gun_or_knife": bool(target),
                           "machine_response_contains_prohibited_item": bool(machine)})
    point, interval = build_external_response_summaries(bootstrap_replicates)
    class_summary, composition = sixray_summary()
    pd.DataFrame(summaries).to_csv(OUT / "policy_capacity.csv", index=False)
    pd.DataFrame(thresholds).to_csv(OUT / "capacity_thresholds.csv", index=False)
    pd.concat(boots, ignore_index=True).to_csv(OUT / "pilot_participant_bootstrap.csv", index=False)
    pd.concat(copulas, ignore_index=True).to_csv(OUT / "copula_numeric_search.csv", index=False)
    previous_search_path = OUT / "copula_numeric_search_20001_precheck.csv"
    pd.concat(initial_copulas, ignore_index=True).to_csv(previous_search_path, index=False)
    if previous_search_path.exists():
        previous_search = pd.read_csv(previous_search_path)
        current_search = pd.concat(copulas, ignore_index=True)
        current_search = current_search.merge(
            previous_search[[
                "condition", "dependence_family", "numerical_minimum_stop_minus_fixed",
                "minimum_parameter", "numerical_maximum_stop_minus_fixed",
                "maximum_parameter", "dense_grid_points",
                "bounded_refinement_starts",
            ]], on=["condition", "dependence_family"], suffixes=("_fine", "_initial"),
            validate="one_to_one",
        )
        convergence = pd.DataFrame({
            "condition": current_search["condition"],
            "dependence_family": current_search["dependence_family"],
            "initial_grid_points": current_search["dense_grid_points_initial"],
            "initial_refinement_intervals": current_search[
                "bounded_refinement_starts_initial"],
            "refined_grid_points": current_search["dense_grid_points_fine"],
            "refined_start_intervals": current_search[
                "bounded_refinement_starts_fine"],
            "initial_minimum_effect_pp":
                100 * current_search["numerical_minimum_stop_minus_fixed_initial"],
            "refined_minimum_effect_pp":
                100 * current_search["numerical_minimum_stop_minus_fixed_fine"],
            "minimum_endpoint_change_pp": 100 * abs(
                current_search["numerical_minimum_stop_minus_fixed_fine"] -
                current_search["numerical_minimum_stop_minus_fixed_initial"]),
            "initial_maximum_effect_pp":
                100 * current_search["numerical_maximum_stop_minus_fixed_initial"],
            "refined_maximum_effect_pp":
                100 * current_search["numerical_maximum_stop_minus_fixed_fine"],
            "maximum_endpoint_change_pp": 100 * abs(
                current_search["numerical_maximum_stop_minus_fixed_fine"] -
                current_search["numerical_maximum_stop_minus_fixed_initial"]),
            "initial_minimum_parameter": current_search["minimum_parameter_initial"],
            "refined_minimum_parameter": current_search["minimum_parameter_fine"],
            "initial_maximum_parameter": current_search["maximum_parameter_initial"],
            "refined_maximum_parameter": current_search["maximum_parameter_fine"],
        })
        convergence.to_csv(OUT / "copula_search_convergence.csv", index=False)
    (OUT / "policy_kernel_validation.json").write_text(
        json.dumps(kernel_checks, indent=2), encoding="utf-8")
    pd.DataFrame(images).to_csv(OUT / "image_unit_map.csv", index=False)
    point.to_csv(OUT / "external_response_summary.csv", index=False)
    interval.to_csv(OUT / "external_participant_bootstrap.csv", index=False)
    class_summary.to_csv(OUT / "sixray_class_image_summary.csv", index=False)
    (OUT / "sixray_image_composition.json").write_text(
        json.dumps(composition, indent=2), encoding="utf-8")
    overlap_audit = participant_overlap_audit()
    (OUT / "public_dataset_overlap_audit.json").write_text(
        json.dumps(overlap_audit, indent=2), encoding="utf-8")
    result = {
        "status": "completed",
        "source_sha256": {path.name: sha256(path) for path in
                          (PRIMARY_FILE, STIMULUS_FILE, EXTERNAL_FILE, SIXRAY_FILE)},
        "pilot_conditions": {
            name: {
                "unique_images": case["image_count"],
                "gun_or_knife_images": case["target_count"],
                "eligible_images_by_first_reader": [int(np.sum(profile["route"] == "U"))
                                                     for profile in case["primary"]],
                "first_stage_routes_by_first_reader": [
                    {route: int(count) for route, count in
                     Counter(profile["route"]).items()}
                    for profile in case["primary"]
                ],
            }
            for name, case in cases.items()
        },
        "new_public_participant_sample": {
            "participants": 232, "images": 42, "trials": 9048,
            "trials_per_participant": 39,
            "behavioral_outcome": "Task correctness and truth-label request only",
            "participant_overlap_with_policy_sample": 0,
            "policy_capacity_role": "Descriptive response evidence; policy capacities use the 20 schedule profiles",
        },
        "SIXray": composition,
        "resampling": {
            "replicates": bootstrap_replicates,
            "unit": "Whole participants resampled within recorded response and visualization groups",
            "order_or_pairing_guarantee": "Not inferred from bootstrap quantiles",
        },
        "independent_kernel_checks": kernel_checks,
        "sequence_resource_checks": validate_sequence_stopping_resources(),
        "numerical_search": {
            "copula_parameter_grid_points_per_family": copula_grid_points,
            "bounded_refinement_intervals_per_family": copula_refinements,
            "global_certificate": False,
        },
    }
    (OUT / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--full", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--copula-grid", type=int, default=COPULA_GRID_POINTS)
    parser.add_argument("--copula-refinements", type=int, default=40)
    args = parser.parse_args()
    cases = prepare_cases()
    if args.smoke:
        result = smoke_check(cases)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    smoke_path = OUT / "smoke.json"
    if not smoke_path.exists():
        raise FileNotFoundError("Run --smoke successfully before the full calculation")
    previous = json.loads(smoke_path.read_text(encoding="utf-8"))
    if not previous.get("passed"):
        raise ArithmeticError("The saved smoke verification did not pass")
    if args.bootstrap < 1 or args.copula_grid < 201 or args.copula_refinements < 2:
        raise ValueError("Full runs require positive resampling and at least 201 copula grid points")
    result = write_full(cases, args.bootstrap, args.copula_grid,
                        args.copula_refinements)
    print(json.dumps({"status": result["status"],
                      "conditions": result["pilot_conditions"],
                      "external_sample": result["new_public_participant_sample"],
                      "SIXray": result["SIXray"]}, indent=2))


if __name__ == "__main__":
    main()
