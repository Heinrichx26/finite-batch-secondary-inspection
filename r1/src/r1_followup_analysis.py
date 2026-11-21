from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product

import numpy as np
import pandas as pd
from mpmath import iv
from openpyxl import load_workbook
from scipy.optimize import linprog
from scipy.stats import beta, binom

from compare_sequential_policies_20261018 import (
    Controller, initial_belief, load_case, normalize, remaining_by_group, terminal_value, transitions,
)
from r1_image_unit_analysis import (
    BASE, EXTERNAL_FILE, OUT as IMAGE_OUT, PRIMARY_FILE, STIMULUS_FILE, copula_matrix, fixed_success, object_set,
    prepare_cases, response_components, sha256, stop_success, transport_bounds,
)
from recent_method_comparison import FAMILIES, continuous_family_extrema
from verify_r1_capacity import extreme


OUT = BASE / "results" / "R1_followup"
FIELD_FILE = BASE / "data" / "checkpoint" / "security_checkpoint_data.xlsx"
TARGET_LABELS = {"gun", "knife"}
COPULA_TOLERANCE = Fraction(1, 1_000_000)
iv.dps = 40


def choose(population: int, sample: int) -> int:
    return math.comb(population, sample) if 0 <= sample <= population else 0


def counting_outcomes(population, positives, n_read, slots, stopping, n_target_records=2):
    samples = choose(population, n_read)
    target_placements = choose(positives, n_target_records)
    both = either = missed_records = missed_image = Fraction(0)
    for selected in range(positives + 1):
        mass = Fraction(choose(positives, selected)
                        * choose(population - positives, n_read - selected), samples)
        admitted = min(selected, slots) if stopping else selected
        neither = Fraction(choose(positives - admitted, n_target_records), target_placements)
        missed_image += mass * neither
        missed_records += mass * (n_target_records - Fraction(n_target_records * admitted, positives))
        if admitted <= slots:
            both += mass * Fraction(choose(admitted, n_target_records), target_placements)
            either += mass * (1 - neither)
    return both, either, missed_records, missed_image


def audit_trial():
    trial = pd.read_excel(PRIMARY_FILE, keep_default_na=False)
    trial = trial.loc[trial["ImageNumber"].ne("")].copy()
    trial["image"] = trial["ImageNumber"].str.casefold().str.removesuffix(".jpg")
    trial["target"] = trial["ground truth"].map(lambda value: bool(object_set(value) & TARGET_LABELS))
    trial["machine"] = trial["model result"].map(lambda value: bool(object_set(value)))
    trial["referral"] = trial["UserDecision"].map(lambda value: bool(object_set(value))) | (
        trial["OpenBag"].astype(str).str.casefold().eq("true"))
    return trial


def repeated_target_analysis(trial):
    assisted = trial.loc[trial["GROUP"].eq("AI")]
    unaided = trial.loc[trial["GROUP"].eq("NoAI")]
    mandatory = assisted["machine"] | assisted["referral"]
    pool = assisted.loc[~mandatory]
    assert (len(assisted), len(unaided), len(pool), int(assisted["target"].sum())) == (1040, 540, 244, 290)
    eligible_targets = pool.loc[pool["target"], "image"]
    assert len(eligible_targets) == 2 and eligible_targets.nunique() == 1
    profiles = unaided.pivot(index="user_identifier", columns="image", values="referral")
    memberships = set(unaided.groupby("image")["user_identifier"].agg(frozenset))
    groups = [profiles.loc[sorted(members)].dropna(axis=1, how="all")
              for members in sorted(memberships, key=sorted)]
    assert len(groups) == 2 and all(group.shape == (10, 27) for group in groups)
    assert set(groups[0]).isdisjoint(groups[1])
    target_referrals = unaided.loc[unaided["image"].eq(eligible_targets.iloc[0]), "referral"]
    assert len(target_referrals) == 10 and target_referrals.all()
    counts = pool["image"].value_counts()
    loads = [group.astype(int).dot(counts.reindex(group.columns, fill_value=0)).tolist() for group in groups]
    recorded_effect = np.asarray([
        counting_outcomes(len(pool), first + second, 218, 111, False)[0]
        - counting_outcomes(len(pool), first + second, 244, 111, False)[0]
        for first, second in product(*loads)
    ], dtype=object).reshape(10, 10)
    recorded_effect = recorded_effect[np.ix_(*(np.argsort(group, kind="stable") for group in loads))]
    records = []
    for capacity in (907, 930):
        slots = capacity - int(mandatory.sum())
        for n_read in (218, 244):
            for policy in ("fixed", "stopping"):
                values = np.asarray([counting_outcomes(len(pool), left + right, n_read,
                                                       slots, policy == "stopping")
                                     for left, right in product(*loads)], dtype=float).reshape(10, 10, 4)
                separate, combined, record_miss, image_miss = values.transpose(2, 0, 1)
                collapsed = np.asarray([counting_outcomes(len(pool) - 1, left + right - 1,
                                                          min(n_read, len(pool) - 1), slots,
                                                          policy == "stopping", n_target_records=1)[::3]
                                        for left, right in product(*loads)], dtype=float).reshape(10, 10, 2)
                distinct_search, distinct_miss = collapsed.transpose(2, 0, 1)
                difference = combined - separate
                assert difference.min() >= 0
                if policy == "fixed":
                    fixed = separate
                effect = separate - fixed
                assert effect.min() >= -1e-12
                records.append({
                    "search_capacity": capacity, "reading_limit": n_read, "policy": policy,
                    "two_targets_lower": extreme(separate), "one_target_lower": extreme(combined),
                    "paired_gain_lower": extreme(difference), "paired_gain_upper": extreme(difference, True),
                    "two_targets_mean_miss_upper": extreme(record_miss, True) / 290,
                    "one_target_mean_miss_upper": extreme(image_miss, True) / 289,
                    "one_reading_and_search_lower": extreme(distinct_search),
                    "one_reading_and_search_mean_miss_upper": extreme(distinct_miss, True) / 289,
                    "stop_minus_matched_fixed_lower": extreme(effect),
                    "stop_minus_matched_fixed_upper": extreme(effect, True),
                })
    assert np.isclose(records[0]["two_targets_lower"], 0.5372, atol=0.00005, rtol=0)
    assert np.isclose(records[2]["two_targets_lower"], 0.5, rtol=0)
    assert np.isclose(records[3]["two_targets_lower"], 0.8745, atol=0.00005, rtol=0)
    return records, recorded_effect


def schedule_workloads(trial):
    records = []
    cohorts = trial.loc[trial["GROUP"].eq("AI")].groupby(["type_of_tv", "num_of_tv"])
    for (condition, schedule), cohort in cohorts:
        per_person_images = cohort.groupby("user_identifier")["image"].agg(frozenset)
        assert per_person_images.nunique() == 1 and len(per_person_images) == 10
        other = cohorts.get_group((condition, "tv1" if schedule == "tv3" else "tv3"))
        assert not set(cohort["user_identifier"]) & set(other["user_identifier"])
        mandatory = cohort["machine"] | cohort["referral"]
        mandatory_count = cohort.assign(mandatory=mandatory).groupby("user_identifier")["mandatory"].sum()
        optional_targets = cohort.assign(eligible=~mandatory & cohort["target"]).groupby("user_identifier")["eligible"].sum()
        target_images = cohort.loc[cohort["target"], "image"].nunique()
        records.append({"condition": condition, "machine_errors": int(schedule[-1]),
                        "participants": len(per_person_images), "images": len(per_person_images.iloc[0]),
                        "shared_images": len(per_person_images.iloc[0] & frozenset(other["image"])),
                        "target_images": target_images, "mandatory_min": int(mandatory_count.min()),
                        "mandatory_max": int(mandatory_count.max()),
                        "profiles_with_optional_target": int(optional_targets.gt(0).sum()),
                        "optional_target_images": int(cohort.loc[~mandatory & cohort["target"], "image"].nunique())})
    return records


def panel_target_outcomes(judgments, target_count):
    target_judgments = judgments.loc[judgments["target"]]
    per_person = target_judgments.groupby("userID")["userCorrect"].agg(["sum", "size"])
    if len(per_person) != judgments["userID"].nunique() or not per_person["size"].ge(target_count).all():
        raise ValueError("Each participant must have enough target judgments for the comparison")
    standardized = [Fraction(choose(correct, target_count), choose(total, target_count))
                    for correct, total in per_person.itertuples(index=False, name=None)]
    return {"participants": len(per_person), "min_target_images": int(per_person["size"].min()),
            "max_target_images": int(per_person["size"].max()),
            "target_judgments": len(target_judgments),
            "correct_target_judgments": int(per_person["sum"].sum()),
            "all_correct_profiles": int(per_person["sum"].eq(per_person["size"]).sum()),
            "standardized_target_count": target_count,
            "standardized_all_correct_probability": float(sum(standardized) / len(standardized))}


def external_panel_analysis(trial):
    responses = pd.read_csv(EXTERNAL_FILE, keep_default_na=False)
    images = pd.read_csv(STIMULUS_FILE, keep_default_na=False)
    images["image"] = images["filename"].str.casefold().str.removesuffix(".jpg")
    target_labels = images.set_index("image")["ground truth"].map(
        lambda value: bool(object_set(value) & TARGET_LABELS))
    responses["image"] = responses["imageID"].str.casefold().str.removesuffix(".jpg")
    panel = responses.loc[responses["beforeOrAfterTV"].eq("afterTV")].copy()
    panel["target"] = panel["image"].map(target_labels)
    assert len(panel) == 5568 and panel["userID"].nunique() == 232
    assert not set(panel["userID"]) & set(trial["user_identifier"])
    assert not panel.duplicated(["userID", "image"]).any() and panel["target"].notna().all()
    assert panel["userCorrect"].isin([0, 1]).all()
    assert panel.groupby("userID")["image"].size().eq(24).all()
    return [{"condition": condition, "images_per_participant": 24,
             **panel_target_outcomes(group, target_count=3)} for condition, group in panel.groupby("type_of_tv")]


def quota_program(sizes, slots, scenarios):
    quotas = [candidate for candidate in product(*(range(size + 1) for size in sizes))
              if sum(candidate) <= slots]
    payoffs = np.array([[min(quota[group], positives[group]) / positives[group] for quota in quotas]
                        for positives, group in scenarios])
    result = linprog(np.r_[np.zeros(len(quotas)), -1.],
                     A_ub=np.c_[-payoffs, np.ones(len(payoffs))], b_ub=np.zeros(len(payoffs)),
                     A_eq=[np.r_[np.ones(len(quotas)), 0.]], b_eq=[1.],
                     bounds=[(0, 1)] * (len(quotas) + 1), method="highs")
    if not result.success:
        raise ArithmeticError(result.message)
    return float(result.x[-1]), quotas, result.x[:-1]


def image_difficulty(trial):
    source = trial.loc[trial["GROUP"].eq("NoAI")]
    assert source.groupby("image")["difficulty level"].nunique().max() == 1
    difficulty = source.drop_duplicates("image").set_index("image")["difficulty level"]
    assert set(difficulty) == {"easy", "difficult"}
    return difficulty


def quota_analysis(case, difficulty):
    image_groups = difficulty.loc[case["image_ids"]].to_numpy()
    data = []
    for primary in case["primary"]:
        eligible = primary["route"] == "U"
        if (eligible & case["targets"]).sum() != 1:
            continue
        masks = [eligible & (image_groups == group) for group in ("difficult", "easy")]
        sizes = tuple(int(mask.sum()) for mask in masks)
        target_group = next(group for group, mask in enumerate(masks) if (mask & case["targets"]).any())
        loads = [tuple(int((mask & secondary["referral"]).sum()) for mask in masks)
                 for secondary in case["secondary"]]
        assert all(load[target_group] > 0 for load in loads)
        for capacity in (24, 25):
            mandatory = int((~eligible).sum())
            slots = capacity - mandatory
            guarantee, quotas, _ = quota_program(sizes, slots,
                                                 [(sizes, group) for group in range(len(sizes))])
            assert np.isclose(guarantee, min(slots, sum(sizes)) / sum(sizes))
            total_quota = min(slots, sum(sizes))
            weights = [math.prod(choose(size, quota) for size, quota in zip(sizes, candidate))
                       / choose(sum(sizes), total_quota) if sum(candidate) == total_quota else 0.0
                       for candidate in quotas]
            assert np.isclose(sum(weights), 1.0)
            supported = [min(quota[target_group], load[target_group]) / load[target_group]
                         for load in loads for quota in quotas]
            fixed_quotas = min(np.array(supported).reshape(len(loads), len(quotas)) @ weights)
            pooled = min(min(slots, sum(load)) / sum(load) for load in loads)
            data.append({"mandatory": mandatory, "difficult_images": sizes[0], "easy_images": sizes[1],
                         "search_capacity": capacity, "available_searches": slots,
                         "placement_guarantee": guarantee, "quota_replay_lower": float(fixed_quotas),
                         "pooled_replay_lower": pooled})
    assert len(data) == 4
    return data


def target_coverage_by_order(target_referral, group_referrals, earlier_referrals, slots):
    if not target_referral:
        return Fraction(0)
    if slots < 0 or group_referrals <= 0 or earlier_referrals < 0:
        raise ValueError("Invalid target-referral counts")
    remaining = max(0, slots - earlier_referrals)
    return Fraction(min(remaining, group_referrals), group_referrals)


def difficulty_order_analysis(cases, difficulty):
    results = []
    for name, capacities in (("false_miss", (24, 25)), ("false_alarm", (24,))):
        case = cases[name]
        labels = difficulty.loc[case["image_ids"]].to_numpy()
        for capacity in capacities:
            for order, priority in (("uniform", None), ("difficult_first", "difficult"),
                                    ("easy_first", "easy")):
                success, risk = [], []
                for primary in case["primary"]:
                    eligible = primary["route"] == "U"
                    target = eligible & case["targets"]
                    target_count = int(target.sum())
                    slots = capacity - int((~eligible).sum())
                    assert slots >= 0 and target_count <= 1
                    success.append([])
                    risk.append([])
                    for secondary in case["secondary"]:
                        coverage = Fraction(1)
                        if target_count:
                            positives = eligible & secondary["referral"]
                            target_group = (np.ones(len(eligible), dtype=bool) if priority is None
                                            else labels == labels[target][0])
                            earlier = (int((positives & ~target_group).sum())
                                       if priority is not None and labels[target][0] != priority else 0)
                            coverage = target_coverage_by_order(
                                bool((positives & target).any()),
                                int((positives & target_group).sum()), earlier, slots,
                            )
                        success[-1].append(float(coverage))
                        risk[-1].append(float((1 - coverage) / case["target_count"]))
                results.append({"condition": name, "search_capacity": capacity,
                                "reading_limit": max(int((item["route"] == "U").sum())
                                                     for item in case["primary"]),
                                "order": order, "success_lower": extreme(np.array(success)),
                                "mean_unreferred_upper": extreme(np.array(risk), True)})
    return results


def full_reading_effect(part, capacity):
    slots = capacity - part["mandatory"]
    targets = part["required_targets"]
    referrals = targets + part["non_target_referrals"]
    if slots < 0 or not part["all_targets_referred"] or slots >= referrals:
        return Fraction(0)
    return Fraction(choose(slots, targets), choose(referrals, targets))


def sorted_full_effect(case, capacity):
    components = [[response_components(case, first, second) for second in range(len(case["secondary"]))]
                  for first in range(len(case["primary"]))]
    primary_order = sorted(range(len(components)), key=lambda index: (
        components[index][0]["mandatory"], tuple(components[index][0]["eligible_target_mask"]),
    ))
    secondary_order = sorted(range(len(components[0])), key=lambda index: (
        int(case["secondary"][index]["referral"].sum()), tuple(components[0][index]["secondary_referrals"]),
    ))
    reading_limit = max(part["population"] for row in components for part in row)
    observed = [[stop_success(part, reading_limit, capacity) - fixed_success(part, reading_limit, capacity)
                 for part in row] for row in components]
    exact = np.array([[full_reading_effect(part, capacity) for part in row] for row in components])
    assert np.allclose(observed, exact.astype(float), rtol=0, atol=1e-12)
    return exact[np.ix_(primary_order, secondary_order)], reading_limit


def copula_effect_terms(effect):
    rows, columns = effect.shape
    constant = effect[-1, -1]
    constant += sum((effect[row - 1, -1] - effect[row, -1]) * Fraction(row, rows)
                    for row in range(1, rows))
    constant += sum((effect[-1, col - 1] - effect[-1, col]) * Fraction(col, columns)
                    for col in range(1, columns))
    terms = []
    for row, col in product(range(1, rows), range(1, columns)):
        weight = effect[row - 1, col - 1] - effect[row, col - 1] - effect[row - 1, col] + effect[row, col]
        if weight:
            terms.append((Fraction(row, rows), Fraction(col, columns), weight))
    return constant, terms


def interval(value):
    return iv.mpf(value.numerator) / value.denominator


def interval_copula(first, second, family, parameter, rotation):
    if parameter == 0:
        return interval(first * second)
    theta = interval(parameter)

    def base(first_rank, second_rank):
        left, right = interval(first_rank), interval(second_rank)
        if family == "Frank":
            return -iv.ln(1 + (iv.exp(-theta * left) - 1) *
                          (iv.exp(-theta * right) - 1) / (iv.exp(-theta) - 1)) / theta
        if family == "Clayton":
            return iv.exp(-iv.ln(iv.exp(-theta * iv.ln(left)) +
                                 iv.exp(-theta * iv.ln(right)) - 1) / theta)
        raise ValueError(f"Unknown copula family: {family}")

    if rotation == 90:
        return interval(second) - base(1 - first, second)
    if rotation == 180:
        return interval(first + second - 1) + base(1 - first, 1 - second)
    if rotation == 270:
        return interval(first) - base(first, 1 - second)
    return base(first, second)


def enclose_copula_extrema(constant, terms, family, rotation, low, high, candidates, tolerance):
    offset = interval(constant)

    def points(parameter):
        values = tuple(interval(weight) * interval_copula(first, second, family, parameter, rotation)
                       for first, second, weight in terms)
        return values, offset + sum(values)

    left, right = points(low), points(high)
    observed = [left[1], right[1], *(points(parameter)[1] for parameter in candidates)]
    attained_min = min(item.b for item in observed)
    attained_max = max(item.a for item in observed)
    segments = [(low, high, left[0], right[0])]
    outer_min, outer_max = attained_min, attained_max
    tolerance = interval(tolerance).a
    checked = 0
    while segments:
        start, end, left_terms, right_terms = segments.pop()
        checked += 1
        paired = list(zip(left_terms, right_terms, strict=True))
        lower = (offset + sum(min(first.a, second.a) for first, second in paired)).a
        upper = (offset + sum(max(first.b, second.b) for first, second in paired)).b
        if lower >= (attained_min - tolerance).b and upper <= (attained_max + tolerance).a:
            outer_min = min(outer_min, lower)
            outer_max = max(outer_max, upper)
            continue
        if checked > 250_000:
            raise ArithmeticError("Copula enclosure did not reach the specified tolerance")
        midpoint = (start + end) / 2
        middle, value = points(midpoint)
        attained_min = min(attained_min, value.b)
        attained_max = max(attained_max, value.a)
        segments.append((start, midpoint, left_terms, middle))
        segments.append((midpoint, end, middle, right_terms))
    assert (attained_min - outer_min).b <= tolerance
    assert (outer_max - attained_max).b <= tolerance
    return outer_min, attained_min, attained_max, outer_max, checked


def enclosure_record(condition, dependence_family, numerical_min, numerical_max, enclosure, tolerance):
    outer_min, min_upper, max_lower, outer_max, checked = enclosure
    minimum_lower = math.nextafter(float(outer_min), -math.inf)
    minimum_upper = math.nextafter(float(min_upper), math.inf)
    maximum_lower = math.nextafter(float(max_lower), -math.inf)
    maximum_upper = math.nextafter(float(outer_max), math.inf)
    assert max(abs(numerical_min - minimum_lower), abs(numerical_min - minimum_upper),
               abs(numerical_max - maximum_lower), abs(numerical_max - maximum_upper)) <= float(tolerance)
    return {"condition": condition, "dependence_family": dependence_family,
            "minimum_lower": minimum_lower, "minimum_upper": minimum_upper,
            "maximum_lower": maximum_lower, "maximum_upper": maximum_upper,
            "numerical_minimum": numerical_min, "numerical_maximum": numerical_max,
            "parameter_subintervals_checked": checked}


def copula_enclosure_analysis(cases, recorded_effect):
    search = pd.read_csv(IMAGE_OUT / "copula_numeric_search.csv", float_precision="round_trip")
    records = []
    for condition, group in search.groupby("condition", sort=False):
        assert group["search_capacity"].nunique() == 1 and len(group) == 5
        capacity = int(group["search_capacity"].iloc[0])
        effect, readings = sorted_full_effect(cases[condition], capacity)
        constant, terms = copula_effect_terms(effect)
        for row in group.itertuples(index=False):
            assert row.common_maximum_readings == readings
            words = row.dependence_family.split()
            family = words[0]
            rotation = int(words[1]) if len(words) > 1 else 0
            low, high = (Fraction(str(value)) for value in (row.parameter_lower, row.parameter_upper))
            candidates = tuple(Fraction(str(value)) for value in (row.minimum_parameter, row.maximum_parameter))
            assert all(low <= parameter <= high for parameter in candidates)
            enclosure = enclose_copula_extrema(constant, terms, family, rotation, low, high,
                                               candidates, COPULA_TOLERANCE)
            records.append(enclosure_record(condition, row.dependence_family,
                                            row.numerical_minimum_stop_minus_fixed,
                                            row.numerical_maximum_stop_minus_fixed, enclosure, COPULA_TOLERANCE))
    assert len(records) == 10
    constant, terms = copula_effect_terms(recorded_effect)
    weights = np.full(10, 0.1)
    tolerance = Fraction(1, 20_000)
    for specification in FAMILIES:
        numerical = continuous_family_extrema(recorded_effect, weights, weights, specification)
        candidates = tuple(Fraction(str(numerical[key])) for key in ("lower_parameter", "upper_parameter"))
        low, high = (Fraction(value) for value in specification.bounds)
        enclosure = enclose_copula_extrema(constant, terms, specification.family, specification.rotation,
                                          low, high, candidates, tolerance)
        record = enclosure_record("recorded_entries", specification.name, numerical["lower"], numerical["upper"],
                                  enclosure, tolerance)
        assert record["maximum_upper"] < 0
        records.append(record)
    return records


def participant_deletion_analysis(cases):
    records = []
    for condition, capacity in (("false_miss", 25), ("false_alarm", 24)):
        effect, _ = sorted_full_effect(cases[condition], capacity)
        effect = effect.astype(float)
        for axis, group in enumerate(("assisted", "unaided")):
            for index in range(effect.shape[axis]):
                reduced = np.delete(effect, index, axis=axis)
                left, right = (np.full(size, 1 / size) for size in reduced.shape)
                lower, upper = transport_bounds(reduced, left, right)
                equal_margins = reduced.repeat(reduced.shape[1], axis=0).repeat(reduced.shape[0], axis=1)
                assert np.allclose((lower, upper), (extreme(equal_margins), extreme(equal_margins, True)),
                                   rtol=0, atol=1e-12)
                records.append({"condition": condition, "removed_from": group, "profile_rank": index + 1,
                                "lower_effect": lower, "upper_effect": upper})
    return records


def feedback_success(worlds, belief, read_limit, search_limit):
    @lru_cache(maxsize=None)
    def solve(state_belief, reads, searches):
        best = terminal_value(state_belief, worlds)[0]
        if reads == read_limit or searches == search_limit:
            return best
        for group, remaining in enumerate(remaining_by_group(state_belief)):
            if not remaining:
                continue
            branches = {}
            for state, weight in state_belief:
                for referral, next_state, probability in transitions(state, group):
                    observation = (referral, next_state[-1] - state[-1])
                    branches.setdefault(observation, []).append((next_state, weight * probability))
            value = sum(sum(mass for _, mass in branch) * solve(normalize(branch), reads + 1, searches + referral)
                        for (referral, _), branch in branches.items())
            best = max(best, value)
        return best

    return solve(belief, 0, 0)


def feedback_analysis(case_name):
    raw, worlds = load_case(case_name)
    belief = initial_belief(worlds, Fraction(1, 2))
    read_limit = raw["read_limit"]
    limits = [raw["search_limit"]]
    if case_name == "full":
        limits.append(3)
    rows = []
    for search_limit in limits:
        binary = Controller(worlds, read_limit, search_limit).solve(belief, 0, 0)[0]
        timely = feedback_success(worlds, belief, read_limit, search_limit)
        assert timely >= binary
        rows.append({"reading_limit": read_limit, "search_limit": search_limit,
                     "binary_referrals": float(binary), "immediate_search_findings": float(timely)})
    return rows


def clock_seconds(value):
    if value is None or str(value).strip() == "":
        return None
    digits = str(value).strip().zfill(6)
    if len(digits) != 6 or not digits.isdecimal():
        raise ValueError("Expected numeric clock time")
    hours, minutes, seconds = map(int, (digits[:2], digits[2:4], digits[4:]))
    if hours >= 24 or minutes >= 60 or seconds >= 60:
        raise ValueError("Clock time outside the usual range")
    return hours * 3600 + minutes * 60 + seconds


def linked_check_times(start, finish, dropoff_times):
    if len(dropoff_times) != 1:
        return None
    dropoff = next(iter(dropoff_times))
    if dropoff > start:
        return None
    return start - dropoff, finish - start, finish - dropoff


def service_records():
    counts = Counter()
    intervals = defaultdict(set)
    workbook = load_workbook(FIELD_FILE, read_only=True, data_only=True)
    for sheet in workbook:
        if sheet.title.startswith("2018_12_06"):
            continue
        counts["sessions"] += 1
        assert sheet.cell(1, 13).value == "Time End Baggage drop off"
        assert [sheet.cell(1, column).value for column in (24, 25)] == [
            "Time Start Baggage Check", "Time End Baggage Check"]
        for row in sheet.iter_rows(min_row=2, min_col=2, max_col=25, values_only=True):
            lane, raw_dropoff, raw_start, raw_finish = row[0], row[11], row[-2], row[-1]
            if lane is None or (not raw_start and not raw_finish):
                continue
            counts["timing_rows"] += 1
            try:
                start, finish = map(clock_seconds, (raw_start, raw_finish))
            except ValueError:
                counts["invalid_clock"] += 1
                continue
            if start is None or finish is None:
                counts["incomplete"] += 1
            elif finish <= start:
                counts["nonpositive"] += 1
            else:
                interval = (sheet.title, lane, start, finish)
                counts["duplicates"] += int(interval in intervals)
                linked_dropoffs = intervals[interval]
                try:
                    dropoff = clock_seconds(raw_dropoff)
                except ValueError:
                    continue
                if dropoff is not None:
                    linked_dropoffs.add(dropoff)
    workbook.close()
    counts["usable_intervals"] = len(intervals)
    assert sum(counts[key] for key in ("incomplete", "invalid_clock", "nonpositive", "duplicates",
                                       "usable_intervals")) == counts["timing_rows"]
    sessions = defaultdict(list)
    linked_times = []
    for (sheet, lane, start, finish), dropoffs in sorted(intervals.items()):
        sessions[sheet, lane].append(finish - start)
        linked = linked_check_times(start, finish, dropoffs)
        if linked is not None:
            linked_times.append(linked)
    return counts, list(sessions.values()), linked_times


def finish_probability(durations, n_jobs, time_limit):
    mass = np.bincount(durations).astype(float) / len(durations)
    total = np.array([1.0])
    for _ in range(n_jobs):
        total = np.convolve(total, mass)[:time_limit + 1]
    return float(total.sum())


def consecutive_service_times(sessions, n_jobs, n_stations):
    times = []
    for session in sessions:
        for start in range(len(session) - n_jobs + 1):
            durations = session[start:start + n_jobs]
            times.append(max(sum(durations[station::n_stations]) for station in range(n_stations)))
    return times


def service_analysis():
    counts, sessions, linked_times = service_records()
    durations = [duration for session in sessions for duration in session]
    elapsed = np.asarray(linked_times, dtype=int)
    assert len(elapsed) == 159 and (elapsed[:, 0] + elapsed[:, 1] == elapsed[:, 2]).all()
    quantiles = np.quantile(durations, [0.5, 0.9, 0.95])
    rows = []
    for workload in (3, 5):
        for workers in (1, 2):
            window_times = sorted(consecutive_service_times(sessions, workload, workers))
            windows = len(window_times)
            service_seconds_95 = window_times[math.ceil(windows * 0.95) - 1]
            loads = [(workload + worker) // workers for worker in range(workers)]
            for deadline in (180, 240, 300, 420):
                probability = math.prod(finish_probability(durations, jobs, deadline) for jobs in loads)
                successes = sum(duration <= deadline for duration in window_times)
                rows.append({"checks": workload, "stations": workers, "available_seconds": deadline,
                             "service_completion": probability, "window_successes": successes,
                             "observed_windows": windows, "window_fraction": successes / windows,
                             "service_seconds_for_95_percent_of_windows": service_seconds_95})
    return ({**counts, "mean_seconds": float(np.mean(durations)),
             "median_seconds": float(quantiles[0]), "p90_seconds": float(quantiles[1]),
             "p95_seconds": float(quantiles[2]), "field_input_sha256": sha256(FIELD_FILE),
             "linked_elapsed_times": {
                 "intervals": len(elapsed),
                 "medians_seconds": dict(zip(("before_search", "search", "dropoff_end_to_search_end"),
                                             np.median(elapsed, axis=0).tolist(), strict=True)),
                 "within_180_seconds_from_dropoff_end": int((elapsed[:, 2] <= 180).sum()),
                 "within_180_seconds_of_search": int((elapsed[:, 1] <= 180).sum()),
             }}, rows,
            [{"duration_seconds": seconds, "observed_intervals": count}
             for seconds, count in sorted(Counter(durations).items())])


def minimum_validation_batches(requirement, alpha, failures):
    if not 0 < requirement < 1:
        raise ValueError("A finite binomial validation sample requires a probability strictly between 0 and 1")
    trials = failures + 1
    while beta.ppf(alpha, trials - failures, failures + 1) < requirement:
        trials += 1
    return trials


def conditional_detection_requirements():
    completion = Fraction(95, 100)
    final_success = Fraction(90, 100)
    single_target_detection = Fraction(99, 100)
    rows = []
    for referral in (Fraction(1), Fraction(95, 100)):
        required_all_targets = final_success / (referral * completion)
        allowed_total_miss = 1 - required_all_targets
        for target_count in (1, 3, 6, 9):
            rows.append({
                "target_units": target_count,
                "referral_and_capacity_bound": float(referral),
                "completion_bound": float(completion),
                "final_success_requirement": float(final_success),
                "required_all_target_detection": float(required_all_targets),
                "maximum_sum_of_target_miss_bounds": float(allowed_total_miss),
                "required_common_target_detection": float(1 - allowed_total_miss / target_count),
                "example_marginal_target_detection": float(single_target_detection),
                "example_all_target_detection_lower": float(max(0, 1 - target_count * (1 - single_target_detection))),
                "example_all_target_detection_upper": float(single_target_detection),
            })
    return rows


def validation_sample_sizes():
    rows = []
    cases = [("completion", 0.95, 1.00, 0.025, 1), ("detection_for_80_percent", 0.80 / 0.95, 1.00, 0.025, 1),
             ("detection_for_90_percent", 0.90 / 0.95, 1.00, 0.025, 1), ("joint_for_80_percent", 0.80, 1.00, 0.05, 1)]
    cases += [("joint_for_90_percent", 0.90 / referral, referral, 0.05, 1)
              for referral in (1.00, 0.975, 0.95, 0.925)]
    cases.append(("joint_for_90_percent", 0.90 / 0.95, 0.95, 0.025, 2))
    for measure, requirement, referral, alpha, strata in cases:
        for failures in (0, 1):
            trials = minimum_validation_batches(requirement, alpha, failures)
            rows.append({"measure": measure, "referral_and_capacity_bound": referral,
                         "required_probability": requirement, "tail_probability": alpha,
                         "allowed_failures": failures, "minimum_batches": trials, "capability_strata": strata})
    return rows


def smoke():
    n_read, slots = 3, 2
    samples = list(combinations(range(5), n_read))
    for stopping in (False, True):
        fixed_paths = [(tuple(item for item in sample if item < 3),) for sample in samples]
        search_paths = [list(combinations(positive[0], min(slots, len(positive[0])))) if stopping
                        else positive for positive in fixed_paths]
        observed = [Fraction(0) for _ in range(4)]
        for paths in search_paths:
            for admitted in paths:
                found = len(set(admitted) & {0, 1})
                values = (found == 2 and len(admitted) <= slots, found >= 1 and len(admitted) <= slots,
                          2 - found, found == 0)
                for index, value in enumerate(values):
                    observed[index] += Fraction(int(value), len(samples) * len(paths))
        assert tuple(observed) == counting_outcomes(5, 3, n_read, slots, stopping)
    assert counting_outcomes(4, 3, 4, 2, True, n_target_records=1) == (
        Fraction(2, 3), Fraction(2, 3), Fraction(1, 3), Fraction(1, 3))
    assert np.isclose(quota_program((1, 5), 3, [((1, 5), 0), ((1, 5), 1)])[0], 0.5)
    feedback = feedback_analysis("smoke")[0]
    assert feedback["binary_referrals"] == feedback["immediate_search_findings"]
    durations = [2, 3, 5]
    exact = Fraction(sum(sum(draws) <= 8 for draws in product(durations, repeat=3)), len(durations) ** 3)
    assert np.isclose(finish_probability(durations, 3, 8), float(exact))
    assert consecutive_service_times([[1, 5, 3, 4], [5, 6]], 3, 2) == [5, 9]
    assert linked_check_times(120, 150, {90}) == (30, 30, 60)
    assert linked_check_times(120, 150, {90, 100}) is None
    assert linked_check_times(120, 150, {130}) is None
    panel = pd.DataFrame({"userID": ["first", "first", "second", "second", "second"],
                          "target": [True, True, True, True, False], "userCorrect": [1, 1, 1, 0, 0]})
    panel_outcomes = panel_target_outcomes(panel, target_count=2)
    assert panel_outcomes["all_correct_profiles"] == 1
    assert panel_outcomes["standardized_all_correct_probability"] == 0.5
    assert panel_target_outcomes(panel, target_count=1)["standardized_all_correct_probability"] == 0.75
    for priority in (True, False):
        orders = [(0, 1, 2), (1, 0, 2)] if priority else [(2, 0, 1), (2, 1, 0)]
        literal = Fraction(sum(0 in order[:2] for order in orders), len(orders))
        assert target_coverage_by_order(True, 2, int(not priority), 2) == literal
    effect = np.array([[Fraction(0), Fraction(1)], [Fraction(1), Fraction(0)]], dtype=object)
    constant, terms = copula_effect_terms(effect)
    for family, low, high, rotation in (("Frank", -2, 2, 0), ("Clayton", 0, 2, 0),
                                       ("Clayton", 0, 2, 90), ("Clayton", 0, 2, 180),
                                       ("Clayton", 0, 2, 270)):
        outer_min, min_upper, max_lower, outer_max, _ = enclose_copula_extrema(
            constant, terms, family, rotation, Fraction(low), Fraction(high), (Fraction(0),), COPULA_TOLERANCE,
        )
        margin = np.full(2, 0.5)
        observed = [float(np.sum(effect.astype(float) * copula_matrix(margin, margin, family, edge, rotation)))
                    for edge in (low, high)]
        assert abs(float(min_upper) - min(observed)) < 1e-12
        assert abs(float(max_lower) - max(observed)) < 1e-12
        assert float(outer_min) <= min(observed) + 1e-12 <= float(outer_max) + 1e-12
    assert clock_seconds(50408) == 5 * 3600 + 4 * 60 + 8
    assert validation_sample_sizes()[0]["minimum_batches"] == math.ceil(math.log(0.025) / math.log(0.95))
    for row in conditional_detection_requirements():
        target_count = row["target_units"]
        threshold = row["required_common_target_detection"]
        misses = np.array(list(product((0, 1), repeat=target_count)))
        all_found = (misses.sum(axis=1) == 0).astype(float)
        total_probability = np.ones((1, len(misses)))
        least = linprog(all_found, A_ub=misses.T, b_ub=np.repeat(1 - threshold, target_count),
                        A_eq=total_probability, b_eq=[1], bounds=(0, None), method="highs")
        assert least.success and np.isclose(least.fun, row["required_all_target_detection"], atol=1e-12, rtol=0)
        for direction, key in ((1, "lower"), (-1, "upper")):
            optimum = linprog(direction * all_found, A_eq=np.vstack((total_probability, misses.T)),
                              b_eq=[1, *np.repeat(0.01, target_count)], bounds=(0, None), method="highs")
            assert optimum.success and np.isclose(direction * optimum.fun,
                                                  row[f"example_all_target_detection_{key}"], atol=1e-12, rtol=0)
    for row in validation_sample_sizes():
        required, alpha = row["required_probability"], row["tail_probability"]
        trials, failures = row["minimum_batches"], row["allowed_failures"]
        assert beta.ppf(alpha, trials - failures, failures + 1) >= required
        assert beta.ppf(alpha, trials - failures - 1, failures + 1) < required
        assert binom.cdf(failures, trials, 1 - required) <= alpha < binom.cdf(failures, trials - 1, 1 - required)
    return {"passed": True, "script_sha256": sha256(BASE / "src" / "r1_followup_analysis.py")}


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report = smoke()
    if args.smoke:
        (OUT / "smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    else:
        saved = json.loads((OUT / "smoke.json").read_text(encoding="utf-8"))
        if saved != report:
            raise ValueError("Run the smoke calculation with the current script")
        trial = audit_trial()
        cases = prepare_cases()
        difficulty = image_difficulty(trial)
        repeated_targets, recorded_effect = repeated_target_analysis(trial)
        summary, replays, histogram = service_analysis()
        tables = {"repeated_target": repeated_targets, "schedule_workloads": schedule_workloads(trial),
                  "external_panel_targets": external_panel_analysis(trial),
                  "quota_sensitivity": quota_analysis(cases["false_miss"], difficulty), "service_completion": replays,
                  "service_duration_counts": histogram, "validation_sample_sizes": validation_sample_sizes(),
                  "conditional_detection_requirements": conditional_detection_requirements(),
                  "search_feedback": feedback_analysis("full"), "difficulty_order": difficulty_order_analysis(cases, difficulty),
                  "copula_extremum_enclosures": copula_enclosure_analysis(cases, recorded_effect),
                  "participant_deletion": participant_deletion_analysis(cases)}
        for name, records in tables.items():
            pd.DataFrame(records).to_csv(OUT / f"{name}.csv", index=False)
        report.update(summary)
        (OUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
