"""Exact finite-horizon sequential inspection comparison.

The constructed example has two observable strata and a hidden target/referral
configuration. A controller may observe only the chosen stratum and its binary
referral response. It never observes target identity or physical-search results.
All probabilities and policy values use exact rational arithmetic.
"""
from __future__ import annotations

import argparse
import itertools
import json
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results" / "sequential_comparison_20261018"
CONFIG = BASE / "data" / "literature" / "sequential_comparison_20261018" / "constructed_two_strata.json"


def load_case(name):
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))[name]
    worlds = tuple(
        (tuple(tuple(map(int, group)) for group in row["groups"]), int(row["targets"]))
        for row in raw["worlds"]
    )
    return raw, worlds


def as_fraction(value):
    if isinstance(value, Fraction):
        return value
    return Fraction(str(value))


def normalize(items):
    merged = {}
    for state, weight in items:
        if weight:
            merged[state] = merged.get(state, Fraction(0)) + weight
    total = sum(merged.values(), Fraction(0))
    if total == 0:
        return ()
    return tuple((state, weight / total) for state, weight in sorted(merged.items()))


def initial_belief(worlds, prior_a):
    prior_a = as_fraction(prior_a)
    weights = (prior_a, 1 - prior_a)
    return normalize((
        ((world, groups[0], groups[1], 0), weights[world])
        for world, (groups, _target_count) in enumerate(worlds)
    ))


def transitions(state, group):
    world, group_1, group_2, found = state
    groups = (group_1, group_2)
    counts = groups[group]
    remaining = sum(counts)
    for category, count in enumerate(counts):
        if count == 0:
            continue
        next_counts = list(counts)
        next_counts[category] -= 1
        next_groups = list(groups)
        next_groups[group] = tuple(next_counts)
        referral = int(category in (0, 2))
        found_target = int(category == 0)
        yield referral, (world, next_groups[0], next_groups[1], found + found_target), Fraction(count, remaining)


def advance(belief, group):
    branch_mass = {0: Fraction(0), 1: Fraction(0)}
    branch_states = {0: {}, 1: {}}
    for state, weight in belief:
        for referral, next_state, probability in transitions(state, group):
            mass = weight * probability
            branch_mass[referral] += mass
            bucket = branch_states[referral]
            bucket[next_state] = bucket.get(next_state, Fraction(0)) + mass
    return tuple(
        (referral, branch_mass[referral], normalize(branch_states[referral].items()))
        for referral in (0, 1) if branch_mass[referral]
    )


def remaining_by_group(belief):
    state = belief[0][0]
    return sum(state[1]), sum(state[2])


def terminal_value(belief, worlds):
    success = Fraction(0)
    caught = Fraction(0)
    for state, weight in belief:
        world, _g1, _g2, found = state
        targets = worlds[world][1]
        success += weight * int(found == targets)
        caught += weight * Fraction(found, targets)
    return success, caught, Fraction(0), Fraction(0)


def better(left, right):
    """Maximize complete success, then target capture, then minimize work."""
    if left[0] != right[0]:
        return left[0] > right[0]
    if left[1] != right[1]:
        return left[1] > right[1]
    if left[2] != right[2]:
        return left[2] < right[2]
    return left[3] < right[3]


class Controller:
    def __init__(self, worlds, read_limit, search_limit):
        self.worlds = worlds
        self.read_limit = read_limit
        self.search_limit = search_limit
        self.actions = {}

    @lru_cache(None)
    def solve(self, belief, reads, searches):
        stop = terminal_value(belief, self.worlds)
        if reads >= self.read_limit or searches >= self.search_limit:
            self.actions[(belief, reads, searches)] = None
            return stop
        candidates = [(stop, None)]
        for group, remaining in enumerate(remaining_by_group(belief)):
            if not remaining:
                continue
            value = [Fraction(0) for _ in range(4)]
            for referral, probability, child in advance(belief, group):
                child_value = self.solve(child, reads + 1, searches + referral)
                for index, metric in enumerate(child_value):
                    value[index] += probability * metric
                value[2] += probability
                value[3] += probability * referral
            candidates.append((tuple(value), group))
        best_value, best_action = candidates[0]
        for value, action in candidates[1:]:
            if better(value, best_value):
                best_value, best_action = value, action
        self.actions[(belief, reads, searches)] = best_action
        return best_value


def uniform_value(worlds, belief, read_limit, search_limit):
    @lru_cache(None)
    def solve(state_belief, reads, searches):
        stop = terminal_value(state_belief, worlds)
        if reads >= read_limit or searches >= search_limit:
            return stop
        remaining = remaining_by_group(state_belief)
        total = sum(remaining)
        value = [Fraction(0) for _ in range(4)]
        for group, count in enumerate(remaining):
            if count == 0:
                continue
            group_probability = Fraction(count, total)
            for referral, observation_probability, child in advance(state_belief, group):
                child_value = solve(child, reads + 1, searches + referral)
                probability = group_probability * observation_probability
                for index, metric in enumerate(child_value):
                    value[index] += probability * metric
                value[2] += probability
                value[3] += probability * referral
        return tuple(value)
    return solve(belief, 0, 0)


def fixed_schedule_value(worlds, belief, schedule, read_limit, search_limit):
    @lru_cache(None)
    def solve(state_belief, schedule_index, reads, searches):
        stop = terminal_value(state_belief, worlds)
        if reads >= read_limit or searches >= search_limit or schedule_index >= len(schedule):
            return stop
        group = schedule[schedule_index]
        if remaining_by_group(state_belief)[group] == 0:
            return solve(state_belief, schedule_index + 1, reads, searches)
        value = [Fraction(0) for _ in range(4)]
        for referral, probability, child in advance(state_belief, group):
            child_value = solve(child, schedule_index + 1, reads + 1, searches + referral)
            for index, metric in enumerate(child_value):
                value[index] += probability * metric
            value[2] += probability
            value[3] += probability * referral
        return tuple(value)
    return solve(belief, 0, 0, 0)


def best_fixed_schedule(worlds, prior_a, read_limit, search_limit):
    belief = initial_belief(worlds, prior_a)
    best = None
    best_schedule = None
    for schedule in itertools.product((0, 1), repeat=read_limit):
        value = fixed_schedule_value(worlds, belief, schedule, read_limit, search_limit)
        if best is None or better(value, best):
            best, best_schedule = value, schedule
    return best, best_schedule


def make_controller(worlds, prior_a, read_limit, search_limit):
    belief = initial_belief(worlds, prior_a)
    controller = Controller(worlds, read_limit, search_limit)
    value = controller.solve(belief, 0, 0)
    return controller, belief, value


def all_category_orders(group_counts):
    records = tuple(category for category, count in enumerate(group_counts) for _ in range(count))
    return tuple(sorted(set(itertools.permutations(records))))


def interleavings(group_size):
    labels = (0,) * group_size + (1,) * group_size
    return tuple(sorted(set(itertools.permutations(labels))))


def direct_evaluate(worlds, policy, trained_prior, actual_prior, read_limit, search_limit,
                    schedule=None, uniform=False):
    """Enumerate labeled item orders independently of the count-state DP."""
    actual_prior = as_fraction(actual_prior)
    train_belief = initial_belief(worlds, trained_prior)
    total = [Fraction(0) for _ in range(4)]
    for world, (groups, targets) in enumerate(worlds):
        world_probability = actual_prior if world == 0 else 1 - actual_prior
        if not world_probability:
            continue
        orders_1 = all_category_orders(groups[0])
        orders_2 = all_category_orders(groups[1])
        if uniform:
            group_sequences = interleavings(sum(groups[0]))
        else:
            group_sequences = (None,)
        denominator = len(orders_1) * len(orders_2) * len(group_sequences)
        for order_1, order_2, group_sequence in itertools.product(orders_1, orders_2, group_sequences):
            pointers = [0, 0]
            reads = searches = found = schedule_index = sequence_index = 0
            belief = train_belief
            while reads < read_limit and searches < search_limit:
                if uniform:
                    if sequence_index >= len(group_sequence):
                        break
                    group = group_sequence[sequence_index]
                    sequence_index += 1
                elif schedule is not None:
                    while schedule_index < len(schedule) and pointers[schedule[schedule_index]] >= len((order_1, order_2)[schedule[schedule_index]]):
                        schedule_index += 1
                    if schedule_index >= len(schedule):
                        break
                    group = schedule[schedule_index]
                    schedule_index += 1
                else:
                    group = policy.actions.get((belief, reads, searches))
                    if group is None:
                        break
                group_order = (order_1, order_2)[group]
                if pointers[group] >= len(group_order):
                    continue
                category = group_order[pointers[group]]
                pointers[group] += 1
                referral = int(category in (0, 2))
                found += int(category == 0)
                reads += 1
                searches += referral
                branch = {y: child for y, _probability, child in advance(belief, group)}
                if referral not in branch:
                    raise AssertionError("direct path has zero probability under the stated training prior")
                belief = branch[referral]
            weight = world_probability / denominator
            assert reads <= read_limit and searches <= search_limit
            total[0] += weight * int(found == targets)
            total[1] += weight * Fraction(found, targets)
            total[2] += weight * reads
            total[3] += weight * searches
    return tuple(total)


def metrics(value):
    return dict(
        complete_target_success=float(value[0]),
        mean_missed_target_fraction=float(1 - value[1]),
        average_readings=float(value[2]),
        average_searches=float(value[3]),
    )


@lru_cache(None)
def action_trees(depth):
    if depth == 0:
        return (None,)
    children = action_trees(depth - 1)
    return (None,) + tuple((group, negative, positive)
                           for group in (0, 1)
                           for negative, positive in itertools.product(children, repeat=2))


def brute_tree_value(worlds, tree, prior_a, read_limit, search_limit):
    """Evaluate an explicit observation-contingent action tree by item orders."""
    total = [Fraction(0) for _ in range(4)]
    for world, (groups, targets) in enumerate(worlds):
        world_probability = prior_a if world == 0 else 1-prior_a
        orders_1 = all_category_orders(groups[0])
        orders_2 = all_category_orders(groups[1])
        denominator = len(orders_1) * len(orders_2)
        for order_1, order_2 in itertools.product(orders_1, orders_2):
            pointers = [0, 0]
            node = tree
            readings = searches = found = 0
            while node is not None and readings < read_limit and searches < search_limit:
                group, negative, positive = node
                order = (order_1, order_2)[group]
                if pointers[group] == len(order):
                    break
                category = order[pointers[group]]
                pointers[group] += 1
                referral = int(category in (0, 2))
                found += int(category == 0)
                readings += 1
                searches += referral
                node = positive if referral else negative
            weight = world_probability / denominator
            total[0] += weight * int(found == targets)
            total[1] += weight * Fraction(found, targets)
            total[2] += weight * readings
            total[3] += weight * searches
    return tuple(total)


def smoke_test():
    raw, worlds = load_case("smoke")
    read_limit, search_limit = raw["read_limit"], raw["search_limit"]
    prior = Fraction(raw["prior_numerator"], raw["prior_denominator"])
    controller, belief, adaptive = make_controller(worlds, prior, read_limit, search_limit)
    uniform = uniform_value(worlds, belief, read_limit, search_limit)
    fixed, schedule = best_fixed_schedule(worlds, prior, read_limit, search_limit)
    direct_adaptive = direct_evaluate(worlds, controller, prior, prior, read_limit, search_limit)
    direct_uniform = direct_evaluate(worlds, controller, prior, prior, read_limit, search_limit, uniform=True)
    direct_fixed = direct_evaluate(worlds, controller, prior, prior, read_limit, search_limit,
                                   schedule=schedule)
    assert adaptive == direct_adaptive
    assert uniform == direct_uniform
    assert fixed == direct_fixed
    assert not better(fixed, adaptive)
    assert adaptive[2] <= read_limit and adaptive[3] <= search_limit
    tree_best = None
    for tree in action_trees(read_limit):
        value = brute_tree_value(worlds, tree, prior, read_limit, search_limit)
        if tree_best is None or better(value, tree_best):
            tree_best = value
    assert adaptive == tree_best
    report = dict(
        status="smoke_passed",
        constructed_worlds=worlds,
        limits=dict(readings=read_limit, searches=search_limit),
        adaptive=metrics(adaptive),
        uniform_referral_stop=metrics(uniform),
        best_precommitted_group_schedule=metrics(fixed),
        best_precommitted_schedule=list(schedule),
        independent_order_enumeration=dict(
            adaptive_matches=True, uniform_matches=True, fixed_schedule_matches=True),
        independent_action_tree_optimum=dict(
            enumerated_trees=len(action_trees(read_limit)), matches=True),
        exact_fraction_values=True,
        hidden_target_identities_used_online=False,
        search_feedback_used_online=False,
        bellman_state_count=len(controller.actions),
    )
    return report


def run_full():
    raw, worlds = load_case("full")
    read_limit, search_limit = raw["read_limit"], raw["search_limit"]
    denominator = raw["prior_denominator"]
    train_priors = tuple(Fraction(x, denominator) for x in raw["training_prior_numerators"])
    actual_priors = tuple(Fraction(x, denominator) for x in raw["actual_prior_numerators"])
    policies = {}
    results = []
    for trained in train_priors:
        controller, belief, adaptive = make_controller(worlds, trained, read_limit, search_limit)
        uniform = uniform_value(worlds, belief, read_limit, search_limit)
        fixed, schedule = best_fixed_schedule(worlds, trained, read_limit, search_limit)
        assert not better(fixed, adaptive), "adaptive Bellman policy must dominate every fixed schedule"
        # Direct enumeration is an independent implementation check at the prior used to optimize.
        direct_adaptive = direct_evaluate(worlds, controller, trained, trained, read_limit, search_limit)
        direct_uniform = direct_evaluate(worlds, controller, trained, trained, read_limit, search_limit,
                                         uniform=True)
        direct_fixed = direct_evaluate(worlds, controller, trained, trained, read_limit, search_limit,
                                       schedule=schedule)
        assert adaptive == direct_adaptive
        assert uniform == direct_uniform
        assert fixed == direct_fixed
        policies[trained] = (controller, schedule)
        for actual in actual_priors:
            actual_adaptive = direct_evaluate(worlds, controller, trained, actual,
                                              read_limit, search_limit)
            actual_fixed = direct_evaluate(worlds, controller, trained, actual,
                                           read_limit, search_limit, schedule=schedule)
            results.append(dict(
                trained_prior_group_A=float(trained), actual_prior_group_A=float(actual),
                adaptive=metrics(actual_adaptive),
                fixed_precommitted_schedule=metrics(actual_fixed),
                uniform_referral_stop=metrics(uniform_value(
                    worlds, initial_belief(worlds, actual), read_limit, search_limit)),
                fixed_schedule=list(schedule),
                adaptive_bellman_states=len(controller.actions),
            ))
    resource_sensitivity = []
    for alternative_reads, alternative_searches in raw["resource_sensitivity"]:
        prior = Fraction(1, 2)
        control, belief, adaptive = make_controller(
            worlds, prior, alternative_reads, alternative_searches)
        uniform = uniform_value(worlds, belief, alternative_reads, alternative_searches)
        fixed, schedule = best_fixed_schedule(
            worlds, prior, alternative_reads, alternative_searches)
        assert adaptive == direct_evaluate(
            worlds, control, prior, prior, alternative_reads, alternative_searches)
        assert uniform == direct_evaluate(
            worlds, control, prior, prior, alternative_reads, alternative_searches, uniform=True)
        assert fixed == direct_evaluate(
            worlds, control, prior, prior, alternative_reads, alternative_searches,
            schedule=schedule)
        resource_sensitivity.append(dict(
            readings_limit=alternative_reads, searches_limit=alternative_searches,
            prior_group_A=0.5, adaptive=metrics(adaptive),
            fixed_precommitted_schedule=metrics(fixed), uniform_referral_stop=metrics(uniform),
            fixed_schedule=list(schedule), independent_order_enumeration=True))
    matched = [row for row in results
               if row["trained_prior_group_A"] == row["actual_prior_group_A"]]
    mismatch_examples = [row for row in results
                         if (row["trained_prior_group_A"], row["actual_prior_group_A"])
                         in ((0.5, 0.1), (0.5, 0.9), (0.1, 0.9))]
    return dict(
        status="completed",
        study_type="Policy comparison in a constructed finite population",
        model=dict(
            strata=2, items_per_stratum=4, hidden_worlds=2, targets_per_world=2,
            categories=["target with referral", "target without referral",
                        "non-target with referral", "non-target without referral"],
            online_observations=["chosen stratum", "binary referral response"],
            online_prohibitions=["target identity", "physical-search outcome"],
            probability_calculation="exact rational hypergeometric updates and finite-horizon Bellman recursion",
        ),
        common_limits=dict(readings=read_limit, searches=search_limit),
        resource_sensitivity=resource_sensitivity,
        prior_sensitivity=dict(
            training_priors=[float(x) for x in train_priors],
            actual_priors=[float(x) for x in actual_priors],
            matched_prior_comparisons=matched,
            prior_misspecification_examples=mismatch_examples,
            cells=results,
        ),
        independent_order_enumeration=True,
        interpretation=("The example compares how choosing the next group using observed referrals "
                        "affects target coverage under common resource limits and different priors."),
    )


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()
    report = smoke_test() if args.smoke else run_full()
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / ("smoke.json" if args.smoke else "exact_comparison.json")
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
