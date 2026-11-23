from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "capacity": "R1_image_unit/policy_capacity.csv",
    "dependence": "R1_followup/copula_extremum_enclosures.csv",
    "detection": "R1_followup/conditional_detection_requirements.csv",
    "omission": "R1_image_unit/target_omission_sensitivity.csv",
    "timing": "R1_followup/consecutive_service_session_bootstrap.csv",
}
FIGURES = ("dependence_effects", "detection_effects", "capacity_and_omission", "service_deadlines")
COLORS = ("#245B73", "#9C493D")
CONDITIONS = ("false_miss", "false_alarm")


def select_capacity(capacity, condition, searches):
    selection = capacity[capacity.condition.eq(condition) & capacity.total_search_capacity.eq(searches)]
    if len(selection) != 1:
        raise ValueError(f"Expected one capacity row for {condition} at {searches} searches")
    return selection.iloc[0]


def coverage_limit(pool_size, targets, requirement):
    total_subsets = math.comb(pool_size, targets)
    return next(pool_size - count for count in range(targets, pool_size + 1)
                if math.comb(count, targets) / total_subsets >= requirement)


def load_inputs(results):
    tables = {key: pd.read_csv(results / path) for key, path in INPUTS.items()}
    capacity = tables["capacity"]
    dependence = tables["dependence"]
    for condition, searches, readings, expected in zip(CONDITIONS, (25, 24), (11, 7), ((0, .1), (.1, .4))):
        row = select_capacity(capacity, condition, searches)
        family = dependence[dependence.condition.eq(condition)]
        endpoints = (row.matched_full_effect_lower, row.matched_full_effect_upper)
        if row.unique_images != 27 or row.common_reading_limit != readings or any(
            not math.isclose(actual, target, abs_tol=1e-12) for actual, target in zip(endpoints, expected)
        ) or len(family) != 5 or family.dependence_family.nunique() != 5 or not (
            family.minimum_lower.le(family.numerical_minimum)
            & family.numerical_minimum.le(family.numerical_maximum)
            & family.numerical_maximum.le(family.maximum_upper)
            & (family.numerical_minimum - family.minimum_lower).le(1.00001e-6)
            & (family.maximum_upper - family.numerical_maximum).le(1.00001e-6)
        ).all():
            raise ValueError(f"Dependence and capacity endpoints disagree for {condition}")
    detection = tables["detection"]
    for policy, expected in (("stopping", .8 / .95), ("full_fixed", .8 / (.9 * .95))):
        row = detection[detection.policy.eq(policy) & detection.final_success_requirement.eq(.8)].iloc[0]
        if row.search_capacity != 25 or row.reading_allowance != 11 or not math.isclose(
            row.required_all_target_detection, expected
        ):
            raise ValueError(f"Unexpected matched detection requirement for {policy}")
    omission = tables["omission"]
    omission = omission[omission.condition.eq("false_miss") & omission.search_capacity.eq(25)]
    crossing = omission[omission.batch_level_target_omission_allowance.eq(.009)]
    if len(omission) != 9 or len(crossing) != 1 or not math.isclose(
        crossing.iloc[0].stopping_mean_miss_upper, .001
    ) or coverage_limit(11, 1, .95) != 0:
        raise ValueError("Unexpected target-coverage or omission limit")
    tables["omission"] = omission.sort_values("batch_level_target_omission_allowance")
    timing = tables["timing"].sort_values(["checks", "available_seconds"])
    expected = [(3, 240, 124, 135), (3, 300, 131, 135), (5, 240, 78, 111),
                (5, 300, 96, 111), (5, 420, 110, 111)]
    if list(timing[["checks", "available_seconds", "successful_windows", "overlapping_windows"]]
            .itertuples(index=False, name=None)) != expected or not (
        timing.bootstrap_2_5.le(timing.observed_completion_fraction)
        & timing.observed_completion_fraction.le(timing.bootstrap_97_5)
        & timing.stations.eq(2)
    ).all():
        raise ValueError("Service completion counts or intervals disagree with the supplement")
    tables["timing"] = timing
    return tables


def save(figure, output, name):
    for axis in figure.axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D8DEE5", linewidth=.5)
        axis.set_axisbelow(True)
    figure.savefig(output / f"{name}.svg", bbox_inches="tight")
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def dependence_plot(tables, output):
    figure, axes = plt.subplots(1, 2, figsize=(7.25, 2.55), layout="constrained")
    for axis, condition, searches, color in zip(axes, CONDITIONS, (25, 24), COLORS):
        rows = tables["dependence"][tables["dependence"].condition.eq(condition)]
        capacity = select_capacity(tables["capacity"], condition, searches)
        names = ["All compatible pairings", "Frank", "Clayton", "Clayton 90°", "Clayton 180°", "Clayton 270°"]
        lower = [capacity.matched_full_effect_lower, *rows.numerical_minimum]
        upper = [capacity.matched_full_effect_upper, *rows.numerical_maximum]
        for position, (left, right) in enumerate(zip(lower, upper)):
            axis.plot([100 * left, 100 * right], [position, position], "|-",
                      color=color, linewidth=2 if position == 0 else 1)
        axis.set(yticks=range(len(names)), yticklabels=names, title=f"{condition.replace('_', ' ').capitalize()}",
                 xlabel="Stopping minus full fixed review (percentage points)")
        axis.invert_yaxis()
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    save(figure, output, FIGURES[0])


def detection_plot(tables, output):
    figure, axes = plt.subplots(1, 2, figsize=(7.25, 2.55), layout="constrained")
    row = select_capacity(tables["capacity"], "false_miss", 25)
    probabilities = range(75, 101)
    for lower, color, line, label in zip((row.stop_lower, row.fixed_full_lower), COLORS, ("-", "--"),
                                        ("Stopping", "Full fixed review")):
        axes[0].plot(probabilities, [value * .95 * lower for value in probabilities],
                     color=color, linestyle=line, label=label)
    for target in (80, 90):
        axes[0].axhline(target, color="#666666", linestyle=":", linewidth=.7)
    axes[0].set(title="(a) Success at 25 searches", xlabel="Conditional all-target detection (%)",
                ylabel="Final batch-success lower bound (%)")
    axes[0].legend(loc="lower right", frameon=False)
    rows = tables["capacity"][tables["capacity"].condition.eq("false_miss")].sort_values("total_search_capacity")
    searches = rows.total_search_capacity.to_numpy()
    lower = rows.matched_full_effect_lower.to_numpy() * 100 * .95**2
    upper = rows.matched_full_effect_upper.to_numpy() * 100 * .95**2
    axes[1].fill_between(searches, lower, upper, color=COLORS[0], alpha=.16)
    axes[1].plot(searches, lower, "o-", searches, upper, "o-", color=COLORS[0], markersize=2.5)
    axes[1].set(title="(b) Paired effect at 95% detection", xlabel="Total search capacity",
                ylabel="Final success difference (percentage points)", xticks=searches)
    save(figure, output, FIGURES[1])


def coverage_plot(tables, output):
    figure, axes = plt.subplots(1, 2, figsize=(7.25, 2.55), layout="constrained")
    requirements = [value / 100 for value in range(50, 101)]
    for pool_size, color, line in zip((6, 11), COLORS, ("-", "--")):
        axes[0].step([100 * value for value in requirements],
                     [coverage_limit(pool_size, 1, value) for value in requirements],
                     where="pre", color=color, linestyle=line, label=f"{pool_size} eligible images")
    axes[0].set(title="(a) Capacity saving with one eligible target",
                xlabel="Required success relative to its ceiling (%)", ylabel="Maximum search places saved")
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].legend(frameon=False)
    rows = tables["omission"]
    axes[1].plot(100 * rows.batch_level_target_omission_allowance, 100 * rows.stopping_mean_miss_upper,
                 "o-", color=COLORS[0], markersize=3)
    axes[1].axhline(.1, color="#666666", linestyle="--", label="0.1% mean-risk limit")
    axes[1].set(title="(b) Target omission at 25 searches", xlabel="Batch-level omission allowance (%)",
                ylabel="Mean unreferred-target upper bound (%)", xscale="symlog", xlim=(-.04, 23))
    axes[1].set_xticks([0, .5, 1, 2.5, 5, 10, 20])
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
    axes[1].legend(frameon=False, loc="upper left")
    save(figure, output, FIGURES[2])


def timing_plot(tables, output):
    figure, axis = plt.subplots(figsize=(5.2, 2.55), layout="constrained")
    for checks, color, marker in zip((3, 5), COLORS, ("o", "s")):
        rows = tables["timing"][tables["timing"].checks.eq(checks)]
        observed = rows.observed_completion_fraction
        axis.errorbar(rows.available_seconds / 60 + (-.035 if checks == 3 else .035), 100 * observed,
                      yerr=[100 * (observed - rows.bootstrap_2_5), 100 * (rows.bootstrap_97_5 - observed)],
                      fmt=marker, color=color, capsize=3, label=f"{checks} checks")
    axis.axhline(95, color="#666666", linestyle="--", label="95% completion")
    axis.set(xticks=[4, 5, 7], ylim=(55, 103), xlabel="Service time available (minutes)",
             ylabel="Check windows completed (%)")
    axis.legend(loc="lower right", frameon=False)
    save(figure, output, FIGURES[3])


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--full", action="store_true")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "article" / "R1_Source" / "figures")
    args = parser.parse_args()
    tables = load_inputs(args.results)
    if args.full:
        args.output.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                             "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 8.5,
                             "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "ress-r1"})
        for plot in (dependence_plot, detection_plot, coverage_plot, timing_plot):
            plot(tables, args.output)
    print(f"Verified saved inputs for {len(FIGURES)} figures.")


if __name__ == "__main__":
    main()
