from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results" / "R1_image_unit"
OUTPUT = BASE / "article" / "R1_Source" / "figures"
CAPACITY_FILE = RESULTS / "policy_capacity.csv"
RISK_FILE = RESULTS / "risk_constrained_capacity.csv"

RULES = (
    ("stopping", "stopping success lower", "solid", "o"),
    ("matched fixed", "same-limit fixed lower", (0, (4, 2)), "s"),
    ("risk-constrained fixed", "risk-constrained fixed lower", "dashdot", "D"),
)
COLORS = {"stopping": "#147A72", "matched fixed": "#B54E4C",
          "risk-constrained fixed": "#475569"}
DISPLAY = {
    "stopping": "Referral stopping",
    "matched fixed": "Fixed count, same maximum readings",
    "risk-constrained fixed": "Best fixed count within the risk limit",
}
CONDITION_TITLES = {"false_miss": "False-miss condition",
                    "false_alarm": "False-alarm condition"}
MEAN_MISS_LIMIT = 0.001


def load_series() -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    source = pd.read_csv(CAPACITY_FILE)
    risk = pd.read_csv(RISK_FILE)
    if set(source["condition"]) != set(CONDITION_TITLES):
        raise ValueError("Expected both prespecified image-level error conditions")
    results = {}
    for condition in CONDITION_TITLES:
        source_case = source[source["condition"] == condition].sort_values(
            "total_search_capacity",
        )
        risk_case = risk[risk["condition"] == condition].sort_values(
            "search_capacity",
        )
        merged = source_case.merge(
            risk_case[["search_capacity", "stopping_success_lower",
                       "stopping_mean_miss_upper", "stopping_jointly_feasible",
                       "risk_constrained_best_fixed_success_lower",
                       "risk_constrained_fixed_feasible",
                       "same_reading_allowance_fixed_mean_miss_upper"]],
            left_on="total_search_capacity", right_on="search_capacity",
            how="inner", validate="one_to_one",
        )
        results[condition] = {
            "stopping": (
                merged["total_search_capacity"].to_numpy(dtype=int),
                np.where(
                    merged["stopping_jointly_feasible"].to_numpy(dtype=bool),
                    merged["stopping_success_lower"].to_numpy(dtype=float), np.nan,
                ),
            ),
            "stopping_raw": (
                merged["total_search_capacity"].to_numpy(dtype=int),
                merged["stopping_success_lower"].to_numpy(dtype=float),
            ),
            "matched fixed": (
                merged["total_search_capacity"].to_numpy(dtype=int),
                np.where(
                    merged["same_reading_allowance_fixed_mean_miss_upper"].to_numpy()
                    .astype(float) <= MEAN_MISS_LIMIT + 1e-12,
                    merged["fixed_full_lower"].to_numpy(dtype=float), np.nan,
                ),
            ),
            "matched fixed raw": (
                merged["total_search_capacity"].to_numpy(dtype=int),
                merged["fixed_full_lower"].to_numpy(dtype=float),
            ),
            "risk-constrained fixed": (
                merged["total_search_capacity"].to_numpy(dtype=int),
                np.where(
                    merged["risk_constrained_fixed_feasible"].to_numpy(dtype=bool),
                    merged["risk_constrained_best_fixed_success_lower"]
                    .to_numpy(dtype=float), np.nan,
                ),
            ),
        }
        if merged.empty:
            raise ValueError(f"No paired image-unit output exists for {condition}")
        for rule, (_, values) in results[condition].items():
            if np.any(values < -1e-10) or np.any(values > 1 + 1e-10):
                raise ArithmeticError(f"Invalid lower probability for {condition}, {rule}")
    return results


def smoke_check(results: dict) -> dict:
    for condition, series in results.items():
        stopping_capacity, stopping_values = series["stopping_raw"]
        fixed_capacity, fixed_values = series["matched fixed raw"]
        if not np.array_equal(stopping_capacity, fixed_capacity):
            raise AssertionError(f"Unmatched capacities for {condition}")
        if np.any(np.diff(stopping_values) < -1e-10):
            raise AssertionError(f"Stopping success decreased with capacity for {condition}")
        if np.any(stopping_values + 1e-10 < fixed_values):
            raise AssertionError(f"Same-order stopping identity failed for {condition}")
    return {
        "passed": True,
        "conditions": list(results),
        "matched_capacity_points": {
            condition: len(values["stopping"][0]) for condition, values in results.items()
        },
        "checks": [
            "Every plotted strategy uses the same capacity axis within its condition.",
            "Stopping probabilities are nondecreasing in the search allowance.",
            "Stopping retains every successful fixed prefix at the same maximum readings.",
        ],
    }


def make_figure(results: dict) -> tuple[Path, Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.7,
        "lines.markersize": 3.2,
    })
    figure, axes = plt.subplots(1, 2, figsize=(7.35, 3.35), sharey=True)
    for axis, condition in zip(axes, CONDITION_TITLES):
        for rule, _, linestyle, marker in RULES:
            capacities, probabilities = results[condition][rule]
            axis.plot(
                capacities, probabilities,
                color=COLORS[rule], linestyle=linestyle, marker=marker,
                markevery=max(1, len(capacities) // 5), label=DISPLAY[rule],
            )
        axis.axhline(0.8, color="#777777", linestyle=(0, (2, 2)), linewidth=0.85,
                     label="80% requirement" if condition == "false_miss" else None)
        axis.set_title(CONDITION_TITLES[condition])
        axis.set_xlabel("Total search-count allowance")
        axis.set_xticks(np.unique(results[condition]["stopping"][0]))
        axis.grid(axis="y", color="#D8DEE5", linewidth=0.55)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#667085")
        axis.spines["bottom"].set_color("#667085")
        axis.tick_params(width=0.65, colors="#394150")
        axis.set_ylim(-0.035, 1.04)
    axes[0].set_ylabel("Referral-and-capacity probability lower bound")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4,
                  frameon=False, bbox_to_anchor=(0.5, -0.045),
                  handlelength=2.6, columnspacing=1.3)
    figure.tight_layout(w_pad=2.15, rect=(0, 0.12, 1, 1))
    svg_path = OUTPUT / "figure_08.svg"
    pdf_path = OUTPUT / "figure_08.pdf"
    figure.savefig(svg_path, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight", metadata={
        "Title": "Image-unit referral and capacity guarantees",
        "Author": "Yanhong Lu; Yuxian Du; Zhaoxin Chen; Tao Wen",
        "Subject": "Exact response-marginal sensitivity analysis",
    })
    plt.close(figure)
    pdf_path.with_suffix(".pdf_tex").write_text(
        "\\begingroup\n"
        "\\ifx\\svgwidth\\undefined\\def\\svgwidth{\\linewidth}\\fi\n"
        "\\includegraphics[width=\\svgwidth]{figures/figure_08.pdf}\n"
        "\\endgroup\n",
        encoding="utf-8", newline="\n",
    )
    return svg_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()
    results = load_series()
    report = smoke_check(results)
    if not report["passed"]:
        raise ArithmeticError("Figure input checks failed")
    if args.smoke:
        print(json.dumps(report, indent=2))
        return
    svg_path, pdf_path = make_figure(results)
    print(json.dumps({**report, "svg": str(svg_path), "pdf": str(pdf_path)}, indent=2))


if __name__ == "__main__":
    main()
