from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results" / "R1_fixed_structure"
OUTPUT = BASE / "article" / "R1_Source" / "figures"
REVERSAL_FILE = RESULTS / "repeated_capacity_reversals.csv"
FEASIBILITY_FILE = RESULTS / "joint_feasibility_family.csv"


def load_verified_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    reversals = pd.read_csv(REVERSAL_FILE)
    reversals = reversals[reversals["L"].eq(4)].sort_values("z")
    feasibility = pd.read_csv(FEASIBILITY_FILE)
    feasibility = feasibility[
        feasibility["N"].eq(2000) & feasibility["n"].ge(1997)
    ].sort_values("n")
    if reversals["z"].tolist() != list(range(1, 9)):
        raise ValueError("The L=4 capacity example must cover capacities 1 through 8")
    effects = reversals["effect"].astype(float).tolist()
    sign_changes = sum(left * right < 0 for left, right in zip(effects, effects[1:]))
    if sign_changes != 7:
        raise ArithmeticError(f"Expected seven strict effect reversals; found {sign_changes}")
    if feasibility["n"].astype(int).tolist() != [1997, 1998, 1999, 2000]:
        raise ValueError("The N=2000 reliability example is incomplete")
    feasible_counts = feasibility.loc[
        feasibility["jointly_feasible"].astype(bool), "n",
    ].astype(int).tolist()
    if feasible_counts != [1998, 2000]:
        raise ArithmeticError(f"Unexpected joint feasible counts: {feasible_counts}")
    return reversals, feasibility


def smoke_check(reversals: pd.DataFrame, feasibility: pd.DataFrame) -> dict:
    return {
        "passed": True,
        "capacity_values": reversals["z"].astype(int).tolist(),
        "strict_effect_sign_changes": 7,
        "review_counts": feasibility["n"].astype(int).tolist(),
        "jointly_feasible_review_counts": [1998, 2000],
        "figure_inputs_are_saved_result_tables": True,
    }


def make_figure(reversals: pd.DataFrame, feasibility: pd.DataFrame) -> tuple[Path, Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.35,
        "lines.markersize": 3.1,
    })
    figure, axes = plt.subplots(1, 2, figsize=(7.25, 2.6))
    axes[0].plot(
        reversals["z"], 100 * reversals["effect"],
        "o-", color="#245B73",
    )
    axes[0].axhline(0, color="#666666", linewidth=0.7)
    axes[0].set(
        xlabel="Remaining inspection capacity",
        ylabel="Partial minus full (percentage points)",
        title="Seven ranking reversals",
        xticks=list(range(1, 9)),
    )
    axes[1].plot(
        feasibility["n"], 100 * feasibility["joint_success"],
        "o-", color="#9C493D",
    )
    axes[1].axhline(99.9, color="#666666", linestyle="--", linewidth=0.8)
    axes[1].set(
        xlabel="Number reviewed",
        ylabel="Batch success (%)",
        title="High-reliability feasible decisions",
        xticks=[1997, 1998, 1999, 2000],
    )
    axes[1].ticklabel_format(useOffset=False, style="plain", axis="x")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#D8DEE5", linewidth=0.5)
        axis.set_axisbelow(True)
    figure.tight_layout(w_pad=2.0)
    svg = OUTPUT / "policy_structure.svg"
    pdf = OUTPUT / "policy_structure.pdf"
    pdf_tex = OUTPUT / "policy_structure.pdf_tex"
    figure.savefig(svg, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight", metadata={
        "Title": "Fixed-count reversals and high-reliability feasible reviews",
        "Author": "Yanhong Lu; Yuxian Du; Zhaoxin Chen; Tao Wen",
    })
    plt.close(figure)
    pdf_tex.write_text(
        "% Saved-table figure rendered as selectable vector artwork.\n"
        "\\begingroup\n"
        "\\ifx\\svgwidth\\undefined\\def\\svgwidth{\\linewidth}\\fi\n"
        "\\includegraphics[width=\\svgwidth]{figures/policy_structure.pdf}\n"
        "\\endgroup\n",
        encoding="utf-8",
        newline="\n",
    )
    return svg, pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--full", action="store_true")
    args = parser.parse_args()
    reversals, feasibility = load_verified_tables()
    report = smoke_check(reversals, feasibility)
    if not report["passed"]:
        raise ArithmeticError("Saved fixed-structure results failed their smoke check")
    if args.smoke:
        print(json.dumps(report, indent=2))
        return
    svg, pdf = make_figure(reversals, feasibility)
    print(json.dumps({**report, "svg": str(svg), "pdf": str(pdf),
                      "pdf_tex": str(OUTPUT / "policy_structure.pdf_tex")}, indent=2))


if __name__ == "__main__":
    main()
