"""Render referral-and-workload comparisons from saved result tables only.

The original study stored J = d_* S. All policy effects, simulated effects,
and their errors therefore become S quantities by division by the same
strictly positive d_*. No response simulation or optimization is rerun here.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "results/manuscript_20261017/comparison"
OUT = DATA / "figures"
# This is the common physical-detection factor used to generate the stored
# comparison outcomes, not an estimate fitted by this plotting script.
STORED_DETECTION_FACTOR = .9999 ** 286 * .99 ** 4


def normalized_comparison_tables():
    """Keep original tables intact and save the exact J-to-S rescaling."""
    specifications = (
        ("yang_copula_sensitivity.csv", "yang_copula_referral_workload.csv",
         ("lower", "upper", "unrestricted_lower", "independent_effect",
          "unrestricted_upper", "lower_uncovered", "upper_uncovered")),
        ("yang_equal_budget_sampling.csv", "yang_referral_workload_sampling.csv",
         ("exact_catalogue_bound", "mean_estimate", "bias", "rmse")),
    )
    tables = []
    sources = {}
    maximum_roundtrip_error = 0.
    for source_name, output_name, columns in specifications:
        columns = list(columns)
        source_path = DATA / source_name
        table = pd.read_csv(source_path)
        original = table.loc[:, columns].to_numpy().copy()
        table.loc[:, columns] = table.loc[:, columns] / STORED_DETECTION_FACTOR
        restored = table.loc[:, columns].to_numpy() * STORED_DETECTION_FACTOR
        error = float(np.max(np.abs(restored - original)))
        maximum_roundtrip_error = max(maximum_roundtrip_error, error)
        if not np.all(np.isfinite(table.loc[:, columns].to_numpy())):
            raise ValueError(f"Non-finite normalized values in {source_name}")
        if not np.array_equal(np.sign(original), np.sign(table.loc[:, columns].to_numpy())):
            raise ValueError(f"Normalization changed an effect sign in {source_name}")
        table.to_csv(DATA / output_name, index=False)
        sources[source_name] = {
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "normalized_output": output_name,
            "divided_columns": list(columns),
            "rows": len(table),
        }
        tables.append(table)
    report = {
        "outcome": "S: every target is referred and all assigned searches fit capacity",
        "conversion": "S = J / stored_detection_factor",
        "stored_detection_factor": STORED_DETECTION_FACTOR,
        "sources": sources,
        "maximum_roundtrip_error": maximum_roundtrip_error,
        "all_effect_signs_preserved": True,
        "new_experiments": 0,
    }
    (DATA / "referral_workload_rescaling.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return tables


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                         "axes.labelsize": 10, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "legend.fontsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), layout="constrained")
    sensitivity, sample = normalized_comparison_tables()
    selected = sensitivity[(sensitivity.review_count == 218) & (sensitivity.inspection_capacity == 907)]
    labels = ["All associations", "Frank", "Clayton", "Clayton 90°", "Clayton 180°", "Clayton 270°"]
    low = [selected.iloc[0].unrestricted_lower, *selected.lower]
    high = [selected.iloc[0].unrestricted_upper, *selected.upper]
    ax = axes[0]
    for index, (left, right) in enumerate(zip(low, high)):
        ax.plot([100 * left, 100 * right], [index, index], color="black", lw=2.6 if index == 0 else 1.6)
        ax.plot([100 * left, 100 * right], [index, index], "|", color="black", markersize=7)
    ax.axvline(0, color=".6", lw=.8, ls="--")
    ax.set_yticks(range(len(labels)), labels); ax.invert_yaxis()
    ax.set_xlabel(r"$S_{218}-S_{244}$ (percentage points)")
    ax.set_title("(a) Assumed association sets", loc="left", fontsize=10)

    ax = axes[1]
    for family, marker in (("Frank", "o"), ("Clayton", "s")):
        for method, style in (("DLMC", "--"), ("ISMC", "-")):
            group = sample[(sample.family == family) & (sample.method == method) & (sample.boundary == "upper")]
            ax.plot(group.conditional_event_evaluations, 100 * group.rmse, color="black" if family == "Frank" else ".5",
                    marker=marker, ls=style, markersize=4, label=f"{family}, {method}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Conditional event evaluations")
    ax.set_ylabel(r"RMSE of upper $S$-effect bound (pp)")
    ax.set_title("(b) Equal simulation budgets", loc="left", fontsize=10)
    ax.legend(frameon=False, loc="lower left")

    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"comparison_methods.{extension}", bbox_inches="tight")
    plt.close(fig)
    print(OUT / "comparison_methods.pdf")


if __name__ == "__main__":
    main()
