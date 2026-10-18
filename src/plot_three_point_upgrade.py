"""Render saved capacity and response-failure requirements without experiments."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results/three_point_upgrade_20261018"
FIG = OUT / "figures"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    coverage_path = BASE / "results/review_capacity_tradeoff_20261018/coverage_savings_curve.csv"
    detection_path = BASE / "results/target_support_upgrade_20261018/target_failure_detection_requirements.csv"
    coverage = pd.read_csv(coverage_path)
    coverage = coverage[coverage.requirement_relative_to_referral_detection_ceiling >= .8]
    detection = pd.read_csv(detection_path)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [coverage_path, detection_path, Path(__file__)]}
    assert sorted(coverage.target_count.unique()) == [1, 2, 5, 10]
    curves = detection[(detection.reviews == 244) & np.isclose(detection.tau, .8)
                       & detection.capacity.isin([949, 969])].copy()
    assert len(curves) == 14
    for capacity, base in [(949, .9), (969, 1.)]:
        one = curves[curves.capacity == capacity].sort_values("epsilon")
        assert np.allclose(one.S_lower, base - one.epsilon)
        assert np.allclose(one.required_dstar, .8 / one.S_lower)
    assert all(np.all(np.diff(one.sort_values(
        "requirement_relative_to_referral_detection_ceiling").maximum_omitted_reviews) <= 0)
        for _, one in coverage.groupby("target_count"))
    OUT.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        (OUT / "plot_smoke.json").write_text(json.dumps(
            dict(passed=True, source_hashes=source_hashes), indent=2), encoding="utf8")
        print("Plot input smoke passed")
        return
    gate = json.loads((OUT / "plot_smoke.json").read_text(encoding="utf8"))
    assert gate["passed"] and gate["source_hashes"] == source_hashes
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.95))
    styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]
    for (m, one), style, marker in zip(coverage.groupby("target_count"), styles, markers):
        one = one.sort_values("requirement_relative_to_referral_detection_ceiling")
        axes[0].plot(100*one.requirement_relative_to_referral_detection_ceiling,
                     one.maximum_omitted_reviews, color="black", linestyle=style,
                     drawstyle="steps-post", linewidth=1.1,
                     label=f"{m} target"+("s" if m > 1 else ""))
    axes[0].set(xlabel="Required fraction of attainable ceiling (%)",
                ylabel="Maximum searches that could be saved",
                xticks=[80, 85, 90, 95, 100], ylim=(0, 51))
    axes[0].legend(frameon=False, fontsize=8)
    for capacity, style, marker in [(949, "-", "o"), (969, "--", "s")]:
        one = curves[curves.capacity == capacity].sort_values("epsilon")
        axes[1].plot(100*one.epsilon, 100*one.required_dstar,
                     linestyle=style, color="black", marker=marker,
                     markersize=3.5, linewidth=1.1, label=f"{capacity} searches")
    axes[1].axhline(100, color="0.5", linestyle=":", linewidth=1.)
    axes[1].set(xlabel="Allowed target-referral failure probability (%)",
                ylabel="Required joint detection probability (%)",
                xticks=[0, 5, 10, 15, 20], ylim=(77, 117))
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    for ax, letter in zip(axes, ["(a)", "(b)"]):
        ax.text(0, 1.04, letter, transform=ax.transAxes, fontweight="bold")
        ax.tick_params(direction="out", length=3)
    fig.tight_layout(w_pad=2.4)
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(FIG / ("capacity_and_referral_requirements."+suffix),
                    bbox_inches="tight", dpi=180)
    plt.close(fig)
    (OUT / "figure_manifest.json").write_text(json.dumps(
        dict(passed=True, source_hashes=source_hashes,
             formula_checks="full-review loss equals allowed target-failure probability",
             outputs=[str(p.relative_to(BASE)) for p in sorted(FIG.iterdir())],
             source_code=str(Path(__file__).relative_to(BASE)),
             experiments_run_by_plot_script=False), indent=2), encoding="utf8")
    print("Saved editable SVG, vector PDF and preview PNG")


if __name__ == "__main__":
    main()
