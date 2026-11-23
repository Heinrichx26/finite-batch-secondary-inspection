# Finite-batch secondary inspection

The [reproducibility files for the revised article](r1/README.md) include
the image-level comparisons, checkpoint service calculations, and conditional
physical-detection requirements. The same files and checksums are available as
`Reproducibility_Materials.zip` in [Release v1.2.1](https://github.com/Heinrichx26/finite-batch-secondary-inspection/releases/tag/v1.2.1).

From `r1/`, run `python src/plot_r1_restored_figures.py --smoke` to verify the
saved inputs and `python src/plot_r1_restored_figures.py --full` to regenerate
Figure 3 and Supplementary Figures S2–S4.

The sections below describe the initial-submission analysis archived at `v1.0.0`.

Analysis code for **Finite-batch reliability of secondary inspection under limited
search capacity: Sequential policies and resource requirements**.

The code evaluates complete target referral, physical-search demand, mean target
misses, incomplete response pairing, and the information requirements of sequential
inspection. Public online screening responses provide the application inputs.
Physical-detection and service conditions are specified numerical scenarios.

## Setup

Use Python 3.11 or later, from the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment using the command appropriate to your shell before
installing packages. `requirements-tested.txt` records the versions used to check
this release. Results are written under `results/`; original and processed inputs
are under `data/`; programs are under `src/`.

## Start with the smoke checks

```bash
python src/run_manuscript_study.py --smoke
python src/capacity_capped_review_study.py --smoke
python src/run_recent_method_comparison.py --smoke
python src/compare_sequential_policies_20261018.py --smoke
python src/stopping_information_theory.py --smoke
python src/stopping_information_theory_independent.py --smoke
```

Run each full calculation only after its smoke check succeeds on the same source
version. The script-specific interfaces are shown below.

## Calculation map

| Result | Program | Full calculation |
|---|---|---|
| Fixed-count reliability and common-distribution policy effects | `run_manuscript_study.py` | no arguments |
| Referral stopping with reading and search limits | `capacity_capped_review_study.py` | `--full` |
| Expanded stopping-capacity and joint-risk tables | `capacity_capped_review_report.py` | `--full` |
| Stopping under marginal-frequency changes | `capacity_stopping_marginal_sensitivity.py` | `--full` |
| Target-omission sensitivity | `target_support_upgrade.py` | `--full` |
| Alternative risk definitions and target-category decisions | `target_support_decision_report.py` | `--full` |
| Finite marginal load bounds | `check_resource_capacity_bounds.py` | no arguments |
| Reweighted full-review resource requirements | `robust_capacity_reweighting.py` | no arguments |
| Physical-detection requirements | `review_detection_requirements.py` | `--full` |
| Conditional completion and dependent detection | `engineering_capability_conditions.py` | no arguments |
| Recent inspector, dependence and decision-diagram comparisons | `run_recent_method_comparison.py` | no arguments |
| Exact adaptive, precommitted and uniform sequential policies | `compare_sequential_policies_20261018.py` | `--full` |
| Information-dependent group-quota theorem | `stopping_information_theory.py` | `--full` |
| Independent literal-placement checks of that theorem | `stopping_information_theory_independent.py` | `--full` |
| Endpoint-definition sensitivity | `endpoint_definition_robustness.py` | `--full` |
| Participant-level response-frequency resampling | `participant_weight_bootstrap.py` | `--full` |

Most programs in this table accept `--smoke`. `capacity_capped_review_report.py`
requires the completed parent stopping study even for its own `--smoke` check.
Run the fixed-count and stopping studies before expanding their reporting tables.
Saved numerical outputs are included for inspection without rerunning experiments.
Some full studies are substantially more expensive than their smoke checks.

`data/curated/full_counterfactual/paired_responses.npz` supplies the processed
response arrays used by the engineering calculations. To regenerate those arrays:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from full_counterfactual_model import build_records; build_records()"
```

`run_capacity_online.py` produces specified service scenarios; its `--refine`
option writes `results/capacity_online_refined/`. Run its `--smoke` check first.
`run_general_policy_bounds.py` reuses those saved service results and does not
repeat service simulations. Independent `validate_*.py` and
`check_finite_population_kernel.py` scripts provide small-population checks.
Use each program's `--help` for its options.

## Figures

Plotting reads saved result tables and is separate from experiment execution:

```bash
python src/plot_stopping_resource_results.py
python src/plot_recent_method_comparison.py
python src/plot_three_point_upgrade.py --smoke
python src/plot_three_point_upgrade.py
```

The first command produces manuscript Figure 2 with each panel's legend below
its own axis, including editable SVG and selectable-text vector PDF outputs.

## Data and interpretation

The five original inputs are from Figshare version 2,
https://doi.org/10.6084/m9.figshare.30316936.v2, under CC BY 4.0. Attribution,
download addresses and checksums are provided in `THIRD_PARTY_NOTICES.md` and
`data/source_provenance.json`.

The application contains 1,040 assisted judgments and two separately observed
unaided groups of ten participants each. Its two eligible target judgments
repeat one knife image. The response combinations preserve each observed
participant's complete pattern within their assigned image group. Conditional
physical-detection and service assumptions are recorded separately from those
response inputs.

The eight-image sequential comparison is a constructed numerical experiment.
It adapts finite-stage belief updating and Bellman recursion from the sequential
inspection method family. Its calibrated and misspecified-prior results quantify
the consequences of the specified information assumptions.

`release_manifest.json` records SHA-256 checksums of the published files. Release
validation logs are under `results/release_validation/`.
