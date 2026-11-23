# Reproducibility materials

This archive contains the public inputs, analysis and plotting scripts, numerical
results, and editable figures for *Secondary inspection under limited search
capacity: Sequential stopping and reliability guarantees*.
All paths are relative to the archive root.

## Data provenance

The five experiment-response and stimulus files are distributed in version 2 of
the Figshare release at https://doi.org/10.6084/m9.figshare.30316936.v2 under the
Creative Commons Attribution 4.0 International license. The positive-image
annotation table is supplied under the same license in that release. `data/`
contains the six inputs. `data/SOURCE_PROVENANCE.json` records their source,
version, license, and checksums.

`data/checkpoint/` contains *Airport Security Checkpoint Data*, version 1, by
Stef Janssen, Régis van der Sommen, Alexander Dilweg, and Alexei Sharpanskykh,
4TU.ResearchData, https://doi.org/10.4121/uuid:6246e6a6-d149-420f-a6a4-7afd08f154b5.
The source workbook and documentation are distributed unchanged under the
Creative Commons Attribution 4.0 International license
(https://creativecommons.org/licenses/by/4.0/).

## Environment

The calculations were checked with Python 3.11. Install the versions listed in
`requirements-tested.txt` in an isolated environment:

```text
python -m venv .venv
```

Activate that environment for the current shell, then run:

```text
python -m pip install -r requirements-tested.txt
```

## Quick verification

These commands check the calculations on small cases:

```text
python src/r1_image_unit_analysis.py --smoke
python src/r1_followup_analysis.py --smoke
python src/r1_external_sensitivity_analysis.py --smoke
python src/r1_revision_validation.py --smoke
python src/verify_r1_capacity.py --smoke
python src/r1_compare_benchmarks.py --smoke
python src/compare_sequential_policies_20261018.py --smoke
python src/plot_r1_image_unit.py --smoke
python src/plot_r1_fixed_structure_examples.py --smoke
python src/plot_r1_restored_figures.py --smoke
```

## Full calculations

Run the image-unit analysis first. Its saved verification record is an input to
the expanded sensitivity analysis:

```text
python src/r1_image_unit_analysis.py --full --bootstrap 2000 --copula-grid 40001 --copula-refinements 40
python src/r1_followup_analysis.py --full
python src/r1_external_sensitivity_analysis.py --full
python src/r1_revision_validation.py --full --bootstrap 2000
python src/verify_r1_capacity.py --full
python src/r1_compare_benchmarks.py --full
python src/compare_sequential_policies_20261018.py --full
python src/plot_r1_image_unit.py --full
python src/plot_r1_fixed_structure_examples.py --full
python src/plot_r1_restored_figures.py --full
```

The analysis writes tables and checks under `results/R1_image_unit/`. The plotting
programs use saved result tables to recreate the figures independently of the
analysis. They write editable SVG, vector PDF, and LaTeX figure files. The source
tables for the fixed-count examples are under `results/R1_fixed_structure/`.
The restored-figure script regenerates Figure 3 and Supplementary Figures S2–S4
from the saved dependence, capacity, omission, detection, and timing results.

## Results by section

`results/R1_image_unit/policy_capacity.csv` and `risk_constrained_thresholds.csv`
give the matched policy comparisons and capacity thresholds in Sections 5.3–5.6.
`results/R1_followup/external_panel_targets.csv` gives the target-image judgment
counts and the comparison at a common target count in Section 6.3 and
Supplementary Section S5.3. `conditional_detection_requirements.csv` reports the
calculated all-target detection requirements for stopping, full fixed review,
group quotas, and pooled allocation at matched resource limits.
`service_completion.csv` and Supplementary Section S5.6 report the recorded
service-stage durations and completion estimates.
`results/R1_followup/external_sensitivity.csv` gives the participant-profile and
lane-session bootstrap ranges and the quota sensitivity to target composition.
`consecutive_service_session_bootstrap.csv` gives a separate lane-session
cluster bootstrap for overlapping consecutive three- and five-check windows;
it retains each observed session's sequence when resampling.
`results/R1_followup/quota_capacity_frontier.csv` gives exact recovery limits
at two through five search places for composition `(1,1)` and all three
two-target compositions. `quota_composition_robustness.csv` quantifies quota
coverage across a defined range of composition frequencies under a 60%
mean-unreferred-target limit. The comparison covers the complete two-target
composition set.
`results/R1_followup/external_image_deletion.csv` gives the response sensitivity
after deleting each target-image identity in turn. `target_category_sensitivity.csv`
and `target_category_capacity_frontier.csv` compare gun-or-knife targets with all
five labeled object classes in the two selected image schedules. They report the
difficulty-group counts and the capacities needed under the article criteria.
`external_matched_target_sensitivity.csv` and
`external_matched_target_deletion.csv` report participant sensitivity on target
images shared with the policy schedules. `external_image_generalisation.csv`
compares shared and other target-image estimates within the same participants.
`external_sensitivity.csv` gives the session-bootstrap interval for five checks
completed within seven minutes.

This archive accompanies the article and is available at
https://github.com/Heinrichx26/finite-batch-secondary-inspection/releases/tag/v1.2.1.

## Contents

- `data/`: licensed public inputs and their provenance.
- `src/`: two-stage finite-image analysis, sensitivity checks, and standalone plotting.
- `results/R1_image_unit/`: application outputs, convergence checks, image mapping,
  and independent calculations.
- `results/R1_followup/`: matched comparisons for repeated targets, copula extremum
  enclosures, participant and order sensitivity, quota and feedback calculations,
  service completion, and quota-composition sensitivity.
- `results/R1_fixed_structure/`: the saved calculations used to render the supplementary
  fixed-count figure.
- `results/R1_benchmarks/` and `results/sequential_comparison_20261018/`: the panel,
  diagram and sequential-policy comparisons.
- `article/R1_Source/figures/`: vector figures, editable figure sources, and LaTeX overlays.
- `checksums.json`: SHA-256 checksums for all files in this archive except this manifest.

Additional theoretical verification scripts are available at
https://github.com/Heinrichx26/finite-batch-secondary-inspection/tree/v1.0.0.
