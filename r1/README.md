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
python src/r1_revision_validation.py --smoke
python src/verify_r1_capacity.py --smoke
python src/r1_compare_benchmarks.py --smoke
python src/compare_sequential_policies_20261018.py --smoke
python src/plot_r1_image_unit.py --smoke
python src/plot_r1_fixed_structure_examples.py --smoke
```

## Full calculations

Run the image-unit analysis first. Its saved verification record is an input to
the expanded sensitivity analysis:

```text
python src/r1_image_unit_analysis.py --full --bootstrap 2000 --copula-grid 40001 --copula-refinements 40
python src/r1_followup_analysis.py --full
python src/r1_revision_validation.py --full --bootstrap 2000
python src/verify_r1_capacity.py --full
python src/r1_compare_benchmarks.py --full
python src/compare_sequential_policies_20261018.py --full
python src/plot_r1_image_unit.py --full
python src/plot_r1_fixed_structure_examples.py --full
```

The analysis writes tables and checks under `results/R1_image_unit/`. The plotting
programs use saved result tables to recreate the figures independently of the
analysis. They write editable SVG, vector PDF, and LaTeX figure files. The source
tables for the fixed-count examples are under `results/R1_fixed_structure/`.

## Results by section

`results/R1_image_unit/policy_capacity.csv` and `risk_constrained_thresholds.csv`
give the matched policy comparisons and capacity thresholds in Sections 5.3–5.6.
`results/R1_followup/external_panel_targets.csv` gives the target-image judgment
counts and the comparison at a common target count in Section 6.3 and
Supplementary Section S5.3. `service_completion.csv`,
`conditional_detection_requirements.csv`, and `validation_sample_sizes.csv` in
the same directory give the workload, detection thresholds, and trial sample
sizes in Section 5.7 and Supplementary Section S5.6.

This archive accompanies the article and is available at
https://github.com/Heinrichx26/finite-batch-secondary-inspection/releases/tag/v1.1.0.

## Contents

- `data/`: licensed public inputs and their provenance.
- `src/`: two-stage finite-image analysis, sensitivity checks, and standalone plotting.
- `results/R1_image_unit/`: application outputs, convergence checks, image mapping,
  and independent calculations.
- `results/R1_followup/`: matched comparisons for repeated targets, copula extremum
  enclosures, participant and order sensitivity, quota and feedback calculations,
  service completion, conditional detection requirements, and validation sample sizes.
- `results/R1_fixed_structure/`: the saved calculations used to render the supplementary
  fixed-count figure.
- `results/R1_benchmarks/` and `results/sequential_comparison_20261018/`: the panel,
  diagram and sequential-policy comparisons.
- `article/R1_Source/figures/`: vector figures, editable figure sources, and LaTeX overlays.
- `checksums.json`: SHA-256 checksums for all files in this archive except this manifest.

Additional theoretical verification scripts are available at
https://github.com/Heinrichx26/finite-batch-secondary-inspection/tree/v1.0.0.
