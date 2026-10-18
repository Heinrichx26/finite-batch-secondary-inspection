"""Export exact policy/resource results for the manuscript; plots are separate."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

from full_counterfactual_model import build_records, complete_batch_matrix, expected_batch, transport_bounds

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'results/manuscript_20261017'


def hashes():
    return {p: hashlib.sha256((BASE/'src'/p).read_bytes()).hexdigest()
            for p in ['run_manuscript_study.py', 'full_counterfactual_model.py']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    state = build_records(False)
    source_hashes = hashes()
    if not args.smoke:
        gate = json.loads((OUT/'study_smoke.json').read_text())
        assert gate['passed'] and gate['source_sha256'] == source_hashes
    n_values = [0, 2, 217, 218, 243, 244] if args.smoke else range(245)
    capacities = [804, 907, 947, 969] if args.smoke else range(804, 981)
    rows, effect_rows, frontier = [], [], []
    for capacity in capacities:
        full = complete_batch_matrix(state, 'gun_knife', 244, capacity)
        for n in n_values:
            kernel = complete_batch_matrix(state, 'gun_knife', n, capacity)
            lo, independent, hi, _, _ = transport_bounds(kernel)
            elo, eind, ehi, _, _ = transport_bounds(kernel-full)
            means = expected_batch(state, 'gun_knife', n)
            rows.append(dict(capacity=capacity, reviews=n, lower=lo,
                             independent=independent, upper=hi, **means))
            effect_rows.append(dict(capacity=capacity, reviews=n,
                                   lower=elo, independent=eind, upper=ehi))
            assert lo >= -1e-12 and hi <= 1+1e-12
            assert lo <= independent+1e-12 and independent <= hi+1e-12
        current = [r for r in rows[-len(n_values):] if r['conditional_miss'] <= .001+1e-12]
        best = max(current, key=lambda r: (r['lower'], -r['reviews']))
        for target in [.5, .8, .9]:
            feasible = [r for r in current if r['lower'] >= target-1e-12]
            frontier.append(dict(capacity=capacity, required_success=target,
                                 best_guarantee=best['lower'], best_reviews=best['reviews'],
                                 feasible_reviews=' '.join(str(r['reviews']) for r in feasible),
                                 minimum_feasible_reviews=min((r['reviews'] for r in feasible), default=np.nan)))
    a = next(r for r in effect_rows if r['capacity']==907 and r['reviews']==218)
    assert abs(a['lower']+.18871112711463423) < 1e-12
    assert abs(a['upper']-.26291398391022514) < 1e-12
    if not args.smoke:
        pd.DataFrame(rows).to_csv(OUT/'source_policy_frontier.csv', index=False)
        pd.DataFrame(effect_rows).to_csv(OUT/'source_paired_effects.csv', index=False)
        pd.DataFrame(frontier).to_csv(OUT/'source_feasible_sets.csv', index=False)
        # Detection/retention are conditional inputs, each with a separate kernel.
        sensitivity = []
        for retention in [1., .95, .9]:
            for n in [217, 218, 244]:
                for capacity in [907, 947, 949, 969]:
                    ker = complete_batch_matrix(state, 'gun_knife', n, capacity, eta=retention)
                    low, ind, high, _, _ = transport_bounds(ker)
                    sensitivity.append(dict(retention=retention, reviews=n, capacity=capacity,
                        lower=low, independent=ind, upper=high,
                        **expected_batch(state, 'gun_knife', n, eta=retention)))
        pd.DataFrame(sensitivity).to_csv(OUT/'retention_sensitivity.csv', index=False)
        metrics = pd.DataFrame(rows)
        subset = metrics[metrics.capacity.eq(907)]
        risk_feasible = subset[subset.conditional_miss.le(.001+1e-12)]
        assert len(risk_feasible) > 0
        best = risk_feasible.sort_values(['lower', 'reviews'], ascending=[False, True]).iloc[0]
        summary = dict(minimum_risk_feasible_reviews=int(risk_feasible.reviews.min()),
                       best_guarantee_capacity907=float(best.lower),
                       best_reviews_capacity907=int(best.reviews),
                       mandatory_inspections=796, eligible_items=244, eligible_targets=2,
                       source_judgments=1040, source_target_judgments=290,
                       recovery_target_images=1, response_group_sizes=[10,10],
                       detection_all_targets=float(.9999**286*.99**4))
        (OUT/'source_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    report = dict(passed=True, smoke=args.smoke, policy_cells=len(rows), effect_cells=len(effect_rows),
                  source_sha256=source_hashes, elapsed_seconds=time.perf_counter()-start,
                  matches_existing_paired_witnesses=True, population_confidence_claim=False,
                  substantive_evaluation='Exact kernel and common-coupling witnesses agree; usable as a conditional finite-batch illustration.')
    (OUT/('study_smoke.json' if args.smoke else 'study_verification.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
