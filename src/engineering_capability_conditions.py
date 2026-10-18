"""Engineering capability conditions, with independent joint-failure checks.

No service or physical-detection capability is estimated from online responses.
The calculations report requirements and scenarios from declared capability inputs.
Plotting and manuscript edits are intentionally separate.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linprog
from scipy.stats import beta, hypergeom

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'results' / 'engineering_conditions_20261018'


def direct_joint_failure_minimum(completion_lower, failure_upper):
    """LP over every within-batch miss pattern and the incomplete-batch state.

    Target miss bounds condition on all-stage completion. Target outcomes and
    completion may be arbitrarily dependent. Only independent repeated trials
    enter the separate calibration calculation.
    """
    eps = np.asarray(failure_upper, float)
    m = len(eps)
    assert 0. <= completion_lower <= 1. and np.all((eps >= 0.) & (eps <= 1.))
    patterns = np.asarray(list(itertools.product([0, 1], repeat=m)), float)
    # Last variable is probability of an incomplete batch.
    objective = np.zeros(2 ** m + 1); objective[0] = 1.
    constraints = []; limits = []
    completion = np.r_[np.ones(2 ** m), 0.]
    constraints.append(-completion); limits.append(-completion_lower)
    for t, upper in enumerate(eps):
        constraints.append(np.r_[patterns[:, t] - upper, 0.]); limits.append(0.)
    result = linprog(objective, A_ub=constraints, b_ub=limits,
                     A_eq=np.ones((1, len(objective))), b_eq=[1.],
                     bounds=(0., None), method='highs')
    if not result.success:
        raise AssertionError(result.message)
    return float(result.fun)


def transportation_lower(matrix, delta):
    """Independent finite LP, allowing total-variation shifts in both margins."""
    matrix = np.asarray(matrix, float)
    first, second = matrix.shape
    count = first * second
    p = np.ones(first) / first; q = np.ones(second) / second
    size = count + first + second
    objective = np.r_[matrix.ravel(), np.zeros(first + second)]
    rows = []; rhs = []
    for i in range(first):
        row = np.zeros(size); row[i * second:(i + 1) * second] = 1.
        row[count + i] = -1.; rows.append(row); rhs.append(p[i])
        row = np.zeros(size); row[i * second:(i + 1) * second] = -1.
        row[count + i] = -1.; rows.append(row); rhs.append(-p[i])
    for j in range(second):
        row = np.zeros(size); row[j:count:second] = 1.
        row[count + first + j] = -1.; rows.append(row); rhs.append(q[j])
        row = np.zeros(size); row[j:count:second] = -1.
        row[count + first + j] = -1.; rows.append(row); rhs.append(-q[j])
    row = np.zeros(size); row[count:count + first] = 1.
    rows.append(row); rhs.append(2. * delta)
    row = np.zeros(size); row[count + first:] = 1.
    rows.append(row); rhs.append(2. * delta)
    equality = np.r_[np.ones(count), np.zeros(first + second)][None, :]
    result = linprog(objective, A_ub=rows, b_ub=rhs, A_eq=equality, b_eq=[1.],
                     bounds=(0., None), method='highs')
    if not result.success:
        raise AssertionError(result.message)
    return float(np.clip(result.fun, 0., 1.))


def exact_lower_all_success(trials, error_allowance):
    """One-sided exact binomial lower limit when every trial succeeds."""
    return float(error_allowance ** (1. / trials)) if trials else 0.


def zero_failure_trials(required_success, error_allowance):
    if required_success <= 0.:
        return 0
    if required_success >= 1.:
        return None
    trials = math.ceil(math.log(error_allowance) / math.log(required_success))
    assert exact_lower_all_success(trials, error_allowance) >= required_success - 1e-14
    assert trials == 1 or exact_lower_all_success(trials - 1, error_allowance) < required_success
    return trials


def checks(smoke):
    rng = np.random.default_rng(20261018)
    maximum_error = 0.; joint_checks = 0
    ms = range(1, 4) if smoke else range(1, 8)
    repetitions = 5 if smoke else 60
    for m in ms:
        candidates = [[0.] * m, [1.] * m, [.2] * m]
        for _ in range(repetitions):
            raw = rng.uniform(.001, 1., m)
            wanted_sum = rng.uniform(.01, min(1.5, float(m)))
            candidates.append(np.minimum(1., raw / raw.sum() * wanted_sum).tolist())
        for eps in candidates:
            for h in [0., .4, 1.]:
                direct = direct_joint_failure_minimum(h, eps)
                expected = h * max(0., 1. - sum(eps))
                maximum_error = max(maximum_error, abs(direct - expected)); joint_checks += 1
    assert maximum_error <= 2e-12
    calibration_checks = 0
    for probability in [.5, .8, 8/9, 16/17, .99, .9999]:
        for alpha in [.05, .025, .05 / 3]:
            trials = zero_failure_trials(probability, alpha)
            assert abs(exact_lower_all_success(trials, alpha) - beta.ppf(alpha, trials, 1)) < 1e-13
            calibration_checks += 1
    # Check workload-stratified kernels by explicit selected-subset enumeration.
    kernel_checks = 0
    limit = 4 if smoke else 8
    for n_pool in range(1, limit + 1):
        for m in range(1, min(n_pool, 3) + 1):
            target = set(range(m)); others = n_pool - m
            for k in range(others + 1):
                referring = target | set(range(m, m + k))
                for n in range(n_pool + 1):
                    direct = 0.
                    for chosen in itertools.combinations(range(n_pool), n):
                        selected = set(chosen)
                        if not target.issubset(selected):
                            continue
                        load = 2 + len(selected & referring)
                        h = max(0., 1. - .03 * load)
                        d = max(0., 1. - m * .01 * load)
                        direct += h * d / math.comb(n_pool, n)
                    reduced = 0.
                    if n >= m:
                        inclusion = math.comb(n, m) / math.comb(n_pool, m)
                        for r in range(k + 1):
                            probability = float(hypergeom.pmf(r, others, k, n - m)) if others else float(r == 0)
                            load = 2 + m + r
                            reduced += inclusion * probability * max(0., 1. - .03 * load) * max(0., 1. - m * .01 * load)
                    maximum_error = max(maximum_error, abs(direct - reduced)); kernel_checks += 1
    assert maximum_error < 2e-12
    return {'passed': True, 'joint_failure_linear_programs': joint_checks,
            'exact_binomial_checks': calibration_checks, 'subset_kernel_checks': kernel_checks,
            'maximum_absolute_error': maximum_error}


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def full_outputs():
    source = np.load(BASE / 'data/curated/full_counterfactual/paired_responses.npz')
    route = source['route']; target = source['gun_knife']; referrals = source['referrals']
    target_counts = [int(np.sum(target & (route == i))) for i in range(3)]
    assert target_counts == [286, 2, 2]
    mandatory = int(np.sum(route < 2)); n_pool = int(np.sum(route == 2)); m = target_counts[2]
    assert (mandatory, n_pool, m) == (796, 244, 2)
    assert bool(referrals[:, (route == 2) & target].all())
    total_non_target = np.sum(referrals[:, (route == 2) & ~target], axis=1).reshape(10, 10)
    eps0, eps1 = .0001, .01
    independent = (1 - eps0) ** target_counts[0] * (1 - eps1) ** sum(target_counts[1:])
    detection_lower = max(0., 1. - target_counts[0] * eps0 - sum(target_counts[1:]) * eps1)
    assert abs(detection_lower - .9314) < 1e-12
    rows = []
    for capacity, delta in [(907, 0.), (930, 0.), (949, 0.), (949, .05), (969, .05)]:
        event = (mandatory + m + total_non_target <= capacity).astype(float)
        lower_s = transportation_lower(event, delta)
        for h in [1., .95, .90]:
            rows.append({'capacity': capacity, 'frequency_radius': delta,
                         'referral_and_count_lower': lower_s, 'completion_conditional_lower': h,
                         'all_target_detection_conditional_lower': detection_lower,
                         'success_lower_arbitrary_target_dependence': lower_s * h * detection_lower,
                         'success_lower_independent_detection_comparator': lower_s * h * independent,
                         'completion_required_for_80_percent': .8 / (lower_s * detection_lower) if lower_s else None,
                         'required_detection_for_80_percent': .8 / (lower_s * h) if lower_s else None})
    write_csv(OUT / 'capability_requirements.csv', rows)
    capacity_rows = []
    candidates = np.unique(mandatory + m + total_non_target)
    for delta in [0., .05]:
        lower_by_capacity = [(int(c), transportation_lower((mandatory + m + total_non_target <= c).astype(float), delta)) for c in candidates]
        for h in [1., .95, .90]:
            feasible = [(c, s * h * detection_lower) for c, s in lower_by_capacity if s * h * detection_lower >= .8 - 1e-12]
            capacity_rows.append({'frequency_radius': delta, 'completion_conditional_lower': h, 'target_success': .8, 'minimum_full_review_capacity': feasible[0][0] if feasible else None, 'success_at_minimum': feasible[0][1] if feasible else None})
    write_csv(OUT / 'minimum_capacity_requirements.csv', capacity_rows)
    # Exact all-success validation designs; alpha controls confidence, not reliability.
    validation = []
    for label, p, alpha, kind in [
        ('joint_detection_C949_radius0', 8/9, .05, 'completed target-matched batch'),
        ('joint_detection_C949_radius005', 16/17, .05, 'completed target-matched batch'),
        ('original_request_search', 1 - eps0, .025, 'target search in original-request route'),
        ('retained_or_optional_search', 1 - eps1, .025, 'target search in alarm-or-optional route')]:
        trials = zero_failure_trials(p, alpha)
        validation.append({'validation_input': label, 'trial_unit': kind,
                           'required_success_probability': p, 'one_sided_error_allowance': alpha,
                           'minimum_all_success_trials': trials,
                           'exact_lower_at_minimum': exact_lower_all_success(trials, alpha),
                           'exact_lower_one_fewer': exact_lower_all_success(trials-1, alpha)})
    write_csv(OUT / 'zero_failure_validation_requirements.csv', validation)
    # Requirement sensitivity to target count, preserving the same route rates.
    composition = []
    for original_targets in [0, 10, 100, 286, 500]:
        for followup_targets in [1, 2, 4, 10]:
            d = max(0., 1. - original_targets * eps0 - followup_targets * eps1)
            composition.append({'original_request_targets': original_targets,
                                'alarm_or_optional_targets': followup_targets,
                                'conditional_detection_lower': d,
                                'minimum_referral_and_completion_probability_for_80_percent': .8/d if d else None})
    write_csv(OUT / 'target_composition_requirements.csv', composition)
    return {'source_route_target_counts': target_counts, 'total_targets': int(target.sum()),
            'source_provenance': 'source_record_structure and paired_responses only; no capability fit',
            'declared_route_miss_bounds': [eps0, eps1],
            'conditional_detection_bound_arbitrary_dependence': detection_lower,
            'independent_product_comparator': independent,
            'absolute_difference': independent - detection_lower,
            'capability_rows': len(rows), 'validation_rows': len(validation),
            'composition_rows': len(composition),
            'measured_detection_or_service_inputs': False}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True); started = time.time()
    if not args.smoke:
        assert json.loads((OUT / 'smoke.json').read_text(encoding='utf-8'))['passed']
    report = checks(args.smoke)
    if not args.smoke:
        report.update(full_outputs())
    report.update({'smoke': args.smoke, 'seconds': time.time() - started})
    destination = OUT / ('smoke.json' if args.smoke else 'verification.json')
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

