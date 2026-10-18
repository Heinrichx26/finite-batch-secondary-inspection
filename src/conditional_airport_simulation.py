"""Conditional FCFS read/physical-inspection queues driven by public hourly counts.

Counts define scenario handoffs, not observed arrival times. Staff, durations,
prevalence, and within-hour patterns are explicit scenario inputs. Required
checks hold bags until completion. This program does not generate figures.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import heapq
import json
import math
import os
import time

import numpy as np
import pandas as pd
from conditional_theory_experiment import BASE, prepare, target_mask, cascade, needed_q

POLICIES = {"original": ("retain_alarm", 0., False),
            "retain_q50": ("retain_alarm", .5, True),
            "retain_target": ("retain_alarm", .8881818181818166, True),
            "retain_full": ("retain_alarm", 1., True),
            "reader_clear_full": ("second_reader_clearance", 1., True)}
_STATE = None


def fcfs(arrivals, durations, capacity):
    if len(arrivals) == 0:
        return np.empty(0), np.empty(0)
    assert capacity >= 1 and np.all(np.diff(arrivals) >= -1e-9)
    servers = [0.]*int(capacity)
    finishes = np.empty(len(arrivals)); waits = np.empty(len(arrivals))
    for i, (arrival, service) in enumerate(zip(arrivals, durations)):
        free = servers[0]
        start = max(free, float(arrival))
        finish = start+float(service)
        heapq.heapreplace(servers, finish)
        finishes[i] = finish; waits[i] = start-arrival
    return finishes, waits


def smoke_queue():
    cases = 0
    for c, expected in [(1, [1., 2., 3.]), (2, [1., 1., 2.]), (3, [1., 1., 1.])]:
        finish, wait = fcfs(np.zeros(3), np.ones(3), c)
        assert np.allclose(finish, expected)
        assert np.allclose(wait, np.array(expected)-1.)
        cases += 1
    f, w = fcfs(np.array([1., 10.]), np.array([2., 3.]), 1)
    assert np.allclose(f, [3., 13.]) and np.allclose(w, 0.)
    assert len(fcfs(np.array([]), np.array([]), 1)[0]) == 0
    # Mandatory checks after horizon stay held, never become releases.
    arrivals = np.zeros(5); f, _ = fcfs(arrivals, np.ones(5)*20, 1)
    horizon = 50
    assert (f > horizon).sum() == 3 and (f <= horizon).sum() == 2
    return dict(passed=True, queue_reference_cases=cases+3)


def build_state():
    first, second, parameters, _, _ = prepare()
    groups = sorted(set(second.groupby("image").person.apply(frozenset)), key=lambda s: sorted(s))
    profile_lookup = []
    import itertools
    for people in itertools.product(*[sorted(g) for g in groups]):
        mapping = second[second.person.isin(people)].set_index("image").refer
        profile_lookup.append(first.image.map(mapping).to_numpy(bool))
    source = pd.read_csv(BASE/"data/curated/airport_scale/tsa_checkpoint_hour_20260927_20261003.csv")
    source = source[~source.checkpoint.str.contains("employee", case=False)].copy()
    source["timestamp"] = pd.to_datetime(source.date+" "+source.hour)
    source["hour_index"] = ((source.timestamp-pd.Timestamp("2026-09-27"))/pd.Timedelta(hours=1)).astype(int)
    p = parameters[parameters.endpoint.eq("gun_knife")].iloc[0].to_dict()
    state = {"flow": source, "p": p, "route": first.route.to_numpy(),
             "positive_indices": np.flatnonzero(target_mask(first, "gun_knife")),
             "empty_indices": np.flatnonzero(~first.truth),
             "refer": np.asarray(profile_lookup)}
    return state


def init_worker():
    global _STATE
    _STATE = build_state()


def lognormal_from_z(z, mean, cv):
    sigma = math.sqrt(math.log1p(cv*cv))
    return mean*np.exp(sigma*z-.5*sigma*sigma)


def simulate(task):
    state = _STATE
    airport, policy_name, factor, pattern, replicate, smoke = task[:6]
    read_mean, physical_mean, physical_cv = task[6:] if len(task) > 6 else (20., 120., 1.)
    policy, q, gate_active = POLICIES[policy_name]
    flow = state["flow"][state["flow"].airport.eq(airport)].copy()
    if smoke:
        flow = flow[(flow.hour_index >= 24) & (flow.hour_index < 26)].copy()
        flow["reported_persons_including_KCM"] = np.minimum(flow.reported_persons_including_KCM, 200)
    rng = np.random.default_rng(20261014000+replicate*1000+["SFO", "DEN", "EWR"].index(airport)*100)
    p = state["p"]
    reference = cascade(p, POLICIES["retain_target"][1], .9999, .99, .99, 1., "retain_alarm")
    totals = {k:0. for k in ["scans", "targets", "reads", "physical", "expected_missed",
              "realized_missed", "expected_released_missed_15m", "held_after_15m",
              "checked_by_15m", "completed_by_15m", "eventual_detected",
              "physical_wait_total", "read_wait_total", "held_at_week_end"]}
    physical_waits = []; read_waits = []; release_delays = []
    station_rows = []; hourly_rows = []
    max_checkpoint_held_anytime = 0; max_checkpoint_held_at_hour_end = 0
    horizon = 7*86400
    for checkpoint, g in flow.groupby("checkpoint", sort=True):
        g = g.sort_values("hour_index")
        counts = g.reported_persons_including_KCM.to_numpy(int)
        hours = g.hour_index.to_numpy(int)
        n = int(counts.sum())
        if not n:
            continue
        # Capacities are hypothetical constants derived from a stated design rule.
        peak = int(counts.max())
        ref_read = max(1, math.ceil(peak*reference["reads_per_scan"]*20/(3600*.85)))
        ref_phys = max(1, math.ceil(peak*reference["physical_per_scan"]*120/(3600*.85)))
        cr = max(1, math.floor(factor*ref_read)); cp = max(1, math.floor(factor*ref_phys))
        h = np.repeat(hours, counts)
        u = rng.random(n)
        offset = u*3600 if pattern == "uniform" else np.where(u < .8, u/.8*900, 900+(u-.8)/.2*2700)
        arrival = h*3600+offset
        order = np.argsort(arrival, kind="stable"); arrival = arrival[order]; h = h[order]
        truth = rng.random(n) < .001
        sample = np.empty(n, int)
        sample[truth] = rng.choice(state["positive_indices"], int(truth.sum()))
        sample[~truth] = rng.choice(state["empty_indices"], int((~truth).sum()))
        route = state["route"][sample]
        profile_by_hour = rng.integers(0, 100, 168)
        referral = state["refer"][profile_by_hour[h], sample]
        selection_mark = rng.random(n)
        selected = (route == "U") & (selection_mark < q)
        gate = (route == "K") & gate_active
        read_needed = selected | (gate & (policy == "second_reader_clearance"))
        physical_needed = (route == "B") | (gate & (policy == "retain_alarm")) | (read_needed & referral)
        zr = rng.standard_normal(n); zp = rng.standard_normal(n)
        read_seconds = lognormal_from_z(zr, read_mean, .5)
        phys_seconds = lognormal_from_z(zp, physical_mean, physical_cv)
        finished = arrival.copy()
        idx_read = np.flatnonzero(read_needed)
        rf, rw = fcfs(arrival[idx_read], read_seconds[idx_read], cr)
        finished[idx_read] = rf
        idx_phys = np.flatnonzero(physical_needed)
        physical_arrival = finished[idx_phys]
        sorted_phys = np.argsort(physical_arrival, kind="stable")
        ordered_idx = idx_phys[sorted_phys]
        pf, pw = fcfs(physical_arrival[sorted_phys], phys_seconds[ordered_idx], cp)
        finished[ordered_idx] = pf
        assert (finished[physical_needed] >= arrival[physical_needed]).all()
        assert (finished[read_needed] >= arrival[read_needed]).all()
        detection = np.where(route == "B", .9999, .99)
        loss = truth.astype(float)*np.where(physical_needed, 1-detection, 1.)
        # Shared physical sensitivity is drawn per 8-hour shift and route.
        shift = np.minimum((h//8), 20)
        theta = np.empty(n)
        for base, mean in [(True, .9999), (False, .99)]:
            values = rng.beta(mean*9, (1-mean)*9, 21)
            mask = (route == "B") if base else (route != "B")
            theta[mask] = values[shift[mask]]
        actual = truth & (~physical_needed | (rng.random(n) > theta))
        timely = finished <= arrival+900
        end_complete = finished <= horizon
        delay = finished-arrival
        totals["scans"] += n; totals["targets"] += int(truth.sum())
        totals["reads"] += len(idx_read); totals["physical"] += len(idx_phys)
        totals["expected_missed"] += float(loss.sum()); totals["realized_missed"] += int(actual.sum())
        totals["expected_released_missed_15m"] += float(loss[timely].sum())
        totals["held_after_15m"] += int((~timely).sum())
        totals["checked_by_15m"] += int((physical_needed & timely).sum())
        totals["completed_by_15m"] += int(timely.sum())
        totals["eventual_detected"] += int((truth & ~actual).sum())
        totals["physical_wait_total"] += float(pw.sum()); totals["read_wait_total"] += float(rw.sum())
        totals["held_at_week_end"] += int((~end_complete).sum())
        physical_waits.append(pw); read_waits.append(rw); release_delays.append(delay)
        active = physical_needed | read_needed
        events = np.r_[arrival[active], finished[active]]
        changes = np.r_[np.ones(int(active.sum()), dtype=int), -np.ones(int(active.sum()), dtype=int)]
        event_order = np.argsort(events, kind="stable")
        checkpoint_peak = int(np.cumsum(changes[event_order]).max()) if len(events) else 0
        max_checkpoint_held_anytime = max(max_checkpoint_held_anytime, checkpoint_peak)
        station_rows.append(dict(checkpoint=checkpoint, scans=n, readers=cr, inspectors=cp,
            peak_hour_count=peak, physical_jobs=len(idx_phys), read_jobs=len(idx_read),
            final_completion_hour=float(finished.max()/3600), held_at_week_end=int((~end_complete).sum())))
        # Report the inventory held at each observed hour boundary.
        completions = np.sort(finished)
        for hour in hours:
            cutoff = (hour+1)*3600
            admitted = np.searchsorted(arrival, cutoff, side="right")
            completed = np.searchsorted(completions, cutoff, side="right")
            max_checkpoint_held_at_hour_end = max(max_checkpoint_held_at_hour_end, int(admitted-completed))
            hourly_rows.append(dict(checkpoint=checkpoint, hour_index=int(hour),
                held_at_hour_end=int(admitted-completed)))
    pw = np.concatenate(physical_waits); rw = np.concatenate(read_waits); delay = np.concatenate(release_delays)
    hour = pd.DataFrame(hourly_rows).groupby("hour_index").held_at_hour_end.sum()
    totals.update(airport=airport, policy=policy_name, capacity_factor=factor, arrival_pattern=pattern,
        replicate=replicate, coverage=q, read_seconds_mean=read_mean, inspect_seconds_mean=physical_mean,
        inspect_seconds_cv=physical_cv, scans_per_reported_person=1, target_prevalence=.001,
        physical_wait_mean_seconds=float(pw.mean()) if len(pw) else 0.,
        physical_wait_p95_seconds=float(np.quantile(pw, .95)) if len(pw) else 0.,
        read_wait_mean_seconds=float(rw.mean()) if len(rw) else 0.,
        read_wait_p95_seconds=float(np.quantile(rw, .95)) if len(rw) else 0.,
        release_delay_p95_seconds=float(np.quantile(delay, .95)),
        release_delay_p99_seconds=float(np.quantile(delay, .99)),
        peak_hour_end_held=int(hour.max()),
        max_checkpoint_held_anytime=max_checkpoint_held_anytime,
        max_checkpoint_held_at_hour_end=max_checkpoint_held_at_hour_end,
        held_fraction_after_15m=totals["held_after_15m"]/totals["scans"],
        expected_misses_per_million_scans=totals["expected_missed"]/totals["scans"]*1e6,
        final_drain_hours=max(0., max(x["final_completion_hour"] for x in station_rows)-168),
        total_readers=sum(x["readers"] for x in station_rows), total_inspectors=sum(x["inspectors"] for x in station_rows))
    return totals


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int); ap.add_argument("--service-stress", action="store_true"); args = ap.parse_args()
    name = "conditional_service_stress" if args.service_stress else "conditional_airport"
    out = BASE/"results"/(name+"_smoke" if args.smoke else name)
    out.mkdir(parents=True, exist_ok=True); started = time.time()
    checks = smoke_queue()
    if not args.smoke:
        assert json.loads((BASE/"results"/(name+"_smoke")/"verification.json").read_text())["passed"]
    init_worker(); p = _STATE["p"]
    # Every available airport contributes to conditional load envelopes.
    aggregate = _STATE["flow"].groupby(["airport", "date", "hour"], as_index=False).reported_persons_including_KCM.sum()
    envelope = []
    for policy, (kind, q, active) in POLICIES.items():
        metrics = cascade(p, q, .9999, .99, .99, 1., kind)
        if not active:
            metrics["physical_per_scan"] = metrics["base_physical_per_scan"]
            metrics["reads_per_scan"] = 0.
            metrics["conditional_final_miss"] = p["B_positive"]*.0001+p["K_positive"]+p["U_positive"]
        for airport, g in aggregate.groupby("airport"):
            total = int(g.reported_persons_including_KCM.sum()); peak = int(g.reported_persons_including_KCM.max())
            envelope.append(dict(airport=airport, policy=policy, reported_week_persons=total,
                peak_hour_persons=peak, expected_week_physical=total*metrics["physical_per_scan"],
                expected_week_reads=total*metrics["reads_per_scan"],
                conditional_expected_misses_per_million=.001*metrics["conditional_final_miss"]*1e6))
    pd.DataFrame(envelope).to_csv(out/"all_airport_conditional_load.csv", index=False)
    import itertools
    tasks = list(itertools.product(["SFO"] if args.smoke else ["SFO", "DEN", "EWR"],
        list(POLICIES), [1.] if args.smoke else [.75, 1., 1.25],
        ["uniform"] if args.smoke else ["uniform", "burst"],
        range(1 if args.smoke else 8), [args.smoke]))
    if args.service_stress:
        tasks = list(itertools.product(["SFO"], ["retain_target", "reader_clear_full"], [1.], ["burst"],
            range(1 if args.smoke else 8), [args.smoke], [10., 20., 40.], [60., 120., 240.], [.5, 1., 1.5]))
    workers = 1 if args.smoke else (args.workers or min(4, max(1, (os.cpu_count() or 4)//4)))
    rows = []
    if workers == 1:
        iterator = map(simulate, tasks)
        for row in iterator:
            rows.append(row)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
            for index, row in enumerate(pool.map(simulate, tasks, chunksize=2), 1):
                rows.append(row)
                if index % 30 == 0:
                    print(f"Completed {index}/{len(tasks)} conditional runs, {time.time()-started:.1f}s", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(out/"simulation_replicates.csv", index=False)
    keys = ["airport", "policy", "capacity_factor", "arrival_pattern"]
    if args.service_stress:
        keys += ["read_seconds_mean", "inspect_seconds_mean", "inspect_seconds_cv"]
    means = results.groupby(keys).mean(numeric_only=True).reset_index()
    means.to_csv(out/"simulation_scenario_means.csv", index=False)
    risks = []
    for policy, (kind, q, active) in POLICIES.items():
        risk = cascade(p, q, .9999, .99, .99, 1., kind)["conditional_final_miss"]
        if not active:
            risk = p["B_positive"]*.0001+p["K_positive"]+p["U_positive"]
        risks.append(dict(policy=policy, analytical_conditional_final_miss=risk,
                          analytical_expected_misses_per_million_scans=risk*1000))
    pd.DataFrame(risks).to_csv(out/"analytical_policy_risk.csv", index=False)
    spread = results.groupby(keys).agg(repetitions=("replicate", "size"),
        miss_mean=("expected_misses_per_million_scans", "mean"),
        miss_sd=("expected_misses_per_million_scans", "std"),
        held_mean=("held_fraction_after_15m", "mean"), held_sd=("held_fraction_after_15m", "std"),
        wait95_mean=("physical_wait_p95_seconds", "mean"),
        wait95_sd=("physical_wait_p95_seconds", "std")).reset_index()
    spread.to_csv(out/"simulation_monte_carlo_spread.csv", index=False)
    assert (results.held_after_15m+results.completed_by_15m).eq(results.scans).all()
    assert (results.eventual_detected+results.realized_missed).eq(results.targets).all()
    assert results.expected_released_missed_15m.le(results.expected_missed+1e-9).all()
    checks.update(smoke=args.smoke, runs=len(rows), workers=workers, seconds=time.time()-started,
        bag_scenario_runs=int(results.scans.sum()), airports_in_load_envelope=aggregate.airport.nunique(),
        no_pending_check_released=True, target_outcomes_conserved=True,
        common_random_inputs_across_policies=True, source_hour_counts_are_not_observed_arrivals=True,
        independent_fields_estimated=False, field_queues_validated=False,
        service_staff_prevalence_within_hour_arrivals_are_scenarios=True,
        source_person_profile_retained_per_checkpoint_hour=True,
        source_clock_preserved=True)
    (out/"verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
