"""Observable quota control: independent replay and three-stage service tests.

Run --smoke before the full experiment. Figures are generated separately.
"""
from pathlib import Path
import argparse
import itertools
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import beta
from numba import njit
from full_counterfactual_model import (BASE, build_records, fast_fcfs, schedule,
    lognormal_from_z, transport_bounds)
from capacity_upgrade_model import stopped_rule


@njit(cache=True)
def online_review(primary_finish, route, referral, durations, readers, allowance):
    """FIFO available images; reserve one quota slot for every running review.

    No target indicator is passed to this controller. Waiting images are released
    when all optional slots have been consumed. Original referrals stay mandatory.
    """
    eligible = np.where(route == 2)[0]
    order = eligible[np.argsort(primary_finish[eligible])]
    finish = primary_finish.copy()
    physical = route != 2
    read = np.zeros(len(route), np.bool_)
    end = np.full(readers, np.inf)
    running = np.full(readers, -1, np.int64)
    # [head, admitted) identifies images available and still waiting.
    head = admitted = used = active = 0
    N = len(order)
    if allowance == 0:
        return finish, physical, read, used
    while head < N or active > 0:
        next_arrival = primary_finish[order[admitted]] if admitted < N else np.inf
        t = min(next_arrival, np.min(end))
        if np.isinf(t):
            break
        while admitted < N and primary_finish[order[admitted]] <= t:
            admitted += 1
        for s in range(readers):
            if end[s] <= t:
                i = running[s]
                finish[i] = t
                physical[i] = referral[i]
                used += int(referral[i])
                active -= 1
                end[s] = np.inf
                running[s] = -1
        if used >= allowance:
            assert active == 0
            for j in range(head, N):
                i = order[j]
                finish[i] = max(primary_finish[i], t)
            break
        for s in range(readers):
            if running[s] < 0 and head < admitted and used + active < allowance:
                i = order[head]
                head += 1
                running[s] = i
                end[s] = t + durations[i]
                read[i] = True
                active += 1
        # If images wait while every slot is provisionally reserved, the next
        # review completion decides whether another image may start.
    assert used <= allowance
    return finish, physical, read, used


def online_schedule(arrival, route, referral, primary, secondary, physical_time,
                    capacities, allowance, primary_cache=None):
    pf = fast_fcfs(arrival, primary, capacities[0])[0] if primary_cache is None else primary_cache[0]
    finish, physical, read, used = online_review(pf, route, referral, secondary,
                                                capacities[1], allowance)
    idx = np.flatnonzero(physical)
    idx = idx[np.argsort(finish[idx], kind='stable')]
    finish[idx] = fast_fcfs(finish[idx], physical_time[idx], capacities[2])[0]
    assert np.all(finish >= arrival)
    assert used == int((physical & (route == 2)).sum())
    return dict(finish=finish, physical_needed=physical, read_needed=read,
                additional_searches=used)


def smoke():
    checks = 0
    for readers in [1, 2, 3]:
        for N in range(1, 7):
            r = np.full(N, 2)
            pf = np.arange(N, dtype=float)*.2
            durations = np.ones(N)
            for mask in itertools.product([False, True], repeat=N):
                e = np.asarray(mask)
                for b in [0, 1, N//2, N, N+1]:
                    f, p, read, used = online_review(pf, r, e, durations, readers, b)
                    assert used <= b
                    assert np.all(f >= pf)
                    assert np.array_equal(p, e & read)
                    if b >= e.sum():
                        # At equality the last referral can terminate remaining
                        # empty readings; every actual referral is still retained.
                        assert np.array_equal(p, e)
                    if readers == 1:
                        cumulative = np.cumsum(e)
                        expected = np.arange(N) <= (np.argmax(cumulative >= b)
                            if b > 0 and e.sum() >= b else N)
                        if b == 0: expected[:] = False
                        assert np.array_equal(read, expected)
                    checks += 1
    return dict(passed=True, observable_controller_checks=checks,
                target_labels_not_controller_inputs=True)


def replay_count(state, reps):
    y = state['targets']['gun_knife']; u = state['route'] == 2
    target = y[u]; m = int(target.sum()); N = len(target); b = 111
    rng = np.random.default_rng(20261015017)
    rows = []
    for profile in range(100):
        referral = state['referrals'][profile, u]
        # Each row is a distinct random order of the actual recorded tasks.
        order = np.argsort(rng.random((reps, N)), axis=1)
        er = referral[order]; yr = target[order]
        before = np.cumsum(er, axis=1) - er
        read = before < b
        inspect = er & read
        caught = (inspect & yr).sum(axis=1)
        k = int((referral & ~target).sum())
        exact = stopped_rule(N, m, k, b)
        success = caught == m
        h = int(success.sum())
        low = float(beta.ppf(.025, h, reps-h+1)) if h else 0.
        high = float(beta.ppf(.975, h+1, reps-h)) if h < reps else 1.
        rows.append(dict(profile=profile,replicates=reps,simulated_success=float(success.mean()),
            exact_success=exact['success'],success_low=low,success_high=high,
            simulated_captured=float(caught.mean()),exact_captured=exact['captured'],
            simulated_readings=float(read.sum(axis=1).mean()),exact_readings=exact['readings'],
            maximum_additional_searches=int(inspect.sum(axis=1).max())))
    result = pd.DataFrame(rows)
    assert result.maximum_additional_searches.le(b).all()
    # Pooled simulation error, including deterministic profiles.
    delta = result.simulated_success-result.exact_success
    se = np.sqrt(np.sum(result.exact_success*(1-result.exact_success)/reps))/100
    assert abs(delta.mean()) < max(6*se, 1e-10)
    return result, dict(replays=100*reps,pooled_success_error=float(delta.mean()),
                        pooled_standard_error=float(se), quota_never_exceeded=True)


def service_run(state, configs, reps):
    y = state['targets']['gun_knife']; route = state['route']; size = len(y)
    u = route == 2; N = int(u.sum()); mandatory = int((~u).sum())
    detection = .9999**int((y & (route==0)).sum())*.99**int((y & (route!=0)).sum())
    deadlines = np.array([3600.,4500.,5400.])
    policies = ['fixed_218','full_244','stop_111']
    rows=[]; effects=[]; cells=[]
    # Cell bounds are simultaneous for the prespecified entire service grid.
    alpha=.05/(len(configs)*len(policies)*len(deadlines)*100)
    for ci,(pattern,readers,inspectors,mean,eta) in enumerate(configs):
        joint=np.zeros((3,3,100),int); completion=np.zeros_like(joint)
        missed=np.zeros((3,100)); reads=np.zeros((3,100)); work=np.zeros((3,100))
        delta_count=np.zeros((2,3,100,2),int)
        for rep in range(reps):
            rng=np.random.default_rng(202610151000+rep)
            marks=rng.random(size)
            arrival=marks*3600 if pattern=='uniform' else np.where(marks<.8,marks/.8*900,900+(marks-.8)/.2*2700)
            order=np.argsort(arrival,kind='stable'); arrival=arrival[order]
            z=rng.standard_normal(size)
            pt=lognormal_from_z(z,mean,1.)[order]
            primary=state['primary_seconds'][order]; r=route[order]; yy=y[order]
            cap=(8,readers,inspectors)
            pc=fast_fcfs(arrival,primary,8)
            chosen=np.zeros(size,bool); chosen[rng.permutation(np.flatnonzero(u))[:218]]=True
            chosen=chosen[order]
            # Same referral discount and random inputs across all decisions.
            retained=rng.random(size)<eta
            sample=np.zeros((3,3,100),bool)
            for profile in range(100):
                e=state['referrals'][profile,order].copy()
                e[yy & (r==2)] &= retained[yy & (r==2)]
                secondary=state['secondary_seconds'][profile,order]
                for pi in range(3):
                    if pi==2:
                        out=online_schedule(arrival,r,e,primary,secondary,pt,cap,111,pc)
                    else:
                        selected=chosen if pi==0 else np.ones(size,bool)
                        out=schedule(arrival,r,e,selected,primary,secondary,pt,cap,'retain',pc)
                    p=out['physical_needed']; included=bool(np.all(p[yy]))
                    done=deadlines>=out['finish'].max()
                    sample[pi,:,profile]=done & included
                    completion[pi,:,profile]+=done
                    # Physical detection is integrated, not re-sampled.
                    missed[pi,profile]+=int((yy & ~p).sum()) + .0001*int((yy&p&(r==0)).sum())+.01*int((yy&p&(r!=0)).sum())
                    reads[pi,profile]+=out['read_needed'].sum()
                    work[pi,profile]+=p.sum()
                    if pi==2: assert p.sum()<=mandatory+111
            joint+=sample
            for vi,other in enumerate([0,1]):
                delta_count[vi,:,:,0]+=sample[2]&~sample[other]
                delta_count[vi,:,:,1]+=sample[other]&~sample[2]
        key=dict(arrival_pattern=pattern,readers=readers,inspectors=inspectors,
                 mean_search_seconds=mean,positive_retention=eta,replicates=reps)
        for pi,policy in enumerate(policies):
            risk=(missed[pi]/reps/y.sum()).reshape(10,10)
            simulated_risk=risk.copy()
            # Fixed random-subset inclusion is known exactly. The stopped
            # available-image procedure is bounded from its simulated mean.
            if pi < 2:
                n=218 if pi==0 else N
                exact_risk=(int((y&(route==0)).sum())*.0001+
                    int((y&(route==1)).sum())*.01+
                    int((y&u).sum())*(1-.99*eta*n/N))/int(y.sum())
                risk[:]=exact_risk
                risk_confidence=risk
            else:
                radius=(int((y&u).sum())*.99/int(y.sum()))*np.sqrt(np.log(1100/.05)/(2*reps))
                risk_confidence=np.minimum(1.,risk+radius)
            for di,deadline in enumerate(deadlines):
                h=joint[pi,di]
                vals=detection*h/reps
                lo=np.where(h>0,beta.ppf(alpha/2,np.maximum(h,1),reps-h+1),0.)*detection
                hi=np.where(h<reps,beta.ppf(1-alpha/2,h+1,np.maximum(reps-h,1)),1.)*detection
                a,c,b,*_=transport_bounds(vals)
                rows.append({**key,'policy':policy,'deadline_minutes':deadline/60,
                    'joint_lower':a,'joint_independent':c,'joint_upper':b,
                    'joint_lower_confidence':transport_bounds(lo)[0],
                    'joint_upper_confidence':transport_bounds(hi)[2],
                    'miss_independent':float(risk.mean()),'miss_upper':transport_bounds(risk)[2],
                    'miss_simulated_independent':float(simulated_risk.mean()),
                    'miss_upper_confidence':transport_bounds(risk_confidence)[2],
                    'mean_readings':float(reads[pi].mean()/reps),
                    'mean_searches':float(work[pi].mean()/reps),
                    'completion_independent':float(completion[pi,di].mean()/reps)})
                for pr in range(100):
                    cells.append({**key,'policy':policy,'deadline_minutes':deadline/60,
                                  'profile':pr,'probability':vals[pr]})
        for vi,other in enumerate([0,1]):
            for di,deadline in enumerate(deadlines):
                pos=delta_count[vi,di,:,0]; neg=delta_count[vi,di,:,1]
                vals=detection*(pos-neg)/reps
                def bounds(h):
                    low=np.where(h>0,beta.ppf(alpha/4,np.maximum(h,1),reps-h+1),0.)
                    high=np.where(h<reps,beta.ppf(1-alpha/4,h+1,np.maximum(reps-h,1)),1.)
                    return low,high
                pl,ph=bounds(pos);nl,nh=bounds(neg)
                a,c,b,*_=transport_bounds(vals)
                effects.append({**key,'contrast':'stop_minus_'+policies[other],
                    'deadline_minutes':deadline/60,'effect_lower':a,'effect_independent':c,'effect_upper':b,
                    'effect_lower_confidence':transport_bounds(detection*(pl-nh))[0],
                    'effect_upper_confidence':transport_bounds(detection*(ph-nl))[2]})
        print(f'Service configurations {ci+1}/{len(configs)} completed',flush=True)
    return pd.DataFrame(rows),pd.DataFrame(effects),pd.DataFrame(cells)


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true')
    p.add_argument('--reps',type=int,default=512);p.add_argument('--refine',action='store_true')
    args=p.parse_args();start=time.time()
    out=BASE/'results'/('capacity_online_smoke' if args.smoke else
                       ('capacity_online_refined' if args.refine else 'capacity_online'))
    out.mkdir(parents=True,exist_ok=True)
    if args.smoke: checks=smoke()
    else:
        checks=json.loads((BASE/'results/capacity_online_smoke/verification.json').read_text())
        assert checks['passed']
    state=build_records(False)
    mc,audit=replay_count(state,32 if args.smoke else 2048)
    mc.to_csv(out/'independent_count_replay.csv',index=False)
    configs=[('uniform',3,28,120.,1.)] if args.smoke else list(itertools.product(
        ['uniform','burst'],[1,3],[28],[90.,120.],[1.]))+[('uniform',3,28,120.,.95)]
    if args.refine:
        configs=[('uniform',1,28,120.,1.),('uniform',3,28,120.,1.)]
    results,effects,cells=service_run(state,configs,2 if args.smoke else args.reps)
    results.to_csv(out/'service_reliability.csv',index=False)
    effects.to_csv(out/'service_effects.csv',index=False)
    cells.to_csv(out/'service_profile_probabilities.csv',index=False)
    checks.update(smoke=args.smoke,count_replay=audit,service_configurations=len(configs),
        service_scenarios=len(configs)*3,service_replicates=2 if args.smoke else args.reps,
        seconds=time.time()-start,physical_detection_integrated=True,
        count_theorem_does_not_claim_deadline_dominance=True,
        provisional_slot_reservations_prevent_parallel_overshoot=True,
        simultaneous_confidence_covers_simulation_only=True)
    (out/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
