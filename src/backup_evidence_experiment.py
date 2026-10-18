"""Audited trial analysis, crossed resampling, and fixed-trajectory rule replay.

Every scenario uses a single observed intervention. Machine-upgrade scenarios
are explicitly conditional on behavior transport, never reported as treatment
effects. Figures are produced by a separate program.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from sklearn.model_selection import GroupKFold

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / 'data/raw/human_backup'
FILES = ['behavior_NRA_group.csv', 'behavior_RA_group.csv',
         'behavior_XAI_presented_beforeAndAfter_trustViolation.csv']

def objset(value):
    text = str(value).strip()
    if text.startswith('['):
        values = json.loads(text)
    else:
        values = text.split(',')
    return frozenset(x.strip().lower() for x in values if x.strip().lower() not in {'', 'none'})

def load_data(smoke=False):
    frames = [pd.read_csv(RAW / f, keep_default_na=False) for f in FILES]
    cols = frames[0].columns.tolist()
    joined = pd.concat(frames, ignore_index=True)
    keys = ['userID', 'taskCaseNum']
    conflicts = joined.groupby(keys)[cols].nunique().gt(1).any(axis=1)
    assert not conflicts.any(), 'Conflicting duplicate trial records'
    main = joined.drop_duplicates(keys).copy()
    assert len(main) == 10608 and main.userID.nunique() == 272
    assert main.groupby('userID').size().eq(39).all()
    stim = pd.read_csv(RAW / 'task_selected_for_study.csv', keep_default_na=False)
    stim['image'] = stim['filename'].str.lower() + '.jpg'
    stim['truth'] = stim['ground truth'].map(objset).map(bool)
    stim['machine'] = stim['model result'].map(objset).map(bool)
    stim['error'] = stim['ground truth'].map(objset).ne(stim['model result'].map(objset))
    main['image'] = main.imageID.str.lower()
    main = main.merge(stim[['image', 'truth', 'machine', 'error']], on='image', validate='many_to_one')
    assert not main['error'].isna().any()
    main['person'] = main.userID
    main['failure'] = main.userCorrect.eq(0)
    main['open'] = main.openBaggage.eq(1)
    main['difficulty'] = main.difficultyLevel.map({'E':'easy','D':'difficult'})
    main['stratum'] = main[['type_of_tv','type_of_responsive','type_of_vis']].astype(str).agg('|'.join, axis=1)
    pilot = pd.read_excel(RAW / 'pilotResult_combinedAIandNoAI.xlsx', keep_default_na=False)
    missing = pilot.ImageNumber.eq('')
    excluded = pilot[missing].groupby('GROUP').size().to_dict()
    pilot = pilot[~missing].copy()
    assert not pilot.duplicated(['user_identifier','OrigColumn']).any()
    assert not pilot.duplicated(['user_identifier','ImageNumber']).any()
    pilot['truth_set'] = pilot['ground truth'].map(objset)
    pilot['machine_set'] = pilot['model result'].map(objset)
    pilot['final_set'] = pilot.UserDecision.map(objset)
    pilot['truth'] = pilot.truth_set.map(bool)
    pilot['machine'] = pilot.machine_set.map(bool)
    pilot['human'] = pilot.final_set.map(bool)
    pilot['error'] = pilot.truth_set.ne(pilot.machine_set)
    pilot['failure'] = pilot.truth_set.ne(pilot.final_set)
    pilot['hazard'] = pilot.truth & ~pilot.human
    pilot['open'] = pilot.OpenBag.astype(str).str.lower().eq('true')
    pilot['image'] = pilot.ImageNumber.str.lower()
    pilot['person'] = pilot.user_identifier
    pilot['difficulty'] = pilot['difficulty level']
    pilot['stratum'] = pilot[['type_of_tv','num_of_tv']].astype(str).agg('|'.join, axis=1)
    assert pilot.failure.eq(pilot.Correct.eq(0)).all()
    ai = pilot[pilot.GROUP.eq('AI')].copy()
    assert (len(ai), ai.person.nunique()) == (1040,40)
    assert (len(pilot[pilot.GROUP.eq('NoAI')]), pilot[pilot.GROUP.eq('NoAI')].person.nunique()) == (540,20)
    assert not set(main.person) & set(pilot.person)
    assert ai[ai.open].failure.sum() == 0
    audit = {
        'raw_main_rows':len(joined), 'deduplicated_main_rows':len(main),
        'identical_duplicate_rows_removed':len(joined)-len(main), 'conflicting_duplicate_keys':int(conflicts.sum()),
        'main_participants':main.person.nunique(), 'main_images':main.image.nunique(),
        'main_trials_per_participant':39, 'pilot_excluded_missing_image':excluded,
        'pilot_AI_trials':len(ai), 'pilot_AI_participants':ai.person.nunique(),
        'pilot_AI_images':ai.image.nunique(), 'main_pilot_shared_images':len(set(main.image)&set(ai.image)),
        'pilot_design_cells':ai[['person','stratum']].drop_duplicates().groupby('stratum').size().to_dict(),
        'pilot_noAI_trials':540,'pilot_noAI_participants':20,
        'main_pilot_shared_participant_ids':0,
        'final_set_correctness_matches_source':True,
        'main_has_final_object_selection':False,
        'machine_status_variants_per_main_image_max':int(main.groupby('image').error.nunique().max()),
        'machine_status_variants_per_pilot_AI_image_max':int(ai.groupby('image').error.nunique().max()),
        'source_sha256':{f:hashlib.sha256((RAW/f).read_bytes()).hexdigest() for f in FILES+['task_selected_for_study.csv','pilotResult_combinedAIandNoAI.xlsx']},
        'license':'CC BY 4.0','data_DOI':'10.6084/m9.figshare.30316936.v2',
    }
    if smoke:
        users = main[['person','stratum']].drop_duplicates().copy()
        users['hash'] = users.person.map(lambda s:hashlib.sha256(('v3:'+s).encode()).hexdigest())
        keep = users.sort_values('hash').groupby('stratum').head(3).person
        main = main[main.person.isin(keep)].copy()
    return main, ai, pilot, audit

def count_cells(main, ai):
    rows=[]
    for name,d,endpoint in [('main_exact',main,'failure'),('pilot_exact',ai,'failure'),
                             ('pilot_false_clearance',ai[ai.truth],'hazard'),
                             ('pilot_false_alarm',ai[~ai.truth],'failure')]:
        for (truth,error),g in d.groupby(['truth','error']):
            failures=int(g[endpoint].sum())
            rows.append(dict(dataset=name,truth_positive=bool(truth),machine_error=bool(error),
                             trials=len(g),final_failures=failures,corrections=len(g)-failures if error else 0,
                             introduced=failures if not error else 0,participants=g.person.nunique(),images=g.image.nunique()))
    return pd.DataFrame(rows)

def metrics(d, endpoint):
    err=d.error.to_numpy(bool); fail=d[endpoint].to_numpy(bool)
    n0=(~err).sum(); n1=err.sum(); a=float((~err & fail).sum()/n0); b=float((err & ~fail).sum()/n1)
    return dict(a=a,b=b,threshold=a/(a+b) if a+b>0 else np.nan,
                machine_risk=float(err.mean()),final_risk=float(fail.mean()))

def bootstrap(d,endpoint,reps,mode,seed):
    """Stratified multinomial cluster bootstrap; crossed weights U_i * I_j.

    Experimental participant cells remain represented. Stimulus resampling
    retains truth, primary-error and difficulty strata. It describes sampling
    uncertainty within this design; it does not transport behavior to new AI.
    """
    rng=np.random.default_rng(seed)
    people=d[['person','stratum']].drop_duplicates().reset_index(drop=True)
    images=d[['image','truth','error','difficulty']].drop_duplicates().reset_index(drop=True)
    assert people.person.is_unique and images.image.is_unique
    u=pd.Categorical(d.person,categories=people.person).codes
    j=pd.Categorical(d.image,categories=images.image).codes
    e=d.error.to_numpy(bool); f=d[endpoint].to_numpy(bool)
    masks=np.stack([~e,e,(~e)&f,e&~f,f,np.ones(len(d),bool)],axis=1).astype(float)
    out=[]
    for start in range(0,reps,128):
        n=min(128,reps-start)
        uw=np.ones((n,len(people))); iw=np.ones((n,len(images)))
        if mode in {'participant','crossed'}:
            for ids in people.groupby('stratum').indices.values():
                uw[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=n)
        if mode in {'image','crossed'}:
            for ids in images.groupby(['truth','error','difficulty']).indices.values():
                iw[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=n)
        sums=(uw[:,u]*iw[:,j])@masks
        with np.errstate(divide='ignore',invalid='ignore'):
            a=sums[:,2]/sums[:,0]; b=sums[:,3]/sums[:,1]
            q=a/(a+b); obs=sums[:,1]/sums[:,5]; final=sums[:,4]/sums[:,5]
        out.append(pd.DataFrame(dict(a=a,b=b,threshold=q,machine_risk=obs,final_risk=final)))
    ans=pd.concat(out,ignore_index=True)
    ans['mode']=mode
    return ans

def interval(values):
    v=np.asarray(values); v=v[np.isfinite(v)]
    return [float(np.quantile(v,.025)),float(np.quantile(v,.975))] if len(v) else [None,None]

def replay(ai):
    d=ai.copy()
    # Retain a machine-positive alarm unless the recorded opening was used.
    d['constrained_alarm']=d.human | (d.machine & ~d.open)
    d['constrained_hazard']=d.truth & ~d.constrained_alarm
    d['constrained_false_alarm']=~d.truth & d.constrained_alarm
    d['observed_false_alarm']=~d.truth & d.human
    d['unconditional_alarm']=d.human | d.machine
    assert (d.constrained_hazard<=d.hazard).all()
    assert (d.constrained_alarm>=d.human).all()
    changed=d[d.constrained_alarm.ne(d.human)].copy()
    changed['effect']=np.where(changed.truth,'prevented_false_clearance','added_false_alarm')
    rows=[]
    for policy,hazard,fa in [('machine',d.truth&~d.machine,~d.truth&d.machine),
                             ('observed_human',d.hazard,d.observed_false_alarm),
                             ('unconditional_alarm_union_replay',d.truth&~d.unconditional_alarm,~d.truth&d.unconditional_alarm),
                             ('alarm_retention_replay',d.constrained_hazard,d.constrained_false_alarm)]:
        rows.append(dict(policy=policy,positive_trials=int(d.truth.sum()),false_clearances=int(hazard.sum()),
                         negative_trials=int((~d.truth).sum()),false_alarms=int(fa.sum()),
                         recorded_openings_on_common_trajectory=int(d.open.sum())))
    keep=['person','UserCaseNum','image','ground truth','model result','UserDecision',
          'OpenBag','TaskTime','truth','machine','human','constrained_alarm','effect']
    return pd.DataFrame(rows),changed[keep],d

def crossed_glmm(d,endpoint,label,prior_scales):
    """Crossed participant and image effects; approximate VB, sensitivity only."""
    fitdata=d.copy()
    fitdata['response']=fitdata[endpoint].astype(int)
    fitdata['primary_error']=fitdata.error.astype(int)
    formula='response ~ primary_error + C(difficulty)'
    if label=='main_exact':
        formula+=' + C(type_of_responsive) + C(type_of_vis) + C(type_of_tv)'
    else:
        # Induced errors occur in the easy stratum; fixed effects are associational.
        formula+=' + C(type_of_tv) + C(num_of_tv)'
    rows=[]; diagnostics=[]
    for scale in prior_scales:
        start=time.time()
        model=BinomialBayesMixedGLM.from_formula(formula,
                 {'person':'0 + C(person)','image':'0 + C(image)'},fitdata,vcp_p=scale,fe_p=2.5)
        with warnings.catch_warnings(record=True) as caught:
            fit=model.fit_vb(minim_opts={'maxiter':450,'gtol':1e-5})
        status=bool(fit.optim_retvals.success)
        for term,mean,sd in zip(model.exog_names,fit.fe_mean,fit.fe_sd):
            rows.append(dict(dataset=label,variance_prior_scale=scale,term=term,log_odds=float(mean),
                             variational_sd=float(sd),odds_ratio=float(np.exp(mean)),
                             approximate_low=float(np.exp(mean-1.96*sd)),approximate_high=float(np.exp(mean+1.96*sd)),
                             optimizer_success=status))
        diagnostics.append(dict(dataset=label,variance_prior_scale=scale,optimizer_success=status,
                       message=str(fit.optim_retvals.message),iterations=int(fit.optim_retvals.nit),
                       seconds=time.time()-start,warnings=[str(w.message) for w in caught],
                       participant_sd=float(np.exp(fit.vcp_mean[0])),image_sd=float(np.exp(fit.vcp_mean[1])),
                       inference='Mean-field variational approximation; conditional association, no causal machine-upgrade identification.'))
        print(label,'prior',scale,'success',status,'seconds',round(time.time()-start,1),flush=True)
    return rows,diagnostics

def holdout_checks(d,endpoint,label):
    """Pre-specified conditional-frequency predictor on held-out people/images.

    These are mechanism portability diagnostics, not in-sample identity checks.
    Intercept-only prevalence predictor is included for context.
    """
    rows=[]
    y=d[endpoint].to_numpy(int); e=d.error.to_numpy(bool)
    for grouping in ['person','image']:
        cv=GroupKFold(n_splits=5)
        for fold,(train,test) in enumerate(cv.split(d,groups=d[grouping]),1):
            pred=np.zeros(len(test))
            for state in [False,True]:
                mask=e[train]==state
                pred[e[test]==state]=(y[train][mask].sum()+.5)/(mask.sum()+1)
            null=np.full(len(test),(y[train].sum()+.5)/(len(train)+1))
            for model,probs in [('conditional_primary_state',pred),('overall_prevalence',null)]:
                rows.append(dict(dataset=label,held_out_unit=grouping,fold=fold,predictor=model,n=len(test),
                                 actual_failures=int(y[test].sum()),predicted_failures=float(probs.sum()),
                                 brier=float(np.mean((probs-y[test])**2)),
                                 log_loss=float(-np.mean(y[test]*np.log(probs)+(1-y[test])*np.log1p(-probs)))))
    return rows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--bootstrap',type=int,default=8000)
    parser.add_argument('--skip-glmm',action='store_true')
    args=parser.parse_args()
    out=BASE/'results'/('evidence_v3_smoke' if args.smoke else 'evidence_v3')
    out.mkdir(parents=True,exist_ok=True)
    main_data,ai,pilot,audit=load_data(args.smoke)
    audit['analyzed_main_rows']=len(main_data); audit['smoke']=args.smoke
    (out/'data_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    counts=count_cells(main_data,ai); counts.to_csv(out/'direct_counts.csv',index=False,encoding='utf-8-sig')
    export=main_data.drop(columns=['userID','imageID']).copy()
    export.to_csv(BASE/'data/curated'/('main_trials_smoke.csv' if args.smoke else 'main_trials.csv'),index=False,encoding='utf-8-sig')
    ai.drop(columns=['truth_set','machine_set','final_set']).to_csv(BASE/'data/curated/pilot_AI_trials.csv',index=False,encoding='utf-8-sig')
    cases=[('main_exact',main_data,'failure'),('pilot_exact',ai,'failure'),
           ('pilot_false_clearance',ai[ai.truth],'hazard'),('pilot_false_alarm',ai[~ai.truth],'failure')]
    summary=[]; scenarios=[]; allboot=[]; checks=[]
    for cidx,(name,d,endpoint) in enumerate(cases):
        point=metrics(d,endpoint)
        for mode in ['participant','image','crossed']:
            boot=bootstrap(d,endpoint,args.bootstrap,mode,20261014+cidx*17)
            boot['dataset']=name; allboot.append(boot)
            valid=boot[['a','b']].notna().all(axis=1)
            for quantity in ['a','b','threshold','machine_risk','final_risk']:
                lo,hi=interval(boot[quantity])
                summary.append(dict(dataset=name,mode=mode,quantity=quantity,estimate=point[quantity],
                                    low=lo,high=hi,replicates=args.bootstrap,valid_replicates=int(valid.sum())))
            for p in [.001,.005,.01,.05,point['machine_risk']]:
                risk=(1-p)*boot.a+p*(1-boot.b)
                lo,hi=interval(risk); dlo,dhi=interval(risk-p)
                scenarios.append(dict(dataset=name,mode=mode,primary_error_probability=p,
                                      risk_point=(1-p)*point['a']+p*(1-point['b']),low=lo,high=hi,
                                      change_point=(1-p)*point['a']-p*point['b'],change_low=dlo,change_high=dhi,
                                      bootstrap_positive_fraction=float(((risk-p)[valid]>0).mean()),
                                      interpretation='Single intervention; behavior held fixed; fraction is resampling frequency.'))
        checks.extend(holdout_checks(d,endpoint,name))
        print(name,point,flush=True)
    pd.DataFrame(summary).to_csv(out/'cluster_uncertainty.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(scenarios).to_csv(out/'conditional_scenarios.csv',index=False,encoding='utf-8-sig')
    pd.concat(allboot).to_csv(out/'bootstrap_draws.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(out/'holdout_checks.csv',index=False,encoding='utf-8-sig')
    policies,changed,replayed=replay(ai)
    policies.to_csv(out/'fixed_trajectory_policies.csv',index=False,encoding='utf-8-sig')
    changed.to_csv(out/'all_changed_trajectories.csv',index=False,encoding='utf-8-sig')
    hetero=[]; leaveout=[]
    for name,d,endpoint in cases:
        for grouping in ['person','image','difficulty','type_of_tv']:
            for value,g in d.groupby(grouping):
                for error,h in g.groupby('error'):
                    hetero.append(dict(dataset=name,grouping=grouping,value=value,machine_error=bool(error),
                                       n=len(h),failures=int(h[endpoint].sum()),rate=float(h[endpoint].mean())))
        if name=='pilot_false_clearance':
            for unit in ['person','image']:
                for value in d[unit].unique():
                    m=metrics(d[d[unit].ne(value)],endpoint)
                    leaveout.append(dict(omitted_unit=unit,omitted_value=value,**m,
                                         conditional_change_at_001=(1-.001)*m['a']-.001*m['b']))
    pd.DataFrame(hetero).to_csv(out/'heterogeneity.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(leaveout).to_csv(out/'rare_event_leave_one_out.csv',index=False,encoding='utf-8-sig')
    # Replaying the final output holds the recorded task/opening path fixed.
    # Cost ratios include threat prevalence explicitly.
    cost=[]
    prev=float(ai.truth.mean())
    gain=float((replayed.hazard & ~replayed.constrained_hazard).sum()/ai.truth.sum())
    added=float((replayed.constrained_false_alarm & ~replayed.observed_false_alarm).sum()/(~ai.truth).sum())
    for pi in [prev,.1,.01,.001]:
        cost.append(dict(threat_prevalence=pi,miss_cost_to_false_alarm_cost_threshold=(1-pi)*added/(pi*gain),
                         eliminated_false_clearances_per_positive=gain,added_false_alarms_per_negative=added))
    pd.DataFrame(cost).to_csv(out/'replay_cost_boundary.csv',index=False,encoding='utf-8-sig')
    drift=[]; a=2/640; b=31/40
    for p in [.001,.005,.01]:
        for a_factor in [0,.1,.2,.25,.5,1,2]:
            for b_factor in [.5,.75,1]:
                at=a*a_factor; bt=b*b_factor
                drift.append(dict(primary_error_probability=p,a_factor=a_factor,b_factor=b_factor,
                                  final_risk=(1-p)*at+p*(1-bt),change=(1-p)*at-p*bt))
    pd.DataFrame(drift).to_csv(out/'behavior_drift_sensitivity.csv',index=False,encoding='utf-8-sig')
    if not args.skip_glmm:
        np.random.seed(20261014)
        params=[]; diagnostics=[]
        for name,d,endpoint in [cases[0],cases[2]]:
            rows,diag=crossed_glmm(d,endpoint,name,[1] if args.smoke else [.5,1,2])
            params.extend(rows); diagnostics.extend(diag)
        pd.DataFrame(params).to_csv(out/'crossed_glmm_sensitivity.csv',index=False,encoding='utf-8-sig')
        (out/'crossed_glmm_diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    result={'version':3,'smoke':args.smoke,'main_rows':len(main_data),'pilot_rows':len(ai),
            'main_exact':metrics(main_data,'failure'),'pilot_false_clearance':metrics(ai[ai.truth],'hazard'),
            'policy_replay':policies.to_dict('records'),
            'verdict':{'mechanism':'Directly observed in public simulated screening trials',
                       'machine_upgrade_causal_effect':'Unidentified: machine output status is fixed within images',
                       'professional_airport_transport':'No directly aligned external trial dataset verified yet',
                       'rule_replay':'Exact fixed-record calculation; new behavioral response is unidentified'}}
    (out/'experiment_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    main()
