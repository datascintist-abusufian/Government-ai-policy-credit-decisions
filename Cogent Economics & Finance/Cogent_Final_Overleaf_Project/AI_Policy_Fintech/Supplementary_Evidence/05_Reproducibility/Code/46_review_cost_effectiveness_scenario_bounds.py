"""STEP 46: Transparent sensitivity to hypothetical review effectiveness and cost.

This script CANNOT estimate the actual effect or resource cost of human review,
because observed review interventions are absent. It provides break-even
requirements under explicitly labelled assumed action effectiveness/cost.

Let r=c_FN/c_FP, c_FP=1, e in [0,1] the fraction of flagged adverse outcomes
that effective intervention would avert, and k the incremental review cost per
flagged loan relative to c_FP=1. Illustrative loss:
    L_model = r*(FN+(1-e)*TP) + FP + k*(TP+FP)
    L_flag_none = r*(TP+FN)
    L_flag_all  = r*(1-e)*(TP+FN) + (TN+FP) + k*N
This is NOT the observed or validated real-world action utility.

Run: python 46_review_cost_effectiveness_scenario_bounds.py --root "E:\\Governance-Policy\\Cogent Economics & Finance"
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_ROOT=Path(r'E:\Governance-Policy\Cogent Economics & Finance')
COUNTS={5:(943,2244,5082),10:(3533,11909,2492),20:(5730,29266,295)}
EXPECTED_N=41145

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    args=parser.parse_args()
    root=args.root.resolve();old=root/'revision_outputs'
    out=root/'Round2_Manuscript_Results'/'01_Tables'
    out.mkdir(parents=True,exist_ok=True)
    y=np.load(old/'cell27_y_test.npy',allow_pickle=False).reshape(-1)
    p=np.load(old/'cell29_histgb_full_test_recalibrated.npy',allow_pickle=False).reshape(-1)
    if len(y)!=EXPECTED_N or int(y.sum())!=6025 or len(p)!=len(y):
        raise RuntimeError('STOP: original test labels/predictions do not reconcile')
    threshold_df=pd.read_csv(old/'cell30_recalibrated_threshold_analysis.csv')
    test=threshold_df[threshold_df.cohort.astype(str).str.lower().eq('test')]
    if len(test)!=3:raise RuntimeError('Expected three saved test thresholds')
    rows=[];bounds=[]
    for r,(tp_expected,fp_expected,fn_expected) in COUNTS.items():
        matches=test.loc[pd.to_numeric(test.fn_cost,errors='coerce').eq(r)]
        if len(matches)!=1:raise RuntimeError('Missing threshold for cost ratio '+str(r))
        t=float(matches.iloc[0].threshold)
        flag=p>=t
        tp=int(np.count_nonzero(flag&(y==1)));fp=int(np.count_nonzero(flag&(y==0)))
        fn=int(np.count_nonzero(~flag&(y==1)));tn=int(np.count_nonzero(~flag&(y==0)))
        if (tp,fp,fn)!=(tp_expected,fp_expected,fn_expected):
            raise RuntimeError(f'Original confusion count mismatch {r}:1')
        flagged=tp+fp;n=len(y);total_events=tp+fn
        for e in (0,0.25,0.5,0.75,1):
            # At effectiveness=0, review flags cannot avert adverse outcomes.
            k_break_even_none=(e*r*tp-fp)/flagged
            for k in (0,0.25,0.5,1.0):
                model=r*(fn+(1-e)*tp)+fp+k*flagged
                none=r*total_events
                all_flag=r*(1-e)*total_events+tn+fp+k*n
                rows.append({
                    'cost_ratio':f'{r}:1','assumed_review_effectiveness':e,
                    'assumed_review_cost_per_flag_relative_to_FP':k,
                    'flagged_count_selected_cohort':flagged,
                    'flagged_share_selected_cohort':flagged/n,
                    'model_scenario_loss':model,'flag_none_scenario_loss':none,
                    'flag_all_scenario_loss':all_flag,
                    'best_policy_of_these_three':min([('model',model),('flag_none',none),('flag_all',all_flag)],key=lambda item:item[1])[0],
                    'no_observed_review_intervention':True,
                })
            bounds.append({
                'cost_ratio':f'{r}:1','assumed_review_effectiveness':e,
                'highest_review_cost_per_flag_to_beat_flag_none':k_break_even_none,
                'minimum_review_effectiveness_if_review_cost_zero':fp/(r*tp),
                'flagged_selected_cohort':flagged,
                'not_a_measured_lender_cost':True,
            })
    scenario=pd.DataFrame(rows)
    fn1=out/'46_hypothetical_review_cost_effectiveness_scenarios.csv'
    fn2=out/'46_break_even_review_cost_bounds.csv'
    scenario.to_csv(fn1,index=False);pd.DataFrame(bounds).to_csv(fn2,index=False)
    print('STEP 46 | hypothetical review scenario analysis (NOT an empirical cost estimate)')
    print(pd.DataFrame(bounds).loc[lambda d:d.assumed_review_effectiveness.eq(1)][[
        'cost_ratio','flagged_selected_cohort',
        'minimum_review_effectiveness_if_review_cost_zero',
        'highest_review_cost_per_flag_to_beat_flag_none']].to_string(index=False))
    print('Saved:',fn1);print('Saved:',fn2)
    print('No actual review actions, lender costs, success rates, or causal effects are observed.')
    print('STEP 46 COMPLETE')

if __name__=='__main__':main()
