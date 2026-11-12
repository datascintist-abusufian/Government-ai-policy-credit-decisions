"""STEP 45: Post-hoc three-window validation sensitivity (NO MODEL RETRAINING).

A genuinely separate chronology within the ORIGINAL selected validation period:
  Model family selection: 2018-07 to 2018-09
  Logistic calibration: 2018-10 to 2018-12
  FN:FP threshold fitting: 2019-01 to 2019-03
  Descriptive evaluation: pre-existing selected 2019-04 to 2019-12 test cohort

The model families were fitted on the original pre-July-2018 training data.
No model is re-fitted here; previously saved out-of-time predictions are used.
This is a POST-HOC sensitivity study, NOT retroactive correction of the
already published validation-selected model, the outcome-selection bias,
or the unverified point-in-time predictor information boundary.

Run from PowerShell:
 python 45_independent_validation_windows_sensitivity.py --root "E:\\Governance-Policy\\Cogent Economics & Finance"
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

ROOT_DEFAULT = Path(r'E:\Governance-Policy\Cogent Economics & Finance')
EXPECTED = {"val": 27374, "test": 41145}
CUTOFF = pd.Timestamp('2020-04-01')
MODELS = ('Logistic_SGD','HistGB','MLP')
COST_RATIOS = (5,10,20)


def reconstruct_val_months(data_path: Path, old_y: np.ndarray) -> np.ndarray:
    """Read only needed source fields and verify original row-order alignment."""
    months = []
    labels = []
    seen = 0
    with pd.read_stata(data_path, columns=['issue_d','term','loan_status','badloan'],
                       convert_categoricals=False, chunksize=200000) as reader:
        for block in reader:
            seen += len(block)
            issue = pd.to_datetime(block.issue_d.astype(str).str.strip(),format='%b-%y',errors='coerce')
            term = pd.to_numeric(block.term.astype(str).str.extract(r'(\d+)')[0],errors='coerce')
            status = block.loan_status.astype(str).str.strip()
            age = (CUTOFF.year-issue.dt.year)*12+(CUTOFF.month-issue.dt.month)
            selected = issue.between('2018-07-01','2019-03-01') & (status.isin(['Fully Paid','Charged Off','Default']) | (age >= term))
            if selected.any():
                months.append(issue.loc[selected].dt.to_period('M').astype(str).to_numpy())
                labels.append(pd.to_numeric(block.loc[selected,'badloan'],errors='coerce').fillna(0).astype(np.int8).to_numpy())
            print(f'Source rows scanned: {seen:,}',flush=True)
    if seen != 2703430:
        raise RuntimeError(f'Source-row count mismatch. Expected 2,703,430; found {seen}')
    if not months:raise RuntimeError('No selected validation observations found')
    month=np.concatenate(months)
    found_y=np.concatenate(labels)
    if len(month) != EXPECTED['val'] or not np.array_equal(found_y,old_y.astype(np.int8)):
        raise RuntimeError('STOP: validation metadata do not match saved prediction ordering/labels. DO NOT INTERPRET RESULTS')
    print('PASS: reconstructed validation rows exactly align to original label array')
    return month


def safe_predictions(path:Path, n:int):
    if not path.is_file():raise FileNotFoundError(path)
    a=np.load(path,allow_pickle=False).reshape(-1).astype(np.float64)
    if len(a)!=n or not np.isfinite(a).all() or (a<0).any() or (a>1).any():
        raise ValueError(f'Invalid predictions {path.name}: len={len(a)}')
    return a


def calibrate(raw,pred_y):
    eps=1e-8
    x=logit(np.clip(raw,eps,1-eps))
    y=pred_y.astype(np.float64)
    if np.unique(y).size!=2:raise RuntimeError('Calibration window contains fewer than 2 outcome classes')
    def fun_grad(coef):
        z=coef[0]+coef[1]*x
        prob=expit(z)
        diff=prob-y
        loss=np.mean(np.logaddexp(0,z)-y*z)
        grad=np.array([np.mean(diff),np.mean(diff*x)],dtype=float)
        return float(loss),grad
    res=minimize(fun_grad,[0.0,1.0],method='L-BFGS-B',jac=True,options={'maxiter':1000,'ftol':1e-13})
    if not res.success:
        raise RuntimeError('Calibration optimisation did not converge: '+str(res.message))
    return res.x


def apply_cal(p,coeff):
    return expit(coeff[0]+coeff[1]*logit(np.clip(p,1e-8,1-1e-8)))


def choose_threshold(p,y,ratio):
    """Exhaustive optimum over unique score boundaries, including flag-none."""
    order=np.argsort(-p,kind='stable')
    ps=p[order]
    ys=y[order].astype(np.int32)
    tp=np.cumsum(ys)
    fp=np.cumsum(1-ys)
    last=np.r_[np.flatnonzero(ps[1:]!=ps[:-1]),len(ps)-1]
    possible_scores=ps[last]
    costs=ratio*(int(ys.sum())-tp[last])+fp[last]
    # First element represents flag no loans at a threshold above max score.
    costs=np.r_[ratio*int(ys.sum()),costs]
    scores=np.r_[np.nextafter(float(ps[0]),np.inf),possible_scores]
    minimum=costs.min()
    ties=np.flatnonzero(costs==minimum)
    bayes=1/(1+ratio)
    chosen=ties[np.argmin(np.abs(scores[ties]-bayes))]
    threshold=float(scores[chosen])
    checkcost=int(ratio*np.count_nonzero((p<threshold)&(y==1))+
                  np.count_nonzero((p>=threshold)&(y==0)))
    if checkcost!=int(minimum):raise RuntimeError('Threshold optimizer inconsistency')
    return threshold,checkcost


def ece10(y,p):
    bins=np.minimum((p*10).astype(int),9)
    return sum(float(np.mean(bins==b)*abs(float(y[bins==b].mean())-float(p[bins==b].mean())))
               for b in range(10) if (bins==b).any())


def performance(y,p,t,ratio):
    flag=p>=t
    tp=int(np.count_nonzero(flag&(y==1)));fp=int(np.count_nonzero(flag&(y==0)))
    fn=int(np.count_nonzero(~flag&(y==1)));tn=int(np.count_nonzero(~flag&(y==0)))
    return dict(adverse=int(y.sum()),n=len(y),tp=tp,fp=fp,fn=fn,tn=tn,
                fnr=fn/int(y.sum()),flag_rate=float(np.mean(flag)),
                classification_cost=ratio*fn+fp,
                flag_none_cost=ratio*int(y.sum()),flag_all_cost=int((y==0).sum()),
                best_trivial_cost=min(ratio*int(y.sum()),int((y==0).sum())))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT_DEFAULT)
    args=parser.parse_args()
    root=args.root.resolve()
    old=root/'revision_outputs'
    data=root/'P2P_Macro_Data.dta'
    if not data.is_file():raise FileNotFoundError(f'Required public dataset (not supplied): {data}')
    dest=root/'Round2_Manuscript_Results'/'01_Tables'
    logs=root/'Round2_Manuscript_Results'/'05_Reproducibility'/'Execution_Logs'
    dest.mkdir(parents=True,exist_ok=True);logs.mkdir(parents=True,exist_ok=True)
    y_val=np.load(old/'cell27_y_val.npy',allow_pickle=False).reshape(-1).astype(np.int8)
    y_test=np.load(old/'cell27_y_test.npy',allow_pickle=False).reshape(-1).astype(np.int8)
    for key, arr in [('val',y_val),('test',y_test)]:
        if len(arr)!=EXPECTED[key] or not np.isin(arr,[0,1]).all():
            raise RuntimeError(f'{key}: unexpected target size or values')
    if y_test.sum()!=6025 or y_val.sum()!=2771:
        raise RuntimeError('Labels/prevalence mismatch from original experiment')
    print('STEP 45: independent chronological windows in original selected validation cohort',flush=True)
    months=reconstruct_val_months(data,y_val)
    windows={
      'select':np.isin(months,['2018-07','2018-08','2018-09']),
      'calibrate':np.isin(months,['2018-10','2018-11','2018-12']),
      'threshold':np.isin(months,['2019-01','2019-02','2019-03']),
    }
    if sum(x.sum() for x in windows.values())!=len(y_val):
        raise RuntimeError('Validation window partition does not cover original selected validation set')
    support=[]
    for k, mask in windows.items():
        pos=int(y_val[mask].sum());n=int(mask.sum())
        support.append({'window':k,'n':n,'adverse_n':pos,'adverse_rate':pos/n,
                        'first_month': {'select':'2018-07','calibrate':'2018-10','threshold':'2019-01'}[k],
                        'last_month': {'select':'2018-09','calibrate':'2018-12','threshold':'2019-03'}[k]})
        if n<300 or pos<30 or (n-pos)<30:raise RuntimeError(f'Insufficient event support in {k}: n={n}, adverse={pos}')
    pd.DataFrame(support).to_csv(dest/'45_three_window_sample_support.csv',index=False)
    print(pd.DataFrame(support).to_string(index=False),flush=True)
    val={m:safe_predictions(old/f'cell27_1304058_{m}_val_pred.npy',len(y_val)) for m in MODELS}
    test={m:safe_predictions(old/f'cell27_1304058_{m}_test_pred.npy',len(y_test)) for m in MODELS}
    selection_aucs={m:float(roc_auc_score(y_val[windows['select']],val[m][windows['select']])) for m in MODELS}
    # Deterministic tie breaker, fixed a priori for this POST-HOC sensitivity: alphabetical model name.
    selected=sorted(MODELS,key=lambda m:(-selection_aucs[m],m))[0]
    coeff=calibrate(val[selected][windows['calibrate']],y_val[windows['calibrate']])
    cal_val=apply_cal(val[selected],coeff)
    cal_test=apply_cal(test[selected],coeff)
    print('Selection-only AUCs:',selection_aucs)
    print('Selected model:',selected,'| separate calibration intercept/slope:',coeff)
    rows=[]
    for ratio in COST_RATIOS:
        threshold,training_cost=choose_threshold(cal_val[windows['threshold']],y_val[windows['threshold']],ratio)
        observed=performance(y_test,cal_test,threshold,ratio)
        threshold_perf=performance(y_val[windows['threshold']],cal_val[windows['threshold']],threshold,ratio)
        rows.append({
            'posthoc_alternative_only':True,
            'selected_model':selected,
            'model_selection_AUC_Logistic_SGD':selection_aucs['Logistic_SGD'],
            'model_selection_AUC_HistGB':selection_aucs['HistGB'],
            'model_selection_AUC_MLP':selection_aucs['MLP'],
            'calibration_intercept':float(coeff[0]),'calibration_slope':float(coeff[1]),
            'fn_to_fp_ratio':f'{ratio}:1','threshold':threshold,
            'threshold_window_n':threshold_perf['n'],
            'threshold_window_cost':training_cost,
            'test_n':observed['n'],'test_adverse_n':observed['adverse'],
            'test_model_auc':roc_auc_score(y_test,test[selected]),
            'test_calibrated_brier':brier_score_loss(y_test,cal_test),
            'test_calibrated_logloss':log_loss(y_test,cal_test),
            'test_calibrated_ece10':ece10(y_test,cal_test),
            'test_predicted_mean':float(cal_test.mean()),
            'test_observed_rate':float(y_test.mean()),
            'test_observed_minus_predicted':float(y_test.mean()-cal_test.mean()),
            'test_fnr':observed['fnr'],'test_flag_rate':observed['flag_rate'],
            'test_tp':observed['tp'],'test_fp':observed['fp'],
            'test_fn':observed['fn'],'test_tn':observed['tn'],
            'test_classification_cost':observed['classification_cost'],
            'test_best_trivial_classification_cost':observed['best_trivial_cost'],
        })
        print(f"{ratio}:1 | threshold={threshold:.6f} | test FNR={observed['fnr']:.2%} "
              f"| flagged={observed['flag_rate']:.2%} "
              f"| model cost={observed['classification_cost']} "
              f"| best trivial={observed['best_trivial_cost']}")
    output=dest/'45_three_window_validation_sensitivity.csv'
    pd.DataFrame(rows).to_csv(output,index=False)
    report=(
      'STEP 45: POST-HOC DISJOINT VALIDATION WINDOWS, WITHOUT REFITTING CLASSIFIERS\n'
      'Selection: Jul-Sep 2018 | Calibration: Oct-Dec 2018 | Threshold: Jan-Mar 2019.\n'
      'The existing fitted models and selected outcome population are unchanged.\n'
      'No model selection/calibration/threshold observation is reused across those stages.\n'
      'This is a retrospective sensitivity prompted by peer review, NOT the original pre-specified procedure.\n'
      'It cannot correct predictor-time leakage, maturity-selection bias, lack of an untouched test period,\n'
      'or the absence of observed review effects. It does not verify original classifier training environment.\n'
      'Results are conditional on the same 41,145 terminal-status-selected loans.\n'
      f'Selected model: {selected}. Validation selection AUC: {selection_aucs}.\n'
      f'Calibration intercept={coeff[0]:.9f}, slope={coeff[1]:.9f}.\n'
      '\n'+pd.DataFrame(rows).to_string(index=False)+'\n'
    )
    (logs/'45_three_window_validation_sensitivity_report.txt').write_text(report,encoding='utf-8')
    print('Saved:',output)
    print('STEP 45 COMPLETE. ORIGINAL FILES UNCHANGED.')

if __name__=='__main__':main()
