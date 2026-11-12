"""Round 2 Reviewer 2, Comment 3. Seed-stability sensitivity at endpoint sample sizes.

Uses the original Cell 26 feature list and Cell 27 preprocessing/model configurations.
Checks an independent re-fit of the 180k, seed-42 benchmark before running new seeds.
Does not alter the raw data or original revision_outputs.

Run from project root:
  python 22_training_seed_stability.py  # from its saved folder
"""
from pathlib import Path
import gc
import platform
import time
import warnings
import sys
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(r"E:\Governance-Policy\Cogent Economics & Finance")
DATA = ROOT / "P2P_Macro_Data.dta"
OLD = ROOT / "revision_outputs"
OUT = ROOT / "Round2_Manuscript_Results"
TABLES = OUT / "01_Tables"
LOGDIR = OUT / "05_Reproducibility" / "Execution_Logs"
TABLES.mkdir(parents=True, exist_ok=True)
LOGDIR.mkdir(parents=True, exist_ok=True)
LOG = LOGDIR / "22_training_seed_stability_report.txt"
CSV = TABLES / "22_training_seed_stability.csv"

class Recorder:
    def __init__(self): self.lines=[]
    def say(self, *parts):
        s = " ".join(map(str,parts)); print(s, flush=True); self.lines.append(s)
    def save(self): LOG.write_text("\n".join(self.lines)+"\n",encoding="utf-8")
r=Recorder()
r.say("ROUND 2 | REVIEWER 2 COMMENT 3 | SEED-STABILITY SENSITIVITY")
r.say("Python:",sys.version.split()[0],"| NumPy:",np.__version__,"| pandas:",pd.__version__,"| sklearn:",sklearn.__version__)
r.say("Dataset:",DATA)
r.say("Original predictions are read-only. No new data is written to revision_outputs.")

SPEC = OLD / "cell26_locked_feature_specification.csv"
for p in (DATA,SPEC,OLD/"cell27_y_val.npy",OLD/"cell27_y_test.npy"):
    if not p.exists(): raise FileNotFoundError(f"Required input absent: {p}")
spec=pd.read_csv(SPEC)
if not {"feature","type"}.issubset(spec.columns): raise RuntimeError(f"Unexpected feature file columns: {spec.columns.tolist()}")
text_exclusions={"emp_title","desc","title"}
features=[str(x) for x in spec["feature"].tolist() if str(x) not in text_exclusions]
if len(features)!=150 or len(features)!=len(set(features)):
    raise RuntimeError(f"Expected 150 distinct model features, got {len(features)}")
types = dict(zip(spec["feature"].astype(str),spec["type"].astype(str)))
cat=[c for c in features if types[c].lower()=="categorical"]
num=[c for c in features if c not in cat]
if (len(num),len(cat))!=(136,14):
    raise RuntimeError(f"Feature types disagree with source specification: {len(num)} numeric, {len(cat)} categorical")
r.say("Feature specification:",len(features),"inputs;",len(num),"numeric;",len(cat),"categorical")

cols=list(dict.fromkeys(["issue_d","term","loan_status","badloan"]+features))
by_split={"train":[],"val":[],"test":[]}
count=0
cutoff=pd.Timestamp("2020-04-01")
r.say("Reconstructing mature cohorts from original Stata file in original row order...")
with pd.read_stata(DATA, columns=cols, convert_categoricals=False, chunksize=200_000) as reader:
    for df in reader:
        count+=len(df)
        issue=pd.to_datetime(df["issue_d"].astype(str).str.strip(), format="%b-%y", errors="coerce")
        term=pd.to_numeric(df["term"].astype(str).str.extract(r"(\d+)")[0],errors="coerce")
        status=df["loan_status"].astype(str).str.strip()
        mature=status.isin(["Fully Paid","Charged Off","Default"]) | (((cutoff.year-issue.dt.year)*12+(cutoff.month-issue.dt.month))>=term)
        split_masks={
            "train":issue.between(pd.Timestamp("2012-08-01"),pd.Timestamp("2018-06-01")) & mature,
            "val":issue.between(pd.Timestamp("2018-07-01"),pd.Timestamp("2019-03-01")) & mature,
            "test":issue.between(pd.Timestamp("2019-04-01"),pd.Timestamp("2019-12-01")) & mature,
        }
        for split,mask in split_masks.items():
            if mask.any():
                block=df.loc[mask,features+["badloan"]].copy()
                by_split[split].append(block)
        r.say(f"  Scanned {count:,} source records")
if count!=2_703_430: raise RuntimeError(f"Unexpected source rows: {count}")
frames={k:pd.concat(v,ignore_index=True) for k,v in by_split.items()}
del by_split; gc.collect()
expected={"train":1_304_058,"val":27_374,"test":41_145}
for k,n in expected.items():
    if len(frames[k])!=n: raise RuntimeError(f"{k} count differs: {len(frames[k])} versus {n}")
    frames[k]["y"]=pd.to_numeric(frames[k].pop("badloan"),errors="coerce").fillna(0).astype(int)
    r.say(k,"n=",len(frames[k]),"adverse rate=",f"{frames[k]['y'].mean():.6f}")
for k, fname in [("val","cell27_y_val.npy"),("test","cell27_y_test.npy")]:
    y0=np.load(OLD/fname)
    if not np.array_equal(frames[k]["y"].to_numpy(),y0):
        raise RuntimeError(f"{k} outcome/order does not match saved predictions. Stop before fitting!")
r.say("PASS: Fixed validation/test labels match original prediction-array order.")

pre=ColumnTransformer(transformers=[
 ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
 ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                  ("onehot",OneHotEncoder(handle_unknown="ignore",max_categories=50,sparse_output=False))]),cat)
],remainder="drop")
r.say("Fitting original full-training preprocessor once (comparison with original, NOT independent sample-specific preprocessing)...")
pre.fit(frames["train"][features])
X_val=pre.transform(frames["val"][features]).astype(np.float32)
X_test=pre.transform(frames["test"][features]).astype(np.float32)
y_val=frames["val"]["y"].to_numpy()
y_test=frames["test"]["y"].to_numpy()
if X_val.shape[1]!=384: raise RuntimeError(f"Unexpected transformed columns {X_val.shape[1]}; expected 384")
r.say("Feature matrix width:",X_val.shape[1],"| fixed val/test dimensions:",X_val.shape,X_test.shape)
del frames["val"],frames["test"]; gc.collect()
r.say("Transforming mature full training matrix once; the same transformer is used for all seeds...")
X_full=pre.transform(frames["train"][features]).astype(np.float32)
y_full=frames["train"]["y"].to_numpy()
del frames,pre; gc.collect()
r.say("Full training matrix shape:",X_full.shape)

# Reproduce the original use of both a sampling permutation and model random_state.
def factory(seed):
    return {
      "Logistic_SGD":SGDClassifier(loss="log_loss",penalty="l2",alpha=1e-5,max_iter=100,tol=1e-4,random_state=seed,n_jobs=-1),
      "HistGB":HistGradientBoostingClassifier(max_depth=6,learning_rate=0.05,max_iter=300,random_state=seed),
      "MLP":MLPClassifier(hidden_layer_sizes=(128,64),activation="relu",solver="adam",max_iter=50,early_stopping=True,validation_fraction=0.10,n_iter_no_change=5,random_state=seed),
    }

def measures(y,p):
    return dict(auc=float(roc_auc_score(y,p)),brier=float(brier_score_loss(y,p)),logloss=float(log_loss(y,p,labels=[0,1])))

rows=[]
def checkpoint():
    pd.DataFrame(rows).to_csv(CSV,index=False)
    r.save()

def evaluate(seed,n,check_baseline=False):
    idx=np.random.default_rng(seed).permutation(len(y_full))[:n]
    X=X_full[idx]  # original notebook transformed after indexing; same fitted transformer
    y=y_full[idx]
    r.say("\nTraining n=",f"{n:,}","seed=",seed,"sample positive rate=",f"{y.mean():.6f}")
    for model_name,model in factory(seed).items():
        start=time.perf_counter()
        model.fit(X,y)
        vp=model.predict_proba(X_val)[:,1]
        tp=model.predict_proba(X_test)[:,1]
        vm=measures(y_val,vp); tm=measures(y_test,tp)
        duration=round(time.perf_counter()-start,1)
        r.say(model_name,"val_auc=",f"{vm['auc']:.6f}","test_auc=",f"{tm['auc']:.6f}","test_brier=",f"{tm['brier']:.6f}","fit_eval_seconds=",duration)
        if check_baseline:
            pth=OLD/f"cell27_{n}_{model_name}_test_pred.npy"
            if not pth.exists(): raise FileNotFoundError(pth)
            ref=np.load(pth)
            if ref.shape!=tp.shape: raise RuntimeError("Original test predictions have different shape")
            reference=measures(y_test,ref)
            aucdelta=abs(reference["auc"]-tm["auc"])
            brierdelta=abs(reference["brier"]-tm["brier"])
            r.say("   benchmark comparison: |AUC difference|=",f"{aucdelta:.6f}","|Brier difference|=",f"{brierdelta:.6f}")
            # Strict check before interpreting newer-seed runs. A version mismatch may make them incomparable.
            if aucdelta>0.002 or brierdelta>0.003:
                checkpoint()
                raise RuntimeError("Seed-42 reproduction differs from the original predictions. Do not interpret or proceed with other seeds; inspect versions and preprocessing.")
        rows.append(dict(seed=int(seed),training_n=int(n),model=model_name,
                         val_auc=vm["auc"],test_auc=tm["auc"],test_brier=tm["brier"],
                         test_logloss=tm["logloss"],fit_eval_seconds=duration,
                         evidence="new_fit"))
        checkpoint()
        del model,vp,tp
        gc.collect()
    del X; gc.collect()

# Do not fit the expensive 1.3M seed-42 baseline again. It already exists as validated saved test predictions.
# Only after the smaller seed-42 reproduction passes do we add these original full-training rows.
r.say("\nPhase 1: validate old seed-42 180k training before new sensitivity fits")
evaluate(42,180_000,check_baseline=True)
r.say("PASS: Original seed-42 180k experiment reproduced within declared tolerances")
for model_name in ["Logistic_SGD","HistGB","MLP"]:
    pval=np.load(OLD/f"cell27_{len(y_full)}_{model_name}_val_pred.npy")
    ptest=np.load(OLD/f"cell27_{len(y_full)}_{model_name}_test_pred.npy")
    vm=measures(y_val,pval);tm=measures(y_test,ptest)
    rows.append(dict(seed=42,training_n=len(y_full),model=model_name,
                     val_auc=vm["auc"],test_auc=tm["auc"],test_brier=tm["brier"],
                     test_logloss=tm["logloss"],fit_eval_seconds=np.nan,
                     evidence="original_saved_predictions"))
checkpoint()

r.say("\nPhase 2: two additional sampling/model random seeds (7 and 101) for 180k and full training")
for seed in [7,101]:
    for n in [180_000,len(y_full)]:
        evaluate(seed,n)

results=pd.DataFrame(rows)
for n in [180_000,len(y_full)]:
    r.say("\nENDPOINT",f"{n:,}")
    p=results.loc[results.training_n==n].pivot(index="seed",columns="model",values="test_auc")
    r.say(p.to_string(float_format=lambda x:f"{x:.6f}"))
    r.say("Top model per seed:",p.idxmax(axis=1).to_dict())
r.say("\nLIMITATIONS: only 3 seeds; sample selection and estimator randomness vary together; original preprocessing fitted on full training data; no hyperparameter re-tuning; outcome selection and predictor-timing concerns remain.")
r.say("Saved metrics:",CSV)
r.say("Saved log:",LOG)
checkpoint()
r.say("STEP 22 COMPLETE")
