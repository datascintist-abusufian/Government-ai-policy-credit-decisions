"""STEP 44: Evidence inventory, multiplicity adjustment and LOCAL replication staging.

This program NEVER changes existing research results or makes a public upload.
It takes read-only snapshots of the existing files and builds a reviewable ZIP.
The ZIP is a reproducibility candidate, NOT evidence of independent clean replay.

Usage in PowerShell from any directory:
  python 44_build_cogent_evidence_package_and_holm.py --root "E:\\Governance-Policy\\Cogent Economics & Finance"

Requires Python, pandas (installed in the verified Round 2 environment).
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import zipfile

import pandas as pd

DEFAULT_ROOT = Path(r"E:\Governance-Policy\Cogent Economics & Finance")
EXPECTED_SCRIPTS = [
    "14_inventory_existing_outputs.py",
    "16_verify_existing_statistical_outputs.py",
    "17_review_all_policy_benchmarks.py",
    "18_brier_baseline_verification.py",
    "19_predictor_provenance_audit.py",
    "20_training_vintage_composition.py",
    "22_training_seed_stability.py",
    "23_validation_auc_selection_uncertainty.py",
    "31_conservative_predictor_sensitivity.py",
    "32_clustered_test_uncertainty_v2.py",  # supersedes original v1
    "33_subgroup_cluster_support_audit.py",
    "34_illustrative_governance_trigger_audit.py",
]

REQUIRED_ORIGINAL = [
    "cell26_locked_feature_specification.csv",
    "cell26_feature_missingness.csv",
    "cell27_training_size_sensitivity.csv",
    "cell27_y_val.npy", "cell27_y_test.npy",
    "cell27_1304058_Logistic_SGD_val_pred.npy",
    "cell27_1304058_HistGB_val_pred.npy",
    "cell27_1304058_MLP_val_pred.npy",
    "cell27_1304058_Logistic_SGD_test_pred.npy",
    "cell27_1304058_HistGB_test_pred.npy",
    "cell27_1304058_MLP_test_pred.npy",
    "cell28_paired_bootstrap_model_comparisons.csv",
    "cell29_histgb_full_val_recalibrated.npy",
    "cell29_histgb_full_test_recalibrated.npy",
    "cell30_recalibrated_threshold_analysis.csv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def holm_bh(df: pd.DataFrame) -> pd.DataFrame:
    """Define one family: six test ROC-AUC contrasts, two training endpoints x three pairs.

    Existing p-values are rounded to 0.001, the original bootstrap's resolution
    floor. Corrected p-values should therefore also be treated as approximate.
    """
    required = {'training_n','model_A','model_B','auc_diff_A_minus_B','auc_bootstrap_p'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing pairwise comparison columns: {required - set(df.columns)}')
    df = df.copy()
    if len(df) != 6 or set(df.training_n.astype(int)) != {180000,1304058}:
        raise ValueError('Expected exactly six original endpoint pairwise comparisons; refusing to redefine family.')
    p = df['auc_bootstrap_p'].astype(float).to_numpy()
    if not all(0 <= x <= 1 for x in p):
        raise ValueError('Invalid p-values')
    import numpy as np
    n = len(p)
    order = np.argsort(p, kind='stable')
    ranked = p[order]
    holm_sorted = np.maximum.accumulate((n-np.arange(n))*ranked)
    holm = np.empty(n)
    holm[order] = np.minimum(1.0, holm_sorted)
    bh_sorted = np.minimum.accumulate((ranked*n/(np.arange(n)+1))[::-1])[::-1]
    bh = np.empty(n)
    bh[order] = np.minimum(1.0, bh_sorted)
    df['holm_adj_bootstrap_p'] = holm
    df['bh_fdr_adj_bootstrap_p'] = bh
    df['holm_significant_at_0p05'] = holm < .05
    df['inference_family'] = 'six endpoint pairwise test ROC-AUC contrasts only'
    df['p_value_qualification'] = 'Original rounded p=0.001 has finite-bootstrap resolution; not an exact tail probability'
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--no-zip', action='store_true', help='Produce audit and tables without staging a ZIP')
    args = parser.parse_args()
    root = args.root.resolve()
    old = root/'revision_outputs'
    base = root/'Round2_Manuscript_Results'
    code = base/'05_Reproducibility'/'Code'
    tables = base/'01_Tables'
    report_dir = base/'05_Reproducibility'/'Evidence_Checks'
    report_dir.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    print('STEP 44 | Cogent reproducibility readiness + multiplicity audit')
    print('Project root:', root)
    if not old.is_dir() or not base.is_dir():
        raise FileNotFoundError('Expected revision_outputs and Round2_Manuscript_Results under project root')

    package_checks = []
    def check(label, path, required):
        exists = path.is_file()
        package_checks.append(dict(label=label, path=str(path), exists=exists, required=required,
                                   size_bytes=(path.stat().st_size if exists else None)))
        print(('PASS ' if exists else 'MISSING ') + label + ': ' + str(path))
        return exists

    for name in EXPECTED_SCRIPTS:
        check('round2_code/'+name, code/name, True)
    # The earlier original v1 is intentionally not part of the recommended execution path.
    for name in REQUIRED_ORIGINAL:
        check('original_output/'+name, old/name, True)
    check('original_dataset/P2P_Macro_Data.dta', root/'P2P_Macro_Data.dta', False)
    check('source/P2P_Macro_Codes.do', root/'P2P_Macro_Codes.do', False)
    check('source/Labels.do', root/'Labels.do', False)

    notebooks = sorted({p.resolve() for pattern in ('*.ipynb', '*/*.ipynb') for p in root.glob(pattern)
                        if p.is_file() and not any(part.startswith('.') for part in p.parts)})
    origin_notebooks = [p for p in notebooks if 'NeurIPS' not in p.name]
    print('Potential original modelling notebooks:', [str(p) for p in origin_notebooks] or 'NOT FOUND')
    if not origin_notebooks:
        package_checks.append(dict(label='original_modelling_notebook',path='Search project root and one subdirectory',
                                   exists=False,required=True,size_bytes=None))

    stat_src = old/'cell28_paired_bootstrap_model_comparisons.csv'
    adjusted = None
    if stat_src.is_file():
        adjusted = holm_bh(pd.read_csv(stat_src))
        out = tables/'44_holm_BH_adjusted_six_test_auc_comparisons.csv'
        adjusted.to_csv(out,index=False)
        print('\nMULTIPLICITY: Holm and BH adjusted results (six comparisons only)')
        print(adjusted[['training_n','model_A','model_B','auc_bootstrap_p','holm_adj_bootstrap_p','bh_fdr_adj_bootstrap_p']].to_string(index=False))
        print('Saved:',out)

    summary = {
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'project_root':str(root),
        'python_current':platform.python_version(),
        'original_training_environment_exact_versions_verified':False,
        'current_environment':{name:(metadata.version(name) if _installed(name) else 'not-installed') for name in
                               ('numpy','pandas','scipy','scikit-learn','matplotlib','statsmodels','lifelines','joblib','threadpoolctl')},
        'checks':package_checks,
        'counts':{
            'round2_python':len(list(base.rglob('*.py'))),
            'round2_csv':len(list(base.rglob('*.csv'))),
            'round2_txt':len(list(base.rglob('*.txt'))),
            'source_prediction_arrays':len(list(old.glob('*.npy'))),
        },
        'missing_required':[row['label'] for row in package_checks if row['required'] and not row['exists']],
        'not_reproducibly_solved_by_this_audit':[
            'outcome-dependent maturity selection',
            'historical decision-time predictor provenance',
            'original validation reuse',
            'original software environment provenance',
            'observed real-world human review and intervention costs',
            'four-jurisdiction high-unemployment subgroup confounding',
            'independent clean-environment end-to-end replay',
            'prospective governance-control effectiveness',
        ]
    }
    audit_json = report_dir/'44_readiness_audit.json'
    audit_json.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    with (report_dir/'44_readiness_audit.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f,fieldnames=['label','path','exists','required','size_bytes'])
        writer.writeheader();writer.writerows(package_checks)
    print('\nSaved audit:', audit_json)

    if args.no_zip:
        return
    # STAGING ONLY: public release requires author review for confidential identifiers/paths and a clean replay.
    staging_root = report_dir/'44_Cogent_replication_candidate'
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)
    def copytree_files(src_dir:Path, dest:Path, extensions:set[str]):
        if not src_dir.is_dir():return
        for p in src_dir.rglob('*'):
            if not p.is_file() or p.suffix.lower() not in extensions:
                continue
            if 'Evidence_Checks' in p.parts:  # avoid packaging previous staging recursively
                continue
            if '32_clustered_test_uncertainty.py'==p.name:  # incorrect version superseded
                continue
            new=dest/p.relative_to(src_dir)
            new.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,new)
    copytree_files(base,staging_root/'Round2_Manuscript_Results',{'.py','.csv','.txt','.md'})
    copytree_files(old,staging_root/'revision_outputs',{'.csv','.npy'})
    for n in origin_notebooks:
        out=staging_root/'original_notebooks'/n.name
        out.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(n,out)
    for name in ('P2P_Macro_Codes.do','Labels.do'):
        p=root/name
        if p.is_file():
            (staging_root/'source_construction').mkdir(exist_ok=True)
            shutil.copy2(p,staging_root/'source_construction'/name)
    readme = staging_root/'README_READ_BEFORE_RELEASING.md'
    readme.write_text("""# Cogent Economics & Finance: candidate replication materials

IMPORTANT: This is a **local candidate archive**, not a verified public release.
It has not been replayed end-to-end in a clean environment. Before publishing,
inspect notebooks and logs for personal information, absolute local paths, and licences.

## Distinguish the experiments
This is **not** the NeurIPS 2026 2015-vintage / 38-feature study.
The Cogent classification experiment uses an outcome-selected 41,145-loan test
cohort and the original 150-feature specification.

## Data
Mendeley Data v3, DOI 10.17632/wb3ndt69gf.3. The raw `P2P_Macro_Data.dta`
is NOT included in this candidate archive.

## Files
- `original_notebooks/`: notebook candidates needing manual provenance checking.
- `revision_outputs/`: original saved prediction arrays and tabular outputs.
- `Round2_Manuscript_Results/`: Round 2 scripts and outputs, including corrected
  `32_clustered_test_uncertainty_v2.py`.
- `SHA256_MANIFEST.csv`: hashes of included files.
- `44_holm_BH_adjusted_six_test_auc_comparisons.csv`: defined six-test AUC family.

## Important limitations
The saved test predictions are conditional on an outcome-selected cohort.
Historical decision-time availability of predictors and fixed-horizon outcomes
have not been independently verified. The Round 2 pip versions reflect the
current environment, not necessarily the original model-training environment.
Observed human review and governance interventions are unavailable.

## Publication prerequisite
Run an independent clean-environment replay, including source data parsing,
preprocessing, model fitting and recomputation of reported metrics. Verify that
all included materials can lawfully be published. Resolve any mismatches and
document them. Do not claim that this ZIP has already passed these tests.
""",encoding='utf-8')
    # Include audit and adjusted table in package.
    shutil.copy2(audit_json,staging_root/'44_readiness_audit.json')
    if adjusted is not None:
        shutil.copy2(tables/'44_holm_BH_adjusted_six_test_auc_comparisons.csv',staging_root/'44_holm_BH_adjusted_six_test_auc_comparisons.csv')
    digest_rows=[]
    for p in sorted(x for x in staging_root.rglob('*') if x.is_file()):
        digest_rows.append((p.relative_to(staging_root).as_posix(),sha256(p),p.stat().st_size))
    with (staging_root/'SHA256_MANIFEST.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['relative_path','sha256','size_bytes']);w.writerows(digest_rows)
    archive = report_dir/'44_Cogent_replication_candidate_NOT_VERIFIED.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in staging_root.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(report_dir))
    print('Files in staging:', len(digest_rows))
    print('Candidate ZIP:',archive)
    print('ZIP size (MB):',round(archive.stat().st_size/(1024**2),2))
    print('WARNING: This is NOT yet an independently verified/public replication package.')
    print('STEP 44 COMPLETE')


def _installed(name):
    try:metadata.version(name);return True
    except metadata.PackageNotFoundError:return False

if __name__=='__main__':main()
