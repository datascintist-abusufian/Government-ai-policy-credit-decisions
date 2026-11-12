"""Audit split-level follow-up support for Cogent competing-risk outcomes.

Requires pandas and numpy. This does NOT calculate classifier AUC or calibration.
Input: cell17_competing_risk_outcomes.csv.gz from the Cogent study.
The duration at or beyond a horizon is a risk-set-support count, not a count
of verified fixed-horizon non-default labels, and does not imply a valid
classifier validation sample.
"""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).with_name('cell17_competing_risk_outcomes.csv.gz')
OUT = Path(__file__).with_name('Cogent_censoring_horizon_support_by_split.csv')
HORIZONS = (6, 9, 12, 18, 24, 36)

x = pd.read_csv(DATA, usecols=['issue_date', 'duration_months', 'event_type'], parse_dates=['issue_date'])
assert set(x.event_type.dropna().unique()).issubset({0., 1., 2.})
assert len(x) == 2_700_690
assert x[['issue_date', 'duration_months', 'event_type']].notna().all().all()
splits = {
    'train': x.issue_date < pd.Timestamp('2018-07-01'),
    'validation': x.issue_date.between('2018-07-01', '2019-03-31'),
    'test': x.issue_date.between('2019-04-01', '2019-12-31'),
}
assert sum(m.sum() for m in splits.values()) == len(x)
rows = []
for name, mask in splits.items():
    d = x.loc[mask, 'duration_months']
    e = x.loc[mask, 'event_type']
    n = len(d)
    for tau in HORIZONS:
        at_risk = int((d >= tau).sum())
        censored_earlier = int(((e == 0) & (d < tau)).sum())
        chargeoff_through = int(((e == 1) & (d <= tau)).sum())
        fullypaid_through = int(((e == 2) & (d <= tau)).sum())
        rows.append({
            'split': name, 'horizon_months': tau, 'n_competing_risk': n,
            'n_duration_at_least_horizon': at_risk,
            'pct_duration_at_least_horizon': round(100 * at_risk / n, 4),
            'n_censored_before_horizon': censored_earlier,
            'pct_censored_before_horizon': round(100 * censored_earlier / n, 4),
            'n_chargeoff_by_horizon': chargeoff_through,
            'n_fullypaid_by_horizon': fullypaid_through,
            'n_event_type_censored': int((e == 0).sum()),
            'max_observed_duration_months': float(d.max()),
        })
result = pd.DataFrame(rows)
result.to_csv(OUT, index=False)
print(result[['split', 'horizon_months', 'n_competing_risk', 'n_duration_at_least_horizon', 'pct_duration_at_least_horizon', 'n_censored_before_horizon','pct_censored_before_horizon','n_chargeoff_by_horizon']].to_string(index=False))
print(f'\nSaved: {OUT}')
