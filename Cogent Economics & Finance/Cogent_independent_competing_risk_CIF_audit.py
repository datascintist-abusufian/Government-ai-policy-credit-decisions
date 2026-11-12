"""Independent Aalen-Johansen check of the saved Cogent Cell 17 output.

Uses integer recorded event/censor durations and the manuscript-stated
same-month terminal-event adjustment (0 -> 1 month). At tied months,
terminal events are counted in the risk set before same-month censorings.
This verifies published numerical horizon estimates conditional on the saved
proxy dates; it does NOT independently verify the original notebook code,
source event dates, or bootstrap confidence intervals.
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / 'cell17_competing_risk_outcomes.csv.gz'
OUT = HERE / 'Cogent_independent_competing_risk_CIF_horizons.csv'
d = pd.read_csv(SRC, usecols=['duration_months','event_type'])
dur = d.duration_months.to_numpy().astype('int32')
ev = d.event_type.to_numpy().astype('int8')
assert len(dur) == 2_700_690 and set(np.unique(ev)) == {0,1,2}
assert dur.min() == 0 and not np.isnan(d.duration_months).any()
zero_term = (dur == 0) & (ev > 0)
assert zero_term.sum() == 12_304
# All 12,304 same-month events enter the risk set and event count at month one.
dur[zero_term] = 1
n = len(dur)
counts = np.bincount(dur)
d1 = np.bincount(dur[ev == 1], minlength=len(counts))
d2 = np.bincount(dur[ev == 2], minlength=len(counts))
S, F1, F2 = 1.0, 0.0, 0.0
at_risk = n
horizons = [6,12,18,24,36]
rows=[]
for month in range(1,len(counts)):
    # Y(month) = subjects whose recorded duration >= month.
    expected = int((dur>=month).sum())
    assert at_risk == expected,(month,at_risk,expected)
    F1 += S * d1[month] / at_risk
    F2 += S * d2[month] / at_risk
    S *= 1. - (d1[month]+d2[month]) / at_risk
    if month in horizons:
        rows.append({'horizon_months':month,'at_risk_at_start':at_risk,
                     'charge_off_CIF':F1,'fully_paid_CIF':F2})
    at_risk -= int(counts[month])
result=pd.DataFrame(rows)
claimed_charge=[.0161,.0413,.0750,.1061,.1419]
claimed_paid=[.0650,.1438,.2370,.3329,.6748]
assert all(round(a,4)==b for a,b in zip(result.charge_off_CIF,claimed_charge))
assert all(round(a,4)==b for a,b in zip(result.fully_paid_CIF,claimed_paid))
assert result.at_risk_at_start.tolist()==[2081403,1092889,637032,522416,221142]
result.to_csv(OUT,index=False,float_format='%.10f')
print(result.to_string(index=False,float_format=lambda z:f'{z:.7f}'))
print('Same-month terminal events moved 0 -> month 1:',int(zero_term.sum()))
print('All five published charge-off and fully-paid CIF horizons reproduced to 4 decimals.')
print('Bootstrap confidence intervals were NOT recomputed; event-date validity not verified.')
print('Output:',OUT)
