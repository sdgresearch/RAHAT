"""Compute resistance, recovery and excess-HI dose from complete windows.

Each home-day has 144 ten-minute readings beginning at 06:00 local time.
Estimates use calendar-date and settlement fixed effects, with equal total
weight per home and standard errors clustered by home.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 32.2
HI_COLUMNS = [f'hi_10min_{i:03d}' for i in range(144)]


def runs(mask):
    transitions = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)))


def outcomes(hi):
    assert len(hi) == 144 and np.isfinite(hi).all()
    day, night = hi[:72], hi[72:]
    # A sustained daytime event is at least three successive 10-minute
    # exceedances; the two readings after 18:00 can confirm late-day entry.
    events = [(start, min(end, 72)) for start, end in runs(hi[:74] >= THRESHOLD)
              if start < 72 and end - start >= 3]
    resistance = 0 if day[0] >= THRESHOLD else (events[0][0] / 6 if events else 12)
    recovery = max(((end - start) / 6 for start, end in runs(night < THRESHOLD)), default=0)
    return {'resistance_h': float(resistance), 'recovery_h': float(recovery),
            'day_excess_hi_c_h': float(np.maximum(day - THRESHOLD, 0).sum() / 6),
            'night_excess_hi_c_h': float(np.maximum(night - THRESHOLD, 0).sum() / 6),
            'day_mean_hi_c': float(day.mean()), 'night_mean_hi_c': float(night.mean()),
            'day_peak_hi_c': float(day.max()), 'night_peak_hi_c': float(night.max())}


def fixed_effect(data, outcome):
    design = dmatrix('1 + C(date) + C(Area) + paint', data, return_type='dataframe')
    weights = 1 / data.groupby('home_id').home_id.transform('size')
    model = sm.WLS(data[outcome], design, weights=weights).fit(
        cov_type='cluster', cov_kwds={'groups': data.home_id, 'use_correction': True},
        use_t=True)
    low, high = model.conf_int().loc['paint']
    return {'outcome': outcome, 'paint_minus_control': float(model.params['paint']),
            'ci_low': float(low), 'ci_high': float(high),
            'p_two_sided': float(model.pvalues['paint'])}


def main():
    source = ROOT / 'derived/complete_day_hi_10min.csv'
    if not source.exists():
        source = ROOT / 'data/analysis_complete_day_hi_10min.csv'
    frame = pd.read_csv(source, dtype={'date': str, 'Area': str})
    assert len(frame) == 619 and frame.home_id.nunique() == 79
    metrics = pd.DataFrame([outcomes(row) for row in frame[HI_COLUMNS].to_numpy(float)])
    daily = pd.concat([frame[['home_id', 'date', 'Area', 'paint']], metrics], axis=1)
    rows = [fixed_effect(daily, name) for name in metrics.columns]
    estimates = pd.DataFrame(rows)
    for name, expected in [('resistance_h', 1.6669610165646274),
                           ('recovery_h', 1.95295038260774)]:
        actual = estimates.loc[estimates.outcome.eq(name), 'paint_minus_control'].iloc[0]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)

    out = ROOT / 'derived'
    out.mkdir(exist_ok=True)
    daily.to_csv(out / 'daily_heat_outcomes.csv', index=False)
    estimates.to_csv(out / 'adjusted_heat_effects.csv', index=False)
    print(json.dumps({'status': 'PASS', 'complete_household_days': len(daily),
                      'homes_with_complete_days': daily.home_id.nunique(),
                      'daytime_resistance_gain_h': float(estimates.iloc[0].paint_minus_control),
                      'nighttime_recovery_gain_h': float(estimates.iloc[1].paint_minus_control)}, indent=2))


if __name__ == '__main__':
    main()
