"""Summarize excess-HI protection by outdoor warmth and dwelling construction.

Within each calendar-date and settlement block, compare mean excess-HI
degree-hours between Control and Cool paint homes. Only blocks containing
both arms within the selected construction category contribute.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def blocks(data, endpoint, outcome, warm_col, factor, level):
    selected = data if factor == 'overall' else data.loc[data[factor].eq(level)]
    means = (selected.groupby(['date', 'Area', 'paint'], as_index=False)[outcome]
             .mean().pivot(index=['date', 'Area'], columns='paint', values=outcome)
             .dropna().reset_index())
    means = means.merge(data[['date', warm_col]].drop_duplicates(), on='date', validate='many_to_one')
    return pd.DataFrame({'endpoint': endpoint, 'factor': factor, 'level': level,
                         'date': means.date, 'Area': means.Area,
                         'control_mean_dose_c_h': means[0],
                         'paint_mean_dose_c_h': means[1],
                         'reduction_c_h': means[0] - means[1],
                         'warm': means[warm_col]})


def main():
    daily = pd.read_csv(ROOT / 'derived/daily_heat_outcomes.csv')
    homes = pd.read_csv(ROOT / 'data/home_characteristics_80homes.csv')
    weather = pd.read_csv(ROOT / 'data/outdoor_daily_context.csv')
    homes['ventilation'] = homes.ventilation.replace(
        {'Ventilation': 'Ventilated', 'No_Ventilation': 'No ventilation'})
    data = daily.merge(homes[['home_id', 'roof', 'wall', 'ventilation']],
                       on='home_id', validate='many_to_one')
    data = data.merge(weather[['date', 'warm_day', 'warm_night']],
                      on='date', validate='many_to_one')
    assert len(data) == 619
    parts = []
    for endpoint, outcome, warm_col in [
        ('daytime', 'day_excess_hi_c_h', 'warm_day'),
        ('nighttime', 'night_excess_hi_c_h', 'warm_night'),
    ]:
        parts.append(blocks(data, endpoint, outcome, warm_col, 'overall', 'Overall'))
        for factor in ('roof', 'wall', 'ventilation'):
            for level in sorted(data[factor].dropna().unique()):
                parts.append(blocks(data, endpoint, outcome, warm_col, factor, level))
    benefit = pd.concat(parts, ignore_index=True)
    benefit = benefit.loc[benefit.factor.ne('ventilation') |
                          benefit.level.ne('Not recorded')].reset_index(drop=True)
    overall = benefit.loc[benefit.factor.eq('overall')]
    assert overall.groupby('endpoint').size().to_dict() == {'daytime': 106, 'nighttime': 106}
    warm = (overall.groupby(['endpoint', 'warm']).reduction_c_h
            .agg(['size', 'mean']).reset_index())
    expected = {('daytime', True): 12.009221, ('daytime', False): 3.435275,
                ('nighttime', True): 8.578346, ('nighttime', False): 5.467476}
    for row in warm.itertuples():
        np.testing.assert_allclose(row.mean, expected[(row.endpoint, row.warm)],
                                   rtol=0, atol=1e-5)
    housing = (benefit.loc[benefit.factor.ne('overall')]
               .groupby(['endpoint', 'factor', 'level']).reduction_c_h
               .agg(['size', 'mean']).reset_index())
    out = ROOT / 'derived'
    out.mkdir(exist_ok=True)
    benefit.to_csv(out / 'context_matched_blocks.csv', index=False)
    warm.to_csv(out / 'warm_period_reductions.csv', index=False)
    housing.to_csv(out / 'housing_reductions.csv', index=False)
    print(json.dumps({'status': 'PASS', 'matched_date_settlement_blocks_per_period': 106,
                      'warm_day_mean_reduction_c_h': expected[('daytime', True)],
                      'warm_night_mean_reduction_c_h': expected[('nighttime', True)]}, indent=2))


if __name__ == '__main__':
    main()
