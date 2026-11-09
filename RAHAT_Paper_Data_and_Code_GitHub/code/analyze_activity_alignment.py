"""Align reported routine intervals with measured resistance and recovery.

The same pooled schedules from 50 women are applied to both monitoring arms.
Panel-a-like arm means use the pooled settlement distribution. Paint–Control
contrasts adjust for date and settlement and give each monitored home equal
total weight. The script calculates point estimates, not bootstrap intervals.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 32.2
ROWS = [('R', 'Cooking'), ('R', 'Cleaning'), ('R', 'Napping'),
        ('R', 'Socializing'), ('Q', 'Socializing'), ('Q', 'Sleep')]
HI_COLUMNS = [f'hi_10min_{i:03d}' for i in range(144)]


def longest_night_run(hi):
    below = hi[72:] < THRESHOLD
    edges = np.flatnonzero(np.diff(np.r_[False, below, False]))
    starts, ends = edges[::2], edges[1::2]
    if not len(starts):
        return 18.0, 18.0
    best = int(np.argmax(ends - starts))
    return 18 + starts[best] / 6, 18 + ends[best] / 6


def overlap_minutes(start, end, thermal_start, thermal_end):
    return 60 * np.maximum(0, np.minimum(thermal_end[:, None], end[None, :])
                           - np.maximum(thermal_start[:, None], start[None, :]))


def main():
    daily = pd.read_csv(ROOT / 'derived/daily_heat_outcomes.csv')
    hi = pd.read_csv(ROOT / 'data/analysis_complete_day_hi_10min.csv')
    source = daily.merge(hi[['home_id', 'date'] + HI_COLUMNS],
                         on=['home_id', 'date'], validate='one_to_one')
    assert len(source) == 619
    values = source[HI_COLUMNS].to_numpy(float)
    night = np.asarray([longest_night_run(row) for row in values])
    np.testing.assert_allclose(night[:, 1] - night[:, 0], source.recovery_h, rtol=0, atol=1e-9)
    routine = pd.read_csv(ROOT / 'data/women_activity_intervals_50.csv')
    assert len(routine) == 250 and routine.respondent_id.nunique() == 50
    assert routine[['respondent_id', 'paint']].drop_duplicates().paint.value_counts().to_dict() == {0: 27, 1: 23}

    meta = source[['home_id', 'date', 'Area', 'paint']]
    home_weights = 1 / meta.groupby('home_id').home_id.transform('size')
    settlement_weights = (meta[['home_id', 'Area']].drop_duplicates()
                          .Area.value_counts(normalize=True).to_dict())
    design = dmatrix('1 + C(date) + C(Area) + paint', meta, return_type='dataframe')
    arms = pd.read_csv(ROOT / 'results/activity_thermal_coverage_arm_means.csv')
    gains = pd.read_csv(ROOT / 'results/activity_thermal_coverage_adjusted_gains.csv')
    ratios = pd.read_csv(ROOT / 'results/activity_thermal_coverage_percentages.csv')
    ratios = ratios.loc[ratios.option_id.eq('1_full_reported_time')]
    output = []
    for component, activity in ROWS:
        eligible = routine.loc[routine.activity.eq(activity)
                               & routine.interval_start_h.notna()
                               & routine.interval_end_h.notna()].copy()
        start, end = eligible.interval_start_h.to_numpy(float), eligible.interval_end_h.to_numpy(float)
        if component == 'R':
            thermal_start = np.full(len(source), 6.0)
            thermal_end = 6 + source.resistance_h.to_numpy(float)
        else:
            thermal_start, thermal_end = night[:, 0], night[:, 1]
        minutes = overlap_minutes(start, end, thermal_start, thermal_end)
        outcome = minutes.mean(axis=1)
        home_means = meta[['home_id', 'Area', 'paint']].copy()
        home_means['minutes'] = outcome
        home_means = home_means.groupby(['home_id', 'Area', 'paint'], as_index=False).minutes.mean()
        published = arms.loc[arms.component.eq(component) & arms.activity.eq(activity)]
        for paint in (0, 1):
            selected = home_means.loc[home_means.paint.eq(paint)]
            mean = sum(weight * selected.loc[selected.Area.eq(area), 'minutes'].mean()
                       for area, weight in settlement_weights.items())
            row = published.loc[published.paint.eq(paint)].iloc[0]
            np.testing.assert_allclose(mean, row.mean_routine_overlap_minutes, rtol=0, atol=1e-8)
            output.append({'component': component, 'activity': activity,
                           'measure': f'arm_mean_paint_{paint}', 'minutes': mean,
                           'n_reference_intervals': len(eligible)})
        contrast = sm.WLS(outcome, design, weights=home_weights).fit().params['paint']
        reference = gains.loc[gains.panel.eq('b') & gains.component.eq(component)
                              & gains.activity.eq(activity)].iloc[0]
        np.testing.assert_allclose(contrast, reference.estimate, rtol=0, atol=1e-8)
        duration = 60 * (end - start).mean()
        percentage = 100 * contrast / duration
        reference_ratio = ratios.loc[ratios.component.eq(component)
                                     & ratios.activity.eq(activity)].iloc[0]
        np.testing.assert_allclose([duration, percentage],
                                   [reference_ratio.denominator_mean_minutes,
                                    reference_ratio.estimate_percent], rtol=0, atol=1e-8)
        output.append({'component': component, 'activity': activity,
                       'measure': 'adjusted_paint_minus_control', 'minutes': contrast,
                       'n_reference_intervals': len(eligible),
                       'percent_of_full_reported_activity_time': percentage})
    out = ROOT / 'derived'
    out.mkdir(exist_ok=True)
    pd.DataFrame(output).to_csv(out / 'activity_alignment_points.csv', index=False)
    print(json.dumps({'status': 'PASS', 'surveyed_women': 50,
                      'activities': len(ROWS), 'panel_a_arm_means_checked': 12,
                      'panel_b_adjusted_gains_checked': 6,
                      'panel_c_percentages_checked': 6}, indent=2))


if __name__ == '__main__':
    main()
