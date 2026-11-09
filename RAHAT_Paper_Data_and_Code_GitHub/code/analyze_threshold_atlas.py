"""Compute the threshold–clock protection atlas and confidence bands."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HI_COLUMNS = [f'hi_10min_{index:03d}' for index in range(144)]
THRESHOLDS = np.arange(28.0, 36.01, 0.5)
BOOTSTRAPS = 2000
SEED = 20261015


def main():
    states = pd.read_csv(ROOT / 'data/analysis_complete_day_hi_10min.csv')
    values = states[HI_COLUMNS].to_numpy(float)
    assert values.shape == (619, 144) and np.isfinite(values).all()
    # Each half-hour cell is the fraction of its three observed 10-minute
    # readings below a candidate heat-index threshold.
    daily = (values[:, None, :].reshape(len(states), 1, 48, 3)
             < THRESHOLDS[None, :, None, None]).mean(axis=3)
    meta = states[['home_id', 'Area', 'paint']].drop_duplicates().reset_index(drop=True)
    home = np.stack([daily[states.home_id.eq(home_id).to_numpy()].mean(axis=0)
                     for home_id in meta.home_id])
    assert len(home) == 79
    areas = sorted(states.Area.unique())
    assert len(areas) == 4
    # Standardize both arms to the pooled distribution of the 79 monitored
    # homes across the four settlements.
    weights = (meta.groupby('Area').size() / len(meta)).to_dict()

    rng = np.random.default_rng(SEED)
    results = {}
    for arm in (0, 1):
        mean = np.zeros((17, 48))
        boot = np.zeros((BOOTSTRAPS, 17, 48))
        for area, weight in weights.items():
            selected = home[(meta.Area.eq(area) & meta.paint.eq(arm)).to_numpy()]
            n = len(selected)
            assert n > 0
            mean += weight * selected.mean(axis=0)
            draws = rng.multinomial(n, np.full(n, 1 / n), size=BOOTSTRAPS)
            boot += weight * (draws @ selected.reshape(n, -1) / n).reshape(BOOTSTRAPS, 17, 48)
        results[arm] = (mean, boot)

    gain = 100 * (results[1][0] - results[0][0])
    draws = 100 * (results[1][1] - results[0][1])
    se = draws.std(axis=0, ddof=1)
    variable = se > 1e-10
    max_statistic = np.max(np.abs((draws[:, variable] - gain[None, variable])
                                  / se[None, variable]), axis=1)
    critical = float(np.quantile(max_statistic, 0.95))
    simultaneous_low = gain - critical * se
    simultaneous_high = gain + critical * se
    pointwise_low, pointwise_high = np.quantile(draws, [0.025, 0.975], axis=0)

    published = pd.read_csv(ROOT / 'results/threshold_time_atlas.csv')
    assert len(published) == 816
    arrays = {
        'control_probability': results[0][0],
        'paint_probability': results[1][0],
        'paint_minus_control_pp': gain,
        'pointwise_low_pp': pointwise_low,
        'pointwise_high_pp': pointwise_high,
        'simultaneous_low_pp': simultaneous_low,
        'simultaneous_high_pp': simultaneous_high,
    }
    maximum_error = 0.0
    for column, array in arrays.items():
        difference = np.abs(published[column].to_numpy(float) - array.reshape(-1))
        maximum_error = max(maximum_error, float(difference.max()))
    # Random draws are applied to the original row order, which is retained
    # while replacing household IDs with unrelated pseudonyms.
    assert maximum_error < 1e-8, maximum_error
    assert (published.simultaneous_positive.to_numpy(bool) ==
            ((variable & (simultaneous_low > 0)).reshape(-1))).all()
    calculated = published[['threshold_hi_c', 'window_hour']].copy()
    for column, array in arrays.items():
        calculated[column] = array.reshape(-1)
    calculated['simultaneous_positive'] = (variable & (simultaneous_low > 0)).reshape(-1)
    output = ROOT / 'derived'
    output.mkdir(exist_ok=True)
    calculated.to_csv(output / 'threshold_time_atlas.csv', index=False)
    peak = published.loc[published.paint_minus_control_pp.idxmax()]
    print(json.dumps({'status': 'PASS', 'atlas_cells': len(published),
                      'bootstrap_replicates': BOOTSTRAPS,
                      'maximum_absolute_difference_from_supplied_table': maximum_error,
                      'peak_threshold_hi_c': float(peak.threshold_hi_c),
                      'peak_clock_hour': float(peak.window_hour),
                      'peak_advantage_percentage_points': float(peak.paint_minus_control_pp)}, indent=2))


if __name__ == '__main__':
    main()
