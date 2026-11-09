"""Rebuild complete 06:00–06:00 HI windows from all 80 homes' logger readings.

Raw temperature and humidity were recorded for all homes. Thirteen implausible
relative-humidity readings were interpolated within home before the HI step;
both the recorded and analysis values are present in the release table.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from heat_index import heat_index_c

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'derived'
HI_COLUMNS = [f'hi_10min_{i:03d}' for i in range(144)]


def main():
    obs = pd.read_csv(DATA / 'indoor_monitoring_80homes.csv', parse_dates=['timestamp'])
    homes = pd.read_csv(DATA / 'home_characteristics_80homes.csv')
    assert len(obs) == 100832 and obs.home_id.nunique() == 80
    assert homes.home_id.nunique() == 80 and homes.paint.value_counts().to_dict() == {0: 40, 1: 40}
    assert set(obs.home_id) == set(homes.home_id)
    assert not obs.duplicated(['home_id', 'timestamp']).any()
    obs = obs.sort_values(['home_id', 'timestamp']).reset_index(drop=True)
    invalid = ~obs.indoor_rh_percent_raw.between(1, 100)
    used_rh = obs.indoor_rh_percent_raw.where(~invalid)
    used_rh = used_rh.groupby(obs.home_id).transform(
        lambda series: series.interpolate(limit=24, limit_direction='both'))
    np.testing.assert_allclose(used_rh, obs.indoor_rh_percent_used, rtol=0, atol=1e-10)
    assert (obs.indoor_rh_qc_flag.eq('observed') == ~invalid).all()
    calculated = heat_index_c(obs.indoor_temp_c, used_rh)
    np.testing.assert_allclose(calculated, obs.indoor_heat_index_c, rtol=0, atol=1e-10)
    assert obs.indoor_rh_qc_flag.ne('observed').sum() == 13

    grid = (obs.set_index('timestamp').groupby('home_id').indoor_heat_index_c
            .resample('10min').mean().reset_index())
    grid['date'] = (grid.timestamp - pd.Timedelta(hours=6)).dt.normalize()
    windows = []
    for (home_id, date), group in grid.groupby(['home_id', 'date'], sort=True):
        clock = pd.date_range(date + pd.Timedelta(hours=6), periods=144, freq='10min')
        values = group.set_index('timestamp').indoor_heat_index_c.reindex(clock).to_numpy(float)
        if np.isfinite(values).all():
            windows.append((int(home_id), date, values))
    assert len(windows) == 620 and len({row[0] for row in windows}) == 79

    full = pd.DataFrame([{'home_id': home_id, 'date': date, **dict(zip(HI_COLUMNS, values))}
                         for home_id, date, values in windows])
    full = full.merge(homes[['home_id', 'settlement', 'paint']], on='home_id', validate='many_to_one')
    supported_dates = full.groupby('date').paint.nunique().eq(2)
    analysis = full.loc[full.date.map(supported_dates)].copy()
    assert len(analysis) == 619 and analysis.home_id.nunique() == 79
    assert analysis.groupby('paint').home_id.nunique().to_dict() == {0: 39, 1: 40}
    assert analysis.groupby('paint').size().to_dict() == {0: 306, 1: 313}
    analysis['date'] = analysis.date.dt.strftime('%Y-%m-%d')
    analysis = analysis.rename(columns={'settlement': 'Area'})
    analysis = analysis[['home_id', 'date', 'Area', 'paint'] + HI_COLUMNS]

    reference = pd.read_csv(DATA / 'analysis_complete_day_hi_10min.csv')
    key = ['home_id', 'date']
    joined = analysis.merge(reference, on=key, validate='one_to_one', suffixes=('_new', '_reference'))
    assert len(joined) == 619
    for field in ['Area', 'paint']:
        assert (joined[field+'_new'] == joined[field+'_reference']).all()
    for field in HI_COLUMNS:
        np.testing.assert_allclose(joined[field+'_new'], joined[field+'_reference'], rtol=0, atol=1e-10)

    OUT.mkdir(exist_ok=True)
    analysis.to_csv(OUT / 'complete_day_hi_10min.csv', index=False)
    print(json.dumps({'status': 'PASS', 'enrolled_and_recorded_homes': 80,
                      'control_homes': 40, 'cool_paint_homes': 40,
                      'logger_readings': len(obs), 'humidity_values_interpolated': 13,
                      'complete_24h_windows': 620,
                      'date_supported_analysis_windows': 619,
                      'homes_in_complete_window_analysis': 79,
                      'home_without_complete_window': 1067}, indent=2))


if __name__ == '__main__':
    main()
