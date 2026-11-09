"""Summarize reported clock-time ranges by activity and allocation arm."""

from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = pd.read_csv(ROOT / 'data/routine_clock_reports.csv')
    assert len(reports) == 226
    reports['first_h'] = reports.start_h.fillna(reports.marker_h)
    reports['last_h'] = reports.end_h.fillna(reports.marker_h)
    summary = (reports.groupby(['activity', 'paint'])
               .agg(n_reports=('report_id', 'size'), earliest_h=('first_h', 'min'),
                    latest_h=('last_h', 'max')).reset_index())
    for activity, paint, expected in [('Cooking', 1, (5.5, 11.0)),
                                      ('Cleaning', 1, (6.5, 12.0)),
                                      ('Napping', 1, (12.0, 18.0)),
                                      ('Napping', 0, (12.0, 17.0))]:
        row = summary.loc[summary.activity.eq(activity) & summary.paint.eq(paint)].iloc[0]
        assert (row.earliest_h, row.latest_h) == expected
    out = ROOT / 'derived'
    out.mkdir(exist_ok=True)
    summary.to_csv(out / 'routine_clock_ranges.csv', index=False)
    print(json.dumps({'status': 'PASS', 'timed_reports': len(reports),
                      'activity_arm_rows': len(summary)}, indent=2))


if __name__ == '__main__':
    main()
