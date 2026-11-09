"""Tabulate roof, wall and ventilation combinations for all 80 homes."""

from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    homes = pd.read_csv(ROOT / 'data/home_characteristics_80homes.csv')
    assert len(homes) == 80 and homes.home_id.nunique() == 80
    assert homes.paint.value_counts().to_dict() == {0: 40, 1: 40}
    paths = (homes.groupby(['paint', 'roof', 'wall', 'ventilation'], dropna=False)
             .size().rename('homes').reset_index())
    assert paths.groupby('paint').homes.sum().to_dict() == {0: 40, 1: 40}
    out = ROOT / 'derived'
    out.mkdir(exist_ok=True)
    paths.to_csv(out / 'housing_combinations.csv', index=False)
    print(json.dumps({'status': 'PASS', 'enrolled_homes': 80,
                      'homes_by_arm': {str(k): int(v) for k, v in
                                       homes.paint.value_counts().items()}}, indent=2))


if __name__ == '__main__':
    main()
