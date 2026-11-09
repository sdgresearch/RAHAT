"""NOAA/NWS heat index used for the indoor monitoring series."""

import numpy as np


def heat_index_c(temperature_c, rh_percent):
    temperature_c = np.asarray(temperature_c, dtype=float)
    rh = np.asarray(rh_percent, dtype=float)
    t = temperature_c * 9 / 5 + 32
    simple = 0.5 * (t + 61 + (t - 68) * 1.2 + rh * 0.094)
    simple = (simple + t) / 2
    use_regression = simple >= 80
    regression = (
        -42.379 + 2.04901523*t + 10.14333127*rh
        - 0.22475541*t*rh - 0.00683783*t**2 - 0.05481717*rh**2
        + 0.00122874*t**2*rh + 0.00085282*t*rh**2
        - 0.00000199*t**2*rh**2
    )
    low = use_regression & (rh < 13) & (t >= 80) & (t <= 112)
    regression = np.where(low, regression - (13-rh)/4 *
                          np.sqrt(np.maximum(0, (17-np.abs(t-95))/17)), regression)
    high = use_regression & (rh > 85) & (t >= 80) & (t <= 87)
    regression = np.where(high, regression + (rh-85)/10 * (87-t)/5, regression)
    hi_f = np.where(use_regression, regression, simple)
    invalid = (~np.isfinite(temperature_c)) | (~np.isfinite(rh)) | (rh < 0) | (rh > 100)
    return np.where(invalid, np.nan, (hi_f - 32) * 5 / 9)
