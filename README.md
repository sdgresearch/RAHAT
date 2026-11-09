# RAHAT paper: monitoring data and core analysis

This repository accompanies the manuscript on the community-engaged cool-paint living lab in Jodhpur, India. It provides the de-identified observations and the main processing steps behind the paper's thermal resistance, nighttime recovery, weather and housing context, and alignment with women's reported routines. It is an **analysis package for the whole paper**.

## Study population and analysis windows

The original allocation has **80 homes: 40 Control and 40 Cool paint**. All 80 have indoor logger observations in `data/indoor_monitoring_80homes.csv`, totaling **100,832 readings**. One Control home has only a partial daytime record. The complete 06:00–06:00 analysis therefore has **620 complete windows from 79 homes**; one window lacked both study arms on its calendar date, leaving **619 supported household-days and 89,136 ten-minute HI values** for the principal analyses. These are different counts for enrollment, observed readings, and complete-window analysis.

The routine survey comprised **50 women**, 27 in Control and 23 in Cool paint households. The released survey tables are de-identified views of the reported clock times and durations. Their respondent identifiers do not link to the monitored home identifiers.

## Files

| File | Contents |
|---|---|
| `data/indoor_monitoring_80homes.csv` | Timestamped indoor air temperature, recorded relative humidity, humidity used after quality control, HI and anonymous home ID for all 80 homes. |
| `data/home_characteristics_80homes.csv` | Allocation, settlement, roof, wall, floor and ventilation for the 80 enrolled homes. |
| `data/analysis_complete_day_hi_10min.csv` | Reference table of the 619 supported complete days; 144 HI readings per 06:00–06:00 window. |
| `data/outdoor_hourly_context.csv` | Hourly outdoor temperature context used in the weather description. |
| `data/outdoor_daily_context.csv` and `data/outdoor_calendar_p80_thresholds.csv` | Daily outdoor maxima/minima, 80th-percentile reference thresholds and warm-day/night flags. |
| `data/routine_clock_reports.csv` | 226 anonymous timed reports used to describe the daily routine ranges. |
| `data/women_activity_intervals_50.csv` | Five activity records for each of the 50 surveyed women, with reported duration and clock interval where available. |
| `results/threshold_time_atlas.csv` | The 816-cell threshold-by-clock protection analysis, including uncertainty bounds. |
| `results/activity_thermal_coverage_*.csv` | Published arm means, adjusted minute contrasts and percentage summaries for activity-aligned resistance/recovery coverage. |

The observed temperature and raw humidity columns retain the logger readings in a common CSV structure. Thirteen humidity values outside 1–100% were interpolated within home; the original value, analysis value and flag are all retained. `code/process_monitoring.py` repeats that quality-control step and checks the HI calculation.

## Run the core analysis

Use Python 3.11 or newer and install the packages in `requirements.txt`. From this folder, run:

```bash
python -m pip install -r requirements.txt
python code/process_monitoring.py
python code/analyze_heat.py
python code/analyze_threshold_atlas.py
python code/analyze_context.py
python code/analyze_routine_timing.py
python code/analyze_activity_alignment.py
python code/summarize_housing.py
```

The scripts write recreated tables into `derived/`. They verify the released observation counts and principal results against the manuscript source tables. In particular, they reproduce the 1.67-hour daytime resistance gain, 1.95-hour nighttime recovery gain, threshold–clock atlas, warm-period degree-hour reductions, housing contrasts, reported routine ranges, and Figure 5 point estimates. The published Figure 5 confidence intervals are retained in `results/`; the streamlined scripts reproduce its point estimates without rerunning the two-cohort bootstrap.

## Definitions and data notes

`home_id` is a release-only random pseudonym consistent across monitoring and housing files. The mapping to original household IDs is not included. `paint` is 0 for Control and 1 for Cool paint. The 144 HI columns in the complete-day table correspond to ten-minute cells from 06:00 to 05:50 the next morning. The operational threshold is **HI = 32.2 °C**. Daytime resistance ends with the first three-reading sustained exceedance between 06:00 and 18:00; nighttime recovery is the longest continuous below-threshold interval between 18:00 and 06:00.

Routine `start_h`, `end_h` and `marker_h` are local clock hours measured from midnight; values above 24 indicate the following morning. The 250-row woman-activity table retains respondent linkage across five activities using a second, unrelated pseudonym. The 226-row clock-report table is a separate report-level view and has no public respondent crosswalk. Activity alignment applies the same pooled reported intervals to both monitored thermal groups.

Daily warmth flags use the supplied 80th-percentile calendar reference for daytime maximum and nighttime minimum outdoor temperature. The daily/hourly outdoor series and percentile thresholds are derived analysis inputs; the full third-party historical weather archive is not redistributed. Original logger workbooks, names, addresses, survey questionnaires, and identifiable photographs are not included. The manuscript figures are supplied in the manuscript/Overleaf package rather than duplicated here.
