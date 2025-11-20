## Meteo Trentino station selection and curated weather dataset

This document describes how we select Meteo Trentino weather stations for each APPA air quality station, build a wide matched weather dataset, analyse its gaps, and finally create a **curated, gap‑free, hourly** weather dataset suitable for modelling.

### 1. Coverage analysis of Meteo Trentino weather stations

Before matching individual APPA stations, we first understand which Meteo Trentino stations have useful data for each weather variable:

- **Script**: `scripts/analyze_weather_station_coverage.py`
- **Input**:
  - Raw Meteo Trentino ZIP/CSV archives for:
    - Temperature and precipitation (`meteo-trentino-storico-completo-temperatura-pioggia`)
    - Wind, pressure, radiation, humidity (`meteo-trentino-storico-completo-vento-pressione-radiazione-umidità`)
  - Station metadata (`data/meteo-trentino/stations.xml`)
- **Main steps**:
  - Extract all raw ZIPs to per‑variable directories (one CSV per station and variable, where possible).
  - For each station × variable pair, compute:
    - Global time coverage and fraction of valid measurements.
    - Basic completeness metrics by year.
  - Build a **coverage cache** (pickled/CSV data frame) summarising which Meteo stations have reliable data for each variable.
- **Outputs**:
  - A coverage cache used by other scripts to quickly filter stations.
  - Optional diagnostic plots/maps under `output/meteo-trentino-appa-matching/` showing spatial coverage and data availability.

This step is purely diagnostic: it does **not** change raw data, but it identifies which Meteo Trentino stations are candidates for matching.

### 2. Interactive matching between APPA and Meteo Trentino stations

The next step is to select, for each APPA air quality station and each weather variable, one Meteo Trentino station to use as the “matched” source of meteorological data.

- **Script**: `scripts/match_appa_meteo_trentino.py`
- **Key utilities reused from the coverage analysis**:
  - Loading APPA and Meteo station metadata (coordinates).
  - Building a distance matrix between APPA and Meteo stations.
  - Loading per‑station/per‑variable time series from the Meteo Trentino archives.
  - Using the precomputed coverage cache to ignore stations with no or very poor data.

#### 2.1 Finding top‑K candidate Meteo stations

For each APPA station and each weather variable:

- Compute distances to all Meteo Trentino stations.
- Filter to stations that:
  - Have that variable available, and
  - Have non‑zero valid coverage in the coverage cache.
- Keep the **K closest candidates** (default `top_k=6`) and compute:
  - Distance in km.
  - Valid coverage percentage.

These candidates are saved to CSV and used to generate **time‑series comparison plots** (per APPA station × variable), so that we can visually inspect which Meteo station’s timeseries best matches the APPA site.

#### 2.2 Manual selection mapping

The final station selection is **manual** and encoded as a hard‑coded mapping:

- **Structure**: `APPA_METEO_SELECTION` in `match_appa_meteo_trentino.py`
- Shape:
  - Keys: APPA station codes (e.g. `"402212"`).
  - Values: dict mapping variable name → Meteo station code, e.g.:
    - `"temperature": "T0408"`
    - `"wind_speed": "T0129"`

Selection criteria are:

- Short distance between APPA and Meteo stations.
- Good data coverage for the variable of interest (few long gaps).
- Consistent behaviour when visually comparing APPA PM10 with local meteorology (using candidate plots).

This mapping is the **single source of truth** for which Meteo station provides each variable for each APPA site.

### 3. Building the wide “selected Meteo Trentino” dataset

Once the APPA–Meteo station mapping is fixed, we build a single **wide** CSV that contains, for each timestamp, all the selected weather variables for all APPA sites.

- **Script**: `scripts/match_appa_meteo_trentino.py` (final step)
- **Output**:
  - `output/meteo-trentino-appa-matching/selected_meteo_trentino_weather_dataset.csv`
- **Structure**:
  - Column `datetime`
  - One column per APPA–Meteo–variable triple, named:
    - `"{appa_code}_{meteo_code}_{variable}"`
    - e.g. `402212_T0408_temperature`, `402206_T0147_wind_speed`, etc.
- **How it is built**:
  - For each entry in `APPA_METEO_SELECTION`:
    - Load the corresponding Meteo Trentino per‑station CSV for that variable.
    - Filter to valid data (`quality == 1` and non‑NaN values).
    - Keep only columns `datetime` and `value`, then rename `value` to the composite column name.
    - Drop duplicate timestamps within the same series (keep the first).
  - Outer‑join all these series on `datetime` to obtain a single, wide dataframe.
  - Sort by `datetime` and write to CSV.

At this point, the dataset typically contains:

- Non‑uniform sampling intervals (depending on Meteo sensors and raw frequency).
- Missing timestamps where no data is present.
- Holes inside each series due to true missing data or quality filtering.

### 4. Gap analysis on the selected dataset

Before curation, we run an analysis to characterise missing data and check for suspicious values.

- **Script**: `scripts/analyze_selected_meteo_trentino_weather_data.py`
- **Inputs**:
  - `selected_meteo_trentino_weather_dataset.csv`
- **Main operations**:

1. **Resampling to hourly**:
   - Parse `datetime`, sort, and use `resample_wide_dataset_to_hourly` to:
     - Resample all series to a **fixed hourly grid** using mean aggregation.
     - Keep the global window **2014‑01‑01 00:00:00 → 2024‑12‑31 23:00:00**.

2. **Per‑series missing statistics**:
   - For each series (column):
     - Define an **effective window** from its first to last non‑NaN value inside 2014‑2024.
     - Within that window:
       - `expected_hours` = number of hourly timestamps.
       - `actual_hours` = count of non‑NaN values.
       - `missing_hours` and `missing_percentage`.
     - Identify contiguous missing periods (in hours).
     - Record:
       - Number of gaps, longest/mean/median gap length.
       - Count of short gaps (≤ 4 hours, useful for linear interpolation).
     - Compute basic value diagnostics:
       - Min/max, and count of negative values (to catch invalid humidity/pressure/etc.).

3. **Global gap distribution**:
   - Aggregate all contiguous gap lengths across all series.
   - Print distribution and high‑level statistics (total gaps, longest/mean/median length, fraction of short gaps).

Outputs include CSV summaries and console diagnostics that guide the interpolation strategy (short vs long gaps, where to trust linear interpolation, where a model is needed).

### 5. Creating the curated, gap‑free meteorological dataset

The final curation step produces an **hourly, gap‑free** version of the selected Meteo Trentino dataset, while tracking how each value was obtained (original, linear interpolation, or regression model).

- **Script**: `scripts/create_curated_selected_meteo_trentino_weather_dataset.py`
- **Inputs**:
  - `selected_meteo_trentino_weather_dataset.csv`
- **Outputs** (under `output/meteo-trentino-appa-matching/curated`):
  - `curated_selected_meteo_trentino_weather_dataset.csv`
    - Clean, gap‑free dataset; same columns as the original wide CSV.
  - `curated_selected_meteo_trentino_weather_dataset_with_metadata.csv`
    - Same data columns, plus companion `"{col}__method"` columns indicating:
      - `"actual"` (original data),
      - `"linear"` (short‑gap linear interpolation),
      - `"regression"` (model‑based or final forward/backward fill).
  - `gap_filling_summary_per_series.csv`
    - Per column: counts and percentages of actual / linear / regression values.
  - `gap_filling_summary_overall.csv`
    - Global counts and percentages per method.
  - `plots/curation_timeseries/timeseries_gap_filling__{series_column}.png`
    - Timeseries plots per APPA–Meteo–variable series with method‑coloured segments.

#### 5.1 Resampling and initial method labelling

1. **Resample to hourly**:
   - Use the same resampling helper as in the analysis script to obtain an hourly dataframe `hourly` on 2014‑01‑01 → 2024‑12‑31.
2. **Initial method matrix**:
   - Build a `method_df` with the same shape as `hourly`:
     - `"actual"` where the resampled value is non‑NaN and comes directly from data.
     - `"missing"` elsewhere (to be filled later).

#### 5.2 Linear interpolation of short gaps

For each series (column):

- Identify contiguous NaN runs in the hourly grid.
- If the run length is **strictly smaller than 5 hours**:
  - Require valid values immediately before and after the gap.
  - Fill the gap by **explicit linear interpolation** between the bounding values.
  - Mark those hours in `method_df` as `"linear"`.
- Longer gaps are left as NaN for regression‑based filling.

This matches the “use linear interpolation only for short gaps” design used in the PM10 curation scripts, adapted here to a 5‑hour threshold.

#### 5.3 Regression model to fill remaining gaps

After linear interpolation, remaining NaNs typically correspond to:

- Longer gaps,
- Periods where a sensor was intermittently off,
- Sparse parts of the record.

For each **variable** (e.g. `temperature`, `rain`, `wind_speed`):

1. Group all series belonging to that variable.
2. For each **target series** in the group:
   - Use all other series of the same variable as **predictors**.
   - Build training data:
     - Rows where the target is non‑NaN and at least one predictor is non‑NaN.
     - Fill predictor NaNs with column means computed on the training subset.
   - Fit a simple **linear regression** (closed‑form least squares with intercept).
   - For rows where the target is NaN but at least one predictor has data:
     - Fill the target with the regression prediction.
     - Mark these cells as `"regression"` in `method_df`.
   - Skip series with too few training samples (e.g. < 50 time points).

This step mimics the spatial/temporal “borrowing strength” ideas from the ARPAL/APPA/ARPAV PM10 curation scripts, but uses a multi‑station regression per variable instead of distance‑weighted interpolation.

#### 5.4 Final fallback to guarantee no gaps

To ensure the curated dataset has **no NaNs at all**:

- For each series, apply a final `ffill().bfill()` to any remaining NaNs.
- Mark these cells as `"regression"` (they are effectively model‑ or imputation‑based, not direct measurements).
- Verify that the curated dataframe contains no missing values.

#### 5.5 Timeseries plots with method colouring

For each series (APPA–Meteo–variable):

- The script creates a time‑series plot where:
  - **Actual data** segments are drawn in black.
  - **Linear interpolation** segments are drawn in blue.
  - **Regression‑filled** (including fallback) segments are drawn in red.
- The legend reports, for that specific series:
  - Percentage of actual values,
  - Percentage of linear interpolation,
  - Percentage of regression‑filled values.

These plots are the main visual QA tool to understand **where** interpolation and model filling occurred in time for each matched series.

### 6. Summary of the pipeline

In compact form, the Meteo Trentino → curated matched dataset process is:

1. **Analyse coverage** of all Meteo Trentino stations per variable (`analyze_weather_station_coverage.py`).
2. **Interactively match** APPA stations to Meteo Trentino stations per variable, based on distance and coverage (`match_appa_meteo_trentino.py` and `APPA_METEO_SELECTION`).
3. **Build a wide matched dataset** with one column per APPA–Meteo–variable series (`selected_meteo_trentino_weather_dataset.csv`).
4. **Analyse gaps and value ranges** in that wide dataset (`analyze_selected_meteo_trentino_weather_data.py`).
5. **Create a curated, hourly, gap‑free dataset** using:
   - Short‑gap linear interpolation (< 5 hours),
   - Per‑variable regressions across stations for longer gaps,
   - Final fallback fill, while tracking the provenance of every value (`create_curated_selected_meteo_trentino_weather_dataset.py`).

The final curated file is designed to be a drop‑in meteorological input for downstream modelling, aligned to APPA stations through the explicit APPA–Meteo station matching. 


