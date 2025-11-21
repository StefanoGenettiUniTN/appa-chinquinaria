## Alto-Adige PM10 station data: processing and curated dataset pipeline

This document describes how we process Alto-Adige (Provincia Autonoma di Bolzano) air‑quality data to obtain a **clean, hourly, gap‑free PM10 dataset** suitable for modelling.  
The pipeline mirrors what we do for ARPAV (Veneto), ARPAL (Lombardia) and APPA (Trento), but is tailored to Alto-Adige’s specific data formats.

### 1. Raw data and directory layout

Alto-Adige meteorological and hydrological data are downloaded via the Open Meteo Data V1 API and stored under:

- **Base folder**: `data/altoadige/altoadige_2000_2025/`
- **Per‑sensor subfolders**: `LT/`, `LF/`, `WG/`, `Q/`, … (one CSV per station×sensor×year)
- **Station metadata**: `data/altoadige/altoadige_2000_2025/stations.csv`

For PM10 air‑quality analysis we work with a pre‑aggregated workbook and a small metadata CSV:

- **Measurement workbook**:  
  - `data/altoadige/PM10/PM10, NO2 E O3 12 stazioni 2008-2024.xlsx`  
  - Sheet `Foglio1` contains 12 stations with hourly **NO2, PM10, O3**.
  - Column `Station:` holds timestamps.
  - Each station has 2–3 columns, e.g. `BZ4`, `BZ4.1`, `BZ4.2`; row 0 indicates the pollutant (`NO2`, `PM10`, `O3`), row 1 the unit.

- **Station metadata CSV** (export from Excel):  
  - `data/altoadige/PM10/metadati_AltoAdige.xlsx - Foglio1.csv`  
  - Columns:
    - `STATION_NAME` (short code: `AB2`, `BZ4`, …)
    - `INDIRIZZO` (address / description)
    - `SEEHOEHE` (altitude in meters)
    - `GEO_LAENGE`, `GEO_BREITE` (longitude/latitude in DMS format, e.g. `"11 20 30.6"`)

These inputs are turned into a long, station×time PM10 table in the next step.

### 2. Merging PM10 data from the Excel workbook

- **Script**: `scripts/merge_altoadige_pm10_hourly.py`
- **Default input paths**:
  - Measurements Excel: `data/altoadige/PM10/PM10, NO2 E O3 12 stazioni 2008-2024.xlsx`
  - Metadata CSV: `data/altoadige/PM10/metadati_AltoAdige.xlsx - Foglio1.csv`
- **Output**:
  - `data/altoadige/PM10/merged_pm10_2008_2024.csv`

#### 2.1 Excel parsing and PM10 column detection

On the sheet `Foglio1`:

- Column `Station:`:
  - Row 0: label `"Messwert:"`
  - Row 1: `"Einheit:"`
  - Rows 2+ : timestamps (e.g. `2008-01-01 01:00:00`).
- All other columns correspond to station×pollutant combinations:
  - Column names: `BZ4`, `BZ4.1`, `BZ4.2`, `BZ5`, `BZ5.1`, `BZ5.2`, …
  - Row 0: pollutant code (`NO2`, `PM10`, `O3`)
  - Row 1: unit (usually `µg/m³` or similar)
  - Rows 2+: numeric values.

The merge script:

- Reads the entire sheet as a DataFrame.
- Uses the **first data row** (row 0) to detect PM10 columns:
  - Any column with `value.upper()` containing `"PM10"` is treated as a PM10 series.
- Uses the part of the column name before the dot as the **station code**:
  - `BZ4.1` → `station_code = "BZ4"`, `LS1.2` → `"LS1"`, etc.
- Converts the `Station:` column (rows 2+) to a `datetime` column.
- Keeps only rows where `datetime` parses successfully and PM10 is numeric.

The result is a long table:

- Columns: `datetime`, `station_code`, `pm10`
- One row per valid PM10 measurement per station.

#### 2.2 Joining station metadata

Metadata are loaded from the CSV and normalized:

- Rename:
  - `STATION_NAME` → `station_code`
  - `INDIRIZZO` → `address`
  - `SEEHOEHE` → `altitude_m`
  - `GEO_LAENGE` → `lon_dms`
  - `GEO_BREITE` → `lat_dms`
- Strip whitespace and drop duplicates on `station_code`.

The merged output CSV thus has:

- `datetime`, `station_code`, `pm10`, `address`, `altitude_m`, `lon_dms`, `lat_dms`

and covers **2008–2024** for 12 original stations (some with sparse PM10).

### 3. Gap and quality analysis of raw PM10 data

- **Script**: `scripts/analyze_altoadige_pm10_data.py`
- **Input**: `data/altoadige/PM10/merged_pm10_2008_2024.csv`
- **Default outputs** (if `output_dir` is set):
  - `pm10_statistics_per_station.csv`
  - `missing_values_per_year.csv`
  - `missing_values_per_station_summary.csv`
  - `contiguous_missing_periods_distribution.csv` (if gaps exist)
  - `station_distance_matrix.csv`

#### 3.1 Basic statistics and invalid values

The script:

- Parses `datetime`, adds `year`.
- For each `station_code`:
  - Computes `count`, `min`, `mean`, `max`, `std` of PM10.
  - Prints a station table with these metrics and the address.
- Computes global statistics (all stations combined).
- Flags:
  - Negative PM10 values (always invalid).
  - Very large values (`pm10 > 1000 µg/m³`, suspicious).

#### 3.2 Missing values per year

To understand coverage over time, the script:

- Determines the global period from the data (`year_min` → `year_max`).
- For each station and each year in this range:
  - Defines the expected hourly window (`year-01-01 00:00` → `year-12-31 23:00`).
  - `expected_hours` = length of this window.
  - `actual_hours` = number of rows for that station in that year.
  - `missing_hours = expected_hours - actual_hours`.
- Outputs:
  - A pivot table of `missing_hours` per `(station_code, year)`.
  - A yearly summary aggregated over all stations (total expected, actual, missing, missing%).
  - A per‑station summary over a global period (e.g. 2010–2025), with expected vs actual vs missing.

#### 3.3 Contiguous missing periods

For each station:

- Builds an expected hourly series between its first and last measurement.
- Identifies all **contiguous runs of missing timestamps** (i.e. periods with no record).
- Records the lengths of these runs (in hours).

From all stations combined:

- Builds a distribution of gap lengths:
  - Total number of gaps.
  - Min/median/mean/max gap length.
  - Per‑station counts and longest/mean gap lengths.

This guides the choice of what constitutes a “short” vs “long” gap for interpolation.

#### 3.4 Distance matrix between stations

Using `lon_dms` and `lat_dms`, the script:

- Converts DMS coordinates to decimal degrees.
- Computes a **Haversine distance matrix** between all stations:
  - Output: symmetric `station_code × station_code` DataFrame in km.
  - Summary: min/mean/max inter‑station distance.

This distance information is reused for distance‑weighted interpolation in the curation step.

### 4. Creating the curated, gap‑free Alto-Adige PM10 dataset (2014–2024)

- **Script**: `scripts/create_curated_altoadige_pm10_dataset.py`
- **Input**: `data/altoadige/PM10/merged_pm10_2008_2024.csv`
- **Outputs**:
  - `data/altoadige/PM10/merged_pm10_hourly_curated.csv`
    - Full metadata and interpolation provenance.
  - `data/altoadige/PM10/merged_pm10_hourly_curated_no_interp_metadata.csv`
    - Same PM10 values but without interpolation metadata columns.

The curated dataset is defined on a **fixed hourly grid** from:

- `2014-01-01 00:00:00` to `2024-12-31 23:00:00`
- For 8 stations (`AB2, BR1, BX1, BZ4, BZ5, LA1, LS1, ME1`)
- With **no missing values**.

#### 4.1 Station filtering and time window

The script:

- Filters the merged dataset to years **2014–2024**.
- Drops stations **AB3** and **CR2**, which have large missing periods / missing years.
- Keeps 8 stations with good coverage for the modelling window.

#### 4.2 Per‑station linear interpolation for short gaps (≤ 6 hours)

For each remaining station:

- Builds a continuous hourly time series over its actual observed span:
  - Either a single year (full calendar) or multi‑year continuous range.
- Identifies contiguous missing periods using `find_contiguous_missing_periods`:
  - A gap is a run of consecutive expected timestamps with no corresponding row.
- Classifies gaps:
  - **Short gaps**: length ≤ 6 hours.
  - **Long gaps**: length > 6 hours.
- For each short gap:
  - Requires valid values immediately before and after the gap.
  - Performs **linear interpolation** across the gap.
  - Marks the affected rows as:
    - `interpolation_method = "linear"`
- Long gaps are left as NaN at this stage.

The first pass summary (on your current data):

- **4,465** hourly values filled by linear interpolation in short gaps.

#### 4.3 Building the complete hourly grid (2014–2024)

After per‑station interpolation, the script constructs a **full hourly grid**:

- All stations × all hours between:
  - `2014-01-01 00:00:00` and `2024-12-31 23:00:00`
- This yields:
  - `96,432` hours per station,
  - `8 × 96,432 = 771,456` rows in total.

It then merges the per‑station interpolated data into this grid:

- Where a station already has `pm10` for a timestamp: keep that value and its `interpolation_method`.
- Where a value is still missing (long gaps or unobserved periods): `pm10` remains NaN for now.

At this point we have:

- A **complete grid of timestamps** for all stations in 2014–2024.
- Several thousand NaNs corresponding to long gaps and unmeasured periods.

#### 4.4 Distance‑weighted interpolation (spatial regressor) for remaining gaps

To remove all remaining gaps, we use a **simple spatial regressor** based on distances between stations:

1. **Station coordinates**:
   - Convert `lon_dms`, `lat_dms` from metadata to decimal degrees.
   - Build a `stations_df` with `station_code`, `latitude`, `longitude`.
2. **Distance matrix**:
   - Compute all‑pairs Haversine distances:
     - `distance_matrix[(s1, s2)]` = distance in km.
3. **For each missing (datetime, station)**:
   - Look at all other stations that have a PM10 value at that timestamp.
   - For each neighbor station:
     - Retrieve its distance `d` to the target station.
   - Compute **scores**:
     - `score = exp(-d / 50 km)` (closer stations → larger scores).
   - Normalize scores with softmax to obtain weights:
     - `weight_i = softmax(score_i)`.
   - Fill the missing PM10 as the **weighted average** of neighbor values.
   - Store:
     - `interpolation_method = "distance_weighted"`,
     - `interpolation_confidence` based on:
       - minimum distance to neighbors,
       - number of neighbors contributing (same idea as ARPAL).

This is analogous to the distance‑weighted interpolation used in `create_curated_arpal_pm10_dataset.py`, specialised to the Alto-Adige station set.

On your current dataset this step:

- Fills the remaining **7,284** missing values.
- Leaves **0 NaNs** in the final 2014–2024 curated dataset.

#### 4.5 Final curated outputs

The curated CSV with metadata, `merged_pm10_hourly_curated.csv`, has columns:

- `datetime` (hourly timestamps, 2014–2024)
- `station_code` (e.g. `BZ4`, `LA1`)
- `station_name` (currently equal to `station_code`)
- `pm10` (curated PM10 in µg/m³)
- `interpolation_method`:
  - `"actual"` (original data),
  - `"linear"` (short‑gap linear interpolation),
  - `"distance_weighted"` (spatial interpolation).
- `interpolation_confidence` (0–1, only for distance‑weighted values)
- `address`, `altitude_m`, `lon_dms`, `lat_dms` (metadata)

The slim version, `merged_pm10_hourly_curated_no_interp_metadata.csv`, keeps:

- `datetime`, `station_code`, `station_name`, `pm10`, `address`, `altitude_m`, `lon_dms`, `lat_dms`

and is suitable as a minimal input for modelling or further merging with other datasets.

### 5. Summary of the Alto-Adige PM10 processing pipeline

In compact form, the Alto-Adige PM10 pipeline is:

1. **Collect and aggregate PM10 data**:
   - Use `merge_altoadige_pm10_hourly.py` to extract PM10 from the multi‑station Excel workbook and join station metadata, producing `merged_pm10_2008_2024.csv`.
2. **Analyse gaps and coverage**:
   - Use `analyze_altoadige_pm10_data.py` to:
     - Inspect PM10 ranges and detect invalid values,
     - Quantify missing hours per year and station,
     - Study the distribution of contiguous gaps,
     - Build a distance matrix between stations.
3. **Curate a gap‑free 2014–2024 dataset**:
   - Use `create_curated_altoadige_pm10_dataset.py` to:
     - Restrict to 2014–2024 and drop AB3/CR2,
     - Fill short gaps (≤ 6 hours) via linear interpolation,
     - Fill all remaining gaps via distance‑weighted interpolation across stations,
     - Track the provenance and confidence of each interpolated value,
     - Export both a full metadata version and a slim version for modelling.

The final curated Alto-Adige PM10 dataset is a **uniform, hourly, gap‑free panel** of PM10 for 8 key stations over 2014–2024, aligned with how we curate PM10 data for the other regions in this project.


