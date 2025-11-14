# ARPAL PM10 Dataset Curation Guide

This document describes the curation process applied to the raw ARPAL PM10 hourly dataset (Lombardia region) to create a high-quality, analysis-ready dataset.

## Overview

The curation process consists of:
1. Converting Excel file to CSV format
2. Extracting station information with coordinates
3. Creating a curated dataset with data quality improvements

## Conversion Process

### Step 1: Excel to CSV Conversion

**Script**: `scripts/convert_arpal_excel_to_csv.py`

**Input**: `data/arpal/PM10/PM10_Orari_Lombardia.xlsx`
- Sheet 1: PM10 hourly data (wide format with datetime + station columns)
- Sheet 2: Station information (names, UTM coordinates, altitude)

**Output**:
- `data/arpal/PM10/merged_pm10_hourly.csv` - Long format CSV (datetime, station_code, station_name, pm10)
- `data/arpal/PM10/arpal_pm10_stations.csv` - Station information with lat/lon coordinates

**Process**:
- Reads Excel file, skipping header rows
- Converts from wide format (one column per station) to long format
- Converts UTM coordinates to latitude/longitude
- Maps station names from column headers to station information

## Curation Steps

**Script**: `scripts/create_curated_arpal_pm10_dataset.py`

The curation script performs the following operations in sequence:

### 1. Drop Data Before 2010

**Reason**: Early years have incomplete coverage across stations. Starting from 2010 ensures better data consistency.

**Action**: All records with datetime before 2010-01-01 00:00:00 are removed. Data from 2010 onwards is retained.

### 2. Drop Stations

**Reason**: Some stations have insufficient data quality or coverage:
- ARPAL_003 (Brescia - via Tartaglia): Too many missing years
- ARPAL_009 (Lecco): Too many missing years
- ARPAL_015 (Milano - viale Marche): Too many missing years
- ARPAL_024 (Sondrio - via Mazzini): Too many missing years

**Action**: All records for these stations are removed from the dataset.

**Result**: ~136,000 rows removed, 24 stations remaining.

### 3. Interpolate Short Gaps (< 4 hours)

**Reason**: Short gaps (1-3 hours) are likely due to temporary sensor issues or data transmission problems. These can be reliably interpolated using linear interpolation.

**Action**: For each station, a complete hourly time series is created from the first to last measurement. Gaps shorter than 4 hours are interpolated linearly using pandas' `interpolate(method='linear')`, while longer gaps remain as NaN.

**Process**:
- Creates expected hourly time series for each station
- Identifies contiguous missing periods
- Interpolates only gaps ≤ 4 hours that have valid values before and after
- Preserves longer gaps as NaN to avoid unreliable estimates

### 4. Final Pass: Interpolate Very Short Gaps (≤ 2 hours)

**Reason**: After the first interpolation pass, some very short gaps (1-2 hours) may remain, especially at boundaries or in edge cases. A final pass ensures these are also interpolated.

**Action**: A second interpolation pass is performed specifically targeting gaps ≤ 2 hours to ensure maximum data completeness.

### 5. Distance-Weighted Interpolation for ALL Remaining Missing Values

**Reason**: After linear interpolation, some missing values may remain (e.g., longer gaps, boundary cases, missing years). **ALL** remaining missing values are filled using spatial correlation with nearby stations via distance-weighted interpolation. This includes missing years for stations like ARPAL_008 and ARPAL_017, which are now handled uniformly with all other missing values.

**Method**: Softmax of Distance-Based Scores

For each missing measurement at station `S` at time `t`:

1. **Find available stations**: Identify all stations with valid PM10 measurements at time `t`
2. **Calculate distances**: Compute Haversine distances from station `S` to all available stations (in km)
3. **Compute scores**: Calculate `score = e^(-distance/scale)` where `scale = 50 km`
   - This gives higher scores to closer stations
   - Example: 1km → score ≈ 0.98, 10km → score ≈ 0.82, 50km → score ≈ 0.37
4. **Apply softmax**: Normalize scores to get weights that sum to 1
   ```
   weights = softmax(scores) = exp(scores) / sum(exp(scores))
   ```
5. **Weighted average**: Compute interpolated value as weighted average
   ```
   PM10_interpolated = Σ(weight_i × PM10_i)
   ```

**Parameters**:
- Maximum distance: 200 km (stations beyond this are not used)
- Minimum stations: 1 (at least one station must be available)
- Distance decay scale: 50 km

**Why This Approach?**
- **Spatial correlation**: PM10 pollution is spatially correlated - nearby stations have similar values
- **Softmax normalization**: Ensures weights sum to 1, making interpretation easier
- **Exponential decay**: Gives much higher weight to very close stations (e.g., 1km vs 10km)
- **Flexible**: Works with any number of available stations
- **Confidence tracking**: Provides quantitative measure of estimate reliability

**Confidence Metric**

The confidence score combines:
- **Distance confidence**: `exp(-min_distance / 20.0)`
  - 1 km → 0.95 confidence
  - 10 km → 0.61 confidence  
  - 50 km → 0.08 confidence
- **Station count confidence**: `min(1.0, n_stations / 5.0)`
  - More stations = higher confidence (capped at 5+ stations)

**Final confidence**: `confidence = distance_confidence × station_count_confidence`

Range: 0.0 (low confidence) to 1.0 (high confidence)

**Confidence Interpretation**:
- **> 0.9**: Very high confidence (station < 2km away, multiple stations)
- **0.7 - 0.9**: High confidence (station < 10km away)
- **0.5 - 0.7**: Moderate confidence (station 10-30km away)
- **0.3 - 0.5**: Low confidence (station 30-70km away)
- **< 0.3**: Very low confidence (station > 70km away or few stations)

Users can filter interpolated values by confidence threshold based on their analysis needs.

**Alternative Approaches Considered**:
- **Inverse Distance Weighting (IDW)**: Simpler but less emphasis on very close stations
- **Gaussian Kernel**: Similar but uses squared distance
- **K-Nearest Neighbors**: Simpler but ignores distance differences among k neighbors
- **OLS Regression**: More complex, requires training data

The current approach balances simplicity, effectiveness, and interpretability.

## Output Dataset

**Filename**: `merged_pm10_hourly_curated.csv`

**Columns**:
- `datetime`: Hourly timestamp (YYYY-MM-DD HH:MM:SS)
- `station_code`: Station identifier (ARPAL_001, ARPAL_002, etc.)
- `station_name`: Station name
- `pm10`: PM10 concentration (µg/m³)
- `interpolation_method`: Method used to fill the value (`None` for original measurements, `'linear'` for linear interpolation, `'distance_weighted'` for distance-weighted interpolation)
- `interpolation_confidence`: Confidence score (0.0-1.0) for distance-weighted interpolations, `NaN` for other values

**Stations**: 24 stations across Lombardia region (after dropping 4 stations)

**Date Range**: 2010-01-01 00:00:00 onwards

**Completeness**: ~99.98% (typically < 1,000 missing values remain out of ~3 million rows)

## Usage

### 1. Convert Excel to CSV:
```bash
python scripts/convert_arpal_excel_to_csv.py
```

### 2. Analyze the dataset:
```bash
python scripts/analyze_arpal_pm10_data.py
```

### 3. Create curated dataset:
```bash
python scripts/create_curated_arpal_pm10_dataset.py
```

## Notes

- The curated dataset maintains the same CSV format as the input, with additional columns for interpolation tracking
- Linear interpolation is used for short temporal gaps (< 4 hours)
- Distance-weighted interpolation is used for remaining missing values, leveraging spatial correlation
- Long gaps (≥ 4 hours) are preserved as NaN to avoid introducing unreliable estimates
- Station coordinates are converted from UTM (Zone 32N) to WGS84 (lat/lon) for distance calculations
- The scripts are idempotent - running them multiple times produces the same result
- Interpolation confidence scores allow users to filter estimates by quality threshold
