# PM10 Dataset Curation Guide

This document describes the curation process applied to the raw merged PM10 hourly dataset to create a high-quality, analysis-ready dataset.

## Overview

The curation script (`scripts/create_curated_pm10_dataset.py`) transforms the raw merged CSV file (`merged_pm10_hourly.csv`) into a curated dataset (`merged_pm10_hourly_curated.csv`) through a series of data quality improvements.

## Curation Steps

### 1. Drop Station Rio Novo (502726)

**Reason**: This station has too many missing years (48% missing globally) and is located far from other stations (26-50 km away), making it unsuitable for spatial analysis.

**Action**: All data from station 502726 (VE Rio Novo) is removed from the dataset.

### 2. Copy Missing Years for TV S Agnese from TV Lancieri

**Reason**: TV S Agnese (502612) is missing data for years 2010-2015, but TV Lancieri (502608) has complete data for these years. These stations are very close (2.2 km apart), making TV Lancieri a reliable proxy.

**Action**: For each missing year, all hourly measurements from TV Lancieri are copied to TV S Agnese, maintaining the same timestamps but updating the station code and name.

### 3. Drop Data Before 2012

**Reason**: Early years (2010-2011) have incomplete coverage across stations, with many stations not yet operational. Starting from 2012 ensures better data consistency.

**Action**: All records with datetime before 2012-01-01 00:00:00 are removed. Data from 2012 onwards is retained.

### 4. Interpolate Missing Hour in VE Tagliamento (2012)

**Reason**: VE Tagliamento (502720) has exactly 1 missing hour in 2012, which can be easily interpolated from surrounding measurements.

**Action**: The missing hour is filled using linear interpolation between the preceding and following hours.

### 5. Interpolate Short Gaps (< 4 hours)

**Reason**: Short gaps (1-3 hours) are likely due to temporary sensor issues or data transmission problems. These can be reliably interpolated using linear interpolation.

**Action**: For each station, a complete hourly time series is created from the first to last measurement. Gaps shorter than 4 hours are interpolated linearly, while longer gaps remain as NaN.

### 6. Final Pass: Interpolate Very Short Gaps (< 2 hours)

**Reason**: After the first interpolation pass, some very short gaps (1-2 hours) may remain, especially at boundaries or in edge cases. A final pass ensures these are also interpolated.

**Action**: A second interpolation pass is performed specifically targeting gaps shorter than 2 hours to ensure maximum data completeness.

## Output Dataset

**Filename**: `merged_pm10_hourly_curated.csv`

**Columns**:
- `datetime`: Hourly timestamp (YYYY-MM-DD HH:MM:SS)
- `station_code`: Station identifier
- `station_name`: Station name
- `pm10`: PM10 concentration (µg/m³)

**Stations Included**:
- 502604: Conegliano
- 502608: TV Lancieri
- 502609: Mansue
- 502612: TV S Agnese
- 502701: Bissuola
- 502720: VE Tagliamento

**Date Range**: 2012-01-01 00:00:00 onwards

## Usage

Run the curation script:

```bash
python scripts/create_curated_pm10_dataset.py
```

The script will:
1. Load the raw merged CSV
2. Apply all curation steps
3. Save the curated dataset
4. Print summary statistics

## Notes

- The curated dataset maintains the same CSV format as the input
- Interpolated values are computed using linear interpolation
- Long gaps (≥ 4 hours) are preserved as NaN to avoid introducing unreliable estimates
- The script is idempotent - running it multiple times produces the same result

