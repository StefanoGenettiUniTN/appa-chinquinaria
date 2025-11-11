## Merge ARPAV PM10 hourly XLSX into a single CSV

This guide explains how to aggregate all ARPAV PM10 hourly Excel files into one CSV keyed by datetime and station code.

### Prerequisites
- Ensure dependencies are installed:

```bash
pip install -r requirements.txt
```

- Download the ARPAV PM10 hourly XLSX archive (see `docs/arpav-download-guide.md`) and place the files under the expected path relative to the repository root:

```
data/arpav/PM10/orari/
  H_2010_502604_Conegliano.xlsx
  H_2010_502701_Bissuola.xlsx
  ...
```

### Run the merge
From the repository root:

```bash
python scripts/merge_arpav_pm10_hourly.py --verbose
```

Or specify paths explicitly:

```bash
python scripts/merge_arpav_pm10_hourly.py \
  --input-dir data/arpav/PM10/orari \
  --output data/arpav/PM10/merged_pm10_hourly.csv \
  --verbose
```

### Outputs
- Merged dataset:
  - Path: `data/arpav/PM10/merged_pm10_hourly.csv`
  - Columns: `datetime, station_code, station_name, pm10`
  - Notes: rows are sorted by `station_code, datetime`, deduplicated on that pair.

- Data quality report (per-station coverage and missing values):
  - Path: `data/arpav/PM10/merged_pm10_hourly_missing_report.csv`
  - Includes coverage over each station’s observed span and over a fixed global period
    (2010-01-01 00:00 to 2025-12-31 23:00).

### Script reference
- Script: `scripts/merge_arpav_pm10_hourly.py`
  - Parses common datetime schemas (`DATA ORA` or `DATA` + `ORA`).
  - Detects PM10 column heuristically if not explicitly labeled.
  - Logs aggregate invalid datetimes, empty PM10 values, and per-station missing-hour percentages.


