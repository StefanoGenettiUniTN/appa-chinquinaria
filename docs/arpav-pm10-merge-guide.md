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


# ARPAV PM10 Hourly Merge Report

## 📄 Processing Summary

| Year | Files Processed |
|------|------------------|
| 2010 | Conegliano, TV_Lancieri, Mansue, Bissuola |
| 2011 | Conegliano, TV_Lancieri, Mansue, Bissuola |
| 2012 | Conegliano, TV_Lancieri, Mansue, Bissuola, VE_Tagliamento |
| 2013 | Conegliano, TV_Lancieri, Mansue, Bissuola, VE_Tagliamento |
| 2014 | Conegliano, TV_Lancieri, Mansue, Bissuola, VE_Tagliamento |
| 2015 | Conegliano, TV_Lancieri, Mansue, TV_S_Agnese, Bissuola, VE_Tagliamento |
| 2016 | Conegliano, TV_Lancieri, Mansue, TV_S_Agnese, Bissuola, VE_Tagliamento |
| 2017 | Conegliano, TV_Lancieri, Mansue, TV_S_Agnese, Bissuola, VE_Tagliamento, VE_Rio_Novo |
| 2018 | Conegliano, TV_Lancieri, Mansue, TV_S_Agnese, Bissuola, VE_Tagliamento, VE_Rio_Novo |
| 2019 | Conegliano, TV_Lancieri, Mansue, TV_S_Agnese, Bissuola, VE_Tagliamento, VE_Rio_Novo |
| 2020 | Conegliano, TV_Lancieri, Mansue, TV_S_Agnese, Bissuola, VE_Tagliamento, VE_Rio_Novo |
| 2021 | Conegliano, TV_Lancieri, Mansue, TV_S_Agnese, Bissuola, VE_Tagliamento, VE_Rio_Novo |
| 2022 | Conegliano, TV_Lancieri, Mansue, TV_S_Agnese, Bissuola, VE_Tagliamento, VE_Rio_Novo |
| 2023 | Conegliano, TV_Lancieri, Mansue, TV_S_Agnese, Bissuola, VE_Tagliamento, VE_Rio_Novo |
| 2024 | Conegliano, TV_Lancieri, Mansue, TV_S_Agnese, Bissuola, VE_Tagliamento, VE_Rio_Novo |
| 2025 | Conegliano, TV_Lancieri, Mansue, TV_S_Agnese, Bissuola, VE_Tagliamento, VE_Rio_Novo |

---

## 📊 Aggregation Results

**Output File:**  
`/Users/federicorubbi/Documents/unitn/public-ai-challenge/appa-chinquinaria/data/arpav/PM10/merged_pm10_hourly.csv`

**Rows Written:** 631,534  

**Aggregate Counts:**  
- Input rows: 848,832  
- Parsed rows: 631,534  
- Invalid datetimes: 35,368  
- Empty PM10: 182,757  

---

## 🧠 Per-Station Completeness Summary

| Station ID | Name | Expected (Observed-Span) | Present | Missing | % Missing | Global Expected | Global Missing | Global % Missing |
|-------------|------|--------------------------|----------|----------|-------------|------------------|----------------|------------------|
| 502604 | Conegliano | 138,632 | 130,008 | 8,624 | 6.22% | 140,256 | 10,248 | 7.31% |
| 502608 | TV Lancieri | 138,527 | 131,365 | 7,162 | 5.17% | 140,256 | 8,891 | 6.34% |
| 502609 | Mansue | 138,445 | 130,035 | 8,410 | 6.07% | 140,256 | 10,221 | 7.29% |
| 502612 | TV S Agnese | 92,123 | 87,011 | 5,112 | 5.55% | 140,256 | 53,245 | 37.96% |
| 502701 | Bissuola | 138,789 | 63,082 | 75,707 | 54.55% | 140,256 | 77,174 | 55.02% |
| 502720 | VE Tagliamento | 121,270 | 56,835 | 64,435 | 53.13% | 140,256 | 83,421 | 59.48% |
| 502726 | VE Rio Novo | 72,849 | 33,198 | 39,651 | 54.43% | 140,256 | 107,058 | 76.33% |

