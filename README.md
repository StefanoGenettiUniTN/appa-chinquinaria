# Trentino Weather & Air Quality ETL Pipeline

## Overview

ETL pipeline integrating meteorological data from MeteoTrentino and air quality (PM10) measurements from APPA monitoring stations across Trentino, Italy. The pipeline also incorporates European Environment Agency (EEA) data matched by geographic proximity.

## Project Structure

```
src/
  ├── merge_appa_meteo_trentino.py      # APPA + MeteoTrentino integration
  ├── filter_eea_by_proximity.py        # EEA proximity filtering
  ├── merge_datasets_by_proximity.py    # APPA-EEA proximity merge
  └── PostProcessing.py                 # Data quality & cleaning utilities

run_pipeline.sh                         # Execute all 3 scripts with logging

input/
  ├── appa_data/                        # APPA air quality measurements
  │   ├── appa_data.csv
  │   ├── appa_metadata.csv
  │   └── statJson.csv
  ├── eea_data/                         # European Environment Agency data
  │   └── eea.csv
  └── meteo-trentino-data/              # Weather station files (T0010, T0118, etc.)

output/
  ├── historical_weather_airPM_trentino.csv      # APPA + Meteo combined
  ├── eea_filtered_by_proximity.csv              # EEA stations near Trentino
  ├── trentino_eea_proximity_mapping.csv         # Station proximity mapping
  ├── merged_appa_eea_by_proximity.csv           # APPA-EEA merged by proximity
  ├── merged_appa_eea_cleaned.csv                # Post-processed clean dataset
  ├── appa_meteo_merge_pipeline_output.txt       # Pipeline logs
  ├── eea_proximity_filter_pipeline_output.txt
  └── merge_appa_eea_proximity_output.txt

notebooks/
  ├── inspect_merged_dataset.ipynb      # Data quality analysis & visualization
  └── mapping_visualization.ipynb       # Station mapping

docs/
  └── Explaining_meteo_trentino_features.md
```

## Key Features

- **3-stage ETL pipeline**: APPA-Meteo merge → EEA filtering → Proximity-based integration
- **Proximity-based matching**: Pairs APPA stations with nearest EEA stations using Haversine distance
- **Data quality controls**: 
  - NaN value analysis and filtering
  - Negative value detection and removal
  - Duplicate row detection
- **Automated logging**: All outputs saved to separate text files
- **Interactive analysis**: Jupyter notebooks with Folium maps and quality reports

## Running the Pipeline

```bash
# Execute complete pipeline with logging
./run_pipeline.sh

# Or run individually
python src/merge_appa_meteo_trentino.py
python src/filter_eea_by_proximity.py
python src/merge_datasets_by_proximity.py
```

## Pipeline Stages

### 1. APPA-MeteoTrentino Merge
**Script:** `src/merge_appa_meteo_trentino.py`

- Integrates APPA air quality measurements with MeteoTrentino weather data
- Temporal alignment of station readings
- Output: `historical_weather_airPM_trentino.csv`

### 2. EEA Proximity Filter
**Script:** `src/filter_eea_by_proximity.py`

- Filters European air quality stations by proximity to Trentino region
- Calculates Haversine distances between stations
- Output: `eea_filtered_by_proximity.csv`, `trentino_eea_proximity_mapping.csv`

### 3. Proximity-Based Merge
**Script:** `src/merge_datasets_by_proximity.py`

- Merges APPA-Meteo data with nearest EEA stations
- Geographic and temporal data integration
- Output: `merged_appa_eea_by_proximity.csv`

### 4. Post-Processing (Optional)
**Script:** `src/PostProcessing.py`

- Data quality analysis (NaN, negatives, duplicates)
- Column/station filtering based on quality thresholds
- Output: `merged_appa_eea_cleaned.csv`

## Data Quality Analysis

Use the inspection notebook for comprehensive quality checks:

```bash
jupyter notebook notebooks/inspect_merged_dataset.ipynb
```

Features:
- Missing value analysis
- Negative value detection
- Station location visualization with Folium
- Data quality scoring

## Output Files

| File | Description | Purpose |
|------|-------------|---------|
| `historical_weather_airPM_trentino.csv` | APPA + MeteoTrentino merged | Weather-AQ integration |
| `eea_filtered_by_proximity.csv` | EEA stations near Trentino | Proximity-filtered EEA data |
| `trentino_eea_proximity_mapping.csv` | Station proximity mappings | Geographic relationships |
| `merged_appa_eea_by_proximity.csv` | Complete APPA-EEA dataset | Full merged data |
| `merged_appa_eea_cleaned.csv` | Quality-controlled dataset | Production-ready data |

## Dependencies

```
pandas
numpy
matplotlib
seaborn
folium
```

Install with:
```bash
pip install -r requirements.txt
```

## Documentation

- `docs/Explaining_meteo_trentino_features.md` - Feature guide & data quality notes
- `notebooks/inspect_merged_dataset.ipynb` - Interactive data quality analysis
- `notebooks/mapping_visualization.ipynb` - Station location mapping
