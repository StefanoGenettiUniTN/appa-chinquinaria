# Trentino Weather & Air Quality ETL Pipeline

## Overview

ETL pipeline integrating meteorological data from MeteoTrentino and air quality (PM10) measurements from APPA monitoring stations across Trentino, Italy.

## Project Structure

```
ETL_appa-meteoTrentino_pipeline.py    # Process and aggregate APPA + Meteo data
ETL_eea_pipeline.py                   # Process EEA European air quality data
merge_datasets.py                     # Merge APPA-Meteo with EEA data
run_pipeline.sh                       # Execute all 3 scripts with logging

input/
  ├── appa_data/                      # APPA air quality measurements
  ├── eea_data/                       # European Environment Agency data
  └── meteo-trentino-data/            # Weather station files

output/
  ├── historical_weather_airPM_trentino.csv      # APPA + Meteo combined
  ├── eea_data_aggregated.csv                    # EEA processed data
  ├── merged_appa_eea.csv                        # Final merged dataset
  ├── ETL_appa-meteoTrentino_pipeline_output.txt # Script logs
  ├── ETL_eea_pipeline_output.txt
  └── merge_datasets_output.txt
```

## Key Features

- **3-stage ETL pipeline**: APPA processing → EEA processing → Dataset merge
- **Spatial-temporal matching**: Pairs air quality with weather stations
- **Validated data**: Annale Idrologico (1923-2025) + Recent measurements (1990-2025)
- **Automated logging**: All outputs saved to separate text files
- **Python scripts** with pandas, numpy for data manipulation

## Running the Pipeline

```bash
# Execute all scripts with logging
./run_pipeline.sh

# Or run individually
python ETL_appa-meteoTrentino_pipeline.py
python ETL_eea_pipeline.py
python merge_datasets.py
```

## Output Files

| File | Description | Records |
|------|-------------|---------|
| `historical_weather_airPM_trentino.csv` | APPA + Meteo Trentino merged | ~2.8M |
| `eea_data_aggregated.csv` | European air quality aggregated | ~1.6M |
| `merged_appa_eea.csv` | Complete dataset APPA-Meteo-EEA | Final merged |

## Dependencies

pandas, numpy, matplotlib, seaborn

## Documentation

- `Explaining_meteo_trentino_features.md` - Feature guide & data quality notes
