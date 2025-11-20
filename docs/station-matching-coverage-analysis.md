# Station Matching and Coverage Analysis

## Overview

This document describes the station matching and weather data coverage analysis system for pairing APPA air quality stations with Meteo Trentino weather stations.

## Script: `analyze_weather_station_coverage.py`

### Purpose

The script performs the following tasks:

1. **Loads APPA air quality stations** with their coordinates from the stations CSV file
2. **Loads Meteo Trentino weather stations** from the XML metadata file
3. **Calculates distance matrix** between all APPA and Meteo Trentino stations using the Haversine formula
4. **Finds closest weather stations** for each APPA station (configurable top N)
5. **Analyzes data coverage** for all weather variables:
   - Temperature
   - Rain (Precipitation)
   - Wind Speed
   - Wind Direction
   - Pressure
   - Radiation
   - Humidity
6. **Generates comprehensive reports** with missing and invalid value statistics

### Usage

#### Basic Usage

```bash
python3 scripts/analyze_weather_station_coverage.py
```

This uses default paths and analyzes the period from 2014-01-01 to 2025-12-31.

#### Advanced Usage

```bash
python3 scripts/analyze_weather_station_coverage.py \
    --appa-stations data/appa-data/appa_monitoring_stations.csv \
    --meteo-stations-xml data/meteo-trentino/stations.xml \
    --temp-rain-dir data/meteo-trentino/meteo-trentino-storico-completo-temperatura-pioggia \
    --wind-pressure-dir "data/meteo-trentino/meteo-trentino-storico-completo-vento-pressione-radiazione-umidità" \
    --start-date 2014-01-01 \
    --end-date 2025-12-31 \
    --top-n 5 \
    --output-dir output/station_matching
```

### Parameters

- `--appa-stations`: Path to APPA stations CSV file (default: `data/appa-data/appa_monitoring_stations.csv`)
- `--meteo-stations-xml`: Path to Meteo Trentino stations XML file (default: `data/meteo-trentino/stations.xml`)
- `--temp-rain-dir`: Directory containing temperature and rain CSV files
- `--wind-pressure-dir`: Directory containing wind, pressure, radiation, humidity CSV files
- `--start-date`: Start date for coverage analysis in YYYY-MM-DD format (default: `2014-01-01`)
- `--end-date`: End date for coverage analysis in YYYY-MM-DD format (default: `2025-12-31`)
- `--top-n`: Number of closest weather stations to analyze for each APPA station (default: `5`)
- `--output-dir`: Output directory for results (default: `output/station_matching`)

### Output Files

The script generates the following output files in the specified output directory:

1. **`distance_matrix.csv`**: Distance matrix with distances (in km) between all APPA stations (rows) and Meteo Trentino stations (columns)

2. **`closest_stations.csv`**: List of closest weather stations for each APPA station with:
   - `appa_station`: APPA station code
   - `rank`: Rank of closeness (1 = closest)
   - `meteo_station`: Meteo Trentino station code
   - `distance_km`: Distance in kilometers

3. **`coverage_analysis.csv`**: Detailed coverage analysis for each variable and station combination:
   - `meteo_station`: Weather station code
   - `variable`: Variable name (temperature, rain, wind_speed, etc.)
   - `available`: Whether the variable is available for this station
   - `total_records`: Total number of records found
   - `valid_records`: Number of records with quality code 1 (good data)
   - `invalid_records`: Number of records with quality codes 140, 145 (uncertain/unvalidated)
   - `missing_records`: Number of missing records (quality codes 151, 255, or NaN)
   - `coverage_percent`: Percentage of expected time period covered
   - `valid_percent`: Percentage of expected time period with valid data
   - `appa_station`: Associated APPA station
   - `rank`: Rank of this weather station for the APPA station
   - `distance_km`: Distance from APPA station

4. **`coverage_summary.csv`**: Aggregated summary by APPA station and weather station:
   - `appa_station`: APPA station code
   - `meteo_station`: Weather station code
   - `rank`: Rank of closeness
   - `distance_km`: Distance in kilometers
   - `available_variables`: Count of available variables
   - `avg_valid_percent`: Average valid data percentage across all variables
   - `avg_coverage_percent`: Average coverage percentage across all variables
   - `total_valid_records`: Total valid records across all variables
   - `total_missing_records`: Total missing records across all variables
   - `total_invalid_records`: Total invalid records across all variables

### Data Quality Codes

The script interprets Meteo Trentino quality codes as follows:

- **1**: Good data (valid)
- **140, 145**: Uncertain/unvalidated data (invalid for our purposes)
- **151**: Missing data
- **255**: No data

### Distance Calculation

Distances are calculated using the Haversine formula, which computes the great-circle distance between two points on Earth given their latitude and longitude coordinates. Distances are reported in kilometers.

### Coverage Analysis

For each weather station and variable combination, the script:

1. Searches for the appropriate CSV file(s) containing the variable data
2. Loads and parses the CSV file (handling the Meteo Trentino format with header rows)
3. Filters data to the specified date range (default: 2014-01-01 to 2025-12-31)
4. Counts records by quality code:
   - Valid records (quality = 1)
   - Invalid records (quality = 140, 145)
   - Missing records (quality = 151, 255, or NaN)
5. Calculates coverage percentages based on expected hourly time series

### Expected Time Series

The script expects hourly data for the entire period from `start-date` to `end-date`. For example, for the period 2014-01-01 to 2025-12-31:
- Expected hours = (365 days × 12 years + leap days) × 24 hours ≈ 105,192 hours

Coverage percentages are calculated as:
- `coverage_percent` = (total_records / expected_hours) × 100
- `valid_percent` = (valid_records / expected_hours) × 100

### File Structure Requirements

The script expects CSV files to be organized as follows:

**Temperature and Rain Directory:**
- CSV files named `{STATION_CODE}.csv` containing temperature or rain data
- Files are identified by checking the header for variable keywords

**Wind, Pressure, Radiation, Humidity Directory:**
- CSV files named `{STATION_CODE}.csv` containing wind, pressure, radiation, or humidity data
- Files are identified by checking the header for variable keywords

The script automatically detects which variable each CSV file contains by examining the header rows.

### Example Workflow

1. **Run the analysis:**
   ```bash
   python3 scripts/analyze_weather_station_coverage.py
   ```

2. **Review the distance matrix** to see all pairwise distances:
   ```bash
   head output/station_matching/distance_matrix.csv
   ```

3. **Check closest stations** for each APPA station:
   ```bash
   head output/station_matching/closest_stations.csv
   ```

4. **Analyze coverage** for candidate stations:
   ```bash
   head output/station_matching/coverage_analysis.csv
   ```

5. **Review summary** to identify best matches:
   ```bash
   head output/station_matching/coverage_summary.csv
   ```

### Interpreting Results

When selecting weather stations for APPA stations, consider:

1. **Distance**: Closer stations are generally better, but data quality is more important
2. **Coverage**: Look for stations with high `valid_percent` (ideally >95%)
3. **Variable availability**: Ensure all required variables are available
4. **Missing data**: Prefer stations with minimal missing records

The `coverage_summary.csv` file provides a quick overview sorted by APPA station and rank, making it easy to identify the best weather station matches.

### Notes

- The script handles missing files gracefully - if a variable is not available for a station, it will be marked as unavailable
- CSV files are expected to be already extracted (not in ZIP format)
- The script uses Latin-1 encoding to handle special characters in station names
- Large datasets may take several minutes to process

