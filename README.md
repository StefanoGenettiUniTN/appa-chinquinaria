# Trentino Weather & Air Quality ETL Pipeline

## Overview

This ETL pipeline integrates meteorological data from MeteoTrentino weather stations with air quality measurements (PM10) from APPA (Provincial Agency for Environmental Protection) monitoring stations across the Trentino region of Italy.

## Data Sources

### MeteoTrentino
- Multiple weather station data files (CSV format)
- Stored as compressed ZIP files in Google Drive
- Contains various meteorological variables (temperature, humidity, precipitation, etc.)
- Temporal resolution: Hourly measurements

### APPA Air Quality
- PM10 particulate matter measurements
- 8 monitoring stations across Trentino
- Temporal resolution: Hourly measurements (aggregated to daily)
- Stations include: Trento (Parco S. Chiara, Via Bolzano), Piana Rotaliana, Rovereto, Borgo Valsugana, Riva del Garda, A22 (Avio), and Monte Gaza

## Pipeline Components

### 1. Data Extraction

**MeteoTrentino Download** (`download_meteo_trentino`)
- Downloads ZIP archives from Google Drive folder
- Extracts CSV files to local directory
- Uses `gdown` library for Google Drive access

**APPA Download** (`appa_download`)
- Downloads two CSV files: measurement data and station metadata
- Retrieves files directly from Google Drive

### 2. Data Transformation

**Weather Data Processing** (`preprocessing_meteo_data`)
- Loads multiple station CSV files
- Standardizes column names and removes unnamed columns
- Parses datetime information (format: `HH:MM:SS DD/MM/YYYY`)
- Adds station identifiers
- Consolidates all stations into single dataframe

**Air Quality Processing** (`preprocessing_appa_data`)
- Filters measurements to PM10 only
- Merges with station metadata (coordinates, station names)
- Aggregates hourly data to daily averages
- Matches each APPA station to nearest MeteoTrentino station using Haversine distance formula
- Cleans and standardizes measurement units (μg/m³)

### 3. Data Integration

**Merge Operation** (`merge_datasets`)
- Joins weather and air quality data on:
  - Station ID (nearest weather station)
  - Date
- Performs inner join to retain only matching records
- Removes duplicate columns
- Outputs unified dataset

## Output

**File**: `output/historical_weather_airPM_trentino.csv`

**Structure**:
- Date: Daily timestamp
- Station: APPA air quality station name
- StazioneMeteo: Corresponding MeteoTrentino station ID
- Valore: PM10 concentration (μg/m³)
- Weather variables: Temperature, humidity, precipitation, wind, etc. (varies by station)
- Coordinates: Latitude and longitude of APPA stations
- Administrative data: Country, Municipality information

## Visualization Component

The pipeline includes an interactive visualization tool (`WeatherPlotter`) for exploring weather variables:
- Station selection dropdown
- Variable selection dropdown
- Time series plots with statistical summaries (mean, min, max)
- Built using matplotlib and ipywidgets for Jupyter/Colab environments

## Technical Requirements

### Dependencies
- pandas: Data manipulation
- numpy: Numerical operations
- gdown: Google Drive downloads
- lxml: XML parsing
- matplotlib: Visualization
- ipywidgets: Interactive widgets
- requests: HTTP requests

### Directory Structure
```
input/
  ├── appa_data/
  │   ├── appa_data.csv
  │   ├── appa_metadata.csv
  │   └── statJson.csv
  └── meteo-trentino-data/
      └── [station CSV files]
output/
  └── historical_weather_airPM_trentino.csv
```

## Usage

Run the main script to execute the complete pipeline:

```python
python ETL_pipeline_appa.py
```

The pipeline will:
1. Create necessary directories
2. Download data if not already present (skips if files exist)
3. Process and clean both datasets
4. Merge datasets based on spatial-temporal matching
5. Save final output to CSV

## Key Features

- **Automated Downloads**: Checks for existing data to avoid redundant downloads
- **Spatial Matching**: Uses Haversine formula to pair air quality stations with nearest weather stations
- **Robust Date Parsing**: Handles multiple date formats with error handling
- **Data Validation**: Converts values to numeric types with coercion for invalid entries
- **Modular Design**: Separate functions for each processing step
- **Type Annotations**: Comprehensive type hints for better code documentation

## Notes

- The pipeline is designed for Jupyter/Colab environments but can be adapted for standard Python execution
- Encoding for CSV files: Latin-1 (ISO-8859-1)
- Date format in output: YYYY-MM-DD
- Missing or invalid numeric values are handled using pandas `coerce` parameter Trentino Weather & Air Quality ETL Pipeline

## Overview

This ETL pipeline integrates meteorological data from MeteoTrentino weather stations with air quality measurements (PM10) from APPA (Provincial Agency for Environmental Protection) monitoring stations across the Trentino region of Italy.

## Data Sources

### MeteoTrentino
- Multiple weather station data files (CSV format)
- Stored as compressed ZIP files in Google Drive
- Contains various meteorological variables (temperature, humidity, precipitation, etc.)
- Temporal resolution: Hourly measurements

### APPA Air Quality
- PM10 particulate matter measurements
- 8 monitoring stations across Trentino
- Temporal resolution: Hourly measurements (aggregated to daily)
- Stations include: Trento (Parco S. Chiara, Via Bolzano), Piana Rotaliana, Rovereto, Borgo Valsugana, Riva del Garda, A22 (Avio), and Monte Gaza

## Pipeline Components

### 1. Data Extraction

**MeteoTrentino Download** (`download_meteo_trentino`)
- Downloads ZIP archives from Google Drive folder
- Extracts CSV files to local directory
- Uses `gdown` library for Google Drive access

**APPA Download** (`appa_download`)
- Downloads two CSV files: measurement data and station metadata
- Retrieves files directly from Google Drive

### 2. Data Transformation

**Weather Data Processing** (`preprocessing_meteo_data`)
- Loads multiple station CSV files
- Standardizes column names and removes unnamed columns
- Parses datetime information (format: `HH:MM:SS DD/MM/YYYY`)
- Adds station identifiers
- Consolidates all stations into single dataframe

**Air Quality Processing** (`preprocessing_appa_data`)
- Filters measurements to PM10 only
- Merges with station metadata (coordinates, station names)
- Aggregates hourly data to daily averages
- Matches each APPA station to nearest MeteoTrentino station using Haversine distance formula
- Cleans and standardizes measurement units (μg/m³)

### 3. Data Integration

**Merge Operation** (`merge_datasets`)
- Joins weather and air quality data on:
  - Station ID (nearest weather station)
  - Date
- Performs inner join to retain only matching records
- Removes duplicate columns
- Outputs unified dataset

## Output

**File**: `output/historical_weather_airPM_trentino.csv`

**Structure**:
- Date: Daily timestamp
- Station: APPA air quality station name
- StazioneMeteo: Corresponding MeteoTrentino station ID
- Valore: PM10 concentration (μg/m³)
- Weather variables: Temperature, humidity, precipitation, wind, etc. (varies by station)
- Coordinates: Latitude and longitude of APPA stations
- Administrative data: Country, Municipality information

## Visualization Component

The pipeline includes an interactive visualization tool (`WeatherPlotter`) for exploring weather variables:
- Station selection dropdown
- Variable selection dropdown
- Time series plots with statistical summaries (mean, min, max)
- Built using matplotlib and ipywidgets for Jupyter/Colab environments

## Technical Requirements

### Dependencies
- pandas: Data manipulation
- numpy: Numerical operations
- gdown: Google Drive downloads
- lxml: XML parsing
- matplotlib: Visualization
- ipywidgets: Interactive widgets
- requests: HTTP requests

### Directory Structure
```
input/
  ├── appa_data/
  │   ├── appa_data.csv
  │   ├── appa_metadata.csv
  │   └── statJson.csv
  └── meteo-trentino-data/
      └── [station CSV files]
output/
  └── historical_weather_airPM_trentino.csv
```

## Usage

Run the main script to execute the complete pipeline:

```python
python ETL_pipeline_appa.py
```

The pipeline will:
1. Create necessary directories
2. Download data if not already present (skips if files exist)
3. Process and clean both datasets
4. Merge datasets based on spatial-temporal matching
5. Save final output to CSV

## Key Features

- **Automated Downloads**: Checks for existing data to avoid redundant downloads
- **Spatial Matching**: Uses Haversine formula to pair air quality stations with nearest weather stations
- **Robust Date Parsing**: Handles multiple date formats with error handling
- **Data Validation**: Converts values to numeric types with coercion for invalid entries
- **Modular Design**: Separate functions for each processing step
- **Type Annotations**: Comprehensive type hints for better code documentation

## Notes

- The pipeline is designed for Jupyter/Colab environments but can be adapted for standard Python execution
- Encoding for CSV files: Latin-1 (ISO-8859-1)
- Date format in output: YYYY-MM-DD
- Missing or invalid numeric values are handled using pandas `coerce` parameter
