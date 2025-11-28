# Web Data Scraping

This directory contains scripts and modules for automated downloading, aggregation, and initial processing of air quality and meteorological datasets from various online sources. The focus is on robust, scalable data acquisition for the APPA Chinquinaria project, supporting downstream analysis and modeling workflows.

## Structure

- **download_from_csv.py**  
  Utility script for batch downloading data using CSV input lists.

- **bulk-download-blh/**  
  Scripts for downloading and processing ERA5 Boundary Layer Height (BLH) data:
  - `download_blh.py`: Automated download of BLH data from Copernicus/ERA5.
  - `build_blh_dataset.py`: Aggregation and structuring of BLH datasets for analysis.

- **bulk-download-eea/**  
  Scripts for downloading European Environment Agency (EEA) air quality data:
  - `bulk_download_eea.py`: Automated retrieval of EEA datasets for selected pollutants and countries.

- **bulk-download-era5-land-all-pm10-stations/**  
  Scripts for downloading and aggregating ERA5-Land timeseries for all PM10 monitoring stations:
  - `bulk_download_era5_land_timeseries.py`: Batch download of ERA5-Land data.
  - `aggregate_era5_land_timeseries.py`: Aggregation and formatting of downloaded timeseries.

- **bulk-download-meteo-arpav/**  
  Scripts for downloading meteorological data from ARPAV (Veneto region):
  - `bulk_download_arpav.py`: Automated download of ARPAV weather data.
  - `test_arpav_functions.py`: Utility and test functions for ARPAV data access.

- **bulk-download-trentino-data/**  
  Scripts for downloading air quality and meteorological data for Trentino:
  - `bulk_download_appa.py`: Bulk download of APPA Trento air quality data.
  - `bulk_download_meteo_trentino.py`: Bulk download of Meteo Trentino weather data.
  - `test_meteo_connection.py`: Connection tests and utilities for Meteo Trentino data access.

## Main Operations

- **Automated Data Acquisition:**  
  - Batch and bulk download of datasets from official APIs, portals, and cloud storage.
  - Support for multiple data sources: ERA5, ERA5-Land, EEA, ARPAV, APPA, Meteo Trentino.

- **Data Aggregation and Structuring:**  
  - Aggregation of downloaded timeseries into analysis-ready formats.
  - Structuring and formatting of raw data for downstream processing.

- **Testing and Validation:**  
  - Utility scripts for connection testing and data integrity checks.

## Usage

Scripts in this directory are designed to be run independently for specific data acquisition tasks. Refer to the docstrings and comments within each script for detailed usage instructions, required parameters, and input formats.

Typical workflow:
1. Use bulk download scripts to acquire raw datasets from online sources.
2. Aggregate and format downloaded data as needed for analysis.
3. Validate data integrity using provided test scripts.

## Notes

- All scripts are written in Python and require standard scientific libraries (pandas, numpy, requests, etc.).
- Some scripts may require API credentials or configuration files (see documentation and comments).
- Output datasets are intended for use in subsequent pre-processing and modeling modules.

For further details on integration with the overall data pipeline, refer to the main project documentation.
