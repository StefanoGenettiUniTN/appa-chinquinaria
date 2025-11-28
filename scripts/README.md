# Web Data Scraping

# Data Acquisition, Curation, Analysis, and Visualization Scripts

This directory contains a comprehensive suite of Python scripts for the APPA Chinquinaria project, supporting the full data pipeline from raw acquisition to advanced analysis and visualization. The scripts are organized to enable robust, scalable workflows for air quality and meteorological data, including:

- Automated bulk and batch data downloads from multiple sources (APPA, ARPAV, Alto Adige, Meteo Trentino, EEA, ERA5).
- Data curation, gap filling, and feature engineering for high-quality datasets.
- Merging and integration of heterogeneous data sources.
- Statistical analysis, correlation studies, and coverage assessment.
- Generation of curated datasets for modeling and reporting.
- Visualization and plotting of time series and curated data.

## Structure

- **download_from_csv.py**  

- **bulk-download-blh/**  
  Scripts for downloading and processing ERA5 Boundary Layer Height (BLH) data:
  - `build_blh_dataset.py`: Aggregation and structuring of BLH datasets for analysis.

- **bulk-download-eea/**  
  - `bulk_download_eea.py`: Automated retrieval of EEA datasets for selected pollutants and countries.

- **bulk-download-era5-land-all-pm10-stations/**  
  - `bulk_download_era5_land_timeseries.py`: Batch download of ERA5-Land data.
  - `aggregate_era5_land_timeseries.py`: Aggregation and formatting of downloaded timeseries.

  Scripts for downloading meteorological data from ARPAV (Veneto region):
  - `bulk_download_arpav.py`: Automated download of ARPAV weather data.
  - `test_arpav_functions.py`: Utility and test functions for ARPAV data access.

- **bulk-download-trentino-data/**  
  - `bulk_download_appa.py`: Bulk download of APPA Trento air quality data.
  - `bulk_download_meteo_trentino.py`: Bulk download of Meteo Trentino weather data.
  - `test_meteo_connection.py`: Connection tests and utilities for Meteo Trentino data access.

- **Data Pre-processing**
  - `PostProcessing.py`: Advanced post-processing utilities for cleaning and transforming merged datasets.
  - `filter_eea_by_proximity.py`: Filters EEA air quality data based on spatial proximity to target locations.
  - `merge_appa_meteo_trentino.py`: Merges APPA air quality data with Meteo Trentino meteorological data for integrated analysis.
  - `merge_datasets_by_proximity.py`: Merges datasets using spatial proximity logic, useful for combining heterogeneous sources.

and more...

## Main Operations

- **Automated Data Acquisition:**  
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
## Notes

- All scripts are written in Python and require standard scientific libraries (pandas, numpy, requests, etc.).
- Some scripts may require API credentials or configuration files (see documentation and comments).
- Output datasets are intended for use in subsequent pre-processing and modeling modules.


For further details on integration with the overall data pipeline, refer to the main project documentation.

---

**Extensive guides and step-by-step instructions for data acquisition, curation, and analysis are provided in the `./docs` directory.**
