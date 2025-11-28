# Data Pre-Processing

This directory contains scripts and notebooks for the preparation, cleaning, and merging of air quality and meteorological datasets used in the APPA Chinquinaria project. The focus is on robust data curation, gap filling, feature engineering, and the creation of analysis-ready datasets for downstream modeling and interpretability.

## Structure

- **bulk-data-pre-processing/**  
  Scripts for initial data analysis, gap detection, interpolation, and creation of curated datasets. Includes:
  - Statistical analysis of PM10 measurements and missing data.
  - Interpolation and gap filling using nearest stations.
  - Creation of complete time series for all monitoring stations.
  - Inspections and correlation analysis notebooks and scripts.

- **daily_dataset/**  
  ETL pipelines and scripts for merging air quality data with meteorological variables on a daily basis. Includes:
  - Merging APPA and MeteoTrentino datasets.
  - Proximity-based dataset filtering and merging.
  - Post-processing and inspection notebooks.

## Main Operations

- **Data Cleaning and Curation:**  
  - Removal of invalid or outlier values.
  - Interpolation of short gaps and imputation from nearest stations.
  - Construction of complete time series for each station.

- **Statistical Analysis:**  
  - Calculation of summary statistics (min, mean, max) for pollutant measurements.
  - Analysis of missing data patterns and contiguous missing periods.
  - Computation of distance matrices between stations.

- **Dataset Merging:**  
  - Integration of air quality and meteorological data for the Trentino region.
  - Automated download and extraction of source data.
  - Creation of merged datasets for modeling and analysis.

- **Correlation Analysis:**  
  - Monthly correlation analysis between pollutants and meteorological variables.
  - Generation of plots and summary statistics for each pollutant.

## Usage

Most scripts are designed to be run independently for specific data preparation tasks. Refer to the docstrings and comments within each script for detailed usage instructions and required input files.

Typical workflow:
1. Run analysis scripts to inspect and clean raw data.
2. Use curation scripts to fill gaps and produce complete time series.
3. Merge datasets as needed for modeling and feature analysis.
4. Perform correlation and inspection analyses to validate data integrity.

## Notes

- All scripts are written in Python and require standard scientific libraries (pandas, numpy, matplotlib, seaborn).
- Some scripts may require additional configuration for input/output paths.
- Notebooks are provided for interactive inspection and visualization of intermediate results.

For further details on the overall data pipeline and integration with modeling and interpretability modules, refer to the main project documentation.
