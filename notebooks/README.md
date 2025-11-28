# Notebooks: Interactive Data Exploration and Analysis

This folder contains Jupyter notebooks designed for interactive exploration, visualization, and analysis of air quality and meteorological datasets within the APPA Chinquinaria project. These notebooks complement the automated pipeline and scripts by enabling flexible, hands-on investigation of data and model results.

## Main Purposes

- **Data Inspection:**
  - Explore merged and curated datasets, inspect data quality, and validate preprocessing steps.

- **Visualization:**
  - Generate plots, maps, and time series to understand trends, distributions, and spatial patterns in the data.

- **Model Prototyping:**
  - Experiment with machine learning models (e.g., LightGBM, LSTM) on sample datasets, tune parameters, and interpret results.

- **Station and Coverage Analysis:**
  - Analyze historic station data, coverage, and variable distributions across regions and time periods.

- **Explainability and Reporting:**
  - Visualize feature importance, model explainability outputs, and generate summary reports for stakeholders.

## Typical Workflow

1. Load and inspect datasets produced by the pipeline or scripts.
2. Visualize data distributions, trends, and spatial relationships.
3. Prototype and evaluate models on selected data subsets.
4. Document findings and generate figures for reports or presentations.

## Notebook List (examples)

- `data_meteo_trentino.ipynb`: Explore and visualize Meteo Trentino weather data.
- `data_visualization.ipynb`: General-purpose data visualization and plotting.
- `Historic_Stations_Data.ipynb`: Analyze historic monitoring station coverage and variables.
- `inspect_merged_dataset.ipynb`: Inspect merged datasets for quality and completeness.
- `LightGBM_fake_data.ipynb`, `LightGBM_sample_APPA.ipynb`: Prototype and evaluate LightGBM models on sample data.
- `mapping_visualization.ipynb`: Visualize spatial data and station locations on maps.
- `visualization_blh_data.ipynb`: Visualize Boundary Layer Height (BLH) data and related variables.

For more details on each notebook, refer to the markdown cells and code comments within the individual files.