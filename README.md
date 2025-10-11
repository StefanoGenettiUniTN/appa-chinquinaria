# APPA Chinquinaria

A comprehensive air quality data collection and analysis toolkit for the Public AI Challenge. This project provides tools for downloading, processing, and analyzing air quality data from multiple sources including APPA Trento and the European Environment Agency (EEA).

## Features

### Data Sources
- **APPA Trento**: Regional air quality data from Trentino, Italy
- **European Environment Agency (EEA)**: European-wide air quality measurements

### Core Functionality
- 📥 **Bulk Data Download**: Automated downloading from multiple air quality data sources
- 📊 **Data Visualization**: Time series plots, station comparisons, and distribution analysis
- 🔍 **Correlation Analysis**: Monthly correlation analysis between stations and pollutants
- 🧩 **Data Processing**: Merging, filtering, and cleaning of air quality datasets
- 📂 **Multiple Formats**: Support for CSV, Parquet, and JSON data formats

## Setup

### Prerequisites
- Python 3.7 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd appa-chinquinaria
   ```

2. **Create and activate virtual environment:**

   **Windows:**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

   **Linux/macOS:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

```bash
# Download APPA Trento data
python scripts/bulk_download_appa.py --start 2025-01-01 --end 2026-01-01

# Download EEA data
python scripts/bulk_download_eea.py \
    --output_csv eea_measurements.csv \
    --metadata ./metadata.csv \
    --api_countries IT \
    --api_pollutants PM10

# Visualize data
python scripts/visualize_data.py --pollutant PM10

# Analyze correlations
python scripts/correlation_analysis.py --pollutant PM10
```

### Detailed Documentation

For comprehensive usage instructions, see the dedicated documentation files:

- **[APPA Download Guide](docs/appa-download-guide.md)**: Complete guide for downloading APPA Trento data
- **[EEA Download Guide](docs/eea-download-guide.md)**: Complete guide for downloading EEA data
- **[Data Analysis Guide](docs/data-analysis-guide.md)**: Guide for visualization and correlation analysis

### Basic Commands

#### APPA Trento Data
```bash
# Download data
python scripts/bulk_download_appa.py --start 2025-01-01 --end 2026-01-01

# Visualize data
python scripts/visualize_data.py --pollutant PM10

# Correlation analysis
python scripts/correlation_analysis.py --pollutant PM10
```

#### EEA Data
```bash
# Download from API
python scripts/bulk_download_eea.py \
    --output_csv eea_measurements.csv \
    --metadata ./metadata.csv \
    --api_countries IT FR DE \
    --api_pollutants PM10 NO2

# Download from CSV URLs
python scripts/download_from_csv.py --csv ParquetFilesUrls.csv
```

## Output Structure

All data is organized in the `data/` directory:

```
data/
├── appa-data/                    # APPA Trento downloads
│   └── appa-aria_YYYY-MM-DD_to_YYYY-MM-DD_csv/
│       ├── csv_*.data.csv
│       ├── merged_data.csv
│       └── state.json
├── eea-data/                     # EEA downloads
│   └── YYYYMMDD_HHMMSS/
│       ├── eea_measurements.csv
│       └── metadata.zip
└── eea-downloads/                # EEA CSV URL downloads
    └── *.parquet files
```

### Plots Output
```
plots/
├── plots_YYYY-MM-DD_to_YYYY-MM-DD_POLLUTANT1_POLLUTANT2/
│   ├── time_series_all.png
│   ├── station_comparison_PM10.png
│   └── pollutant_distributions.png
└── correlations_YYYY-MM-DD_to_YYYY-MM-DD_POLLUTANT1_POLLUTANT2/
    ├── correlation_series_PM10.png
    ├── correlation_heatmap.png
    └── monthly_correlations.csv
```

## Data Sources

### APPA Trento
- **Source**: Regional air quality monitoring network in Trentino, Italy
- **Data Types**: PM10, PM2.5, NO2, O3, SO2, CO
- **Format**: CSV files with time series data

### European Environment Agency (EEA)
- **Source**: [EEA Air Quality Dissemination Portal](https://discomap.eea.europa.eu/App/AQViewer/index.html?fqn=Airquality_Dissem.b2g.measurements#)
- **Data Types**: Multiple pollutants across European countries
- **Format**: Parquet files with comprehensive metadata

## Requirements

### Dependencies
```
pandas
requests
pyarrow
matplotlib
seaborn
numpy
tqdm
```

Install all dependencies via:
```bash
pip install -r requirements.txt
```

## Notes

- The EEA script currently supports pollutants mapped in `pollutant_dict`. Extend the dictionary for more pollutants.
- All scripts create timestamped output folders to avoid overwriting previous results.
- The visualization scripts automatically detect available data and generate appropriate plots.
- Correlation analysis uses 30-day rolling windows for monthly correlation calculations.

## Contributing

This project is part of the Public AI Challenge. For contributions or issues, please refer to the project repository.

## License

[Add appropriate license information]