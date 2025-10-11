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

### APPA Trento Data

#### Download Data
```bash
python scripts/bulk_download_appa.py --start 2025-01-01 --end 2026-01-01
```

For more options:
```bash
python scripts/bulk_download_appa.py --help
```

#### Visualize APPA Data
Create time series plots and station comparisons:

```bash
# Visualize all pollutants (auto-generated output folder)
python scripts/visualize_data.py

# Visualize specific pollutant
python scripts/visualize_data.py --pollutant PM10

# Custom data folder
python scripts/visualize_data.py --data-folder appa-data --pollutant NO2
```

**What it plots:**
- **Time Series**: Daily average concentrations over time for each station and pollutant
- **Station Comparison**: Min-max ranges and daily means for each station
- **Distribution**: Histograms showing concentration frequency distributions

#### APPA Correlation Analysis
Analyze monthly correlations between stations:

```bash
# Analyze all pollutants (auto-generated output folder)
python scripts/correlation_analysis.py

# Analyze specific pollutant
python scripts/correlation_analysis.py --pollutant PM10

# Custom output directory
python scripts/correlation_analysis.py --output-dir my_correlations
```

**What it plots:**
- **Monthly Correlation Series**: Average correlation between stations over 30-day intervals
- **Correlation Heatmap**: Correlation strength across pollutants and time periods
- **Correlation Distribution**: Histograms of correlation values for each pollutant
- **Station Pairwise Correlations**: Individual correlation time series for each station pair

### EEA Data

#### Download EEA Data

**Example: Fetch data from API**
```bash
python bulk_download_eea.py \
    --output_folder ./output \
    --output_csv eea_measurements.csv \
    --metadata ./metadata.csv \
    --api_countries IT FR DE \
    --api_pollutants PM10 NO2 \
    --api_dateTimeStart 2024-01-01 \
    --api_dateTimeEnd 2024-01-31 \
    --api_aggregationType hour
```

**Example: Use a local zip file**
```bash
python bulk_download_eea.py \
    --zip_path ./data/measurements.zip \
    --output_folder ./output \
    --output_csv eea_measurements.csv \
    --metadata ./metadata.csv
```

#### EEA Command-line Arguments
| Argument                | Required | Description                                                                                          |
| ----------------------- | -------- | ---------------------------------------------------------------------------------------------------- |
| `--zip_path`            | No       | Path to an already-downloaded `.zip` file (if not provided, the script will fetch data via the API). |
| `--output_folder`       | ✅ Yes    | Folder where results will be stored (a timestamped subfolder is created).                            |
| `--output_csv`          | ✅ Yes    | Name of the final CSV file (inside the output folder).                                               |
| `--metadata`            | ✅ Yes    | Path to metadata CSV (download from the EEA AQ portal).                                              |
| `--api_countries`       | No       | List of ISO2 country codes (e.g., `IT FR DE`).                                                       |
| `--api_cities`          | No       | List of cities to filter by (e.g., `Rome Paris Berlin`).                                             |
| `--api_pollutants`      | No       | List of pollutants (e.g., `PM10 NO2 PM2.5`).                                                         |
| `--api_dateTimeStart`   | No       | Start date (`YYYY-MM-DD`).                                                                           |
| `--api_dateTimeEnd`     | No       | End date (`YYYY-MM-DD`).                                                                             |
| `--api_aggregationType` | No       | Aggregation type (e.g., `hour`, `day`).                                                              |

## Output Structure

### APPA Data Output
**Default output folders:**
- `plots/plots_YYYY-MM-DD_to_YYYY-MM-DD_POLLUTANT1_POLLUTANT2/` (all pollutants)
- `plots/plots_YYYY-MM-DD_to_YYYY-MM-DD_PM10/` (specific pollutant)
- `plots/correlations_YYYY-MM-DD_to_YYYY-MM-DD_POLLUTANT1_POLLUTANT2/` (all pollutants)
- `plots/correlations_YYYY-MM-DD_to_YYYY-MM-DD_PM10/` (specific pollutant)

### EEA Data Output
The EEA script generates:

1. **Raw combined CSV**: From all Parquet files
2. **Merged CSV**: With station metadata
3. **Filtered CSV**: Containing only the most useful fields:
   - station-id
   - Start, End
   - Value, Unit
   - AggType
   - Country
   - Air Pollutant
   - Longitude, Latitude, Altitude
   - Altitude Unit
   - Air Quality Station Area
   - Air Quality Station Type
   - Municipality
   - Duration Unit
   - Cadence Unit

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