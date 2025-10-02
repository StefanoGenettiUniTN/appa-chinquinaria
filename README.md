# APPA Chinquinaria

APPA Aria bulk downloader for Public AI Challenge.

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

### Usage

#### Download Data

```bash
python scripts/bulk_download_appa.py --start 2025-01-01 --end 2026-01-01
```

For more options:
```bash
python scripts/bulk_download_appa.py --help
```

#### Visualize Data

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

**Default output folders:**
- `plots/plots_YYYY-MM-DD_to_YYYY-MM-DD_POLLUTANT1_POLLUTANT2/` (all pollutants)
- `plots/plots_YYYY-MM-DD_to_YYYY-MM-DD_PM10/` (specific pollutant)

#### Correlation Analysis

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

**Default output folders:**
- `plots/correlations_YYYY-MM-DD_to_YYYY-MM-DD_POLLUTANT1_POLLUTANT2/` (all pollutants)
- `plots/correlations_YYYY-MM-DD_to_YYYY-MM-DD_PM10/` (specific pollutant)
