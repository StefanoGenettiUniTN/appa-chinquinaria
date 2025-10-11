# Data Analysis Guide

This guide explains how to use the visualization and correlation analysis scripts to analyze air quality data.

## Overview

The analysis toolkit includes two main scripts:

1. **`scripts/visualize_data.py`**: Creates time series plots, station comparisons, and distribution analysis
2. **`scripts/correlation_analysis.py`**: Analyzes correlations between stations and pollutants

## Data Visualization (`visualize_data.py`)

### Features

- 📈 **Time Series Plots**: Daily average concentrations over time
- 🏢 **Station Comparison**: Min-max ranges and daily means for each station
- 📊 **Distribution Analysis**: Histograms showing concentration frequency distributions
- 🎨 **Automatic Styling**: Professional plots with proper labels and legends

### Basic Usage

```bash
# Visualize all pollutants (auto-detects data folder)
python scripts/visualize_data.py

# Visualize specific pollutant
python scripts/visualize_data.py --pollutant PM10

# Custom data folder
python scripts/visualize_data.py --data-folder data/appa-data --pollutant NO2
```

### Command Line Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--data-folder` | No | Path to data directory | `data/appa-data` |
| `--pollutant` | No | Specific pollutant to analyze | All pollutants |
| `--output-dir` | No | Custom output directory | `plots/` |
| `--start-date` | No | Start date for analysis | All available data |
| `--end-date` | No | End date for analysis | All available data |

### Output Structure

```
plots/
└── plots_YYYY-MM-DD_to_YYYY-MM-DD_POLLUTANT1_POLLUTANT2/
    ├── time_series_all.png           # All pollutants time series
    ├── time_series_PM10.png          # Specific pollutant time series
    ├── station_comparison_PM10.png   # Station comparison
    └── pollutant_distributions.png   # Distribution histograms
```

### Plot Types

#### 1. Time Series Plots
- **Purpose**: Show temporal trends in air quality
- **Features**: 
  - Daily averages with trend lines
  - Multiple stations on same plot
  - Seasonal patterns visible
  - Outlier detection

#### 2. Station Comparison Plots
- **Purpose**: Compare air quality across different monitoring stations
- **Features**:
  - Min-max ranges for each station
  - Daily mean values
  - Box plots showing distribution
  - Geographic context

#### 3. Distribution Plots
- **Purpose**: Understand the statistical distribution of pollutant concentrations
- **Features**:
  - Histograms for each pollutant
  - Normal distribution overlay
  - Percentile markers
  - Outlier identification

## Correlation Analysis (`correlation_analysis.py`)

### Features

- 🔗 **Monthly Correlations**: Average correlation between stations over 30-day intervals
- 🗺️ **Correlation Heatmaps**: Correlation strength across pollutants and time periods
- 📊 **Correlation Distributions**: Histograms of correlation values
- 🔄 **Station Pairwise Correlations**: Individual correlation time series for each station pair

### Basic Usage

```bash
# Analyze all pollutants
python scripts/correlation_analysis.py

# Analyze specific pollutant
python scripts/correlation_analysis.py --pollutant PM10

# Custom output directory
python scripts/correlation_analysis.py --output-dir my_correlations
```

### Command Line Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--data-folder` | No | Path to data directory | `data/appa-data` |
| `--pollutant` | No | Specific pollutant to analyze | All pollutants |
| `--output-dir` | No | Custom output directory | `plots/` |
| `--window-size` | No | Correlation window size (days) | 30 |
| `--min-stations` | No | Minimum stations required | 2 |

### Output Structure

```
plots/
└── correlations_YYYY-MM-DD_to_YYYY-MM-DD_POLLUTANT1_POLLUTANT2/
    ├── correlation_series_PM10.png           # Monthly correlation trends
    ├── correlation_heatmap.png               # Correlation matrix
    ├── correlation_distributions.png         # Correlation histograms
    ├── station_pairwise_correlations.png     # Individual station pairs
    └── monthly_correlations.csv              # Raw correlation data
```

### Analysis Types

#### 1. Monthly Correlation Series
- **Purpose**: Track how correlations between stations change over time
- **Features**:
  - 30-day rolling correlation windows
  - Seasonal patterns in correlations
  - Trend analysis
  - Confidence intervals

#### 2. Correlation Heatmaps
- **Purpose**: Visualize correlation strength across all station pairs
- **Features**:
  - Color-coded correlation values
  - Hierarchical clustering
  - Statistical significance markers
  - Interactive hover information

#### 3. Correlation Distributions
- **Purpose**: Understand the statistical properties of correlations
- **Features**:
  - Histograms of correlation values
  - Normal distribution fits
  - Outlier detection
  - Confidence intervals

#### 4. Station Pairwise Correlations
- **Purpose**: Detailed analysis of individual station relationships
- **Features**:
  - Time series of correlations for each pair
  - Geographic distance vs correlation
  - Pollutant-specific patterns
  - Statistical significance testing

## Data Requirements

### Input Data Format

The scripts expect CSV files with the following columns:

#### APPA Data Format
- **Data**: Date (YYYY-MM-DD)
- **Ora**: Time (HH:MM)
- **Stazione**: Station ID
- **Inquinante**: Pollutant name
- **Valore**: Measured value
- **Unità**: Unit of measurement

#### EEA Data Format
- **station-id**: Station identifier
- **Start**: Start time
- **End**: End time
- **Value**: Measured value
- **Unit**: Unit of measurement
- **Air Pollutant**: Pollutant name

### Data Quality Checks

The scripts include automatic data quality checks:

- **Missing values**: Handled with interpolation or exclusion
- **Outliers**: Detected using statistical methods
- **Data consistency**: Validated across stations and time periods
- **Unit conversion**: Automatic unit standardization

## Advanced Usage

### Custom Analysis Periods

```bash
# Analyze specific time period
python scripts/visualize_data.py \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --pollutant PM10
```

### Multiple Pollutants Analysis

```bash
# Analyze multiple pollutants
python scripts/correlation_analysis.py \
    --pollutant PM10 NO2 O3 \
    --output-dir multi_pollutant_analysis
```

### Custom Correlation Windows

```bash
# Use 60-day correlation windows
python scripts/correlation_analysis.py \
    --window-size 60 \
    --pollutant PM10
```

## Interpretation Guide

### Time Series Analysis

1. **Trends**: Look for long-term increases or decreases
2. **Seasonality**: Identify seasonal patterns (winter peaks for PM, summer peaks for O3)
3. **Outliers**: Identify unusual events or measurement errors
4. **Station Differences**: Compare urban vs rural stations

### Correlation Analysis

1. **High Correlations (>0.7)**: Stations likely influenced by similar sources
2. **Low Correlations (<0.3)**: Stations may have different pollution sources
3. **Seasonal Patterns**: Correlations may change with weather patterns
4. **Geographic Patterns**: Nearby stations typically show higher correlations

### Distribution Analysis

1. **Normal Distributions**: Indicate stable pollution sources
2. **Skewed Distributions**: May indicate episodic pollution events
3. **Multiple Peaks**: Could indicate different pollution sources or conditions

## Troubleshooting

### Common Issues

1. **No data found**: Check data folder path and file format
2. **Missing stations**: Ensure all expected stations are present in data
3. **Date format errors**: Verify date columns are in correct format
4. **Memory issues**: For large datasets, consider filtering by date range

### Error Messages

- **"No valid data found"**: Check data format and column names
- **"Insufficient data for correlation"**: Need at least 30 days of overlapping data
- **"Station not found"**: Verify station IDs in your data

### Performance Tips

- Use specific date ranges to reduce processing time
- Filter by pollutant to focus analysis
- Consider data sampling for very large datasets
- Use SSD storage for better I/O performance

## Examples

### Example 1: Complete Analysis Workflow

```bash
# 1. Download data
python scripts/bulk_download_appa.py --start 2024-01-01 --end 2024-12-31

# 2. Visualize data
python scripts/visualize_data.py --pollutant PM10

# 3. Analyze correlations
python scripts/correlation_analysis.py --pollutant PM10
```

### Example 2: Seasonal Analysis

```bash
# Winter analysis
python scripts/visualize_data.py \
    --start-date 2024-12-01 \
    --end-date 2024-02-28 \
    --pollutant PM10

# Summer analysis
python scripts/visualize_data.py \
    --start-date 2024-06-01 \
    --end-date 2024-08-31 \
    --pollutant O3
```

### Example 3: Multi-Pollutant Correlation

```bash
# Analyze correlations between PM10 and NO2
python scripts/correlation_analysis.py \
    --pollutant PM10 NO2 \
    --output-dir pm10_no2_correlation
```

## Output Interpretation

### Plot Interpretation

1. **Time Series**: Look for trends, seasonality, and outliers
2. **Correlation Heatmaps**: Identify strong/weak relationships
3. **Distribution Plots**: Understand data characteristics
4. **Station Comparisons**: Identify spatial patterns

### Statistical Significance

- **p-values**: Indicate statistical significance of correlations
- **Confidence Intervals**: Show uncertainty in estimates
- **Effect Sizes**: Indicate practical significance

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Verify your data format matches the expected structure
3. Ensure you have sufficient data for analysis
4. Check the script's error messages for specific guidance
5. Review the generated plots for data quality issues
