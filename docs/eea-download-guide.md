# EEA Data Download Guide

This guide explains how to download air quality data from the European Environment Agency (EEA) using the bulk download scripts.

## Overview

The EEA data download system consists of two main scripts:

1. **`scripts/bulk_download_eea.py`**: Downloads and processes data from the EEA API
2. **`scripts/download_from_csv.py`**: Downloads files from URLs listed in a CSV file

Note: the CSV file is obtained from the EEA download service: https://www.eea.europa.eu/en/datahub/datahubitem-view/778ef9f5-6293-4846-badd-56a29c70880d
For large downloads only CSV is available.

## EEA API Download (`bulk_download_eea.py`)

### Features

- 📥 Download Parquet datasets from the EEA API
- 📂 Extract .parquet files from .zip archives
- 🔄 Convert multiple Parquet files into a single CSV
- 🧩 Merge measurement data with metadata
- ✂️ Output a filtered CSV with clean and relevant columns

### Prerequisites

1. **Download metadata file**: Get station metadata from the [EEA Air Quality Dissemination Portal](https://discomap.eea.europa.eu/App/AQViewer/index.html?fqn=Airquality_Dissem.b2g.measurements#) and save it as a `.csv` file.

### Basic Usage

#### Download from API

```bash
python scripts/bulk_download_eea.py \
    --output_folder ./output \
    --output_csv eea_measurements.csv \
    --metadata ./metadata.csv \
    --api_countries IT FR DE \
    --api_pollutants PM10 NO2 \
    --api_dateTimeStart 2024-01-01 \
    --api_dateTimeEnd 2024-01-31 \
    --api_aggregationType hour
```

#### Use Local Zip File

```bash
python scripts/bulk_download_eea.py \
    --zip_path ./data/measurements.zip \
    --output_folder ./output \
    --output_csv eea_measurements.csv \
    --metadata ./metadata.csv
```

### Command Line Arguments

| Argument | Required | Description | Example |
|----------|----------|-------------|---------|
| `--zip_path` | No | Path to downloaded .zip file | `./data/measurements.zip` |
| `--output_folder` | ✅ Yes | Output directory | `./output` |
| `--output_csv` | ✅ Yes | Output CSV filename | `eea_measurements.csv` |
| `--metadata` | ✅ Yes | Path to metadata CSV | `./metadata.csv` |
| `--api_countries` | No | ISO2 country codes | `IT FR DE` |
| `--api_cities` | No | City names | `Rome Paris Berlin` |
| `--api_pollutants` | No | Pollutant names | `PM10 NO2 PM2.5` |
| `--api_dateTimeStart` | No | Start date (YYYY-MM-DD) | `2024-01-01` |
| `--api_dateTimeEnd` | No | End date (YYYY-MM-DD) | `2024-01-31` |
| `--api_aggregationType` | No | Aggregation type | `hour`, `day` |

### Output Structure

The script generates three files in a timestamped subfolder:

```
data/eea-data/
└── YYYYMMDD_HHMMSS/
    ├── eea_measurements.csv          # Final filtered CSV
    ├── raw_combined.csv              # Raw combined data
    └── metadata.zip                  # Downloaded zip (if from API)
```

### Output CSV Columns

The final CSV contains these filtered columns:

- **station-id**: Unique station identifier
- **Start, End**: Measurement time range
- **Value, Unit**: Measured value and unit
- **AggType**: Aggregation type
- **Country**: Country code
- **Air Pollutant**: Pollutant name
- **Longitude, Latitude, Altitude**: Station coordinates
- **Altitude Unit**: Altitude measurement unit
- **Air Quality Station Area**: Station area type
- **Air Quality Station Type**: Station type
- **Municipality**: City/municipality name
- **Duration Unit, Cadence Unit**: Time units

## CSV URL Download (`download_from_csv.py`)

### Features

- 📥 Download files from URLs listed in CSV files
- 🔄 Automatic retry with exponential backoff
- 📊 Progress tracking with tqdm
- 🛡️ Error handling and validation
- ⏭️ Skip existing files option

### Usage

```bash
# Basic usage
python scripts/download_from_csv.py --csv ParquetFilesUrls.csv --output ./downloads

# With custom settings
python scripts/download_from_csv.py \
    --csv urls.csv \
    --output ./data/eea-downloads \
    --timeout 120 \
    --retries 5 \
    --skip-existing
```

### Command Line Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--csv` | ✅ Yes | Path to CSV file with URLs | - |
| `--output` | ✅ Yes | Download directory | - |
| `--timeout` | No | Request timeout (seconds) | 60 |
| `--retries` | No | Maximum retry attempts | 3 |
| `--skip-existing` | No | Skip existing files | False |

### CSV Format Support

The script supports multiple CSV formats:

1. **Simple line-separated URLs**:
   ```
   https://example.com/file1.parquet
   https://example.com/file2.parquet
   ```

2. **Traditional CSV with headers**:
   ```csv
   URL,Description
   https://example.com/file1.parquet,File 1
   https://example.com/file2.parquet,File 2
   ```

3. **Multiple columns** (URL in any column):
   ```csv
   Name,URL,Size
   File1,https://example.com/file1.parquet,1MB
   ```

### Output Structure

```
data/eea-downloads/
├── file_abc123.parquet
├── file_def456.parquet
└── file_ghi789.parquet
```

## Data Sources

### EEA Air Quality Portal

- **URL**: [EEA Air Quality Dissemination Portal](https://discomap.eea.europa.eu/App/AQViewer/index.html?fqn=Airquality_Dissem.b2g.measurements#)
- **Data Format**: Parquet files
- **Coverage**: European countries
- **Update Frequency**: Daily

### Supported Pollutants

The script supports these pollutants (extend `pollutant_dict` for more):

- **PM10**: Particulate matter 10μm
- **PM2.5**: Particulate matter 2.5μm
- **NO2**: Nitrogen dioxide
- **O3**: Ozone
- **SO2**: Sulfur dioxide
- **CO**: Carbon monoxide

## Troubleshooting

### Common Issues

1. **API Rate Limiting**: The EEA API may have rate limits. The script includes retry logic.

2. **Metadata File**: Ensure you have the correct metadata CSV file from the EEA portal.

3. **Network Issues**: Use the `--retries` parameter to handle network timeouts.

4. **Disk Space**: Parquet files can be large. Ensure sufficient disk space.

### Error Messages

- **"No URLs found in CSV file"**: Check CSV format and encoding
- **"Request timeout"**: Increase `--timeout` value
- **"Failed to download"**: Check internet connection and URL validity

### Performance Tips

- Use specific date ranges to reduce download size
- Filter by countries/pollutants to get relevant data only
- Use `--skip-existing` to resume interrupted downloads
- Monitor disk space for large downloads

## Examples

### Example 1: Download Italian PM10 Data

```bash
python scripts/bulk_download_eea.py \
    --output_folder ./output \
    --output_csv italy_pm10.csv \
    --metadata ./metadata.csv \
    --api_countries IT \
    --api_pollutants PM10 \
    --api_dateTimeStart 2024-01-01 \
    --api_dateTimeEnd 2024-01-31 \
    --api_aggregationType day
```

### Example 2: Download Multiple Countries

```bash
python scripts/bulk_download_eea.py \
    --output_folder ./output \
    --output_csv european_data.csv \
    --metadata ./metadata.csv \
    --api_countries IT FR DE ES \
    --api_pollutants PM10 NO2 O3 \
    --api_dateTimeStart 2024-01-01 \
    --api_dateTimeEnd 2024-03-31 \
    --api_aggregationType hour
```

### Example 3: Download from CSV URLs

```bash
# First, get URLs from EEA portal and save to CSV
python scripts/download_from_csv.py \
    --csv ParquetFilesUrls.csv \
    --output ./data/eea-downloads \
    --skip-existing
```

## Data Processing Pipeline

### Typical Workflow

1. **Get metadata**: Download station metadata from EEA portal
2. **Download data**: Use API or CSV URL download
3. **Process data**: Script automatically converts and merges
4. **Analyze data**: Use the generated CSV for analysis

### Data Quality

- The script includes data validation and error handling
- Unmatched records are saved separately for review
- Metadata merging ensures station information is included
- Filtered output contains only relevant columns

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Verify your metadata file is correct
3. Check the EEA website for service status
4. Review script error messages for specific guidance
5. Ensure you have the latest version of required packages
