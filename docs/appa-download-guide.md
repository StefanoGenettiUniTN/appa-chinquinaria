# APPA Trento Data Download Guide

This guide explains how to download air quality data from APPA Trento using the bulk download script.

## Overview

The APPA Trento bulk downloader (`scripts/bulk_download_appa.py`) is designed to download air quality data from the [APPA Trento Open Data portal](https://bollettino.appa.tn.it/aria/opendata/). It automatically handles:

- **Chunked downloads**: Splits large date ranges into 90-day chunks (APPA's limit)
- **Resume capability**: Can resume interrupted downloads
- **Multiple formats**: Supports CSV, JSON, and XML formats
- **Station filtering**: Download data from specific monitoring stations
- **Automatic merging**: Combines downloaded CSV files into a single file

## Quick Start

### Basic Usage

```bash
# Download data for a specific date range
python scripts/bulk_download_appa.py --start 2025-01-01 --end 2026-01-01

# Download with custom format and stations
python scripts/bulk_download_appa.py --start 2024-01-01 --end 2024-12-31 --format csv --stations "2,4,6,8"
```

### Command Line Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--start` | ✅ Yes | Start date (YYYY-MM-DD) | - |
| `--end` | ✅ Yes | End date (YYYY-MM-DD) | - |
| `--format` | No | Output format (csv, json, xml) | csv |
| `--stations` | No | Comma-separated station IDs | 2,4,6,8,9,15,22,23 |
| `--out` | No | Custom output folder name | auto-generated |

## Station Configuration

### Available Stations

The default configuration includes these monitoring stations:
- **Station 2**: Trento Centro
- **Station 4**: Rovereto
- **Station 6**: Riva del Garda
- **Station 8**: Borgo Valsugana
- **Station 9**: Pergine Valsugana
- **Station 15**: Arco
- **Station 22**: Mezzolombardo
- **Station 23**: Cles

### Station Syntax

```bash
# Download from specific stations
--stations "2,4,6"

# Download specific pollutants from stations (advanced)
--stations "2[48,53],4,6"
# Where 48 and 53 are pollutant IDs for station 2
```

## Output Structure

### Default Output Location

All downloads are saved to the `data/appa-data/` directory with the following structure:

```
data/appa-data/
└── appa-aria_YYYY-MM-DD_to_YYYY-MM-DD_format/
    ├── csv_YYYY-MM-DD_to_YYYY-MM-DD.data.csv
    ├── csv_YYYY-MM-DD_to_YYYY-MM-DD.data.csv
    ├── merged_data.csv
    └── state.json
```

### Files Generated

- **Chunk files**: Individual CSV files for each 90-day period
- **merged_data.csv**: Combined data from all chunks
- **state.json**: Download state and metadata for resume capability

## Advanced Usage

### Resume Interrupted Downloads

If a download is interrupted, simply re-run the same command:

```bash
# This will resume from where it left off
python scripts/bulk_download_appa.py --start 2024-01-01 --end 2024-12-31
```

The script automatically detects existing files and skips completed chunks.

### Custom Output Directory

```bash
# Use a custom folder name
python scripts/bulk_download_appa.py --start 2024-01-01 --end 2024-12-31 --out my_custom_download
```

This creates: `data/appa-data/my_custom_download/`

### Different Formats

```bash
# Download as JSON
python scripts/bulk_download_appa.py --start 2024-01-01 --end 2024-12-31 --format json

# Download as XML
python scripts/bulk_download_appa.py --start 2024-01-01 --end 2024-12-31 --format xml
```

## Data Format

### CSV Structure

The downloaded CSV files contain the following columns:

- **Data**: Date of measurement
- **Ora**: Time of measurement
- **Stazione**: Station ID
- **Inquinante**: Pollutant name
- **Valore**: Measured value
- **Unità**: Unit of measurement
- **Stato**: Data quality status

### Pollutants Available

Common pollutants in the dataset:
- **PM10**: Particulate matter 10μm
- **PM2.5**: Particulate matter 2.5μm
- **NO2**: Nitrogen dioxide
- **O3**: Ozone
- **SO2**: Sulfur dioxide
- **CO**: Carbon monoxide

## Troubleshooting

### Common Issues

1. **Network timeouts**: The script includes retry logic with exponential backoff
2. **Date range too large**: Automatically splits into 90-day chunks
3. **Invalid station IDs**: Check the APPA website for current station IDs
4. **Disk space**: Ensure sufficient space for large downloads

### Error Messages

- **"Existing state.json belongs to a different job"**: Use a different `--out` folder name
- **"end date must be >= start date"**: Check your date parameters
- **"No CSV files found to merge"**: Check if downloads completed successfully

### Performance Tips

- Use specific date ranges to avoid downloading unnecessary data
- Filter by stations if you only need data from certain locations
- Monitor disk space for large downloads
- Use the resume feature for large datasets

## API Reference

### Endpoint Format

The script uses the APPA Open Data API:
```
https://bollettino.appa.tn.it/aria/opendata/{FORMAT}/{DATE_RANGE}/{STATIONS}
```

Where:
- `FORMAT`: csv, json, or xml
- `DATE_RANGE`: YYYY-MM-DD,YYYY-MM-DD
- `STATIONS`: Comma-separated station IDs (optional)

### Rate Limiting

The script includes built-in delays to respect the server's rate limits and avoid being blocked.

## Examples

### Example 1: Download Recent Data

```bash
# Download last 3 months of data
python scripts/bulk_download_appa.py --start 2024-10-01 --end 2024-12-31
```

### Example 2: Download Specific Stations

```bash
# Download only from Trento and Rovereto
python scripts/bulk_download_appa.py --start 2024-01-01 --end 2024-12-31 --stations "2,4"
```

### Example 3: Long-term Download

```bash
# Download 2 years of data (will be automatically chunked)
python scripts/bulk_download_appa.py --start 2022-01-01 --end 2023-12-31 --out historical_data
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your internet connection and date parameters
3. Check the APPA website for service status
4. Review the script's error messages for specific guidance




