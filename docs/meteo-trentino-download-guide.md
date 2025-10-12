# Meteo Trentino Data Download Guide

## Overview

This guide explains how to use the Meteo Trentino bulk download script to retrieve meteorological data from the Meteo Trentino Hydstra web portal using a browser-based approach.

## Prerequisites

- Python 3.7 or higher
- Required Python packages (install with `pip install -r requirements.txt`):
  - `requests>=2.25.0` - HTTP requests
  - `beautifulsoup4>=4.9.0` - HTML parsing
  - `lxml>=4.6.0` - XML parsing
  - `tqdm>=4.60.0` - Progress bars
  - `pandas>=1.3.0` - Data manipulation

## Service Information

- **Base URL**: http://storico.meteotrentino.it
- **Data Portal**: Hydstra web interface
- **Data Type**: Meteorological data (temperature, precipitation, wind, etc.)
- **Download Method**: Browser-based ZIP file generation

## Script Usage

### Basic Usage

```bash
python scripts/bulk_download_meteo_trentino.py --stations "T0038,T0129" --variables "Pioggia,Temperatura aria"
```

### Advanced Usage

```bash
python scripts/bulk_download_meteo_trentino.py \
    --all-stations \
    --all-variables \
    --out custom_output_folder
```

### Parameters

- `--stations`: Comma-separated list of station codes (default: "T0038,T0129,T0139")
- `--all-stations`: Download from all available stations
- `--variables`: Comma-separated list of variables (default: "Pioggia,Temperatura aria,Umid.relativa aria,Direzione vento media,Veloc. vento media,Pressione atmosferica,Radiazione solare totale")
- `--all-variables`: Download all available variables for each station
- `--out`: Custom output folder name (optional)
- `--no-extract`: Don't extract CSV files from ZIP archives

## Data Organization

### Output Structure

```
data/meteo-trentino/
└── meteo-trentino_20250115_143022/
    ├── state.json                    # Download state and metadata
    ├── T0038_Pioggia.zip             # ZIP archive with CSV data
    ├── T0038_Pioggia.csv             # Extracted CSV file
    ├── T0038_Temperatura_aria.zip
    ├── T0038_Temperatura_aria.csv
    ├── T0129_Pioggia.zip
    ├── T0129_Pioggia.csv
    └── ...
```

### File Naming Convention

Individual files follow the pattern:
```
{STATION_CODE}_{VARIABLE_NAME}.zip
{STATION_CODE}_{VARIABLE_NAME}.csv
```

### State File

The `state.json` file contains:
- Download metadata (stations, variables)
- Progress tracking for each download
- Resume information for interrupted downloads

## Features

### Browser-Based Download

- Mimics the web browser interface to trigger data exports
- Uses the same ZIP file generation system as the web portal
- Automatically extracts CSV files from ZIP archives

### Resume Capability

- The script can resume interrupted downloads
- Progress is tracked in the `state.json` file
- Only incomplete downloads will be retried on subsequent runs

### Error Handling

- Automatic retry with exponential backoff for network errors
- Graceful handling of service errors and missing data
- Detailed error logging and reporting

### Data Organization

- Each station-variable combination is downloaded as a separate ZIP file
- CSV files are automatically extracted for easy access
- Maintains both original ZIP and extracted CSV files

## Common Station Codes

Some common meteorological monitoring stations in Trentino:

- `T0038` - San Michele Alladige
- `T0129` - Trento (Laste)
- `T0139` - Santorsola Terme
- `T0144` - Monte Bondone
- `T0147` - Rovereto
- `T0152` - Brentonico
- `T0356` - Trento (Aeroporto)

*Note: The script includes a comprehensive list of stations. Use `--all-stations` to download from all available stations.*

## Common Variables

Typical meteorological variables available:

- `Pioggia` - Precipitation (millimeters)
- `Temperatura aria` - Air temperature (Celsius)
- `Umid.relativa aria` - Relative humidity (percentage)
- `Direzione vento media` - Average wind direction (degrees)
- `Veloc. vento media` - Average wind speed (meters/second)
- `Pressione atmosferica` - Atmospheric pressure (hectopascals)
- `Radiazione solare totale` - Total solar radiation (kJoule/square meter)

*Note: Available variables depend on the specific station and may vary. The script automatically filters out "Annale Idrologico" variants to avoid duplicates.*

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Check internet connection
   - Verify the Meteo Trentino website is accessible
   - Ensure requests and beautifulsoup4 libraries are properly installed

2. **No Data Retrieved**
   - Verify station codes are correct
   - Check if variables are available for the selected stations
   - Some stations may not have data for all variables

3. **Partial Downloads**
   - Re-run the same command to resume
   - Check network connectivity
   - Review error messages in the output

4. **ZIP Extraction Errors**
   - Some ZIP files may be corrupted or empty
   - Check the ZIP file manually
   - Use `--no-extract` to skip CSV extraction if needed

### Debug Mode

For detailed debugging, you can modify the script to add more verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Data Format

### CSV Structure

Each CSV file contains time series data with the following typical structure:

```csv
Date,Time,Value,Quality
01/01/2025,00:00,12.5,good
01/01/2025,01:00,13.2,good
01/01/2025,02:00,12.8,good
...
```

### ZIP Archive Structure

Each ZIP file contains:
- One or more CSV files with the actual data
- Metadata files (if available)
- The exact structure depends on the station and variable

### State File Format

The state file contains:

```json
{
  "meta": {
    "stations": ["T0038", "T0129"],
    "variables": ["Pioggia", "Temperatura aria"],
    "created_at": "2025-01-15T10:30:00",
    "version": 1
  },
  "downloads": [
    {
      "station": "T0038",
      "variable": "Pioggia",
      "filename": "T0038_Pioggia.zip",
      "status": "done",
      "attempts": 1,
      "updated_at": "2025-01-15T10:35:00",
      "csv_file": "T0038_Pioggia.csv"
    }
  ]
}
```

## Performance Considerations

- **Rate Limiting**: 2-second delay between requests to be polite to the server
- **Retry Logic**: Automatic retry for failed requests
- **Progress Tracking**: Real-time progress bars when tqdm is available
- **Memory Usage**: Data is processed file by file to minimize memory usage

## Integration with Existing Workflow

This script follows the same patterns as the existing APPA downloader:

- Similar command-line interface
- Consistent output directory structure
- State management for resumable downloads
- Error handling and retry logic
- Progress reporting

## How It Works

The script works by:

1. **Station Discovery**: Uses a predefined list of station codes (can be extended)
2. **Variable Discovery**: Fetches available variables for each station from XML endpoints
3. **Download Trigger**: Mimics browser requests to trigger ZIP file generation
4. **File Extraction**: Downloads ZIP files and optionally extracts CSV data
5. **State Management**: Tracks progress and enables resume functionality

## Support and Documentation

- **Meteo Trentino**: Contact for service-specific questions
- **Hydstra Documentation**: Available from Kisters Pty Ltd
- **Web Scraping**: General web scraping documentation and tutorials

## License and Usage

Please ensure compliance with Meteo Trentino's terms of service and data usage policies when using this script.
