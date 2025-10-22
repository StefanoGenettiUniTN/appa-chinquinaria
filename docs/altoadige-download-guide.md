# Alto Adige Open Meteo Data Downloader Guide

This guide explains how to use the Alto Adige (Provincia Autonoma di Bolzano) bulk data downloader.

## Overview

The Alto Adige downloader fetches meteorological and hydrological data from the Alto Adige Open Data Platform using their REST API v1. The downloader is robust, resumable, and organizes data efficiently by sensor type and year.

## API Information

### Base URL
```
http://daten.buergernetz.bz.it/services/meteo/v1
```

### Available Endpoints

1. **Stations Metadata** - `/stations`
   - Returns all available weather and hydrometric stations
   - Parameters: `output_format` (JSON or CSV), `coord_sys` (optional)

2. **Sensors Metadata** - `/sensors`
   - Returns available sensors for a specific station
   - Parameters: `station_code` (required), `sensor_code` (optional), `output_format`

3. **Timeseries Data** - `/timeseries`
   - Returns measurement data for a specific station and sensor
   - Parameters: `station_code`, `sensor_code`, `date_from`, `date_to`, `output_format`
   - **Limit**: Maximum 120,000 records per request

## Sensor Types

The API provides the following sensor types:

| Code | Description (Italian) | Unit | Description (English) |
|------|----------------------|------|---------------------|
| LT | Temperatura dell'aria | °C | Air temperature |
| LF | Umidità dell'aria | % | Air humidity |
| N | Pioggia | mm | Precipitation |
| WG | Velocità media del vento | m/s | Average wind speed |
| WG.BOE | Raffica vento | m/s | Wind gust |
| WR | Direzione media del vento | gradi | Average wind direction |
| LD.RED | Pressione aria ridotta | hPa | Reduced air pressure |
| SD | Tempo di soleggiamento | secondi | Sunshine duration |
| GS | Radiazione globale | W/m² | Global radiation |
| HS | Altezza neve | cm | Snow height |
| W | Livello dell'acqua | cm | Water level |
| Q | Portata | m³/s | Flow rate |

## Installation

Make sure you have the required dependencies installed:

```bash
pip install requests pandas tqdm
```

Or install from the project's requirements file:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Download data for a specific year range:

```bash
python scripts/bulk_download_altoadige.py --start 2023 --end 2024
```

### Advanced Options

**Download specific sensors only:**
```bash
python scripts/bulk_download_altoadige.py --start 2023 --end 2024 --sensors "LT,N,Q"
```

**Custom output folder:**
```bash
python scripts/bulk_download_altoadige.py --start 2020 --end 2025 --out my_custom_folder
```

### Command-Line Arguments

- `--start` (required): Start year (e.g., 2020)
- `--end` (required): End year (e.g., 2025)
- `--sensors` (optional): Comma-separated list of sensor codes to download (default: all available)
- `--out` (optional): Custom output folder name (default: `altoadige_STARTYEAR_ENDYEAR`)

## Output Structure

The downloader creates the following directory structure:

```
data/altoadige/altoadige_YYYY_YYYY/
├── stations.csv              # All station metadata
├── sensors/                  # Sensor information per station
│   ├── 19850PG.json
│   ├── 19851PG.json
│   └── ...
├── LT/                       # Temperature data
│   ├── 2023/
│   │   ├── 19850PG.csv
│   │   ├── 19851PG.csv
│   │   └── ...
│   └── 2024/
│       └── ...
├── Q/                        # Flow rate data
│   ├── 2023/
│   │   └── ...
│   └── 2024/
│       └── ...
├── N/                        # Precipitation data
│   └── ...
└── state.json               # Download progress tracking
```

### File Formats

**stations.csv** - Station metadata:
```csv
SCODE,NAME_D,NAME_I,NAME_L,NAME_E,ALT,LONG,LAT
89940PG,ETSCH BEI SALURN,ADIGE A SALORNO,,,205.57,11.20262,46.243333
```

**sensors/*.json** - Sensor information for each station:
```json
[
  {
    "SCODE": "19850PG",
    "TYPE": "Q",
    "DESC_I": "Portata",
    "UNIT": "m³/s",
    "DATE": "2025-02-25T15:30:00CET",
    "VALUE": 18.6
  }
]
```

**{sensor}/{year}/{station}.csv** - Timeseries data:
```csv
station_code,sensor_code,year,DATE,VALUE
19850PG,Q,2024,2024-01-01T01:00:00CET,18.4
19850PG,Q,2024,2024-01-01T02:00:00CET,17.9
```

## Resume Capability

The downloader maintains a `state.json` file that tracks download progress. If the download is interrupted:

1. The state file records which station-sensor-year combinations have been completed
2. Simply re-run the same command to resume from where it stopped
3. Already downloaded files are skipped automatically

**State file structure:**
```json
{
  "meta": {
    "start_year": 2023,
    "end_year": 2024,
    "created_at": "2025-10-20T10:30:00",
    "version": 1
  },
  "tasks": [
    {
      "station_code": "19850PG",
      "sensor_code": "Q",
      "year": 2024,
      "status": "done",
      "attempts": 1,
      "updated_at": "2025-10-20T10:35:00",
      "records": 8760
    }
  ]
}
```

## Rate Limiting

The downloader implements polite rate limiting:
- Random sleep between 0.5-2.0 seconds between requests
- Automatic retry with exponential backoff on failures
- Maximum 3 retry attempts per request

## Error Handling

The downloader handles various error scenarios:

- **Network errors**: Automatic retry with backoff
- **Empty responses**: Logged as warnings, task marked as failed
- **API limits**: Years are downloaded sequentially to avoid hitting the 120,000 record limit
- **Invalid stations/sensors**: Gracefully skipped with warnings

## Testing

Before running a full download, test the API connection:

```bash
python scripts/test_altoadige_connection.py
```

This test script verifies:
1. Station metadata can be fetched
2. Sensor information is accessible
3. Timeseries data downloads correctly
4. API record limits are understood

## Examples

### Example 1: Download Temperature and Precipitation Data

```bash
python scripts/bulk_download_altoadige.py \
  --start 2022 \
  --end 2024 \
  --sensors "LT,N"
```

### Example 2: Download All Hydrological Data

```bash
python scripts/bulk_download_altoadige.py \
  --start 2020 \
  --end 2025 \
  --sensors "W,Q"
```

### Example 3: Complete Historical Download

```bash
python scripts/bulk_download_altoadige.py \
  --start 2010 \
  --end 2025 \
  --out complete_historical
```

## Performance Notes

- **Stations**: The API provides ~174 stations (as of 2025)
- **Download speed**: ~1-2 requests per second (rate-limited)
- **Estimated time**: For all sensors and stations, 2-3 years of data takes ~2-4 hours
- **Storage**: Approximately 50-100 MB per year for all sensors

## Troubleshooting

### "Connection timeout" errors
- The server may be temporarily unavailable
- Re-run the command to resume - the state file will track progress

### "No data returned" warnings
- Some stations may not have data for all sensor types
- Some stations may have gaps in their historical data
- This is normal and the downloader will continue with other stations

### "Existing state.json belongs to a different job" error
- The output folder already contains a download job with different parameters
- Either:
  - Delete the existing folder to start fresh
  - Use a different `--out` folder name

## Data Quality Notes

1. **Temporal resolution**: Most sensors provide data at 10-minute intervals
2. **Data gaps**: Historical data may have gaps due to sensor maintenance or failures
3. **Station availability**: Not all stations have all sensor types
4. **Date ranges**: Station operational periods vary - some are recent, others historical

## API Documentation

Official API documentation:
- Portal: http://daten.buergernetz.bz.it/
- API Base: http://daten.buergernetz.bz.it/services/meteo/v1

## Support

For issues related to:
- **The downloader script**: Check the state.json file and error messages
- **API availability**: Contact Alto Adige Open Data support
- **Data quality**: Refer to the official data portal documentation

