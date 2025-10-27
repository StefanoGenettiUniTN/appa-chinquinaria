# ARPAV Download Guide

Guide for downloading meteorological data from ARPAV (Agenzia Regionale per la Prevenzione e Protezione Ambientale del Veneto) monitoring stations in the Veneto region.


## TLDR;

cd /Users/federicorubbi/Documents/unitn/public-ai-challenge/appa-chinquinaria
source venv/bin/activate
python scripts/bulk_download_arpav.py \
  --start-year 2000 --end-year 2025 \
  --out arpav_2000_2025 \
  --connect-timeout 30 --read-timeout 240

## Overview

The ARPAV bulk downloader (`bulk_download_arpav.py`) collects weather data from meteorological stations across the Veneto region. The script supports:

- **Resume capability**: Interrupted downloads can be resumed
- **State tracking**: JSON-based state management
- **Multiple sensors**: Temperature, precipitation, humidity, radiation, wind, pressure, and more
- **Organized output**: One merged CSV file per year with all stations and sensors

## Data Source

- **Provider**: ARPAV - Dipartimento Regionale per la Sicurezza del Territorio
- **License**: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/deed.it)
- **Endpoints**:
  - Station metadata: `https://www.ambienteveneto.it/datiorari/getXmlStazioniOrari.php`
  - Station data: `https://www.ambienteveneto.it/datiorari/datiSensOrari.php`

## Available Sensors

The downloader supports the following sensor types:

| Sensor Code | Description | Unit |
|-------------|-------------|------|
| `TEMPMIN` | Air temperature at 2m | °C |
| `PREC` | Precipitation | mm |
| `UMID` | Relative humidity | % |
| `RADSOL` | Global solar radiation | W/m² |
| `VVENTOMEDIO` | Average wind speed | m/s |
| `LIVIDRO` | Hydrometric level | m |
| `PORT` | Flow rate | m³/s |
| `PRESSMARE` | Sea level pressure | hPa |

## Installation

Ensure you have the required dependencies:

```bash
cd appa-chinquinaria
source venv/bin/activate
pip install requests beautifulsoup4 pandas tqdm
```

## Usage

### Basic Usage

Download all available sensors for a year range:

```bash
python scripts/bulk_download_arpav.py --start-year 2023 --end-year 2024
```

This will:
1. Collect station metadata for 2023-2024
2. Download data for all available sensors
3. Save output to `data/arpav/arpav_2023_2024/`

### Download Specific Sensors

Download only temperature and precipitation data:

```bash
python scripts/bulk_download_arpav.py \
    --start-year 2023 \
    --end-year 2023 \
    --sensors "TEMPMIN,PREC"
```

### Custom Output Directory

Specify a custom output folder:

```bash
python scripts/bulk_download_arpav.py \
    --start-year 2020 \
    --end-year 2023 \
    --out my_arpav_data
```

Output will be saved to `data/arpav/my_arpav_data/`

## Output Structure

```
data/arpav/
└── arpav_2023_2024/
    ├── metadata.json           # Station metadata (collected once)
    ├── state.json             # Download progress tracking
    ├── arpav_2023.csv         # Merged data for 2023
    └── arpav_2024.csv         # Merged data for 2024
```

### Metadata File (`metadata.json`)

Contains station information collected from XML endpoints:
- Station IDs and names
- Geographic coordinates (latitude, longitude)
- Elevation (quota)
- Province and commune
- Available sensor IDs for each station

### State File (`state.json`)

Tracks download progress with entries for each station-sensor-year combination:
- `status`: `pending`, `done`, or `failed`
- `attempts`: Number of download attempts
- `updated_at`: Last update timestamp

### Data Files (`arpav_YYYY.csv`)

Merged CSV files containing all downloaded data for a year:

| Column | Description |
|--------|-------------|
| `station_id` | ARPAV station ID |
| `year` | Year of measurement |
| `sensor_id` | Internal sensor ID |
| `timestamp` | Date and time (format: YYYY-MM-DD HH:00:00) |
| `Temp. aria a 2 m (°C)_med` | Average air temperature |
| `Pioggia (mm)_tot` | Total precipitation |
| `Umidità rel. a 2 m (%)_min` | Minimum relative humidity |
| `Umidità rel. a 2 m (%)_max` | Maximum relative humidity |
| `Radiazione globale (W/m²)_tot` | Total global radiation |
| `Vento a 10 m_Velocità med (m/s)` | Average wind speed |
| `Vento a 10 m_Direz. preval. (gradi)` | Prevailing wind direction (degrees) |
| `Vento a 10 m_Direz. preval. (settore)` | Prevailing wind direction (sector) |

**Note**: Column names may vary depending on the sensors available at each station.

## Resuming Downloads

If a download is interrupted (network issues, manual cancellation), simply re-run the same command:

```bash
python scripts/bulk_download_arpav.py --start-year 2023 --end-year 2024
```

The script will:
1. Load existing metadata (skip re-fetching)
2. Load download state
3. Resume from where it left off

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--start-year` | Start year (required) | - |
| `--end-year` | End year (required) | - |
| `--sensors` | Comma-separated sensor types | All sensors |
| `--out` | Custom output folder name | `arpav_STARTYEAR_ENDYEAR` |
| `--verify-ssl` | Enable SSL certificate verification | Disabled |

## Two-Phase Download Process

### Phase 1: Metadata Collection

The script first queries the XML endpoint for each year and sensor type to collect:
- Available stations for each year
- Station coordinates and metadata
- Sensor IDs for data download

This information is cached in `metadata.json` and reused on subsequent runs.

### Phase 2: Data Download

For each station-sensor-year combination:
1. Download HTML data using sensor ID
2. Parse HTML table to extract measurements
3. Convert to pandas DataFrame
4. Track progress in `state.json`

After all downloads for a year complete, data is merged into a single CSV file.

## Performance Considerations

- **Rate limiting**: The script adds 0.5s delay between requests to be polite to the server
- **Download time**: ~5 seconds per station on average
- **Example**: 200 stations × 1 sensor × 1 year ≈ 20 minutes

For 2023 with all 8 sensors:
- ~200 stations
- ~1,600 sensor-year combinations
- Estimated time: ~2.5 hours

## Troubleshooting

### SSL Certificate Errors

If you see SSL certificate verification errors, the script automatically disables SSL verification by default. To enable it:

```bash
python scripts/bulk_download_arpav.py --start-year 2023 --end-year 2023 --verify-ssl
```

### No Data for Specific Stations

Some stations may not have data for all years or sensors. The script will:
- Mark these as "failed" in state.json
- Log warnings to the console
- Continue with other stations

### Memory Issues

For large downloads (many years), the script processes one year at a time and writes CSV files incrementally to manage memory usage.

### Empty or Malformed HTML

If a station returns empty HTML:
- The script marks it as failed
- You can retry by re-running the same command
- Check the ARPAV website to verify the station has data for that year

## Example Workflows

### Download Recent Year (Fast Test)

```bash
# Download only temperature data for 2024
python scripts/bulk_download_arpav.py \
    --start-year 2024 \
    --end-year 2024 \
    --sensors "TEMPMIN"
```

### Complete Historical Download

```bash
# Download all sensors for 2020-2024
python scripts/bulk_download_arpav.py \
    --start-year 2020 \
    --end-year 2024
```

### Retry Failed Downloads

After a complete run, check for failures:

```bash
# Re-run the same command - only failed tasks will be retried
python scripts/bulk_download_arpav.py \
    --start-year 2023 \
    --end-year 2024
```

## Data Quality Notes

From ARPAV's disclaimer:

> "I dati della Rete Idrografica sono esposti nelle tabelle e nei grafici in modo automatico, senza validazione preventiva. ARPAV non assume responsabilità alcuna per usi diversi dalla pura informazione sulle condizioni dei corsi d'acqua. In seguito a validazione i dati possono subire modifiche anche notevoli, oppure i dati possono essere invalidati e quindi non riportati negli archivi definitivi."

Translation: Data is exposed automatically without prior validation. After validation, data may be modified significantly or invalidated.

**Recommendation**: Always verify data quality before analysis, especially for research purposes.

## Integration with Other Data Sources

The ARPAV data can be combined with:
- **APPA Trento** air quality data (PM10, NO₂, etc.)
- **Meteo Trentino** meteorological data
- **ERA5 BLH** boundary layer height data

Example analysis:
- Cross-regional weather patterns (Veneto + Trentino)
- Pollutant transport modeling between regions
- Validation of meteorological models

## Further Information

- **ARPAV Website**: https://www.arpa.veneto.it/
- **Data Portal**: https://www.ambienteveneto.it/datiorari/
- **License**: https://creativecommons.org/licenses/by/4.0/deed.it

## Testing

Before running large downloads, test the implementation:

```bash
# Run unit tests for key components
python scripts/test_arpav_functions.py
```

This will verify:
- XML metadata parsing
- HTML data download and parsing
- Multi-sensor metadata collection

