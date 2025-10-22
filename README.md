# APPA Chinquinaria - Air Quality Data Analysis

A comprehensive data collection and analysis framework for air quality monitoring in Trentino, Italy. This project integrates multiple data sources including APPA Trento air quality measurements, Meteo Trentino meteorological data, European Environment Agency (EEA) data, and ERA5 reanalysis data for boundary layer height.

## 🎯 Project Overview

This repository provides tools to:
- Download and process air quality data from multiple sources
- Analyze correlations between meteorological conditions and pollutant concentrations
- Visualize temporal and spatial patterns in air quality
- Study the relationship between boundary layer height and pollutant dispersion

## 📊 Data Sources

| Source | Type | Description |
|--------|------|-------------|
| **APPA Trento** | Air Quality | PM10, NO₂, and other pollutant measurements from monitoring stations across Trentino |
| **Meteo Trentino** | Meteorological | Temperature, precipitation, wind, pressure, radiation, and humidity data from Trentino |
| **ARPAV** | Meteorological | Weather data from monitoring stations across Veneto region |
| **Alto Adige** | Meteorological | Weather and hydrological data from 174 monitoring stations in Alto Adige/Südtirol |
| **EEA** | Air Quality | European air quality data from multiple countries and stations |
| **ERA5** | Reanalysis | Boundary Layer Height (BLH) data from Copernicus Climate Data Store |

## 📁 Project Structure

```
appa-chinquinaria/
├── data/                          # Data storage directory
│   ├── data-samples/              # Sample datasets for testing and examples
│   │   └── sample_blh_hourly_stations.csv
│   ├── meteo-trentino/            # Meteo Trentino meteorological data
│   ├── arpav/                     # ARPAV (Veneto) meteorological data
│   ├── altoadige/                 # Alto Adige meteorological data
│   ├── eea-data/                  # European Environment Agency data
│   ├── eu-dtm/                    # Digital Terrain Model data
│   └── data_blh/                  # ERA5 Boundary Layer Height data (created by scripts)
│
├── docs/                          # Documentation
│   ├── appa-download-guide.md     # APPA Trento data download instructions
│   ├── meteo-trentino-download-guide.md
│   ├── altoadige-download-guide.md # Alto Adige data download instructions
│   ├── eea-download-guide.md
│   ├── blh-download-guide.md      # ERA5 BLH download instructions
│   └── data-analysis-guide.md
│
├── scripts/                       # Data processing scripts
│   ├── bulk_download_appa.py      # APPA Trento bulk downloader
│   ├── bulk_download_meteo_trentino.py
│   ├── bulk_download_arpav.py     # ARPAV (Veneto) bulk downloader
│   ├── bulk_download_altoadige.py # Alto Adige bulk downloader
│   ├── bulk_download_eea.py
│   ├── download_blh.py            # ERA5 BLH downloader (requires CDS credentials)
│   ├── build_blh_dataset.py       # Process ERA5 ZIP files into structured datasets
│   ├── download_from_csv.py
│   ├── list_station_variables.py
│   ├── test_meteo_connection.py
│   ├── test_arpav_functions.py    # Test ARPAV downloader functions
│   ├── test_altoadige_connection.py # Test Alto Adige API connection
│   ├── correlation_analysis.py    # Correlation analysis tools
│   └── visualize_data.py
│
├── notebooks/                     # Jupyter notebooks for analysis and visualization
│   └── visualization_blh_data.ipynb
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- For ERA5 data downloads: [CDS API credentials](https://cds.climate.copernicus.eu/api-how-to)

### Installation

1. **Clone the repository**
   ```bash
   cd appa-chinquinaria
   ```

2. **Create and activate virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure CDS API** (for ERA5 data only)
   
   Create `~/.cdsapirc` with your Copernicus credentials:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: YOUR_API_KEY
   ```

## 📖 Usage

### Downloading APPA Trento Air Quality Data

Download air quality measurements from APPA Trento monitoring stations:

```bash
# Basic download for a date range
python scripts/bulk_download_appa.py --start 2024-01-01 --end 2024-12-31

# Download specific stations with custom output directory
python scripts/bulk_download_appa.py \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --stations "2,4,6,8" \
    --output data/appa-2024
```

See [`docs/appa-download-guide.md`](docs/appa-download-guide.md) for detailed usage.

### Downloading Meteo Trentino Data

Download meteorological data (temperature, wind, precipitation, etc.):

```bash
# Download all available data for a station
python scripts/bulk_download_meteo_trentino.py \
    --station-code T0129 \
    --start-date 2024-01-01 \
    --end-date 2024-12-31
```

See [`docs/meteo-trentino-download-guide.md`](docs/meteo-trentino-download-guide.md) for details.

### Downloading ARPAV (Veneto) Data

Download meteorological data from ARPAV stations in Veneto region:

```bash
# Download all sensors for 2023-2024
python scripts/bulk_download_arpav.py \
    --start-year 2023 \
    --end-year 2024

# Download specific sensors only
python scripts/bulk_download_arpav.py \
    --start-year 2023 \
    --end-year 2023 \
    --sensors "TEMPMIN,PREC,UMID"
```

Available sensors: TEMPMIN, PREC, UMID, RADSOL, VVENTOMEDIO, LIVIDRO, PORT, PRESSMARE

See [`docs/arpav-download-guide.md`](docs/arpav-download-guide.md) for detailed information.

### Downloading Alto Adige Data

Download meteorological and hydrological data from Alto Adige monitoring stations:

```bash
# Download all sensors for 2023-2024
python scripts/bulk_download_altoadige.py \
    --start 2023 \
    --end 2024

# Download specific sensors only
python scripts/bulk_download_altoadige.py \
    --start 2023 \
    --end 2024 \
    --sensors "LT,N,Q"
```

Available sensors: LT (temperature), LF (humidity), N (precipitation), WG (wind speed), WR (wind direction), LD.RED (pressure), SD (sunshine), GS (radiation), HS (snow height), W (water level), Q (flow rate)

See [`docs/altoadige-download-guide.md`](docs/altoadige-download-guide.md) for detailed information.

### Downloading EEA Data

Download European air quality data:

```bash
python scripts/bulk_download_eea.py \
    --year 2024 \
    --pollutant PM10 \
    --country IT
```

See [`docs/eea-download-guide.md`](docs/eea-download-guide.md) for more information.

### Downloading ERA5 Boundary Layer Height Data

Download and process ERA5 boundary layer height reanalysis data:

**Step 1: Download ERA5 data** (requires CDS credentials)

```bash
# Download BLH data for 2020-2025
python scripts/download_blh.py \
    --start-year 2020 \
    --end-year 2025 \
    --chunk-years 2 \
    --area "47.67,4.61,43.54,16.12"
```

**Step 2: Build structured dataset**

```bash
# Process ZIP files and create hourly/daily datasets
python scripts/build_blh_dataset.py \
    --in-dir data_blh \
    --out-dir data_blh/processed \
    --csv
```

This creates:
- `blh_hourly_all_stations.parquet` - Hourly BLH values
- `blh_daily_all_stations.parquet` - Daily averaged BLH values
- Optional CSV files if `--csv` flag is used

See [`docs/blh-download-guide.md`](docs/blh-download-guide.md) for detailed information.

### Data Visualization and Analysis

Explore the notebooks for interactive analysis:

```bash
# Start Jupyter
jupyter notebook notebooks/visualization_blh_data.ipynb
```

The visualization notebook includes:
- Geographic visualization of monitoring stations and ERA5 grid cells
- Time series analysis of boundary layer height
- Station-to-grid mapping and distance calculations

### Correlation Analysis

Analyze relationships between meteorological variables and pollutants:

```bash
python scripts/correlation_analysis.py \
    --aq-data data/appa-data.csv \
    --meteo-data data/meteo-trentino/processed.csv \
    --output results/
```

## 🔍 Key Features

### Boundary Layer Height Analysis

The **Mixing Layer Height (MLH)** or **Planetary Boundary Layer Height (PBLH)** is crucial for understanding pollutant dispersion:

- **Low values (100–300 m)**: Stable atmosphere, pollutant accumulation
- **High values (1000–2000 m)**: Strong mixing, efficient dispersion

This metric helps:
- Identify thermal inversion episodes
- Predict air stagnation events
- Correlate meteorological conditions with PM10/NO₂ concentrations

### Resume-able Downloads

All download scripts support:
- State tracking with JSON files
- Automatic resume of interrupted downloads
- Chunk-based downloading to handle API limits

### Multiple Data Formats

- CSV, JSON, XML support for APPA data
- NetCDF for ERA5 data
- Parquet for efficient data storage and processing

## 📚 Documentation

Detailed guides are available in the `docs/` directory:

- **[APPA Download Guide](docs/appa-download-guide.md)** - APPA Trento air quality data
- **[Meteo Trentino Guide](docs/meteo-trentino-download-guide.md)** - Meteorological data (Trentino)
- **[ARPAV Download Guide](docs/arpav-download-guide.md)** - Meteorological data (Veneto)
- **[Alto Adige Download Guide](docs/altoadige-download-guide.md)** - Meteorological data (Alto Adige)
- **[EEA Download Guide](docs/eea-download-guide.md)** - European air quality data
- **[BLH Download Guide](docs/blh-download-guide.md)** - ERA5 boundary layer height
- **[Data Analysis Guide](docs/data-analysis-guide.md)** - Analysis methodologies

## 🛠️ Troubleshooting

### CDS API Issues

If ERA5 downloads fail:
1. Verify your `~/.cdsapirc` credentials
2. Check your CDS API quota at https://cds.climate.copernicus.eu
3. Ensure you've accepted the ERA5 license terms

### Memory Issues

For large datasets:
- Use Parquet format instead of CSV
- Process data in chunks
- Increase chunk size in download scripts

### Missing Dependencies

If you encounter import errors:
```bash
pip install --upgrade -r requirements.txt
```

## 📊 Sample Data

Sample datasets are provided in `data/data-samples/` for testing and exploring the notebooks without downloading full datasets.

## 🤝 Contributing

When adding new features:
1. Place scripts in `scripts/`
2. Add documentation to `docs/`
3. Update this README
4. Keep data files in `data/` (ensure `.gitignore` excludes large files)

## 📝 Notes

- Large data files (*.nc, *.zip, *.parquet) are excluded from version control
- Always verify downloaded data integrity before analysis
- ERA5 downloads may take significant time due to CDS queue times

## 📄 License

This project is part of the University of Trento public AI challenge.

## 🙏 Acknowledgments

- **APPA Trento** for air quality monitoring data
- **Meteo Trentino** for meteorological observations from Trentino region
- **ARPAV** for meteorological data from Veneto region
- **Provincia Autonoma di Bolzano - Alto Adige** for meteorological and hydrological data
- **European Environment Agency** for European air quality data
- **Copernicus Climate Data Store** for ERA5 reanalysis data

