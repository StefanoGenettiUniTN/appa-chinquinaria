# bulk_download_eea.py

`bulk_download_eea.py` is a Python utility for bulk downloading, extracting, and processing air quality data from the [European Environment Agency (EEA)](https://discomap.eea.europa.eu/App/AQViewer/index.html?fqn=Airquality_Dissem.b2g.measurements#).

It automates:

- Downloading measurement datasets (Parquet format) from the EEA API or using a local `.zip` file.

- Extracting `.parquet` files and converting them into a single CSV.

- Merging the dataset with metadata (station details, location, altitude, station type, etc.).

- Filtering to keep only relevant columns for easier analysis.

## Features

- 📥 Download Parquet datasets from the EEA API (optionally filter by country, city, pollutant, date, aggregation).

- 📂 Extract .parquet files from .zip archives.

- 🔄 Convert multiple Parquet files into a single CSV.

- 🧩 Merge measurement data with metadata (Sampling Point Id, station location, etc.).

- ✂️ Output a filtered CSV with clean and relevant columns.

## Requirements

- Dependencies:

```
pip install pandas requests pyarrow
```

(or install all required dependencies via `requirements.txt`)

## Usage
### 1. Download metadata file

Download station metadata from the [EEA Air Quality Dissemination Portal](https://discomap.eea.europa.eu/App/AQViewer/index.html?fqn=Airquality_Dissem.b2g.measurements#) and save it as a `.csv`.

### 2. Run the script
**Example: Fetch data from API**
```
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
```
python bulk_download_eea.py \
    --zip_path ./data/measurements.zip \
    --output_folder ./output \
    --output_csv eea_measurements.csv \
    --metadata ./metadata.csv
```

### Command-line Arguments
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

### Output

The script generates:

1. A raw combined CSV (from all Parquet files).

2. A merged CSV (with station metadata).

3. A filtered CSV containing only the most useful fields:

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

## Notes
The script currently supports pollutants mapped in pollutant_dict. Extend the dictionary for more pollutants.
