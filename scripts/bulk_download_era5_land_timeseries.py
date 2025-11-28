#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ERA5-Land hourly timeseries bulk downloader (per point, ZIP of CSVs).

This script downloads ERA5-Land hourly timeseries for a list of stations
defined by latitude/longitude coordinates (typically air quality stations)
using the CDS API "reanalysis-era5-land-timeseries" dataset.

The goal is to obtain, for each station:
    - 2m temperature (t2m → temperature_2m in aggregated output)
    - surface pressure (sp → surface_pressure in aggregated output)
    - total precipitation (tp → total_precipitation in aggregated output)
    - surface solar radiation downwards (ssrd → solar_radiation_downwards in aggregated output)
    - 10m u component of wind (u10 → wind_u_10m in aggregated output)
    - 10m v component of wind (v10 → wind_v_10m in aggregated output)
    
Note: Column names are expanded to readable forms during aggregation
by `aggregate_era5_land_timeseries.py`.

Usage example:

    python bulk_download_era5_land_timeseries.py \\
        --stations-csv data/data_blh/era5_land_timeseries_stations.csv \\
        --start-date 2014-01-01 --end-date 2025-01-01 \\
        --out era5_land_timeseries

Requirements:
    - `cdsapi` installed (pip install cdsapi)
    - A valid `~/.cdsapirc` with CDS credentials.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    import cdsapi
except ImportError:
    print("This script requires the 'cdsapi' package. Install with: pip install cdsapi", file=sys.stderr)
    sys.exit(1)


DEFAULT_DATASET = "reanalysis-era5-land-timeseries"
DEFAULT_VARIABLES = [
    "2m_temperature",
    "surface_pressure",
    "total_precipitation",
    "surface_solar_radiation_downwards",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]


def ensure_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bulk download ERA5-Land hourly timeseries for multiple stations (point locations)."
    )
    parser.add_argument(
        "--stations-csv",
        type=Path,
        default=Path("data/data_blh/era5_land_timeseries_stations.csv"),
        help="CSV with columns: region,station_code,station_name,latitude,longitude",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2014-01-01",
        help="Start date (YYYY-MM-DD, inclusive).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-01-01",
        help="End date (YYYY-MM-DD, inclusive or exclusive as per CDS docs).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="era5_land_timeseries",
        help="Output subdirectory under project 'data' folder (default: era5_land_timeseries).",
    )
    parser.add_argument(
        "--variables",
        type=str,
        default=",".join(DEFAULT_VARIABLES),
        help=(
            "Comma-separated list of ERA5-Land variables to request "
            f"(default: {','.join(DEFAULT_VARIABLES)})"
        ),
    )
    return parser.parse_args()


def build_output_dir(out_name: str) -> Path:
    """
    Build the base output directory under the project root.

    We place all ERA5-Land timeseries files under:
        data/era5-land/{out_name}
    """
    project_root = Path(__file__).resolve().parent.parent
    base_dir = project_root / "data" / "era5-land"
    return base_dir / out_name


def load_stations(stations_csv: Path) -> pd.DataFrame:
    if not stations_csv.exists():
        raise FileNotFoundError(f"Stations CSV not found: {stations_csv}")

    df = pd.read_csv(stations_csv)
    expected_cols = {"station_code", "station_name", "latitude", "longitude"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Stations CSV must contain columns: {sorted(expected_cols)}; missing: {sorted(missing)}"
        )

    # Ensure numeric lat/lon
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    if df["latitude"].isna().any() or df["longitude"].isna().any():
        raise ValueError("Some station coordinates could not be parsed as floats.")

    return df


def build_request(
    variables: List[str],
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> dict:
    """
    Build the CDS request payload for one station.

    The CDS API for ERA5-Land timeseries expects:
        - dataset: "reanalysis-era5-land-timeseries"
        - "variable": list of variable names
        - "location": {"longitude": lon, "latitude": lat}
        - "date": ["YYYY-MM-DD/YYYY-MM-DD"]
        - "data_format": "csv"
    """
    return {
        "variable": variables,
        "location": {"longitude": float(lon), "latitude": float(lat)},
        "date": [f"{start_date}/{end_date}"],
        "data_format": "csv",
    }


def main() -> None:
    args = parse_args()

    stations_csv: Path = args.stations_csv
    start_date: str = args.start_date
    end_date: str = args.end_date
    out_name: str = args.out
    variables: List[str] = [v.strip() for v in args.variables.split(",") if v.strip()]

    out_dir = build_output_dir(out_name)
    ensure_folder(out_dir)

    print("=" * 80)
    print("ERA5-Land hourly timeseries bulk download")
    print("=" * 80)
    print(f"Stations CSV: {stations_csv}")
    print(f"Output directory: {out_dir}")
    print(f"Date range: {start_date} → {end_date}")
    print(f"Variables: {variables}")
    print("=" * 80)

    stations_df = load_stations(stations_csv)

    # Exclude stations with insufficient PM10 data from ERA5-Land downloads.
    # These were filtered out in the PM10 curation workflow and should not be
    # part of the ERA5-Land dataset used alongside it.
    excluded_station_codes = {"AB3", "CR2"}
    if not excluded_station_codes.isdisjoint(set(stations_df["station_code"].astype(str))):
        before = len(stations_df)
        stations_df = stations_df[~stations_df["station_code"].astype(str).isin(excluded_station_codes)]
        after = len(stations_df)
        print(
            f"Filtered stations metadata: removed {before - after} stations "
            f"with station_code in {sorted(excluded_station_codes)}."
        )

    print(f"Found {len(stations_df)} stations in CSV after filtering.")

    client = cdsapi.Client()

    for _, row in stations_df.iterrows():
        station_code = str(row["station_code"])
        station_name = str(row["station_name"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])

        safe_code = station_code.replace(" ", "_")
        # The CDS API returns a ZIP archive containing multiple CSVs,
        # so we use a .zip extension for clarity.
        out_file = out_dir / f"era5_land_timeseries_{safe_code}_{start_date}_{end_date}.zip"

        if out_file.exists():
            print(f"[SKIP] {station_code} ({station_name}) → {out_file.name} already exists.")
            continue

        print(f"[REQ ] {station_code} ({station_name}) at lat={lat:.5f}, lon={lon:.5f}")

        request = build_request(
            variables=variables,
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date,
        )

        try:
            # cdsapi will write the file to the current working directory by default.
            # We set the target path explicitly to out_file.
            client.retrieve(DEFAULT_DATASET, request, str(out_file))
            print(f"[DONE] Saved: {out_file}")
        except Exception as exc:
            print(f"[FAIL] {station_code} ({station_name}) → {exc}", file=sys.stderr)

    print("\nAll requests processed.")


if __name__ == "__main__":
    main()


