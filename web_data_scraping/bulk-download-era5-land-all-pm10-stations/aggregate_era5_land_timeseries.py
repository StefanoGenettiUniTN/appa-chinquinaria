#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate ERA5-Land hourly timeseries (per-station ZIP files) into a single CSV.

The files produced by `bulk_download_era5_land_timeseries.py` are ZIP archives
with a `.zip` extension (older runs may have used `.csv` as extension).
Each archive contains four CSV members:

    - wind          → columns: valid_time, u10, v10, latitude, longitude
    - 2m temperature → columns: valid_time, t2m, latitude, longitude
    - radiation     → columns: valid_time, ssrd, latitude, longitude
    - pressure/precip → columns: valid_time, sp, tp, latitude, longitude

This script:
    1. Reads the station metadata from `era5_land_timeseries_stations.csv`.
    2. Scans an input directory for all `era5_land_timeseries_*.csv` files.
    3. For each file:
         - Opens it as a ZIP archive.
         - Reads all CSV members.
         - Merges them on `valid_time`.
         - Renames `valid_time` to `datetime`.
         - Adds station metadata (code, name, region, latitude, longitude).
    4. Concatenates all stations into one long table and saves to a single CSV.

The final dataset is in "long" format with columns:

    datetime, station_code, station_name, region, latitude, longitude,
    t2m, sp, tp, ssrd, u10, v10

Usage example:

    python scripts/aggregate_era5_land_timeseries.py \\
        --in-dir data/era5-land/era5_land_timeseries \\
        --stations-csv data/data_blh/era5_land_timeseries_stations.csv \\
        --output data/era5-land/era5_land_timeseries_all_stations_long.csv
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from functools import reduce
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate ERA5-Land hourly timeseries files (ZIP/CSV) into one CSV."
    )
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=Path("data/era5-land/era5_land_timeseries"),
        help="Directory containing per-station ERA5-Land timeseries files (.zip; legacy .csv also supported).",
    )
    parser.add_argument(
        "--stations-csv",
        type=Path,
        default=Path("data/data_blh/era5_land_timeseries_stations.csv"),
        help="Stations metadata CSV (region,station_code,station_name,latitude,longitude).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/era5-land/era5_land_timeseries_all_stations_long.csv"),
        help="Output CSV path for the aggregated long-format dataset.",
    )
    return parser.parse_args()


def load_stations_metadata(stations_csv: Path) -> pd.DataFrame:
    if not stations_csv.exists():
        raise FileNotFoundError(f"Stations CSV not found: {stations_csv}")

    df = pd.read_csv(stations_csv)
    required = {"station_code", "station_name", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Stations CSV must contain columns {sorted(required)}, missing: {sorted(missing)}"
        )

    df["station_code"] = df["station_code"].astype(str)
    return df


def station_code_from_filename(path: Path) -> str:
    """
    Extract station_code from a filename of the form:

        era5_land_timeseries_<station_code>_<start>_<end>.(zip|csv)

    where <station_code> itself may contain underscores.
    """
    stem = path.stem  # without extension
    parts = stem.split("_")
    # stem always starts with "era5", "land", "timeseries"
    # last two parts are start_date and end_date
    if len(parts) < 5:
        raise ValueError(f"Unexpected filename format: {path.name}")
    code_parts = parts[3:-2]
    if not code_parts:
        raise ValueError(f"Could not parse station_code from: {path.name}")
    return "_".join(code_parts)


def read_member_csv(zf: zipfile.ZipFile, member_name: str) -> pd.DataFrame:
    """
    Read a CSV member from a ZipFile into a DataFrame.
    """
    with zf.open(member_name) as f:
        return pd.read_csv(f)


def aggregate_single_file(path: Path) -> pd.DataFrame:
    """
    Aggregate all CSV members inside one ERA5-Land timeseries ZIP file into a
    single dataframe with columns:

        valid_time, t2m, sp, tp, ssrd, u10, v10, latitude, longitude
    """
    with zipfile.ZipFile(path, "r") as zf:
        members = zf.namelist()
        if not members:
            raise ValueError(f"No members found in ZIP file: {path}")

        dfs: List[pd.DataFrame] = []
        for name in members:
            df = read_member_csv(zf, name)
            if "valid_time" not in df.columns:
                raise ValueError(f"'valid_time' column not found in member {name} of {path}")
            # Keep only relevant columns per member
            keep_cols = [c for c in df.columns if c in ("valid_time", "t2m", "sp", "tp", "ssrd", "u10", "v10")]
            if "latitude" in df.columns:
                keep_cols.append("latitude")
            if "longitude" in df.columns:
                keep_cols.append("longitude")
            df = df[keep_cols].copy()
            dfs.append(df)

        # Merge all member dataframes on valid_time
        merged = reduce(
            lambda left, right: pd.merge(left, right, on=["valid_time", "latitude", "longitude"], how="outer"),
            dfs,
        )

        # Drop any duplicate columns created during merges (e.g., latitude_x/y)
        # (Covered by using the same column names in the merge keys.)

        # Sort by time
        merged = merged.sort_values("valid_time").reset_index(drop=True)
        return merged


def main() -> None:
    args = parse_args()
    in_dir: Path = args.in_dir
    stations_csv: Path = args.stations_csv
    output: Path = args.output

    print("=" * 80)
    print("Aggregating ERA5-Land timeseries files into a single CSV")
    print("=" * 80)
    print(f"Input directory: {in_dir}")
    print(f"Stations metadata: {stations_csv}")
    print(f"Output file: {output}")
    print("=" * 80)

    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {in_dir}")

    stations_meta = load_stations_metadata(stations_csv)

    # Support both legacy .csv (ZIP) files and new .zip files
    files_zip = list(in_dir.glob("era5_land_timeseries_*.zip"))
    files_csv = list(in_dir.glob("era5_land_timeseries_*.csv"))
    files = sorted({p for p in files_zip + files_csv})
    if not files:
        print("No files matching 'era5_land_timeseries_*.zip' or '*.csv' found in input directory.")
        sys.exit(0)

    print(f"Found {len(files)} per-station files.")

    all_rows: List[pd.DataFrame] = []

    for path in files:
        station_code = station_code_from_filename(path)
        print(f"[READ] {path.name} (station_code={station_code})")

        try:
            df_station = aggregate_single_file(path)
        except Exception as exc:
            print(f"[WARN] Skipping {path.name} due to error: {exc}", file=sys.stderr)
            continue

        # Rename time column and add station code
        df_station = df_station.rename(columns={"valid_time": "datetime"})
        df_station["datetime"] = pd.to_datetime(df_station["datetime"])
        df_station["station_code"] = station_code

        # Attach metadata (region, station_name, latitude, longitude)
        meta_row = stations_meta[stations_meta["station_code"] == station_code]
        if meta_row.empty:
            print(
                f"[WARN] No metadata found for station_code={station_code} in {stations_csv}; "
                "columns region/station_name/latitude/longitude will be NaN for this station.",
                file=sys.stderr,
            )
            df_station["region"] = pd.NA
            df_station["station_name"] = pd.NA
            df_station["latitude_meta"] = pd.NA
            df_station["longitude_meta"] = pd.NA
        else:
            meta_row = meta_row.iloc[0]
            df_station["region"] = meta_row.get("region", pd.NA)
            df_station["station_name"] = meta_row.get("station_name", pd.NA)
            df_station["latitude_meta"] = meta_row.get("latitude", pd.NA)
            df_station["longitude_meta"] = meta_row.get("longitude", pd.NA)

        all_rows.append(df_station)

    if not all_rows:
        print("No station data could be aggregated (all files failed).", file=sys.stderr)
        sys.exit(1)

    df_all = pd.concat(all_rows, ignore_index=True)

    # Reorder columns to a sensible order
    desired_cols = [
        "datetime",
        "region",
        "station_code",
        "station_name",
        "latitude_meta",
        "longitude_meta",
        "latitude",
        "longitude",
        "t2m",
        "sp",
        "tp",
        "ssrd",
        "u10",
        "v10",
    ]
    cols_existing = [c for c in desired_cols if c in df_all.columns]
    other_cols = [c for c in df_all.columns if c not in cols_existing]
    df_all = df_all[cols_existing + other_cols]

    # Sort by station and datetime
    df_all = df_all.sort_values(["station_code", "datetime"]).reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(output, index=False)
    print(f"\nSaved aggregated dataset with {len(df_all):,} rows to: {output}")


if __name__ == "__main__":
    main()


