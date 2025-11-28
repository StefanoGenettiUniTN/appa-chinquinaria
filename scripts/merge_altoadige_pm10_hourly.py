#!/usr/bin/env python3
"""
Merge Alto-Adige PM10 hourly data for all stations into a single CSV.

Input:
  - One Excel file containing PM10, NO2 and O3 for multiple stations:
        "PM10, NO2 E O3 12 stazioni 2008-2024.xlsx"
    Structure (sheet 'Foglio1'):
        - Column 'Station:':
            * Row 0: 'Messwert:' (measurement type header)
            * Row 1: 'Einheit:'  (unit header)
            * Rows 2+: timestamps like '2008-01-01 01:00:00'
        - For each station there are 2–3 columns whose column names look like:
              BZ4, BZ4.1, BZ5, BZ5.1, BZ5.2, ...
          Row 0 of each column contains the pollutant name (e.g. 'NO2', 'PM10', 'O3').
          Row 1 contains the unit (e.g. 'µg/m³', 'µg /m³').
          Rows 2+ contain numeric values.
    We only keep columns where the first data row (row index 0) is 'PM10'.

  - One CSV file with station metadata:
        "metadati_AltoAdige.xlsx - Foglio1.csv"
    Columns:
        STATION_NAME, INDIRIZZO, SEEHOEHE, GEO_LAENGE, GEO_BREITE
    STATION_NAME corresponds to the station code used in the Excel columns
    (e.g. BZ4, LS1, AB2, ...).

Output:
  - A CSV with columns:
        datetime, station_code, pm10, address, altitude_m, lon_dms, lat_dms
    Sorted by station_code, datetime, with duplicates removed.

The script follows the same style and CLI pattern as `merge_arpav_pm10_hourly.py`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def default_paths() -> Tuple[Path, Path, Path]:
    """
    Compute default input/output paths based on repository layout.

    Assumes the following structure (from repo root):
        data/altoadige/PM10/
            PM10, NO2 E O3 12 stazioni 2008-2024.xlsx
            metadati_AltoAdige.xlsx - Foglio1.csv
    """
    root = Path(__file__).resolve().parents[1]  # scripts/ -> repo root
    input_excel = root / "data" / "altoadige" / "PM10" / "PM10, NO2 E O3 12 stazioni 2008-2024.xlsx"
    metadata_csv = root / "data" / "altoadige" / "PM10" / "metadati_AltoAdige.xlsx - Foglio1.csv"
    output_csv = root / "data" / "altoadige" / "PM10" / "merged_pm10_2008_2024.csv"
    return input_excel, metadata_csv, output_csv


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """
    Load station metadata CSV and normalize column names.
    """
    if not metadata_path.exists():
        logging.warning("Metadata CSV does not exist: %s", metadata_path)
        return pd.DataFrame(columns=["station_code"])

    meta = pd.read_csv(metadata_path, dtype=str)

    # Strip whitespace from column names and values
    meta.columns = [str(c).strip() for c in meta.columns]
    for col in meta.columns:
        meta[col] = meta[col].astype(str).str.strip()

    # Expected original columns:
    #   STATION_NAME, INDIRIZZO, SEEHOEHE, GEO_LAENGE, GEO_BREITE
    rename_map = {
        "STATION_NAME": "station_code",
        "INDIRIZZO": "address",
        "SEEHOEHE": "altitude_m",
        "GEO_LAENGE": "lon_dms",
        "GEO_BREITE": "lat_dms",
    }
    for old, new in rename_map.items():
        if old in meta.columns:
            meta = meta.rename(columns={old: new})

    # Ensure station_code exists
    if "station_code" not in meta.columns:
        logging.warning(
            "Metadata CSV missing 'STATION_NAME'/'station_code' column; "
            "metadata will not be joined."
        )
        return pd.DataFrame(columns=["station_code"])

    # Keep only the relevant columns
    keep_cols: List[str] = [
        "station_code",
        "address",
        "altitude_m",
        "lon_dms",
        "lat_dms",
    ]
    meta = meta[[c for c in keep_cols if c in meta.columns]].drop_duplicates(subset=["station_code"])

    logging.info("Loaded metadata for %d stations from %s", len(meta), metadata_path)
    return meta


def extract_pm10_timeseries(excel_path: Path) -> pd.DataFrame:
    """
    Read the Alto-Adige PM10/NO2/O3 Excel file and return a long-format
    DataFrame with columns: datetime, station_code, pm10.
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"Input Excel file does not exist: {excel_path}")

    logging.info("Reading Excel file: %s", excel_path)
    xls = pd.ExcelFile(excel_path)

    # The provided file has a single sheet 'Foglio1', but keep this generic.
    sheet_name = xls.sheet_names[0]
    df_raw = pd.read_excel(xls, sheet_name=sheet_name)

    if "Station:" not in df_raw.columns:
        raise ValueError(
            "Expected a 'Station:' column in the Excel sheet; found columns: "
            f"{list(df_raw.columns)}"
        )

    # Drop completely empty rows
    df_raw = df_raw.dropna(how="all")

    if len(df_raw) < 3:
        raise ValueError("Excel sheet looks too short to contain header + data rows.")

    # Header rows:
    #   row 0: measurement names per column (e.g. 'NO2', 'PM10', 'O3')
    #   row 1: units (e.g. 'µg/m³')
    header_measure = df_raw.iloc[0]
    # header_units = df_raw.iloc[1]  # unused, but kept for clarity

    # Data rows start from index 2
    df_data = df_raw.iloc[2:].copy()
    df_data = df_data.reset_index(drop=True)

    # Parse datetime column
    dt_series = pd.to_datetime(df_data["Station:"], errors="coerce")
    df_data["datetime"] = dt_series

    # Drop rows with invalid datetime
    before = len(df_data)
    df_data = df_data.dropna(subset=["datetime"])
    after = len(df_data)
    if after < before:
        logging.debug("Dropped %d rows with invalid datetime", before - after)

    if df_data.empty:
        logging.warning("No valid datetime rows found in Excel data.")
        return pd.DataFrame(columns=["datetime", "station_code", "pm10"])

    # Identify PM10 columns: measurement row (index 0) equals 'PM10'
    pm10_columns: List[str] = []
    for col in df_raw.columns:
        if col == "Station:":
            continue
        measure = str(header_measure[col]).strip().upper()
        if "PM10" in measure:
            pm10_columns.append(col)

    if not pm10_columns:
        logging.warning("No PM10 columns found in Excel file.")
        return pd.DataFrame(columns=["datetime", "station_code", "pm10"])

    logging.info("Detected %d PM10 columns: %s", len(pm10_columns), ", ".join(pm10_columns))

    records: List[pd.DataFrame] = []
    for col in pm10_columns:
        # Station code is the part before any dot, e.g. 'BZ4.1' -> 'BZ4'
        base_name = str(col).split(".")[0].strip()
        station_code = base_name

        series = pd.to_numeric(df_data[col], errors="coerce")
        tmp = pd.DataFrame(
            {
                "datetime": df_data["datetime"],
                "station_code": station_code,
                "pm10": series,
            }
        )
        # Drop rows where pm10 is NaN
        before_station = len(tmp)
        tmp = tmp.dropna(subset=["pm10"])
        dropped = before_station - len(tmp)
        if dropped > 0:
            logging.debug(
                "Station %s: dropped %d rows with missing PM10", station_code, dropped
            )

        if not tmp.empty:
            records.append(tmp)

    if not records:
        logging.warning("All PM10 columns resulted in empty data after cleaning.")
        return pd.DataFrame(columns=["datetime", "station_code", "pm10"])

    df_pm10 = pd.concat(records, ignore_index=True)

    # Normalize datetime and sort
    df_pm10["datetime"] = pd.to_datetime(df_pm10["datetime"], errors="coerce")
    df_pm10 = df_pm10.dropna(subset=["datetime"])
    df_pm10 = df_pm10.sort_values(["station_code", "datetime"]).reset_index(drop=True)

    # Drop duplicated station_code + datetime keeping first
    before = len(df_pm10)
    df_pm10 = df_pm10.drop_duplicates(subset=["station_code", "datetime"], keep="first")
    if len(df_pm10) < before:
        logging.debug("Dropped %d duplicate station_code+datetime rows", before - len(df_pm10))

    logging.info(
        "Extracted %d PM10 records across %d stations",
        len(df_pm10),
        df_pm10["station_code"].nunique(),
    )
    return df_pm10


def merge_pm10_with_metadata(
    excel_path: Path,
    metadata_path: Path,
) -> pd.DataFrame:
    """
    Produce the final merged DataFrame:
        datetime, station_code, pm10, [metadata...]
    """
    df_pm10 = extract_pm10_timeseries(excel_path)
    if df_pm10.empty:
        return df_pm10

    meta = load_metadata(metadata_path)
    if meta.empty:
        logging.warning("Station metadata is empty or missing; proceeding without metadata join.")
        return df_pm10

    df = df_pm10.merge(meta, on="station_code", how="left")

    # Reorder columns for readability
    ordered_cols: List[str] = ["datetime", "station_code", "pm10"]
    for col in ["address", "altitude_m", "lon_dms", "lat_dms"]:
        if col in df.columns:
            ordered_cols.append(col)

    # Append any remaining columns (if any)
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    ordered_cols.extend(remaining_cols)

    df = df[ordered_cols]
    return df


def parse_args() -> argparse.Namespace:
    in_default, meta_default, out_default = default_paths()
    parser = argparse.ArgumentParser(
        description="Merge Alto-Adige PM10 hourly data for all stations into one CSV."
    )
    parser.add_argument(
        "--input-excel",
        type=Path,
        default=in_default,
        help=f"Input Excel file with PM10/NO2/O3 data (default: {in_default})",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=meta_default,
        help=f"CSV file with station metadata (default: {meta_default})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=out_default,
        help=f"Output CSV path (default: {out_default})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    excel_path: Path = args.input_excel
    metadata_path: Path = args.metadata
    output_csv: Path = args.output

    if not excel_path.exists():
        logging.error("Input Excel file does not exist: %s", excel_path)
        raise SystemExit(2)

    if not metadata_path.exists():
        logging.warning("Metadata CSV does not exist: %s", metadata_path)

    df = merge_pm10_with_metadata(excel_path, metadata_path)
    if df.empty:
        logging.warning("No data collected; nothing to write.")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logging.info("Wrote %d rows to %s", len(df), output_csv)


if __name__ == "__main__":
    main()


