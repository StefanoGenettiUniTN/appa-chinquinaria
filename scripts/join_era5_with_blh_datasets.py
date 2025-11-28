#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Join the aggregated ERA5-Land timeseries dataset with BLH and multi-level ERA5
station datasets from `data/data_blh`.

Inputs:
    1) Aggregated ERA5-Land long-format dataset
       (from `aggregate_era5_land_timeseries.py`), e.g.:
           data/era5-land/era5_land_timeseries_all_stations_long.csv

       Columns (at least):
           datetime, station_code,
           temperature_2m, surface_pressure, total_precipitation,
           solar_radiation_downwards, wind_u_10m, wind_v_10m, ...

    2) BLH station dataset in wide format:
           data/data_blh/df_blh_stations.csv
       Columns:
           valid_time, <station_code>_blh, ..., year

    3) ERA5 multi-level station dataset in wide format:
           data/data_blh/df_era5_stations.csv
       Columns:
           valid_time, <station_code>_<var>, ...

Output:
    A long-format dataset with all variables joined on (datetime, station_code),
    and a short completeness / NaN report.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd


# Mapping from EU codes (IT1930A, etc.) to IT codes (402212, etc.) for APPA stations
EU_TO_IT_CODE_MAPPING = {
    "IT1930A": "402212",  # PIANA ROTALIANA
    "IT0753A": "402204",  # RIVA GAR
    "IT1191A": "402203",  # MONTE GAZZA
    "IT1037A": "402209",  # TRENTO PSC
    "IT0591A": "402206",  # ROVERETO LGP
    "IT1859A": "402211",  # TRENTO VBZ
    "IT0703A": "402201",  # BORGO VAL
    # AVIO A22 has no EU code, uses 402213 directly
}


def normalize_station_code(code: str) -> str:
    """
    Normalize station code: convert EU codes (IT1930A, etc.) to IT codes (402212, etc.)
    for APPA stations. Other codes are returned as-is.
    
    Args:
        code: Station code (may be EU code or IT code)
    
    Returns:
        Normalized IT code
    """
    code_str = str(code).strip()
    # Check if it's an EU code that needs mapping
    if code_str in EU_TO_IT_CODE_MAPPING:
        return EU_TO_IT_CODE_MAPPING[code_str]
    # Otherwise return as-is (already IT code or other region)
    return code_str


def extract_station_code_from_column(col_name: str, valid_station_codes: Optional[Set[str]] = None) -> str:
    """
    Extract station code from a column name like "ARPAL_001_humidity_550" or "502604_t850".
    
    Station codes can contain underscores (e.g., "ARPAL_001"), so we need to find the
    longest prefix that matches a valid station code.
    
    Args:
        col_name: Column name like "<station_code>_<variable>"
        valid_station_codes: Optional set of valid station codes to match against
    
    Returns:
        Extracted station code (normalized)
    """
    parts = col_name.split("_")
    
    if valid_station_codes is not None:
        # Try progressively longer prefixes until we find a match
        for i in range(1, len(parts) + 1):
            candidate = "_".join(parts[:i])
            normalized = normalize_station_code(candidate)
            if normalized in valid_station_codes:
                return normalized
        # If no match found, return the first part (fallback)
        return normalize_station_code(parts[0])
    else:
        # Without valid_station_codes, we can't determine the split point reliably
        # Use a heuristic: if it starts with "ARPAL_", take first two parts
        if len(parts) >= 2 and parts[0] == "ARPAL":
            return normalize_station_code("_".join(parts[:2]))
        # Otherwise, take first part
        return normalize_station_code(parts[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join aggregated ERA5-Land long dataset with BLH and ERA5 station datasets."
    )
    parser.add_argument(
        "--era5-long",
        type=Path,
        default=Path("data/era5-land/era5_land_timeseries_all_stations_long.csv"),
        help="Aggregated ERA5-Land long-format dataset.",
    )
    parser.add_argument(
        "--blh",
        type=Path,
        default=Path("data/data_blh/df_blh_stations.csv"),
        help="BLH station dataset (wide format, columns: valid_time, <station_code>_blh, ..., year).",
    )
    parser.add_argument(
        "--era5-extra",
        type=Path,
        default=Path("data/data_blh/df_era5_stations.csv"),
        help="ERA5 station dataset with additional variables at different levels (wide format).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/era5-land/era5_land_blh_joined_long.csv"),
        help="Output CSV path for the joined long-format dataset.",
    )
    return parser.parse_args()


def load_era5_long(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ERA5-Land long dataset not found: {path}")
    print(f"[LOAD] Aggregated ERA5-Land long dataset: {path}")
    df = pd.read_csv(path, parse_dates=["datetime"])
    # Ensure station_code is string for consistent joins
    if "station_code" in df.columns:
        df["station_code"] = df["station_code"].astype(str)
    return df


def melt_blh_wide_to_long(path: Path, valid_station_codes: Optional[Set[str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"BLH dataset not found: {path}")
    print(f"[LOAD] BLH stations dataset (wide): {path}")
    df = pd.read_csv(path, parse_dates=["valid_time"])

    id_cols: List[str] = ["valid_time"]
    # If there is a 'year' column, keep it as an ID (it can be useful for debug)
    if "year" in df.columns:
        id_cols.append("year")

    value_cols = [c for c in df.columns if c not in id_cols]

    print(f"       BLH value columns detected: {len(value_cols)}")
    
    # Extract station codes from column names and filter if needed
    if valid_station_codes is not None:
        blh_station_codes = set()
        filtered_value_cols = []
        for col in value_cols:
            if col.endswith("_blh"):
                station_code_raw = col.replace("_blh", "")
                # Normalize station code (convert EU codes to IT codes)
                station_code = normalize_station_code(station_code_raw)
                if station_code in valid_station_codes:
                    filtered_value_cols.append(col)
                    blh_station_codes.add(station_code)
                else:
                    print(f"       [FILTER] Dropping BLH column for station {station_code_raw} → {station_code} (not in main dataset)")
        
        if len(filtered_value_cols) < len(value_cols):
            print(f"       Filtered BLH columns: {len(filtered_value_cols)} kept, {len(value_cols) - len(filtered_value_cols)} dropped")
            value_cols = filtered_value_cols
        else:
            blh_station_codes = {col.replace("_blh", "") for col in value_cols}
        
        print(f"       BLH stations in dataset: {sorted(blh_station_codes)}")
    else:
        blh_station_codes_raw = {col.replace("_blh", "") for col in value_cols}
        blh_station_codes = {normalize_station_code(code) for code in blh_station_codes_raw}
        print(f"       BLH stations detected (after normalization): {sorted(blh_station_codes)}")

    df_long = df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="station_blh",
        value_name="blh",
    )

    # Extract station_code from "<code>_blh"
    df_long["station_code"] = df_long["station_blh"].str.replace("_blh", "", regex=False)
    df_long = df_long.drop(columns=["station_blh"])
    df_long = df_long.rename(columns={"valid_time": "datetime"})

    # Normalize station codes (convert EU codes to IT codes for APPA stations)
    df_long["station_code"] = df_long["station_code"].apply(normalize_station_code)
    df_long["station_code"] = df_long["station_code"].astype(str)
    df_long["datetime"] = pd.to_datetime(df_long["datetime"])

    # Drop rows where BLH is entirely missing
    before = len(df_long)
    df_long = df_long.dropna(subset=["blh"])
    after = len(df_long)
    print(f"       BLH long-format rows: {after:,} (dropped {before - after:,} all-NaN rows)")

    return df_long


def melt_era5_extra_wide_to_long(path: Path, valid_station_codes: Optional[Set[str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Extra ERA5 station dataset not found: {path}")
    print(f"[LOAD] Extra ERA5 stations dataset (wide): {path}")
    df = pd.read_csv(path, parse_dates=["valid_time"])

    id_cols: List[str] = ["valid_time"]
    # Some files may also have a 'year' or similar flag; keep them as ID columns
    for col in ("year",):
        if col in df.columns:
            id_cols.append(col)

    value_cols = [c for c in df.columns if c not in id_cols]
    print(f"       Extra ERA5 value columns detected: {len(value_cols)}")
    
    # Extract station codes from column names and filter if needed
    if valid_station_codes is not None:
        era5_extra_station_codes = set()
        filtered_value_cols = []
        for col in value_cols:
            # Column format: "<station_code>_<variable>" where station_code may contain underscores
            station_code = extract_station_code_from_column(col, valid_station_codes=valid_station_codes)
            if station_code in valid_station_codes:
                filtered_value_cols.append(col)
                era5_extra_station_codes.add(station_code)
            else:
                print(f"       [FILTER] Dropping ERA5-extra column {col} (station {station_code} not in main dataset)")
        
        if len(filtered_value_cols) < len(value_cols):
            print(f"       Filtered ERA5-extra columns: {len(filtered_value_cols)} kept, {len(value_cols) - len(filtered_value_cols)} dropped")
            value_cols = filtered_value_cols
        else:
            # Re-extract station codes from filtered columns (should all be valid now)
            era5_extra_station_codes = {extract_station_code_from_column(col, valid_station_codes=valid_station_codes) for col in value_cols if "_" in col}
        
        print(f"       ERA5-extra stations in dataset (after normalization): {sorted(era5_extra_station_codes)}")
    else:
        # Without valid_station_codes, use heuristic extraction
        era5_extra_station_codes = {extract_station_code_from_column(col) for col in value_cols if "_" in col}
        print(f"       ERA5-extra stations detected (after normalization): {sorted(era5_extra_station_codes)}")

    df_long = df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="station_var",
        value_name="value",
    )

    df_long = df_long.rename(columns={"valid_time": "datetime"})
    df_long["datetime"] = pd.to_datetime(df_long["datetime"])

    # Split "station_var" into station_code and variable name:
    #   e.g., "502604_t850" or "ARPAL_001_humidity_550" -> station_code="502604" or "ARPAL_001", var_name="t850" or "humidity_550"
    # Since station codes can contain underscores, we need to extract them properly
    if valid_station_codes is not None:
        # Use the extraction function to get station codes
        df_long["station_code"] = df_long["station_var"].apply(
            lambda x: extract_station_code_from_column(x, valid_station_codes=valid_station_codes)
        )
        # Extract variable name by removing the station code prefix
        def extract_var_name(col_name: str, station_code: str) -> str:
            # Remove the station code prefix (with underscore) from the column name
            prefix = station_code + "_"
            if col_name.startswith(prefix):
                return col_name[len(prefix):]
            # Fallback: try to find where variable starts
            parts = col_name.split("_")
            # If station code has underscores, skip those parts
            code_parts = station_code.split("_")
            if len(parts) > len(code_parts):
                return "_".join(parts[len(code_parts):])
            return col_name  # Shouldn't happen
        
        df_long["extra_var"] = df_long.apply(
            lambda row: extract_var_name(row["station_var"], row["station_code"]), axis=1
        )
    else:
        # Fallback: simple split (may not work correctly for codes with underscores)
        sv = df_long["station_var"].str.split("_", n=1, expand=True)
        station_code_raw = sv[0].astype(str)
        df_long["station_code"] = station_code_raw.apply(normalize_station_code)
        df_long["extra_var"] = sv[1].astype(str)
    
    df_long = df_long.drop(columns=["station_var"])

    # Pivot back to wide by variable per (datetime, station_code)
    df_pivot = df_long.pivot_table(
        index=["datetime", "station_code"],
        columns="extra_var",
        values="value",
        aggfunc="first",
    ).reset_index()

    print(f"       Extra ERA5 long-format records: {len(df_long):,}")
    print(f"       Extra ERA5 pivoted rows: {len(df_pivot):,}, variables: {len(df_pivot.columns) - 2}")

    return df_pivot


def interpolate_to_hourly(df: pd.DataFrame, datetime_col: str = "datetime", station_col: str = "station_code") -> pd.DataFrame:
    """
    Interpolate 3-hourly (or other non-hourly) data to hourly frequency using linear interpolation.
    
    This function:
    1. Detects the actual frequency of the data (by checking time differences)
    2. For each station, creates a complete hourly time series
    3. Interpolates missing hourly values using linear interpolation
    
    Args:
        df: DataFrame with datetime and station_code columns, plus data columns to interpolate
        datetime_col: Name of datetime column (default: "datetime")
        station_col: Name of station code column (default: "station_code")
    
    Returns:
        DataFrame with hourly frequency (interpolated where needed)
    """
    print(f"\n[INTERPOLATE] Checking data frequency and interpolating to hourly if needed...")
    
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    # Detect frequency by checking time differences
    # Sample a few stations to determine frequency
    sample_stations = df[station_col].unique()[:min(5, len(df[station_col].unique()))]
    time_diffs = []
    for station in sample_stations:
        station_data = df[df[station_col] == station].sort_values(datetime_col)
        if len(station_data) > 1:
            diffs = station_data[datetime_col].diff().dropna()
            # Get most common time difference (in hours)
            most_common_diff_hours = diffs.mode()[0].total_seconds() / 3600 if len(diffs.mode()) > 0 else None
            if most_common_diff_hours is not None:
                time_diffs.append(most_common_diff_hours)
    
    if time_diffs:
        detected_freq_hours = max(set(time_diffs), key=time_diffs.count)
        print(f"       Detected data frequency: {detected_freq_hours:.1f} hours")
        
        if detected_freq_hours == 1.0:
            print(f"       Data is already hourly - no interpolation needed")
            return df
    else:
        detected_freq_hours = 1.0
        print(f"       Could not detect frequency - assuming hourly")
        return df
    
    # Get data columns (exclude datetime and station_code)
    data_cols = [c for c in df.columns if c not in [datetime_col, station_col]]
    
    if not data_cols:
        print(f"       No data columns to interpolate")
        return df
    
    print(f"       Interpolating {len(data_cols)} variables for {df[station_col].nunique()} stations...")
    
    # Get global time range
    global_start = df[datetime_col].min()
    global_end = df[datetime_col].max()
    
    # Create complete hourly time series for each station
    all_stations_interpolated = []
    total_interpolated = 0
    
    for station_code in sorted(df[station_col].unique()):
        station_df = df[df[station_col] == station_code].copy().sort_values(datetime_col)
        
        # Create complete hourly time series for this station's date range
        station_start = station_df[datetime_col].min()
        station_end = station_df[datetime_col].max()
        
        # Round to hour boundaries
        station_start_hour = pd.Timestamp(station_start).replace(minute=0, second=0, microsecond=0)
        station_end_hour = pd.Timestamp(station_end).replace(minute=0, second=0, microsecond=0)
        
        hourly_times = pd.date_range(start=station_start_hour, end=station_end_hour, freq="h")
        
        # Create complete DataFrame with hourly timestamps
        complete_df = pd.DataFrame({datetime_col: hourly_times})
        complete_df[station_col] = station_code
        
        # Merge with existing data (only merge on datetime, then set station_code)
        # This avoids duplicate column issues
        complete_df = complete_df.merge(
            station_df[[datetime_col] + data_cols],
            on=datetime_col,
            how="left"
        )
        
        # Ensure station_code is correctly set
        complete_df[station_col] = station_code
        
        # Interpolate data columns using linear interpolation
        for col in data_cols:
            if col in complete_df.columns:
                # Use linear interpolation
                # For 3-hourly data, we need to fill 2 consecutive NaNs between each pair of values
                # Set limit to detected_freq_hours to ensure we can fill all gaps
                max_gap_hours = int(detected_freq_hours)  # Allow interpolation for gaps up to the detected frequency
                complete_df[col] = complete_df[col].interpolate(
                    method="linear",
                    limit_direction="both",
                    limit=max_gap_hours
                )
        
        # Count interpolated values
        before_count = station_df[data_cols].notna().sum().sum()
        after_count = complete_df[data_cols].notna().sum().sum()
        interpolated_count = after_count - before_count
        
        all_stations_interpolated.append(complete_df)
        total_interpolated += interpolated_count
    
    result_df = pd.concat(all_stations_interpolated, ignore_index=True)
    result_df = result_df.sort_values([station_col, datetime_col]).reset_index(drop=True)
    
    print(f"       Interpolated {total_interpolated:,} values")
    print(f"       Result: {len(result_df):,} rows (hourly frequency)")
    
    return result_df


def check_completeness_and_nans(df: pd.DataFrame) -> None:
    """
    Simple completeness / NaN report:
      - Expected hourly coverage per station for 2014-2024
      - NaN counts per column
    """
    print("\n" + "=" * 80)
    print("Completeness & NaN Check (Joined ERA5 + BLH + Extra ERA5)")
    print("=" * 80)

    if "datetime" not in df.columns or "station_code" not in df.columns:
        print("Dataset is missing 'datetime' or 'station_code' columns; skipping completeness check.")
        return

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["station_code"] = df["station_code"].astype(str)

    # Restrict to 2014-2024 range for completeness check (consistent with PM10 usage)
    expected_start = pd.Timestamp("2014-01-01 00:00:00")
    expected_end = pd.Timestamp("2024-12-31 23:00:00")
    expected_times = pd.date_range(start=expected_start, end=expected_end, freq="h")
    expected_count = len(expected_times)

    print(f"\nExpected completeness window: {expected_start} → {expected_end}")
    print(f"Expected hours per station in this window: {expected_count:,}")

    rows_in_window = df[(df["datetime"] >= expected_start) & (df["datetime"] <= expected_end)].copy()
    unique_stations = sorted(rows_in_window["station_code"].unique())
    print(f"Stations in completeness window: {len(unique_stations)}")

    summary_rows = []
    for sc in unique_stations:
        sub = rows_in_window[rows_in_window["station_code"] == sc]
        actual_times = pd.to_datetime(sub["datetime"]).sort_values().unique()
        actual_set = set(actual_times)
        actual_count = sum(1 for t in expected_times if t in actual_set)
        missing_count = expected_count - actual_count
        missing_pct = (missing_count / expected_count * 100) if expected_count > 0 else 0.0
        summary_rows.append(
            {
                "station_code": sc,
                "actual_hours": actual_count,
                "missing_hours": missing_count,
                "missing_pct": missing_pct,
            }
        )

    completeness_df = pd.DataFrame(summary_rows)
    print("\nPer-station completeness in 2014–2024 window (hours):")
    print(completeness_df.to_string(index=False))

    # NaN statistics per column
    print("\nNaN statistics per column:")
    data_cols = [c for c in df.columns if c != "datetime"]
    print(f"{'Column':<40s} {'NaNs':>12s} {'Non-NaNs':>15s} {'% NaN':>10s}")
    print("-" * 80)

    total_cells = len(df) * len(data_cols) if data_cols else 0
    total_nans = 0

    stats = []
    for col in data_cols:
        missing = df[col].isna().sum()
        non_missing = df[col].notna().sum()
        pct = (missing / len(df) * 100) if len(df) > 0 else 0.0
        total_nans += missing
        stats.append((col, missing, non_missing, pct))

    # Sort by number of NaNs descending
    stats.sort(key=lambda x: x[1], reverse=True)
    for col, missing, non_missing, pct in stats[:30]:
        print(f"{col:<40s} {missing:>12,} {non_missing:>15,} {pct:>9.2f}%")
    if len(stats) > 30:
        print(f"... ({len(stats) - 30} more columns)")

    if total_cells > 0:
        print("-" * 80)
        print(f"Total NaNs: {total_nans:,} / {total_cells:,} ({total_nans / total_cells * 100:.2f}%)")


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("Join ERA5-Land Long Dataset with BLH and Extra ERA5 Station Datasets")
    print("=" * 80)
    print(f"ERA5-Land long: {args.era5_long}")
    print(f"BLH stations:   {args.blh}")
    print(f"Extra ERA5:     {args.era5_extra}")
    print(f"Output file:    {args.output}")
    print("=" * 80)

    # Load all components
    era5_long = load_era5_long(args.era5_long)
    
    # Get valid station codes from main ERA5-long dataset
    valid_station_codes = set(era5_long["station_code"].astype(str).unique())
    print(f"\n[INFO] Main ERA5-Land dataset contains {len(valid_station_codes)} stations:")
    print(f"       {sorted(valid_station_codes)}")
    
    # Note: BLH and ERA5-extra datasets may use EU codes (IT1930A, etc.) for APPA stations.
    # These will be automatically normalized to IT codes (402212, etc.) during processing.
    print(f"\n[INFO] Normalizing station codes: EU codes (IT1930A, etc.) → IT codes (402212, etc.)")
    print(f"       Mapping: {EU_TO_IT_CODE_MAPPING}")
    
    # Filter BLH and ERA5-extra to only include stations in main dataset
    blh_long = melt_blh_wide_to_long(args.blh, valid_station_codes=valid_station_codes)
    era5_extra = melt_era5_extra_wide_to_long(args.era5_extra, valid_station_codes=valid_station_codes)
    
    # Interpolate ERA5-extra to hourly frequency if needed (e.g., if it's 3-hourly)
    era5_extra = interpolate_to_hourly(era5_extra, datetime_col="datetime", station_col="station_code")
    
    # Print summary of all stations involved
    blh_stations = set(blh_long["station_code"].astype(str).unique())
    era5_extra_stations = set(era5_extra["station_code"].astype(str).unique())
    
    print(f"\n[INFO] Station coverage summary:")
    print(f"       ERA5-Land main: {len(valid_station_codes)} stations")
    print(f"       BLH:             {len(blh_stations)} stations")
    print(f"       ERA5-extra:      {len(era5_extra_stations)} stations")
    print(f"\n       All stations in final joined dataset:")
    all_stations = sorted(valid_station_codes | blh_stations | era5_extra_stations)
    print(f"       {all_stations}")

    print("\n[JOIN] Joining ERA5-Land with BLH (on datetime, station_code)...")
    merged = pd.merge(
        era5_long,
        blh_long[["datetime", "station_code", "blh"]],
        on=["datetime", "station_code"],
        how="left",
    )

    print("[JOIN] Joining with extra ERA5 variables (on datetime, station_code)...")
    merged = pd.merge(
        merged,
        era5_extra,
        on=["datetime", "station_code"],
        how="left",
    )

    print(f"\nJoined dataset shape: {merged.shape[0]:,} rows × {merged.shape[1]} columns")

    # Check completeness and NaNs
    check_completeness_and_nans(merged)

    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"\nSaved joined dataset to: {args.output}")
    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()


