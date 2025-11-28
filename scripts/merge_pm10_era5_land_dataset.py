#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge all PM10 air pollution data and ERA5-Land weather data into a single CSV.

This script creates a wide-format dataset where:
- The key column is `datetime` (hourly timestamps)
- PM10 measurements from all stations are columns: `pm10_<region>_<station_code>`
- ERA5-Land weather variables for each station are columns: `<var>_<region>_<station_code>`
  where <var> is one of: t2m, sp, tp, ssrd, u10, v10

Regions included:
- Veneto (ARPAV)
- Lombardia (ARPAL)
- Trentino (APPA)
- Alto-Adige

Usage:
    python scripts/merge_pm10_era5_land_dataset.py
    python scripts/merge_pm10_era5_land_dataset.py --output data/merged_pm10_era5_land.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge PM10 and ERA5-Land data into a single wide-format CSV."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/merged_pm10_era5_land.csv"),
        help="Output CSV path for the merged dataset.",
    )
    parser.add_argument(
        "--appa-pm10",
        type=Path,
        default=Path("data/appa-data/merged_pm10_hourly_curated.csv"),
        help="Path to APPA (Trentino) curated PM10 dataset.",
    )
    parser.add_argument(
        "--arpav-pm10",
        type=Path,
        default=Path("data/arpav/PM10/merged_pm10_hourly_curated.csv"),
        help="Path to ARPAV (Veneto) curated PM10 dataset.",
    )
    parser.add_argument(
        "--arpal-pm10",
        type=Path,
        default=Path("data/arpal/PM10/merged_pm10_hourly_curated.csv"),
        help="Path to ARPAL (Lombardia) curated PM10 dataset.",
    )
    parser.add_argument(
        "--altoadige-pm10",
        type=Path,
        default=Path("data/altoadige/PM10/merged_pm10_hourly_curated.csv"),
        help="Path to Alto-Adige curated PM10 dataset.",
    )
    parser.add_argument(
        "--era5-land",
        type=Path,
        default=Path("data/era5-land/era5_land_timeseries_all_stations_long.csv"),
        help="Path to aggregated ERA5-Land dataset.",
    )
    return parser.parse_args()


def load_pm10_data(file_path: Path, region: str) -> pd.DataFrame:
    """
    Load PM10 data from a curated dataset file.
    
    Args:
        file_path: Path to the curated PM10 CSV file
        region: Region name (Veneto, Lombardia, Trentino, Alto-Adige)
    
    Returns:
        DataFrame with columns: datetime, station_code, pm10
    """
    if not file_path.exists():
        print(f"[WARN] PM10 file not found: {file_path}", file=sys.stderr)
        return pd.DataFrame()
    
    print(f"[LOAD] Loading PM10 data from {region}: {file_path.name}")
    df = pd.read_csv(file_path, parse_dates=["datetime"])
    
    # Ensure we have the required columns
    required_cols = ["datetime", "station_code", "pm10"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in {file_path}: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )
    
    # Select only required columns and add region
    df = df[["datetime", "station_code", "pm10"]].copy()
    df["region"] = region
    
    # Remove any rows with missing datetime or station_code
    df = df.dropna(subset=["datetime", "station_code"])
    
    # Filter to 2014-2024 range (inclusive, drop 2025)
    df = df[(df["datetime"] >= "2014-01-01") & (df["datetime"] < "2025-01-01")].copy()
    
    # Convert station_code to string for consistency
    df["station_code"] = df["station_code"].astype(str)
    
    print(f"       Loaded {len(df):,} rows, {df['station_code'].nunique()} stations")
    if len(df) > 0:
        print(f"       Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    return df


def load_era5_land_data(file_path: Path) -> pd.DataFrame:
    """
    Load ERA5-Land weather data.
    
    Args:
        file_path: Path to the aggregated ERA5-Land CSV file
    
    Returns:
        DataFrame with columns: datetime, region, station_code, and weather variables
    """
    if not file_path.exists():
        raise FileNotFoundError(f"ERA5-Land file not found: {file_path}")
    
    print(f"[LOAD] Loading ERA5-Land data: {file_path.name}")
    df = pd.read_csv(file_path, parse_dates=["datetime"])
    
    # Required columns
    required_cols = ["datetime", "region", "station_code"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in {file_path}: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )
    
    # Weather variables
    weather_vars = ["t2m", "sp", "tp", "ssrd", "u10", "v10"]
    available_vars = [v for v in weather_vars if v in df.columns]
    
    # Select columns
    cols = ["datetime", "region", "station_code"] + available_vars
    df = df[cols].copy()
    
    # Remove rows with missing datetime or station_code
    df = df.dropna(subset=["datetime", "station_code"])
    
    # Filter to 2014-2024 range (inclusive, drop 2025)
    df = df[(df["datetime"] >= "2014-01-01") & (df["datetime"] < "2025-01-01")].copy()
    
    # Convert station_code to string for consistency
    df["station_code"] = df["station_code"].astype(str)
    
    print(f"       Loaded {len(df):,} rows, {df['station_code'].nunique()} stations")
    if len(df) > 0:
        print(f"       Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"       Weather variables: {', '.join(available_vars)}")
    return df


def pivot_pm10_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert PM10 data from long format to wide format.
    
    Args:
        df: DataFrame with columns: datetime, region, station_code, pm10
    
    Returns:
        DataFrame with datetime as index and columns: pm10_<region>_<station_code>
    """
    if df.empty:
        return pd.DataFrame()
    
    # Create column name: pm10_<region>_<station_code>
    # Normalize region name: lowercase, replace spaces and hyphens with underscores
    region_normalized = df["region"].str.lower().str.replace(" ", "_").str.replace("-", "_")
    df["col_name"] = "pm10_" + region_normalized + "_" + df["station_code"]
    
    # Pivot to wide format
    df_wide = df.pivot_table(
        index="datetime",
        columns="col_name",
        values="pm10",
        aggfunc="first"  # In case of duplicates, take first
    )
    
    return df_wide


def pivot_era5_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert ERA5-Land data from long format to wide format.
    
    Args:
        df: DataFrame with columns: datetime, region, station_code, and weather variables
    
    Returns:
        DataFrame with datetime as index and columns: <var>_<region>_<station_code>
    """
    if df.empty:
        return pd.DataFrame()
    
    # Get weather variables (exclude datetime, region, station_code)
    weather_vars = [c for c in df.columns if c not in ["datetime", "region", "station_code"]]
    
    # Melt to long format first
    df_melted = df.melt(
        id_vars=["datetime", "region", "station_code"],
        value_vars=weather_vars,
        var_name="variable",
        value_name="value"
    )
    
    # Create column name: <var>_<region>_<station_code>
    # Normalize region name: lowercase, replace spaces and hyphens with underscores
    region_normalized = df_melted["region"].str.lower().str.replace(" ", "_").str.replace("-", "_")
    df_melted["col_name"] = (
        df_melted["variable"] + "_" +
        region_normalized + "_" +
        df_melted["station_code"]
    )
    
    # Pivot to wide format
    df_wide = df_melted.pivot_table(
        index="datetime",
        columns="col_name",
        values="value",
        aggfunc="first"  # In case of duplicates, take first
    )
    
    return df_wide


def normalize_region_name(region: str) -> str:
    """
    Normalize region names for consistent column naming.
    
    Args:
        region: Region name
    
    Returns:
        Normalized region name (lowercase, spaces and hyphens replaced with underscores)
    """
    return region.lower().replace(" ", "_").replace("-", "_")


def pretty_print_stations(df: pd.DataFrame, data_type: str) -> None:
    """
    Pretty print all station codes grouped by region.
    
    Args:
        df: DataFrame with 'region' and 'station_code' columns
        data_type: Type of data (e.g., "PM10", "ERA5-Land")
    """
    if df.empty or "region" not in df.columns or "station_code" not in df.columns:
        return
    
    print(f"\n   {data_type} Stations:")
    print("   " + "-" * 76)
    
    stations_by_region = df.groupby("region")["station_code"].unique()
    for region in sorted(stations_by_region.index):
        stations = sorted(stations_by_region[region])
        print(f"   {region:15s}: {len(stations):2d} stations")
        # Print stations in columns (3 per line)
        for i in range(0, len(stations), 3):
            line_stations = stations[i:i+3]
            print(f"      {'  '.join(f'{s:15s}' for s in line_stations)}")
    
    total_stations = df["station_code"].nunique()
    print(f"   {'Total':15s}: {total_stations:2d} unique stations")
    print("   " + "-" * 76)


def pretty_print_columns(columns: List[str], max_per_line: int = 3) -> None:
    """
    Pretty print column names in a formatted way.
    
    Args:
        columns: List of column names
        max_per_line: Maximum number of columns to print per line
    """
    print("\n   All Columns:")
    print("   " + "-" * 76)
    
    # Group columns by prefix (pm10_ or weather variable)
    pm10_cols = [c for c in columns if c.startswith("pm10_")]
    weather_cols = [c for c in columns if c not in ["datetime"] + pm10_cols]
    
    print(f"   datetime (1 column)")
    print(f"   PM10 columns ({len(pm10_cols)} columns):")
    for i in range(0, len(pm10_cols), max_per_line):
        line_cols = pm10_cols[i:i+max_per_line]
        print(f"      {'  '.join(f'{c:30s}' for c in line_cols)}")
    
    print(f"\n   ERA5-Land columns ({len(weather_cols)} columns):")
    for i in range(0, len(weather_cols), max_per_line):
        line_cols = weather_cols[i:i+max_per_line]
        print(f"      {'  '.join(f'{c:30s}' for c in line_cols)}")
    
    print("   " + "-" * 76)


def check_data_completeness(df: pd.DataFrame) -> None:
    """
    Check that all hourly records are present for 2014-2024 and check for missing values.
    
    Args:
        df: DataFrame with 'datetime' column
    """
    print("\n   Checking data completeness...")
    
    # Expected datetime range: 2014-01-01 00:00:00 to 2024-12-31 23:00:00 (inclusive)
    expected_start = pd.Timestamp("2014-01-01 00:00:00")
    expected_end = pd.Timestamp("2024-12-31 23:00:00")
    
    # Generate expected hourly timestamps
    expected_datetimes = pd.date_range(start=expected_start, end=expected_end, freq="h")
    expected_count = len(expected_datetimes)
    
    print(f"   Expected datetime range: {expected_start} to {expected_end}")
    print(f"   Expected number of hourly records: {expected_count:,}")
    
    # Check actual datetime range
    actual_start = df["datetime"].min()
    actual_end = df["datetime"].max()
    actual_count = len(df)
    
    print(f"   Actual datetime range: {actual_start} to {actual_end}")
    print(f"   Actual number of records: {actual_count:,}")
    
    # Check if all expected datetimes are present
    actual_datetimes_set = set(df["datetime"])
    expected_datetimes_set = set(expected_datetimes)
    
    missing_datetimes = expected_datetimes_set - actual_datetimes_set
    extra_datetimes = actual_datetimes_set - expected_datetimes_set
    
    if missing_datetimes:
        print(f"\n   ⚠ WARNING: {len(missing_datetimes):,} missing hourly records")
        if len(missing_datetimes) <= 10:
            print(f"   Missing datetimes: {sorted(missing_datetimes)}")
        else:
            print(f"   First 10 missing datetimes: {sorted(list(missing_datetimes))[:10]}")
    else:
        print(f"\n   ✓ All expected hourly records are present")
    
    if extra_datetimes:
        print(f"   ⚠ WARNING: {len(extra_datetimes):,} extra records outside expected range")
        if len(extra_datetimes) <= 10:
            print(f"   Extra datetimes: {sorted(extra_datetimes)}")
    
    # Check for missing values (NaN)
    print(f"\n   Checking for missing values (NaN)...")
    data_cols = [c for c in df.columns if c != "datetime"]
    
    missing_stats = []
    for col in data_cols:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100 if len(df) > 0 else 0
        missing_stats.append({
            "column": col,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "non_missing_count": df[col].notna().sum()
        })
    
    # Sort by missing count (descending)
    missing_stats.sort(key=lambda x: x["missing_count"], reverse=True)
    
    print(f"   Missing value statistics:")
    print(f"   {'Column':<40s} {'Missing':>12s} {'Non-Missing':>15s} {'% Missing':>12s}")
    print("   " + "-" * 80)
    
    total_missing = sum(s["missing_count"] for s in missing_stats)
    total_values = len(df) * len(data_cols)
    
    for stat in missing_stats[:20]:  # Show top 20 columns with most missing values
        print(f"   {stat['column']:<40s} {stat['missing_count']:>12,} {stat['non_missing_count']:>15,} {stat['missing_pct']:>11.2f}%")
    
    if len(missing_stats) > 20:
        print(f"   ... ({len(missing_stats) - 20} more columns)")
    
    print("   " + "-" * 80)
    print(f"   Total missing values: {total_missing:,} / {total_values:,} ({total_missing/total_values*100:.2f}%)")
    
    # Summary by data type
    pm10_cols = [c for c in data_cols if c.startswith("pm10_")]
    era5_cols = [c for c in data_cols if c not in pm10_cols]
    
    if pm10_cols:
        pm10_missing = sum(df[c].isna().sum() for c in pm10_cols)
        pm10_total = len(df) * len(pm10_cols)
        print(f"\n   PM10 columns: {pm10_missing:,} / {pm10_total:,} missing ({pm10_missing/pm10_total*100:.2f}%)")
    
    if era5_cols:
        era5_missing = sum(df[c].isna().sum() for c in era5_cols)
        era5_total = len(df) * len(era5_cols)
        print(f"   ERA5-Land columns: {era5_missing:,} / {era5_total:,} missing ({era5_missing/era5_total*100:.2f}%)")


def main() -> None:
    args = parse_args()
    
    print("=" * 80)
    print("Merge PM10 and ERA5-Land Dataset")
    print("=" * 80)
    print(f"Output file: {args.output}")
    print("=" * 80)
    
    # ============================================================================
    # Step 1: Load PM10 data from all regions
    # ============================================================================
    print("\n1. Loading PM10 data from all regions...")
    
    pm10_datasets = {
        "Trentino": args.appa_pm10,
        "Veneto": args.arpav_pm10,
        "Lombardia": args.arpal_pm10,
        "Alto-Adige": args.altoadige_pm10,
    }
    
    pm10_dfs = []
    for region, file_path in pm10_datasets.items():
        df = load_pm10_data(file_path, region)
        if not df.empty:
            pm10_dfs.append(df)
    
    if not pm10_dfs:
        print("[ERROR] No PM10 data loaded!", file=sys.stderr)
        sys.exit(1)
    
    # Combine all PM10 data
    pm10_all = pd.concat(pm10_dfs, ignore_index=True)
    print(f"\n   Total PM10 rows: {len(pm10_all):,}")
    print(f"   Total PM10 stations: {pm10_all['station_code'].nunique()}")
    if len(pm10_all) > 0:
        print(f"   Date range: {pm10_all['datetime'].min()} to {pm10_all['datetime'].max()}")
    
    # Pretty print PM10 stations
    pretty_print_stations(pm10_all, "PM10")
    
    # ============================================================================
    # Step 2: Load ERA5-Land data
    # ============================================================================
    print("\n2. Loading ERA5-Land weather data...")
    era5_df = load_era5_land_data(args.era5_land)
    
    if era5_df.empty:
        print("[ERROR] No ERA5-Land data loaded!", file=sys.stderr)
        sys.exit(1)
    
    if len(era5_df) > 0:
        print(f"   Date range: {era5_df['datetime'].min()} to {era5_df['datetime'].max()}")
    
    # Pretty print ERA5-Land stations
    pretty_print_stations(era5_df, "ERA5-Land")
    
    # ============================================================================
    # Step 3: Convert to wide format
    # ============================================================================
    print("\n3. Converting to wide format...")
    
    print("   Pivoting PM10 data...")
    pm10_wide = pivot_pm10_to_wide(pm10_all)
    print(f"      Created {len(pm10_wide.columns)} PM10 columns")
    
    print("   Pivoting ERA5-Land data...")
    era5_wide = pivot_era5_to_wide(era5_df)
    print(f"      Created {len(era5_wide.columns)} ERA5-Land columns")
    
    # ============================================================================
    # Step 4: Merge on datetime
    # ============================================================================
    print("\n4. Merging datasets on datetime...")
    
    # Get all unique datetimes from both datasets
    all_datetimes = pd.Index(
        pm10_wide.index.union(era5_wide.index)
    ).sort_values()
    
    print(f"   Total unique datetimes: {len(all_datetimes):,}")
    print(f"   Date range: {all_datetimes.min()} to {all_datetimes.max()}")
    
    # Merge on datetime (outer join to keep all timestamps)
    merged = pd.merge(
        pm10_wide,
        era5_wide,
        left_index=True,
        right_index=True,
        how="outer",
        sort=True
    )
    
    # Reset index to make datetime a column
    merged = merged.reset_index()
    merged = merged.rename(columns={"index": "datetime"})
    
    # Sort by datetime
    merged = merged.sort_values("datetime").reset_index(drop=True)
    
    print(f"   Merged dataset shape: {merged.shape[0]:,} rows × {merged.shape[1]} columns")
    
    # ============================================================================
    # Step 5: Reorder columns (datetime first, then PM10, then ERA5-Land)
    # ============================================================================
    print("\n5. Reordering columns...")
    
    # Get column names
    pm10_cols = [c for c in merged.columns if c.startswith("pm10_")]
    era5_cols = [c for c in merged.columns if c not in ["datetime"] + pm10_cols]
    
    # Sort columns within each group
    pm10_cols.sort()
    era5_cols.sort()
    
    # Reorder: datetime first, then PM10, then ERA5-Land
    column_order = ["datetime"] + pm10_cols + era5_cols
    merged = merged[column_order]
    
    print(f"   Column order: datetime ({len(pm10_cols)} PM10 columns, {len(era5_cols)} ERA5-Land columns)")
    
    # Pretty print all columns
    pretty_print_columns(column_order)
    
    # ============================================================================
    # Step 6: Filter to 2014-2024 range and check completeness
    # ============================================================================
    print("\n6. Filtering to 2014-2024 range and checking completeness...")
    
    # Ensure datetime is datetime type
    merged["datetime"] = pd.to_datetime(merged["datetime"])
    
    # Filter to 2014-2024 range (inclusive, drop 2025)
    filter_start = pd.Timestamp("2014-01-01 00:00:00")
    filter_end = pd.Timestamp("2025-01-01 00:00:00")
    merged = merged[
        (merged["datetime"] >= filter_start) & 
        (merged["datetime"] < filter_end)
    ].copy()
    
    print(f"   Filtered dataset shape: {merged.shape[0]:,} rows × {merged.shape[1]} columns")
    
    # Check data completeness
    check_data_completeness(merged)
    
    # ============================================================================
    # Step 7: Save output
    # ============================================================================
    print("\n7. Saving merged dataset...")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    merged.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved merged dataset to: {output_path}")
    print(f"  Shape: {merged.shape[0]:,} rows × {merged.shape[1]} columns")
    print(f"  PM10 stations: {len(pm10_cols)}")
    print(f"  ERA5-Land stations: {len(set(c.split('_')[2] for c in era5_cols if '_' in c))}")
    print(f"  Date range: {merged['datetime'].min()} to {merged['datetime'].max()}")


if __name__ == "__main__":
    main()

