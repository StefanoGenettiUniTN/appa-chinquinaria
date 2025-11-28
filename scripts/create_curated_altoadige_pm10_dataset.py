"""
Script to create a curated PM10 dataset from the merged Alto-Adige CSV.

This script is analogous in structure to:
  - scripts/create_curated_arpal_pm10_dataset.py
  - scripts/create_curated_pm10_dataset.py
  - scripts/create_curated_appa_pm10_dataset.py

Operations:
1. Load merged Alto-Adige PM10 dataset for all stations (2008–2024)
2. Restrict to years 2014–2024 (inclusive)
3. Drop stations AB3 and CR2 (they have missing years)
4. For each remaining station:
   - Build a complete hourly time series over its observed span
   - Linearly interpolate all gaps of length <= 6 hours
5. Build a full 2014–2024 hourly grid for all remaining stations
6. For all remaining missing values (longer gaps and unobserved periods),
   train a simple multi-station regression model on the whole dataset
   and fill gaps using regression predictions (no distance-weighted copying).
7. Save two curated datasets:
   - merged_pm10_hourly_curated.csv (with interpolation metadata)
   - merged_pm10_hourly_curated_no_interp_metadata.csv (without metadata)

The final curated dataset is intended to have no gaps in 2014–2024 for
the remaining stations and to track whether each value is original,
linearly interpolated, or regression-filled.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path to import from chinquinaria if needed
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_dms_to_decimal(dms: str | float | int | None) -> float | None:
    """
    Convert a DMS string like '11 20 30.6' or '46 28 56,3' to decimal degrees.

    Returns None if parsing fails.
    """
    if dms is None or (isinstance(dms, float) and np.isnan(dms)):
        return None

    s = str(dms).strip()
    if not s:
        return None

    # Replace common separators / odd characters
    s = s.replace("\xa0", " ")  # non-breaking space
    s = s.replace(",", ".")
    parts = [p for p in s.split() if p]
    if len(parts) < 3:
        return None

    try:
        deg = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        sign = 1.0
        if deg < 0:
            sign = -1.0
            deg = abs(deg)
        decimal = sign * (deg + minutes / 60.0 + seconds / 3600.0)
        return decimal
    except Exception:
        return None


def find_contiguous_missing_periods(actual_times_set: set[pd.Timestamp], expected_times: pd.DatetimeIndex):
    """
    Find all contiguous missing periods and return their start/end indices.

    Args:
        actual_times_set: Set of actual measurement timestamps
        expected_times: Series of expected hourly timestamps

    Returns:
        List of tuples: (start_idx, end_idx, length_in_hours)
    """
    missing_periods: list[tuple[int, int, int]] = []
    current_start_idx: int | None = None

    for i, t in enumerate(expected_times):
        if t not in actual_times_set:
            if current_start_idx is None:
                current_start_idx = i
        else:
            if current_start_idx is not None:
                length = i - current_start_idx
                missing_periods.append((current_start_idx, i - 1, length))
                current_start_idx = None

    # Handle case where series ends with missing values
    if current_start_idx is not None:
        length = len(expected_times) - current_start_idx
        missing_periods.append((current_start_idx, len(expected_times) - 1, length))

    return missing_periods


def interpolate_station_data(station_df: pd.DataFrame, max_gap_hours: int = 6):
    """
    Interpolate gaps of length <= max_gap_hours for a single station.

    Args:
        station_df: DataFrame with columns ['datetime', 'pm10', 'station_code', 'station_name']
        max_gap_hours: Maximum gap length to interpolate (inclusive)

    Returns:
        complete_df: DataFrame with continuous hourly series for this station
        interpolated_count: number of hours in gaps flagged for interpolation
        actual_interpolated: number of values actually filled by interpolation
    """
    if station_df.empty:
        return station_df.copy(), 0, 0

    # Get actual measurement times
    actual_times = pd.to_datetime(station_df["datetime"]).sort_values().unique()
    station_start = actual_times.min()
    station_end = actual_times.max()

    # Cover full years spanned by this station
    year_start = pd.Timestamp(station_start.year, 1, 1, 0, 0, 0)
    year_end = pd.Timestamp(station_end.year, 12, 31, 23, 0, 0)
    if station_start.year != station_end.year:
        range_start = pd.Timestamp(
            station_start.year, station_start.month, station_start.day, 0, 0, 0
        )
        range_end = pd.Timestamp(station_end.year, station_end.month, station_end.day, 23, 0, 0)
    else:
        range_start = year_start
        range_end = year_end

    # Create complete hourly time series
    expected_times = pd.date_range(start=range_start, end=range_end, freq="h")

    # Base frame with all timestamps
    complete_df = pd.DataFrame({"datetime": expected_times})

    station_df_sorted = station_df.sort_values("datetime").reset_index(drop=True)
    station_df_sorted["datetime"] = pd.to_datetime(station_df_sorted["datetime"])

    complete_df = complete_df.merge(
        station_df_sorted[["datetime", "pm10", "station_code", "station_name"]],
        on="datetime",
        how="left",
    )

    complete_df["station_code"] = complete_df["station_code"].ffill().bfill()
    complete_df["station_name"] = complete_df["station_name"].ffill().bfill()

    # Identify missing periods
    actual_times_set = set(actual_times)
    missing_periods = find_contiguous_missing_periods(actual_times_set, expected_times)

    short_gaps: list[tuple[int, int, int]] = []
    long_gap_indices: set[int] = set()

    for start_idx, end_idx, length in missing_periods:
        if length <= max_gap_hours:
            if start_idx > 0 and end_idx < len(complete_df) - 1:
                before_val = complete_df.loc[start_idx - 1, "pm10"]
                after_val = complete_df.loc[end_idx + 1, "pm10"]
                if (before_val is not None and not pd.isna(before_val)) and (
                    after_val is not None and not pd.isna(after_val)
                ):
                    short_gaps.append((start_idx, end_idx, length))
        else:
            for idx in range(start_idx, end_idx + 1):
                long_gap_indices.add(idx)

    pm10_series = complete_df["pm10"].copy()

    # Track interpolation method
    complete_df["interpolation_method"] = "actual"

    # Mark long gaps with sentinel
    for idx in long_gap_indices:
        pm10_series.loc[idx] = -999999

    # Interpolate linearly for allowed gaps
    pm10_series = pm10_series.interpolate(
        method="linear", limit_direction="both", limit=max_gap_hours
    )

    # Mark interpolated values in short gaps
    for start_idx, end_idx, _ in short_gaps:
        for idx in range(start_idx, end_idx + 1):
            if not pd.isna(pm10_series.loc[idx]):
                complete_df.loc[idx, "interpolation_method"] = "linear"

    # Restore long gaps to NaN
    pm10_series[pm10_series == -999999] = np.nan
    complete_df["pm10"] = pm10_series

    # Counters
    interpolated_count = sum(length for _, _, length in short_gaps)
    actual_interpolated = 0
    for start_idx, end_idx, _ in short_gaps:
        gap_values = pm10_series.loc[start_idx:end_idx]
        actual_interpolated += gap_values.notna().sum()

    return complete_df, interpolated_count, actual_interpolated


def regression_fill(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform regression-based interpolation for remaining missing values.

    A separate linear regression model is trained for each target station,
    using all other stations as predictors on the full 2014–2024 dataset.
    Only currently-missing values are replaced by regression predictions.

    Args:
        result_df: DataFrame with full 2014–2024 hourly grid, may contain NaN pm10

    Returns:
        Updated result_df with additional interpolated values and method labels.
    """
    df = result_df.copy()

    # Ensure interpolation columns exist
    if "interpolation_method" not in df.columns:
        df["interpolation_method"] = None
    if "interpolation_confidence" not in df.columns:
        df["interpolation_confidence"] = np.nan

    # Wide matrix: datetime × station_code
    wide = df.pivot_table(
        index="datetime",
        columns="station_code",
        values="pm10",
        aggfunc="first",
    )

    station_codes = list(wide.columns)
    predictions_per_station: dict[str, pd.Series] = {
        code: pd.Series(index=wide.index, dtype="float64") for code in station_codes
    }

    for target_code in station_codes:
        y = wide[target_code]
        X = wide.drop(columns=[target_code])

        # Training mask: target observed and at least one predictor available
        valid_y = y.notna()
        has_feature = X.notna().any(axis=1)
        train_mask = valid_y & has_feature

        if train_mask.sum() < 50:
            # Not enough data to train a meaningful regressor
            continue

        X_train = X[train_mask].copy()
        y_train = y[train_mask].astype(float).values

        # Fill NaNs in predictors with column means on training set
        col_means = X_train.mean(axis=0, skipna=True)
        X_train = X_train.fillna(col_means)

        # Design matrix with intercept
        X_mat = np.c_[np.ones(len(X_train)), X_train.values]
        try:
            beta, *_ = np.linalg.lstsq(X_mat, y_train, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # Prediction mask: target missing but at least one predictor is available
        pred_mask = (~valid_y) & has_feature
        if pred_mask.sum() == 0:
            continue

        X_pred = X[pred_mask].copy()
        X_pred = X_pred.fillna(col_means)
        Xp_mat = np.c_[np.ones(len(X_pred)), X_pred.values]
        y_pred = Xp_mat @ beta

        predictions_per_station[target_code].loc[pred_mask] = y_pred

    # Build long-form predictions DataFrame
    pred_frames = []
    for code, series in predictions_per_station.items():
        s = series.dropna()
        if s.empty:
            continue
        tmp = s.to_frame(name="pm10_reg")
        tmp["station_code"] = code
        tmp = tmp.reset_index().rename(columns={"index": "datetime"})
        pred_frames.append(tmp)

    if not pred_frames:
        return df

    pred_df = pd.concat(pred_frames, ignore_index=True)
    pred_df["datetime"] = pd.to_datetime(pred_df["datetime"])
    pred_df["station_code"] = pred_df["station_code"].astype(str)

    # Merge predictions into original long df
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["station_code"] = df["station_code"].astype(str)

    df = df.merge(
        pred_df,
        on=["datetime", "station_code"],
        how="left",
    )

    # Apply regression only where pm10 is currently missing and prediction is available
    use_reg = df["pm10"].isna() & df["pm10_reg"].notna()
    n_reg = int(use_reg.sum())

    if n_reg > 0:
        df.loc[use_reg, "pm10"] = df.loc[use_reg, "pm10_reg"]
        df.loc[use_reg, "interpolation_method"] = "regression"
        # Optional: could set interpolation_confidence based on model diagnostics

    # Drop helper column
    df = df.drop(columns=["pm10_reg"])

    print(f"\n   Regression-based interpolation filled {n_reg:,} additional values")
    return df


def plot_station_timeseries(result_df: pd.DataFrame, plots_dir: Path) -> None:
    """
    Generate per-station timeseries plots showing:
      - Original data (actual)
      - Linearly interpolated segments
      - Regression-filled segments
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    method_colors = {
        "actual": "black",
        "linear": "blue",
        "regression": "red",
    }

    for station_code, station_df in result_df.groupby("station_code"):
        station_df = station_df.sort_values("datetime")

        plt.figure(figsize=(14, 4))
        for method, color in method_colors.items():
            mask = station_df["interpolation_method"] == method
            if not mask.any():
                continue
            y = station_df["pm10"].where(mask)
            plt.plot(
                station_df["datetime"],
                y,
                color=color,
                linewidth=0.8,
                label=method,
            )

        plt.title(f"Alto-Adige PM10 gap filling — station {station_code}")
        plt.xlabel("Datetime")
        plt.ylabel("PM10 [µg/m³]")
        plt.legend()
        plt.tight_layout()

        out_path = plots_dir / f"altoadige_pm10_gap_filling__{station_code}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()


def create_curated_altoadige_dataset(
    input_file: Path,
    output_file: Path | None = None,
    max_gap_hours: int = 6,
    start_year: int = 2014,
    end_year: int = 2024,
):
    """
    Main function to create curated Alto-Adige PM10 dataset.

    Args:
        input_file: Path to merged_pm10_2008_2024.csv
        output_file: Optional base output file path (CSV with interpolation metadata)
        max_gap_hours: Maximum gap length to interpolate linearly (inclusive)
        start_year: First year to keep (inclusive)
        end_year: Last year to keep (inclusive)
    """
    print("=" * 80)
    print("Alto-Adige PM10 Dataset Curation")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Step 1: Load data
    # -------------------------------------------------------------------------
    print(f"\n1. Loading data from: {input_file}")
    df = pd.read_csv(input_file, parse_dates=["datetime"])
    print(f"   Loaded {len(df):,} rows")
    print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    df["station_code"] = df["station_code"].astype(str)

    # Build station metadata with decimal coordinates
    station_meta = (
        df.groupby("station_code")
        .agg(
            {
                "address": "first",
                "altitude_m": "first",
                "lon_dms": "first",
                "lat_dms": "first",
            }
        )
        .reset_index()
    )
    station_meta["station_name"] = station_meta["station_code"]
    station_meta["latitude"] = station_meta["lat_dms"].apply(parse_dms_to_decimal)
    station_meta["longitude"] = station_meta["lon_dms"].apply(parse_dms_to_decimal)

    # -------------------------------------------------------------------------
    # Step 2: Restrict to desired years and drop AB3, CR2
    # -------------------------------------------------------------------------
    print(f"\n2. Restricting to years {start_year}-{end_year} and dropping AB3, CR2...")
    df["year"] = df["datetime"].dt.year
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()
    print(f"   After year filter: {len(df):,} rows")

    # Drop AB3 and CR2
    stations_to_drop = ["AB3", "CR2"]
    rows_before = len(df)
    df = df[~df["station_code"].isin(stations_to_drop)].copy()
    rows_after = len(df)
    print(
        f"   Dropped stations {stations_to_drop}: {rows_before - rows_after:,} rows removed"
    )

    # Update metadata
    station_meta = station_meta[~station_meta["station_code"].isin(stations_to_drop)].copy()

    unique_stations = sorted(df["station_code"].unique())
    print(f"   Remaining stations: {unique_stations}")

    # -------------------------------------------------------------------------
    # Step 3: Interpolate short gaps per station (<= max_gap_hours)
    # -------------------------------------------------------------------------
    print(
        f"\n3. Interpolating gaps shorter than or equal to {max_gap_hours} hours "
        "(per station, within its observed span)..."
    )

    interpolated_dfs = []
    total_interpolated = 0
    total_actual_interpolated = 0

    for station_code in unique_stations:
        station_df = df[df["station_code"] == station_code].copy()
        station_name = station_code
        station_df["station_name"] = station_name

        print(f"   Processing {station_code}...", end=" ")
        interp_df, gap_count, actual_count = interpolate_station_data(
            station_df, max_gap_hours=max_gap_hours
        )
        interpolated_dfs.append(interp_df)
        total_interpolated += gap_count
        total_actual_interpolated += actual_count
        print(f"interpolated {actual_count} values in short gaps")

    # Combine per-station results
    combined_df = pd.concat(interpolated_dfs, ignore_index=True)
    combined_df = combined_df.sort_values(["station_code", "datetime"]).reset_index(drop=True)

    print(f"\n   Total rows after per-station interpolation: {len(combined_df):,}")
    print(f"   Total gaps flagged as short: {total_interpolated:,} hours")
    print(f"   Total values interpolated linearly: {total_actual_interpolated:,}")

    # -------------------------------------------------------------------------
    # Step 4: Create complete 2014–2024 time series for all stations
    # -------------------------------------------------------------------------
    print(
        f"\n4. Creating complete time series for all stations "
        f"({start_year}-{end_year}, hourly)..."
    )

    global_start = pd.Timestamp(f"{start_year}-01-01 00:00:00")
    global_end = pd.Timestamp(f"{end_year}-12-31 23:00:00")
    all_datetimes = pd.date_range(start=global_start, end=global_end, freq="h")
    expected_hours_per_station = len(all_datetimes)
    print(f"   Expected hours per station: {expected_hours_per_station:,}")

    complete_rows = []
    for station_code in unique_stations:
        station_name = station_code
        for dt in all_datetimes:
            complete_rows.append(
                {
                    "datetime": dt,
                    "station_code": station_code,
                    "station_name": station_name,
                    "pm10": np.nan,
                    "interpolation_method": None,
                    "interpolation_confidence": np.nan,
                }
            )

    complete_df = pd.DataFrame(complete_rows)

    # Merge combined per-station interpolated data into complete grid
    combined_df_for_merge = combined_df[["datetime", "station_code", "pm10", "interpolation_method"]]

    complete_df = complete_df.merge(
        combined_df_for_merge,
        on=["datetime", "station_code"],
        how="left",
        suffixes=("", "_existing"),
    )

    # Fill with existing values where present
    complete_df["pm10"] = complete_df["pm10_existing"].combine_first(complete_df["pm10"])
    complete_df["interpolation_method"] = complete_df["interpolation_method_existing"].combine_first(
        complete_df["interpolation_method"]
    )

    # Drop helper columns
    complete_df = complete_df[
        ["datetime", "station_code", "station_name", "pm10", "interpolation_method", "interpolation_confidence"]
    ]

    print(f"   Created complete grid: {len(complete_df):,} rows")
    print(f"   Non-null values after merge: {complete_df['pm10'].notna().sum():,}")
    print(f"   Missing values to fill: {complete_df['pm10'].isna().sum():,}")

    # -------------------------------------------------------------------------
    # Step 5: Regression-based interpolation for remaining missing values
    # -------------------------------------------------------------------------
    print("\n5. Regression-based interpolation for remaining missing values...")

    result_df = regression_fill(complete_df)

    remaining_missing = result_df["pm10"].isna().sum()
    print(f"\n   Remaining missing values after regression-based interpolation: {remaining_missing:,}")
    if remaining_missing > 0:
        print("   ⚠️  Warning: could not fill all gaps; some NaNs remain.")
    else:
        print("   All gaps filled successfully; no missing values remain.")

    # -------------------------------------------------------------------------
    # Step 6: Save curated datasets and plots
    # -------------------------------------------------------------------------
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / "merged_pm10_hourly_curated.csv"

    print(f"\n6. Saving curated dataset with interpolation metadata to: {output_file}")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Attach metadata columns from station_meta
    result_df = result_df.merge(
        station_meta[["station_code", "address", "altitude_m", "lon_dms", "lat_dms"]],
        on="station_code",
        how="left",
    )

    # Reorder columns
    output_columns = [
        "datetime",
        "station_code",
        "station_name",
        "pm10",
        "interpolation_method",
        "interpolation_confidence",
        "address",
        "altitude_m",
        "lon_dms",
        "lat_dms",
    ]
    result_df[output_columns].to_csv(output_file, index=False)
    print(f"   ✓ Saved {len(result_df):,} rows")

    # Version without interpolation metadata
    output_file_no_interp = output_file.parent / "merged_pm10_hourly_curated_no_interp_metadata.csv"
    print(
        f"\n6b. Saving curated dataset (without interpolation metadata) to: "
        f"{output_file_no_interp}"
    )
    output_columns_no_interp = [
        "datetime",
        "station_code",
        "station_name",
        "pm10",
        "address",
        "altitude_m",
        "lon_dms",
        "lat_dms",
    ]
    result_df[output_columns_no_interp].to_csv(output_file_no_interp, index=False)
    print(f"   ✓ Saved {len(result_df):,} rows")

    # Generate per-station timeseries plots (actual vs linear vs regression)
    plots_dir = output_file.parent / "plots"
    print(f"\n6c. Saving per-station gap-filling plots to: {plots_dir}")
    plot_station_timeseries(result_df, plots_dir)

    # -------------------------------------------------------------------------
    # Step 7: Summary
    # -------------------------------------------------------------------------
    print(f"\n7. Summary:")
    print(f"   Original dataset rows (2014–2024, all stations): {len(df):,}")
    print(f"   Curated dataset rows: {len(result_df):,}")
    print(f"   Stations in curated dataset: {len(result_df['station_code'].unique())}")
    print(f"   Date range: {result_df['datetime'].min()} to {result_df['datetime'].max()}")
    print(f"   Short-gap (<= {max_gap_hours}h) linear interpolation: {total_actual_interpolated:,} values")
    print(f"   Regression-based interpolation: {(result_df['interpolation_method'] == 'regression').sum():,} values")

    # Per-station completeness
    print(f"\n   Per-station row counts and completeness:")
    for station_code in sorted(result_df["station_code"].unique()):
        station_name = station_code
        station_data = result_df[result_df["station_code"] == station_code]
        count = len(station_data)
        missing = station_data["pm10"].isna().sum()
        completeness = ((count - missing) / count * 100) if count > 0 else 0.0
        print(
            f"     {station_code}: {count:,} rows, {missing:,} missing "
            f"({completeness:.2f}% complete)"
        )

    print("\n" + "=" * 80)
    print("Dataset curation complete!")
    print("=" * 80)

    return result_df


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    input_file = project_root / "data" / "altoadige" / "PM10" / "merged_pm10_2008_2024.csv"
    output_file = project_root / "data" / "altoadige" / "PM10" / "merged_pm10_hourly_curated.csv"

    create_curated_altoadige_dataset(
        input_file,
        output_file=output_file,
        max_gap_hours=6,
        start_year=2014,
        end_year=2024,
    )


