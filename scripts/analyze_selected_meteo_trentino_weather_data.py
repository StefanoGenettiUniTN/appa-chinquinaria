"""
Analysis of gaps for the selected Meteo Trentino weather dataset used to
match APPA air quality stations.

Input dataset:
    output/meteo-trentino-appa-matching/selected_meteo_trentino_weather_dataset.csv

This CSV is produced by `match_appa_meteo_trentino.py` and has the following
structure:
    - Column `datetime` (timestamp)
    - One column per APPA–Meteo–variable combination, with name:
          {appa_code}_{meteo_code}_{variable_name}
      e.g. "402212_T0408_temperature"

Goals of this script:
    1. Resample all series to hourly frequency (if they are sub-hourly),
       aggregating within each hour (by mean) so that we have at most one
       value per hour.
    2. For each series, compute:
         - Expected number of hourly samples (based on global time range)
         - Actual number of non-missing hourly samples
         - Missing hours and missing percentage
         - Distribution of contiguous missing periods (in hours)
    3. Aggregate gap information across all series so we can understand
       how many gaps are short (e.g. <= 3–4 hours) and therefore good
       candidates for linear interpolation.

Outputs (if an output directory is provided):
    - hourly_resampled_dataset.csv
        The full wide dataset resampled to hourly frequency (no interpolation).
    - missing_values_per_series_summary.csv
        One row per series (column) with missing statistics and longest gap.
    - contiguous_missing_periods_per_series.csv
        One row per contiguous missing period, including the series it
        belongs to and its length in hours.
    - contiguous_missing_periods_distribution.csv
        Aggregated distribution of gap lengths across all series.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_series_column_name(col: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse a column name of the form "{appa_code}_{meteo_code}_{variable}".

    Returns:
        (appa_code, meteo_code, variable_name) or (None, None, None)
        if the pattern does not match.
    """
    if col == "datetime":
        return None, None, None

    parts = col.split("_", 2)
    if len(parts) != 3:
        return None, None, None

    appa_code, meteo_code, var_name = parts
    return appa_code, meteo_code, var_name


def resample_wide_dataset_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample the wide Meteo Trentino dataset to hourly frequency.

    Strategy:
        - Use the global min/max datetime in the dataset.
        - Set datetime as index and resample with freq="1H".
        - For each series (column), we aggregate multiple values within
          an hour using the mean (numeric only). If at least one valid
          value exists in that hour, the resampled value is non-NaN;
          otherwise it is NaN.
    """
    if "datetime" not in df.columns:
        raise ValueError("Input dataframe must contain a 'datetime' column.")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    df = df.set_index("datetime")

    # Ensure all non-index columns are numeric where possible
    value_cols = [c for c in df.columns if c != "datetime"]
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Resample to hourly with mean aggregation
    hourly = df.resample("1H").mean(numeric_only=True)
    hourly = hourly.sort_index()

    return hourly


def find_contiguous_missing_periods_from_hourly(
    series: pd.Series,
) -> List[Dict[str, object]]:
    """
    Given an hourly series (with a DatetimeIndex and fixed 1H frequency),
    find all contiguous missing value periods.

    Returns:
        List of dicts with keys:
            - start_time
            - end_time
            - length_hours
    """
    if series.empty:
        return []

    # We rely on the series already being at 1H frequency.
    is_missing = series.isna()

    if not is_missing.any():
        return []

    # Identify contiguous segments where is_missing stays True.
    # We create a group id that increments every time the missing flag changes.
    change_points = is_missing.ne(is_missing.shift()).cumsum()
    df_flag = pd.DataFrame({"is_missing": is_missing, "group": change_points})
    df_flag["datetime"] = series.index

    missing_groups = df_flag[df_flag["is_missing"]].groupby("group")

    periods: List[Dict[str, object]] = []
    for _, grp in missing_groups:
        start_time = grp["datetime"].min()
        end_time = grp["datetime"].max()
        length_hours = int(grp.shape[0])  # 1 row per hour
        periods.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "length_hours": length_hours,
            }
        )

    return periods


def analyze_selected_meteo_weather_data(
    data_file: Path,
    output_dir: Optional[Path] = None,
    max_gap_for_interp: int = 4,
) -> None:
    """
    Main analysis function for the selected Meteo Trentino weather dataset.

    Args:
        data_file: Path to selected_meteo_trentino_weather_dataset.csv
        output_dir: Optional directory to save results (CSV files)
        max_gap_for_interp: Threshold (hours) used only for reporting how many
            gaps are short enough to be good candidates for linear interpolation.
    """
    print("=" * 80)
    print("Selected Meteo Trentino Weather Data – Gap Analysis")
    print("=" * 80)

    print("\n1. Loading data...")
    print(f"   Loading selected Meteo Trentino dataset from: {data_file}")
    df = pd.read_csv(data_file)
    if "datetime" not in df.columns:
        raise ValueError(
            "Expected a 'datetime' column in the selected Meteo Trentino dataset."
        )

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    n_rows, n_cols = df.shape
    print(f"   Loaded {n_rows:,} rows and {n_cols - 1} data series columns.")
    print(f"   Date range: {df['datetime'].min()}  →  {df['datetime'].max()}")

    # Restrict the analysis to a fixed period: 2014–01–01 to 2024–12–31 (inclusive).
    analysis_start = pd.Timestamp("2014-01-01 00:00:00")
    analysis_end = pd.Timestamp("2024-12-31 23:00:00")
    print(
        f"   Target analysis window (global): {analysis_start}  →  {analysis_end}"
    )

    # Identify all series columns (exclude datetime)
    series_cols = [c for c in df.columns if c != "datetime"]
    print(f"   Number of series to analyze: {len(series_cols)}")

    # ----------------------------------------------------------------------
    # 2. Resample to hourly frequency
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. Resampling to hourly frequency")
    print("=" * 80)

    hourly = resample_wide_dataset_to_hourly(df)
    # Clamp resampled dataset to the desired global analysis window
    hourly = hourly.loc[analysis_start:analysis_end]
    hourly_index = hourly.index
    if len(hourly_index) == 0:
        print("   No data within the requested analysis window; nothing to do.")
        return
    print(
        f"   Hourly index in analysis window: {hourly_index.min()} → {hourly_index.max()}"
    )
    print(f"   Total hourly timestamps in analysis window: {len(hourly_index):,}")

    # Optionally save the hourly-resampled dataset (no interpolation)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        hourly_out = output_dir / "hourly_resampled_dataset.csv"
        print(f"\n   Saving hourly-resampled dataset to: {hourly_out}")
        hourly.reset_index().rename(columns={"index": "datetime"}).to_csv(
            hourly_out, index=False
        )

    # ----------------------------------------------------------------------
    # 3. Per-series missing value statistics and contiguous gaps
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. Per-series missing value statistics and gap lengths")
    print("=" * 80)

    summary_rows: List[Dict[str, object]] = []
    gap_rows: List[Dict[str, object]] = []
    all_gap_lengths: List[int] = []
    total_negative_values = 0
    series_with_negative: List[str] = []

    for col in series_cols:
        if col not in hourly.columns:
            # This should not happen, but guard just in case.
            continue

        # Work on a copy of the hourly series for this column
        series = hourly[col].copy()
        # Ensure numeric and keep as float
        series = pd.to_numeric(series, errors="coerce")

        appa_code, meteo_code, var_name = parse_series_column_name(col)

        # Determine the effective analysis window for this series:
        # intersection of the global window and the period where the
        # series actually has (non-NaN) data. This avoids counting long
        # leading/trailing periods before/after the sensor was active.
        valid_times = series[series.notna()].index
        if len(valid_times) == 0:
            expected_hours = 0
            actual_hours = 0
            missing_hours = 0
            missing_pct = 0.0
            periods: List[Dict[str, object]] = []
            longest_gap = 0
            mean_gap = 0.0
            median_gap = 0.0
            n_short_gaps = 0
            min_val = np.nan
            max_val = np.nan
            negative_count = 0
        else:
            series_start = max(valid_times.min(), analysis_start)
            series_end = min(valid_times.max(), analysis_end)
            if series_start > series_end:
                # No overlap between this series' data and the analysis window
                expected_hours = 0
                actual_hours = 0
                missing_hours = 0
                missing_pct = 0.0
                periods = []
                longest_gap = 0
                mean_gap = 0.0
                median_gap = 0.0
                n_short_gaps = 0
                min_val = np.nan
                max_val = np.nan
                negative_count = 0
            else:
                series_window = series.loc[series_start:series_end]
                expected_hours = int(series_window.shape[0])

                valid_mask = series_window.notna()
                actual_hours = int(valid_mask.sum())
                missing_hours = int(expected_hours - actual_hours)
                missing_pct = (
                    float(missing_hours) / float(expected_hours) * 100.0
                    if expected_hours > 0
                    else 0.0
                )

                # Basic value diagnostics (min/max and negative values)
                if series_window.notna().any():
                    min_val = float(series_window.min())
                    max_val = float(series_window.max())
                else:
                    min_val = np.nan
                    max_val = np.nan

                negative_count = int((series_window < 0).sum())
                if negative_count > 0:
                    total_negative_values += negative_count
                    series_with_negative.append(col)

                # Find contiguous missing periods (in hours) within the
                # effective series window only.
                periods = find_contiguous_missing_periods_from_hourly(series_window)
                if periods:
                    gap_lengths = [p["length_hours"] for p in periods]
                    longest_gap = int(max(gap_lengths))
                    mean_gap = float(np.mean(gap_lengths))
                    median_gap = float(np.median(gap_lengths))
                    all_gap_lengths.extend(int(x) for x in gap_lengths)

                    # Store detailed rows
                    for p in periods:
                        gap_rows.append(
                            {
                                "series_column": col,
                                "appa_code": appa_code,
                                "meteo_code": meteo_code,
                                "variable": var_name,
                                "start_time": p["start_time"],
                                "end_time": p["end_time"],
                                "length_hours": p["length_hours"],
                            }
                        )
                else:
                    longest_gap = 0
                    mean_gap = 0.0
                    median_gap = 0.0

                # Count how many gaps are "short" (<= max_gap_for_interp)
                n_short_gaps = 0
                if periods:
                    n_short_gaps = sum(
                        1
                        for p in periods
                        if int(p["length_hours"]) <= max_gap_for_interp
                    )

        summary_rows.append(
            {
                "series_column": col,
                "appa_code": appa_code,
                "meteo_code": meteo_code,
                "variable": var_name,
                "expected_hours": expected_hours,
                "actual_hours": actual_hours,
                "missing_hours": missing_hours,
                "missing_percentage": missing_pct,
                "n_contiguous_gaps": len(periods),
                "n_short_gaps_leq_threshold": n_short_gaps,
                "gap_length_threshold_hours": max_gap_for_interp,
                "longest_gap_hours": longest_gap,
                "mean_gap_length_hours": mean_gap,
                "median_gap_length_hours": median_gap,
                "min_value": min_val,
                "max_value": max_val,
                "negative_value_count": negative_count,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    print("\nPer-series missing value summary:")
    if not summary_df.empty:
        # Show a concise view in the console
        display_cols = [
            "series_column",
            "appa_code",
            "meteo_code",
            "variable",
            "missing_percentage",
            "longest_gap_hours",
            "n_contiguous_gaps",
            "n_short_gaps_leq_threshold",
            "negative_value_count",
        ]
        existing_display_cols = [c for c in display_cols if c in summary_df.columns]
        print(summary_df[existing_display_cols].to_string(index=False))
    else:
        print("  No series found for summary.")

    # Simple global check for negative values across all series
    print("\nNegative value check (all series, analysis window):")
    if total_negative_values > 0:
        print(f"   Total negative values found: {total_negative_values:,}")
        print(
            f"   Number of series containing negative values: "
            f"{len(set(series_with_negative))}"
        )
        # Show a few series as examples
        neg_example = summary_df[summary_df["negative_value_count"] > 0][
            ["series_column", "variable", "negative_value_count", "min_value"]
        ].head(10)
        print("\n   Example series with negative values:")
        print(neg_example.to_string(index=False))
    else:
        print("   No negative values found in the analysis window.")

    # ----------------------------------------------------------------------
    # 4. Global distribution of contiguous missing period lengths
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. Global distribution of contiguous missing gap lengths")
    print("=" * 80)

    if all_gap_lengths:
        length_counts = pd.Series(all_gap_lengths).value_counts().sort_index()
        total_periods = int(len(all_gap_lengths))

        print("\nDistribution of contiguous missing period lengths (all series):")
        print(f"{'Length (hours)':<20} {'Frequency':<15} {'Percentage':<15}")
        print("-" * 50)
        for length, count in length_counts.items():
            pct = (int(count) / total_periods * 100.0) if total_periods > 0 else 0.0
            print(f"{int(length):<20} {int(count):<15} {pct:>8.2f}%")

        # Also report how many gaps are "short"
        n_short = sum(1 for l in all_gap_lengths if l <= max_gap_for_interp)
        short_pct = (n_short / total_periods * 100.0) if total_periods > 0 else 0.0
        print(
            f"\nGaps with length <= {max_gap_for_interp} hours: "
            f"{n_short:,} ({short_pct:.2f}%)"
        )

        print(f"\nTotal contiguous missing periods: {total_periods:,}")
        print(f"Longest contiguous missing period: {max(all_gap_lengths)} hours")
        print(f"Shortest contiguous missing period: {min(all_gap_lengths)} hours")
        print(
            f"Mean contiguous missing period length: {np.mean(all_gap_lengths):.2f} hours"
        )
        print(
            f"Median contiguous missing period length: {np.median(all_gap_lengths):.2f} hours"
        )
    else:
        print("No missing periods found in the hourly-resampled dataset.")

    # ----------------------------------------------------------------------
    # 5. Save results to disk (if requested)
    # ----------------------------------------------------------------------
    if output_dir is not None:
        assert output_dir is not None  # for type checkers

        summary_out = output_dir / "missing_values_per_series_summary.csv"
        gaps_out = output_dir / "contiguous_missing_periods_per_series.csv"
        dist_out = output_dir / "contiguous_missing_periods_distribution.csv"

        print("\n" + "=" * 80)
        print("Saving gap analysis results to CSV files...")
        print("=" * 80)

        summary_df.to_csv(summary_out, index=False)
        print(f"  Saved per-series summary: {summary_out}")

        if gap_rows:
            gap_df = pd.DataFrame(gap_rows)
            gap_df.to_csv(gaps_out, index=False)
            print(f"  Saved per-period details: {gaps_out}")

            length_counts = (
                pd.Series(all_gap_lengths)
                .value_counts()
                .sort_index()
                .rename_axis("length_hours")
                .reset_index(name="frequency")
            )
            length_counts["percentage"] = (
                length_counts["frequency"] / length_counts["frequency"].sum() * 100.0
            )
            length_counts.to_csv(dist_out, index=False)
            print(f"  Saved gap length distribution: {dist_out}")
        else:
            print("  No gap periods to save (no missing data found).")

    print("\n" + "=" * 80)
    print("Gap analysis complete.")
    print("=" * 80)


if __name__ == "__main__":
    # Default paths relative to the project root (one level up from scripts/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    data_file = (
        project_root
        / "output"
        / "meteo-trentino-appa-matching"
        / "selected_meteo_trentino_weather_dataset.csv"
    )
    output_dir = project_root / "output" / "meteo-trentino-appa-matching" / "gap_analysis"

    analyze_selected_meteo_weather_data(data_file=data_file, output_dir=output_dir)


