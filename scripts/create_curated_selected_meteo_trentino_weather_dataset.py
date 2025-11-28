"""
Create a curated Meteo Trentino weather dataset (for APPA-matched stations)
from the wide CSV produced by `match_appa_meteo_trentino.py`.

Steps:
    1. Load `selected_meteo_trentino_weather_dataset.csv` (wide format:
       datetime + one column per APPA–Meteo–variable combination).
    2. Restrict to the analysis window 2014-01-01 .. 2024-12-31.
    3. Resample to hourly frequency (if needed) using mean aggregation.
    4. For each series (column):
         - Fill contiguous gaps with length < 5 hours using *explicit*
           linear interpolation between the bounding observations.
    5. For each variable (temperature, rain, etc.):
         - For each series in that variable group, train a simple linear
           regression model using all *other* series of the same variable
           as predictors (across time).
         - Use the model to estimate remaining missing values in that
           series (where at least one predictor has data).
    6. Optionally apply a final forward/backward fill per series to
       remove any residual NaNs (very rare edge cases), counting them
       as "regression" for reporting.
    7. Save:
         - A curated dataset with the same structure as the original
           wide CSV, but with *no gaps*.
         - A version with per-cell interpolation metadata.
         - CSV summaries and plots describing the fraction and temporal
           location of data filled by linear interpolation vs regression.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse helper functions from the analysis script
from analyze_selected_meteo_trentino_weather_data import (  # type: ignore
    parse_series_column_name,
    resample_wide_dataset_to_hourly,
)


def fill_short_gaps_with_linear(
    series: pd.Series,
    max_gap_hours: int = 4,
) -> Tuple[pd.Series, pd.Series]:
    """
    Fill contiguous NaN gaps in a series using linear interpolation,
    *only* when the gap length is strictly smaller than max_gap_hours + 1.

    Note: With hourly data, a gap of length L means L consecutive NaN
    hours. For the user's requirement "strictly smaller than 5 hours",
    we call this function with max_gap_hours=4.

    Args:
        series: Hourly time series with DatetimeIndex.
        max_gap_hours: Maximum gap length (in hours) to fill via linear
            interpolation. Gaps with length <= max_gap_hours are filled,
            longer gaps are left as NaN.

    Returns:
        (filled_series, is_linear_mask)
        - filled_series: copy of input with short gaps filled.
        - is_linear_mask: boolean Series indicating where values came
          from linear interpolation.
    """
    s = series.copy()
    is_linear = pd.Series(False, index=s.index)

    is_na = s.isna()
    if not is_na.any():
        return s, is_linear

    # Identify contiguous NaN runs via group IDs
    grp_id = is_na.ne(is_na.shift()).cumsum()
    na_groups = grp_id[is_na].groupby(grp_id[is_na])

    index_array = s.index.to_list()
    n = len(index_array)

    for _, idx_positions in na_groups:
        # idx_positions is a Series of integer positions of NaNs
        positions = idx_positions.index
        # positions is an Index of timestamps; get integer locations
        locs = [s.index.get_loc(ts) for ts in positions]
        if not locs:
            continue
        start_loc = min(locs)
        end_loc = max(locs)
        length = end_loc - start_loc + 1

        if length > max_gap_hours:
            # Long gaps are left untouched for regression-based filling
            continue

        # Need valid values just before and after the gap
        prev_loc = start_loc - 1
        next_loc = end_loc + 1
        if prev_loc < 0 or next_loc >= n:
            continue

        y0 = s.iloc[prev_loc]
        y1 = s.iloc[next_loc]
        if pd.isna(y0) or pd.isna(y1):
            continue

        # Linear interpolation across the gap
        for k, loc in enumerate(range(start_loc, end_loc + 1), start=1):
            frac = k / (length + 1)
            val = float(y0) + (float(y1) - float(y0)) * frac
            ts = index_array[loc]
            s.loc[ts] = val
            is_linear.loc[ts] = True

    return s, is_linear


def fit_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    ridge_alpha: float = 0.0,
) -> np.ndarray:
    """
    Fit a simple linear regression (optionally with ridge regularization)
    using closed-form least squares.

    Args:
        X: 2D design matrix (n_samples, n_features)
        y: 1D target vector (n_samples,)
        ridge_alpha: L2 regularization strength (0.0 = ordinary least squares)

    Returns:
        Coefficient vector (including intercept as the first element).
    """
    # Add intercept term
    X_design = np.column_stack([np.ones(X.shape[0]), X])

    if ridge_alpha > 0.0:
        n_features = X_design.shape[1]
        I = np.eye(n_features)
        I[0, 0] = 0.0  # Do not regularize intercept
        A = X_design.T @ X_design + ridge_alpha * I
        b = X_design.T @ y
        coef = np.linalg.solve(A, b)
    else:
        coef, *_ = np.linalg.lstsq(X_design, y, rcond=None)

    return coef


def predict_linear_regression(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply a fitted linear regression model (with intercept) to new data."""
    X_design = np.column_stack([np.ones(X.shape[0]), X])
    return X_design @ coef


def create_curated_selected_meteo_dataset(
    data_file: Path,
    output_dir: Optional[Path] = None,
    max_linear_gap_hours: int = 4,
    start_year: int = 2014,
    end_year: int = 2024,
    appa_pm10_file: Optional[Path] = None,
) -> None:
    """
    Main function to create a curated selected Meteo Trentino dataset.

    Args:
        data_file: Path to selected_meteo_trentino_weather_dataset.csv
        output_dir: Directory to save curated datasets, metadata, and plots.
        max_linear_gap_hours: Maximum contiguous gap length (< this + 1)
            to fill using linear interpolation (e.g. 4 → gaps of 1–4 hours).
        start_year: Start year for the analysis window (inclusive).
        end_year: End year for the analysis window (inclusive).
    """
    print("=" * 80)
    print("Creating curated Meteo Trentino weather dataset (selected stations)")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\n1. Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    if "datetime" not in df.columns:
        raise ValueError("Input CSV must contain a 'datetime' column.")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    n_rows, n_cols = df.shape
    print(f"   Loaded {n_rows:,} rows and {n_cols - 1} data series columns.")
    print(f"   Raw date range: {df['datetime'].min()}  →  {df['datetime'].max()}")

    # Global analysis window
    analysis_start = pd.Timestamp(f"{start_year}-01-01 00:00:00")
    analysis_end = pd.Timestamp(f"{end_year}-12-31 23:00:00")
    print(f"   Target analysis window: {analysis_start}  →  {analysis_end}")

    # Restrict raw data to a slightly expanded window (for resampling safety)
    df = df[(df["datetime"] >= analysis_start) & (df["datetime"] <= analysis_end)].copy()

    # Identify series columns
    series_cols = [c for c in df.columns if c != "datetime"]
    print(f"   Number of series to curate: {len(series_cols)}")

    # ------------------------------------------------------------------
    # 2. Resample to hourly frequency
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. Resampling to hourly frequency")
    print("=" * 80)

    hourly = resample_wide_dataset_to_hourly(df)
    hourly = hourly.loc[analysis_start:analysis_end]
    if hourly.empty:
        print("   No data in the requested analysis window; nothing to do.")
        return

    hourly_index = hourly.index
    print(f"   Hourly index: {hourly_index.min()}  →  {hourly_index.max()}")
    print(f"   Number of hourly timestamps: {len(hourly_index):,}")

    # Prepare metadata frame to track how each cell was obtained
    method_df = pd.DataFrame("missing", index=hourly.index, columns=hourly.columns)
    for col in series_cols:
        if col in hourly.columns:
            method_df.loc[hourly[col].notna(), col] = "actual"

    # ------------------------------------------------------------------
    # 3. Fill short gaps with linear interpolation
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. Filling short gaps with linear interpolation")
    print("=" * 80)

    linear_filled_count = 0

    for col in series_cols:
        if col not in hourly.columns:
            continue

        series = hourly[col]
        filled_series, is_linear = fill_short_gaps_with_linear(
            series, max_gap_hours=max_linear_gap_hours
        )

        # Count how many values were actually filled
        newly_filled_mask = series.isna() & filled_series.notna() & is_linear
        n_new = int(newly_filled_mask.sum())
        linear_filled_count += n_new

        # Update data and method metadata
        hourly[col] = filled_series
        method_df.loc[newly_filled_mask, col] = "linear"

    print(f"   Total values filled via linear interpolation: {linear_filled_count:,}")

    # ------------------------------------------------------------------
    # 4. Regression-based filling for remaining gaps (per variable)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. Regression-based filling for remaining gaps")
    print("=" * 80)

    # Map: variable_name -> list of series columns
    variable_to_cols: Dict[str, List[str]] = {}
    for col in series_cols:
        appa_code, meteo_code, var_name = parse_series_column_name(col)
        if var_name is None:
            continue
        variable_to_cols.setdefault(var_name, []).append(col)

    regression_filled_count = 0

    for var_name, cols in variable_to_cols.items():
        print(f"\n   Variable '{var_name}': processing {len(cols)} series")

        # Subset for this variable
        df_var = hourly[cols].copy()

        for target_col in cols:
            y = df_var[target_col]

            # Remaining NaNs after linear interpolation
            missing_mask = y.isna()
            if not missing_mask.any():
                continue

            feature_cols = [c for c in cols if c != target_col]
            if not feature_cols:
                continue  # single-series variable (nothing to regress on)

            X = df_var[feature_cols]

            # Training data: rows where target is non-NaN and at least one feature is non-NaN
            train_mask = y.notna() & X.notna().any(axis=1)
            if train_mask.sum() < 50:
                # Too few samples to train a reliable model
                continue

            X_train = X[train_mask]
            y_train = y[train_mask].astype(float)

            # Impute feature NaNs with per-column means computed on training data
            col_means = X_train.mean(axis=0)
            X_train_filled = X_train.fillna(col_means)

            # Fit regression model
            coef = fit_linear_regression(X_train_filled.to_numpy(), y_train.to_numpy())

            # Prediction for missing target values where we have at least one feature
            pred_mask = y.isna() & X.notna().any(axis=1)
            if not pred_mask.any():
                continue

            X_pred = X[pred_mask]
            X_pred_filled = X_pred.fillna(col_means)

            y_pred = predict_linear_regression(coef, X_pred_filled.to_numpy())

            # Apply predictions
            hourly.loc[pred_mask, target_col] = y_pred
            method_df.loc[pred_mask, target_col] = "regression"
            n_pred = int(pred_mask.sum())
            regression_filled_count += n_pred

            print(
                f"      {target_col}: filled {n_pred:,} values via regression "
                f"(train samples={int(train_mask.sum()):,})"
            )

    print(f"\n   Total values filled via regression: {regression_filled_count:,}")

    # ------------------------------------------------------------------
    # 5. Final fallback: forward/backward fill any residual NaNs
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("5. Final fallback fill for any remaining gaps")
    print("=" * 80)

    fallback_filled_count = 0
    for col in series_cols:
        if col not in hourly.columns:
            continue
        s = hourly[col]
        mask_before = s.isna()
        if not mask_before.any():
            continue
        s_filled = s.ffill().bfill()
        mask_after = mask_before & s_filled.notna()

        n_fallback = int(mask_after.sum())
        if n_fallback > 0:
            hourly[col] = s_filled
            method_df.loc[mask_after, col] = "regression"  # count as model-based
            fallback_filled_count += n_fallback

    print(f"   Total values filled via fallback (ffill/bfill): {fallback_filled_count:,}")

    # Sanity check: no NaNs should remain
    if hourly.isna().any().any():
        n_remaining = int(hourly.isna().sum().sum())
        print(f"   WARNING: {n_remaining:,} NaNs remain after curation.")
    else:
        print("   No NaNs remain in the curated dataset.")

    # ------------------------------------------------------------------
    # 6. Save curated datasets and metadata
    # ------------------------------------------------------------------
    if output_dir is None:
        output_dir = data_file.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curated_clean_path = output_dir / "curated_selected_meteo_trentino_weather_dataset.csv"
    curated_with_meta_path = (
        output_dir / "curated_selected_meteo_trentino_weather_dataset_with_metadata.csv"
    )
    per_series_summary_path = output_dir / "gap_filling_summary_per_series.csv"
    per_method_summary_path = output_dir / "gap_filling_summary_overall.csv"

    # Clean curated dataset (no metadata columns), same structure as original
    curated_clean = hourly.copy()
    curated_clean = curated_clean.reset_index().rename(columns={"index": "datetime"})
    curated_clean.to_csv(curated_clean_path, index=False)
    print(f"\n   Saved curated dataset (no metadata): {curated_clean_path}")

    # Dataset with per-cell method metadata: keep one extra column per data column
    meta_cols = {}
    for col in series_cols:
        if col in hourly.columns:
            meta_cols[f"{col}__method"] = method_df[col]
    method_meta_df = pd.DataFrame(meta_cols, index=hourly.index)

    curated_with_meta = pd.concat([hourly, method_meta_df], axis=1)
    curated_with_meta = curated_with_meta.reset_index().rename(columns={"index": "datetime"})
    curated_with_meta.to_csv(curated_with_meta_path, index=False)
    print(f"   Saved curated dataset with metadata: {curated_with_meta_path}")

    # ------------------------------------------------------------------
    # 7. Numeric summaries and per-series timeseries plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("6. Creating summaries and per-series timeseries plots")
    print("=" * 80)

    # Per-series summary
    summary_rows: List[Dict[str, object]] = []

    for col in series_cols:
        if col not in hourly.columns:
            continue
        appa_code, meteo_code, var_name = parse_series_column_name(col)
        methods = method_df[col]
        n_actual = int((methods == "actual").sum())
        n_linear = int((methods == "linear").sum())
        n_reg = int((methods == "regression").sum())
        n_total = n_actual + n_linear + n_reg
        if n_total == 0:
            continue
        summary_rows.append(
            {
                "series_column": col,
                "appa_code": appa_code,
                "meteo_code": meteo_code,
                "variable": var_name,
                "n_actual": n_actual,
                "n_linear": n_linear,
                "n_regression": n_reg,
                "pct_actual": n_actual / n_total * 100.0,
                "pct_linear": n_linear / n_total * 100.0,
                "pct_regression": n_reg / n_total * 100.0,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(per_series_summary_path, index=False)
    print(f"   Saved per-series summary: {per_series_summary_path}")

    # Overall summary per method (still useful numerically, even if not plotted)
    overall_counts = {
        "actual": int((method_df == "actual").sum().sum()),
        "linear": int((method_df == "linear").sum().sum()),
        "regression": int((method_df == "regression").sum().sum()),
    }
    overall_total = sum(overall_counts.values())
    overall_rows = []
    for method, cnt in overall_counts.items():
        overall_rows.append(
            {
                "method": method,
                "count": cnt,
                "percentage": cnt / overall_total * 100.0 if overall_total > 0 else 0.0,
            }
        )
    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(per_method_summary_path, index=False)
    print(f"   Saved overall method summary: {per_method_summary_path}")

    # Per-series timeseries plots: original vs linear vs regression
    plots_dir = output_dir / "plots" / "curation_timeseries"
    plots_dir.mkdir(parents=True, exist_ok=True)

    color_map = {
        "actual": "#000000",      # black
        "linear": "#1f77b4",      # blue
        "regression": "#d62728",  # red
    }

    for _, row in summary_df.iterrows():
        col = row["series_column"]
        if col not in hourly.columns:
            continue

        series = hourly[col]
        methods = method_df[col]

        pct_actual = row["pct_actual"]
        pct_linear = row["pct_linear"]
        pct_reg = row["pct_regression"]

        plt.figure(figsize=(12, 4))

        # Plot each method as its own colored line over time
        for method_name, color in color_map.items():
            mask = methods == method_name
            if not mask.any():
                continue
            plt.plot(
                hourly_index[mask],
                series[mask],
                linestyle="-",
                linewidth=0.6,
                color=color,
                label=method_name,  # temporary, legend text overridden below
            )

        # Build legend text with percentages
        legend_entries = []
        if pct_actual > 0:
            legend_entries.append(
                f"Actual ({pct_actual:.1f}%)"
            )
        if pct_linear > 0:
            legend_entries.append(
                f"Linear interp. ({pct_linear:.1f}%)"
            )
        if pct_reg > 0:
            legend_entries.append(
                f"Model regression ({pct_reg:.1f}%)"
            )

        # Rebuild legend handles to match color_map ordering
        handles = []
        labels = []
        if pct_actual > 0:
            handles.append(
                plt.Line2D([0], [0], color=color_map["actual"], linewidth=1.2)
            )
            labels.append(f"Actual ({pct_actual:.1f}%)")
        if pct_linear > 0:
            handles.append(
                plt.Line2D([0], [0], color=color_map["linear"], linewidth=1.2)
            )
            labels.append(f"Linear interp. ({pct_linear:.1f}%)")
        if pct_reg > 0:
            handles.append(
                plt.Line2D([0], [0], color=color_map["regression"], linewidth=1.2)
            )
            labels.append(f"Model regression ({pct_reg:.1f}%)")

        if handles:
            plt.legend(handles, labels, fontsize=8, loc="upper right")

        appa_code = row["appa_code"]
        meteo_code = row["meteo_code"]
        var_name = row["variable"]

        title_parts = [col]
        if appa_code:
            title_parts.append(f"APPA {appa_code}")
        if meteo_code:
            title_parts.append(f"Meteo {meteo_code}")
        if var_name:
            title_parts.append(var_name)

        plt.title(" – ".join(title_parts))
        plt.xlabel("Datetime")
        plt.ylabel(var_name if isinstance(var_name, str) else "value")
        plt.grid(alpha=0.3, linewidth=0.3)
        plt.tight_layout()

        plot_path = plots_dir / f"timeseries_gap_filling__{col}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"   Saved timeseries plot for {col}: {plot_path}")

    # ------------------------------------------------------------------
    # 8. Join curated Meteo dataset with curated APPA PM10 dataset
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("7. Joining curated APPA PM10 dataset with curated Meteo dataset")
    print("=" * 80)

    if appa_pm10_file is None:
        # Default to ML-filled APPA curated dataset without interpolation metadata
        appa_pm10_file = (
            output_dir.parent.parent
            / "data"
            / "appa-data"
            / "merged_pm10_hourly_curated_ml_filled_no_interp_metadata.csv"
        )

    appa_pm10_file = Path(appa_pm10_file)
    if not appa_pm10_file.exists():
        print(
            f"   WARNING: APPA PM10 file not found at {appa_pm10_file}. "
            "Skipping final joined Trentino dataset."
        )
    else:
        print(f"   Loading curated APPA PM10 dataset from: {appa_pm10_file}")
        appa_df = pd.read_csv(appa_pm10_file)
        if "datetime" not in appa_df.columns:
            raise ValueError(
                "Expected a 'datetime' column in the APPA PM10 dataset."
            )

        appa_df["datetime"] = pd.to_datetime(appa_df["datetime"])

        # Restrict APPA data to the same analysis window
        appa_df = appa_df[
            (appa_df["datetime"] >= analysis_start)
            & (appa_df["datetime"] <= analysis_end)
        ].copy()

        # Pivot APPA PM10 to wide format: one column per station_code
        if "station_code" not in appa_df.columns or "pm10" not in appa_df.columns:
            raise ValueError(
                "APPA PM10 dataset must contain 'station_code' and 'pm10' columns."
            )

        appa_wide = appa_df.pivot_table(
            index="datetime", columns="station_code", values="pm10", aggfunc="first"
        ).sort_index()

        # Rename columns to make them self-describing (pm10_{station_code})
        appa_wide.columns = [f"pm10_{str(c)}" for c in appa_wide.columns]

        # Prepare curated Meteo dataframe with datetime as index
        meteo_wide = curated_clean.set_index("datetime").copy()

        # Join on datetime (inner join to keep only timestamps present in both)
        joined = meteo_wide.join(appa_wide, how="inner")
        joined = joined.sort_index()

        # Sanity check
        print(
            f"   Joined dataset shape: {joined.shape[0]:,} rows × {joined.shape[1]:,} columns"
        )

        # Save final Trentino-wide dataset
        final_trentino_path = (
            output_dir / "trentino_pm10_with_meteo_hourly_curated.csv"
        )
        joined.reset_index().to_csv(final_trentino_path, index=False)
        print(f"   Saved final Trentino dataset: {final_trentino_path}")

    print("\n" + "=" * 80)
    print("Curation complete.")
    print("=" * 80)


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    data_file = (
        project_root
        / "output"
        / "meteo-trentino-appa-matching"
        / "selected_meteo_trentino_weather_dataset.csv"
    )
    output_dir = (
        project_root
        / "output"
        / "meteo-trentino-appa-matching"
        / "curated"
    )

    appa_pm10_file = (
        project_root
        / "data"
        / "appa-data"
        / "merged_pm10_hourly_curated_ml_filled_no_interp_metadata.csv"
    )

    create_curated_selected_meteo_dataset(
        data_file=data_file,
        output_dir=output_dir,
        max_linear_gap_hours=4,
        start_year=2014,
        end_year=2024,
        appa_pm10_file=appa_pm10_file,
    )


