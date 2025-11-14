#!/usr/bin/env python3
"""
Fill remaining APPA PM10 gaps using a gradient boosting regressor that learns from
the full multivariate time series (all stations simultaneously).

Steps:
1. Load curated dataset (with interpolation metadata).
2. Pivot to wide format (datetime x station_code).
3. Engineer temporal features.
4. Train one HistGradientBoostingRegressor per station on rows with real values.
5. Predict missing values and write them back to the long-format dataset.
6. Save new CSV versions (with and without interpolation metadata) plus summary stats.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score


@dataclass
class StationModelResult:
    station_code: str
    mae: float
    confidence: float
    n_train: int
    n_predicted: int


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical temporal features based on the datetime index."""
    df = df.copy()
    dt_index = df.index

    hour = dt_index.hour
    dow = dt_index.dayofweek
    month = dt_index.month
    doy = dt_index.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 366)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 366)
    df["is_weekend"] = (dow >= 5).astype(float)

    return df


def train_and_predict_station(
    station_code: str,
    features_df: pd.DataFrame,
    model_params: Dict,
) -> StationModelResult:
    """Train a model for a single station and fill its missing values."""
    y = features_df[station_code]
    X = features_df.drop(columns=[station_code])

    missing_mask = y.isna()
    n_missing = missing_mask.sum()
    n_train = (~missing_mask).sum()

    if n_missing == 0 or n_train < 500:
        return StationModelResult(station_code, mae=np.nan, confidence=0.0, n_train=n_train, n_predicted=0)

    model = HistGradientBoostingRegressor(**model_params)
    model.fit(X[~missing_mask], y[~missing_mask])

    # Cross-validated MAE for confidence estimation
    try:
        scores = cross_val_score(
            model,
            X[~missing_mask],
            y[~missing_mask],
            scoring="neg_mean_absolute_error",
            cv=min(5, max(2, n_train // 2000)),
        )
        mae = -scores.mean()
    except Exception:
        mae = float(np.abs(y[~missing_mask] - model.predict(X[~missing_mask])).mean())

    # Convert MAE into confidence (relative to 50 µg/m3 range)
    confidence = float(np.clip(1 - mae / 50.0, 0.0, 1.0))

    preds = model.predict(X[missing_mask])
    features_df.loc[missing_mask, station_code] = preds

    return StationModelResult(
        station_code=station_code,
        mae=mae,
        confidence=confidence,
        n_train=int(n_train),
        n_predicted=int(n_missing),
    )


def fill_gaps_with_regressor(
    curated_file: Path,
    output_with_metadata: Path,
    output_without_metadata: Path,
    summary_dir: Path,
) -> None:
    print("=" * 80)
    print("ML Gap Filling for APPA PM10")
    print("=" * 80)

    df = pd.read_csv(curated_file)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["station_code"] = df["station_code"].astype(str)
    df = df.sort_values(["datetime", "station_code"]).reset_index(drop=True)

    missing_mask = df["pm10"].isna()
    total_missing = int(missing_mask.sum())
    print(f"Loaded {len(df):,} rows ({df['station_code'].nunique()} stations)")
    print(f"Missing values before ML filling: {total_missing:,}")

    if total_missing == 0:
        print("No gaps to fill. Exiting.")
        return

    # Pivot to wide format
    wide = df.pivot_table(
        index="datetime",
        columns="station_code",
        values="pm10",
        aggfunc="first",
    ).sort_index()

    # Preserve station order
    station_codes = list(wide.columns)

    features_df = add_time_features(wide)

    model_params = dict(
        max_depth=6,
        learning_rate=0.08,
        max_iter=400,
        l2_regularization=0.5,
        random_state=42,
    )

    results: List[StationModelResult] = []
    for station in station_codes:
        res = train_and_predict_station(station, features_df, model_params)
        results.append(res)
        if res.n_predicted > 0:
            print(
                f"  {station}: predicted {res.n_predicted:,} values | "
                f"train={res.n_train:,} | MAE={res.mae:.2f} | confidence={res.confidence:.3f}"
            )

    # Prepare filled values
    filled_wide = features_df[station_codes].copy()
    filled_long = (
        filled_wide
        .reset_index()
        .melt(id_vars="datetime", var_name="station_code", value_name="pm10_filled")
    )

    df = df.merge(
        filled_long,
        on=["datetime", "station_code"],
        how="left",
    )

    station_conf_lookup = {
        r.station_code: r.confidence for r in results if not np.isnan(r.confidence)
    }

    # Apply predictions back to long df
    ml_fill_mask = df["pm10"].isna() & df["pm10_filled"].notna()
    df.loc[ml_fill_mask, "pm10"] = df.loc[ml_fill_mask, "pm10_filled"]

    # Track interpolation metadata
    if "interpolation_method" not in df.columns:
        df["interpolation_method"] = "actual"

    df.loc[ml_fill_mask, "interpolation_method"] = "ml_predicted"
    df["prediction_confidence"] = df["station_code"].map(station_conf_lookup).fillna(0.0)

    remaining_missing = int(df["pm10"].isna().sum())
    print(f"\nMissing values after ML filling: {remaining_missing:,}")

    # Save outputs
    output_with_metadata.parent.mkdir(parents=True, exist_ok=True)
    output_without_metadata.parent.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_with_metadata, index=False)
    print(f"✓ Saved with metadata: {output_with_metadata}")

    df_no_meta = df.drop(columns=["interpolation_method", "pm10_filled"], errors="ignore")
    df_no_meta = df_no_meta.drop(columns=["prediction_confidence"], errors="ignore")
    df_no_meta.to_csv(output_without_metadata, index=False)
    print(f"✓ Saved without metadata: {output_without_metadata}")

    # Save summary metrics
    summary_df = pd.DataFrame([r.__dict__ for r in results])
    summary_df.to_csv(summary_dir / "ml_gap_filling_summary.csv", index=False)
    print(f"✓ Saved summary metrics to {summary_dir / 'ml_gap_filling_summary.csv'}")

    print("\nSample of ML-filled rows:")
    print(df[ml_fill_mask].head(10)[["datetime", "station_code", "pm10", "prediction_confidence"]])

    print("\nDone!")


def parse_args():
    parser = argparse.ArgumentParser(description="Fill APPA PM10 gaps using ML regressor.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/appa-data/merged_pm10_hourly_curated.csv"),
        help="Curated dataset (with interpolation metadata).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/appa-data/merged_pm10_hourly_curated_ml_filled.csv"),
        help="Output CSV with ML predictions and metadata.",
    )
    parser.add_argument(
        "--output-no-meta",
        type=Path,
        default=Path("data/appa-data/merged_pm10_hourly_curated_ml_filled_no_interp_metadata.csv"),
        help="Output CSV without interpolation metadata columns.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("output/appa_ml_gap_filling"),
        help="Directory to store summary metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fill_gaps_with_regressor(
        curated_file=args.input,
        output_with_metadata=args.output,
        output_without_metadata=args.output_no_meta,
        summary_dir=args.summary_dir,
    )


