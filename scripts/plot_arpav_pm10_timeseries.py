#!/usr/bin/env python3
"""
Plot ARPAV PM10 Curated Dataset

This script mirrors the APPA visualization workflow (minus ML overlays) and
produces the same set of figures for the ARPAV curated dataset.
"""

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def categorize_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Add helper boolean columns for interpolation categories."""
    if "interpolation_method" not in df.columns:
        df["interpolation_method"] = "actual"

    df["is_actual"] = df["interpolation_method"] == "actual"
    df["is_linear"] = df["interpolation_method"] == "linear"
    df["is_missing"] = df["interpolation_method"] == "missing"

    return df


def plot_arpav_curated_data(data_file, output_dir=None):
    """
    Create visualizations for the ARPAV curated PM10 dataset.

    Args:
        data_file: Path to the curated PM10 CSV file (with interpolation metadata)
        output_dir: Optional directory to save plots (if None, displays interactively)
    """
    print("=" * 80)
    print("ARPAV PM10 Curated Dataset Visualization")
    print("=" * 80)

    # Load data
    print(f"\n1. Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Convert station_code to string for consistent sorting
    df["station_code"] = df["station_code"].astype(str)

    df = df.sort_values(["station_code", "datetime"])
    df = categorize_methods(df)

    print(f"   Loaded {len(df):,} rows")
    print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"   Stations: {df['station_code'].nunique()}")

    # Create output directory if needed
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n2. Saving plots to: {output_dir}")
    else:
        print(f"\n2. Displaying plots interactively")

    # Get list of stations
    stations = sorted(df["station_code"].unique())
    has_interpolation = "interpolation_method" in df.columns

    # ========================================================================
    # Plot 1: Overview - All stations on one plot
    # ========================================================================
    print(f"\n3. Creating overview plot (all stations)...")

    fig, ax = plt.subplots(figsize=(16, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(stations)))

    for i, station_code in enumerate(stations):
        station_df = df[df["station_code"] == station_code]
        station_name = station_df["station_name"].iloc[0]

        actual_data = station_df[station_df["is_actual"]] if has_interpolation else station_df[station_df["pm10"].notna()]

        if len(actual_data) > 0:
            ax.plot(
                actual_data["datetime"],
                actual_data["pm10"],
                color=colors[i],
                alpha=0.7,
                linewidth=0.5,
                label=f"{station_code} ({station_name[:30]})",
            )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("PM10 (µg/m³)", fontsize=12)
    ax.set_title(
        "ARPAV PM10 Curated Dataset - All Stations Overview\n(Actual Measurements Only)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()

    if output_dir:
        plt.savefig(output_dir / "arpav_overview_all_stations.png", dpi=300, bbox_inches="tight")
        print(f"   Saved: arpav_overview_all_stations.png")
    else:
        plt.show()

    plt.close()

    # ========================================================================
    # Plot 2: Individual station plots (grid layout)
    # ========================================================================
    print(f"\n4. Creating individual station plots...")

    n_stations = len(stations)
    n_cols = 3
    n_rows = (n_stations + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten() if n_stations > 1 else [axes]

    for i, station_code in enumerate(stations):
        ax = axes[i]
        station_df = df[df["station_code"] == station_code].copy()
        station_name = station_df["station_name"].iloc[0]

        actual_data = station_df[station_df["is_actual"]]
        linear_data = station_df[station_df["is_linear"]]

        if len(actual_data) > 0:
            ax.plot(
                actual_data["datetime"],
                actual_data["pm10"],
                color="blue",
                alpha=0.6,
                linewidth=0.5,
                label="Actual",
            )

        if len(linear_data) > 0:
            ax.scatter(
                linear_data["datetime"],
                linear_data["pm10"],
                color="green",
                alpha=0.4,
                s=2,
                label="Linear interp.",
            )

        ax.set_title(f"{station_code}\n{station_name[:40]}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Date", fontsize=8)
        ax.set_ylabel("PM10 (µg/m³)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc="upper right")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.tick_params(axis="y", labelsize=7)

        total_points = len(station_df)
        actual_points = len(actual_data)
        linear_points = len(linear_data)
        interp_points = linear_points
        interp_pct = (interp_points / total_points * 100) if total_points > 0 else 0

        textstr = f"Total: {total_points:,}\nActual: {actual_points:,}"
        if has_interpolation:
            textstr += f"\nLinear: {linear_points:,}"
            textstr += f"\nInterp%: {interp_pct:.1f}%"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.3)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=6, verticalalignment="top", bbox=props)

    for i in range(n_stations, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("ARPAV PM10 Curated Dataset - Individual Stations", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()

    if output_dir:
        plt.savefig(output_dir / "arpav_individual_stations.png", dpi=300, bbox_inches="tight")
        print(f"   Saved: arpav_individual_stations.png")
    else:
        plt.show()

    plt.close()

    # ========================================================================
    # Plot 3: Monthly averages for all stations
    # ========================================================================
    print(f"\n5. Creating monthly averages plot...")

    df["year_month"] = df["datetime"].dt.to_period("M").dt.to_timestamp()
    monthly_avg = df.groupby(["station_code", "station_name", "year_month"])["pm10"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(16, 8))

    for i, station_code in enumerate(stations):
        station_data = monthly_avg[monthly_avg["station_code"] == station_code]
        station_name = station_data["station_name"].iloc[0]

        ax.plot(
            station_data["year_month"],
            station_data["pm10"],
            color=colors[i],
            alpha=0.8,
            linewidth=1.5,
            label=f"{station_code} ({station_name[:30]})",
            marker="o",
            markersize=2,
        )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("PM10 Monthly Average (µg/m³)", fontsize=12)
    ax.set_title("ARPAV PM10 Curated Dataset - Monthly Averages by Station", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()

    if output_dir:
        plt.savefig(output_dir / "arpav_monthly_averages.png", dpi=300, bbox_inches="tight")
        print(f"   Saved: arpav_monthly_averages.png")
    else:
        plt.show()

    plt.close()

    # ========================================================================
    # Plot 4: Interpolation statistics
    # ========================================================================
    if has_interpolation:
        print(f"\n6. Creating interpolation statistics plot...")

        fig, axes = plt.subplots(2, 1, figsize=(16, 12))

        # Plot 4a: Interpolation breakdown per station (stacked)
        ax = axes[0]

        breakdown = []
        for station_code in stations:
            station_data = df[df["station_code"] == station_code]
            station_name = station_data["station_name"].iloc[0]
            total = len(station_data)
            breakdown.append(
                {
                    "station_code": station_code,
                    "station_name": station_name,
                    "actual": station_data["is_actual"].sum(),
                    "linear": station_data["is_linear"].sum(),
                    "missing": station_data["is_missing"].sum(),
                    "total": total,
                }
            )

        breakdown_df = pd.DataFrame(breakdown)
        indices = np.arange(len(breakdown_df))
        width = 0.6

        bottoms = np.zeros(len(breakdown_df))
        colors_map = {"actual": "#1f77b4", "linear": "#2ca02c", "missing": "#d62728"}

        for key in ["actual", "linear", "missing"]:
            values = breakdown_df[key] / breakdown_df["total"] * 100
            ax.bar(indices, values, width, bottom=bottoms, color=colors_map[key], label=key.capitalize())
            bottoms += values

        ax.set_xticks(indices)
        ax.set_xticklabels(
            [f"{row['station_code']}\n{row['station_name'][:20]}" for _, row in breakdown_df.iterrows()],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_ylabel("Percentage of measurements (%)", fontsize=12)
        ax.set_title("Measurement breakdown by method per station", fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper right")

        # Plot 4b: Yearly breakdown for linear fills
        ax = axes[1]
        df["year"] = df["datetime"].dt.year
        yearly = df.groupby("year").agg({"is_linear": "sum", "pm10": "count"}).reset_index()
        yearly["linear_pct"] = yearly["is_linear"] / yearly["pm10"] * 100

        ax.plot(
            yearly["year"],
            yearly["linear_pct"],
            marker="o",
            linewidth=2,
            color=colors_map["linear"],
            label="Linear",
        )
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Percentage of measurements (%)", fontsize=12)
        ax.set_title("Linear interpolation percentage over time", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

        for _, row in yearly.iterrows():
            ax.text(
                row["year"],
                row["linear_pct"],
                f"{row['linear_pct']:.1f}%",
                fontsize=8,
                color=colors_map["linear"],
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        if output_dir:
            plt.savefig(output_dir / "arpav_interpolation_statistics.png", dpi=300, bbox_inches="tight")
            print(f"   Saved: arpav_interpolation_statistics.png")
        else:
            plt.show()

        plt.close()

    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ARPAV curated dataset with interpolation breakdown.")
    workspace_default = Path(__file__).parent.parent
    parser.add_argument(
        "--data-file",
        type=Path,
        default=workspace_default / "data" / "arpav" / "PM10" / "merged_pm10_hourly_curated_with_interp_metadata.csv",
        help="Curated dataset CSV (with interpolation metadata).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace_default / "output" / "arpav_plots",
        help="Directory to store generated plots.",
    )
    args = parser.parse_args()

    plot_arpav_curated_data(
        data_file=args.data_file,
        output_dir=args.output_dir,
    )

