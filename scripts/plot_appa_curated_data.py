#!/usr/bin/env python3
"""
Plot APPA PM10 Curated Dataset

This script creates visualizations of the curated APPA PM10 dataset.
Generates both individual station plots and an overview plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import numpy as np


def categorize_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Add helper boolean columns for interpolation categories."""
    if "interpolation_method" not in df.columns:
        df["interpolation_method"] = "actual"

    df["is_actual"] = df["interpolation_method"] == "actual"
    df["is_linear"] = df["interpolation_method"] == "linear"
    df["is_copied"] = df["interpolation_method"].str.startswith("copied_from_", na=False)
    df["is_ml"] = df["interpolation_method"] == "ml_predicted"

    return df


def plot_appa_curated_data(data_file, output_dir=None, ml_summary_file=None):
    """
    Create visualizations for the APPA curated PM10 dataset.
    
    Args:
        data_file: Path to the curated PM10 CSV file
        output_dir: Optional directory to save plots (if None, displays interactively)
        ml_summary_file: Optional path to ml_gap_filling_summary.csv for overlay stats
    """
    print("=" * 80)
    print("APPA PM10 Curated Dataset Visualization")
    print("=" * 80)
    
    # Load data
    print(f"\n1. Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Convert station_code to string to avoid type comparison errors
    df['station_code'] = df['station_code'].astype(str)
    
    df = df.sort_values(['station_code', 'datetime'])
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
    
    # Get list of stations (now all strings, so sorting works)
    stations = sorted(df['station_code'].unique())
    
    # Check if interpolation_method column exists
    has_interpolation = 'interpolation_method' in df.columns

    # Load ML summary if provided
    if ml_summary_file:
        summary_path = Path(ml_summary_file)
        if summary_path.exists():
            ml_summary = pd.read_csv(summary_path)
            print(f"\nLoaded ML summary from {summary_path}:")
            print(ml_summary.to_string(index=False))
        else:
            print(f"\n⚠️  ML summary file not found at {summary_path}. Continuing without it.")
            ml_summary = None
    else:
        ml_summary = None
    
    # ========================================================================
    # Plot 1: Overview - All stations on one plot
    # ========================================================================
    print(f"\n3. Creating overview plot (all stations)...")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot each station with a different color
    colors = plt.cm.tab20(np.linspace(0, 1, len(stations)))
    
    for i, station_code in enumerate(stations):
        station_df = df[df['station_code'] == station_code]
        station_name = station_df['station_name'].iloc[0]
        
        # Plot actual measurements
        if has_interpolation:
            actual_data = station_df[station_df['is_actual']]
        else:
            actual_data = station_df[station_df['pm10'].notna()]
        
        if len(actual_data) > 0:
            ax.plot(actual_data['datetime'], actual_data['pm10'], 
                   color=colors[i], alpha=0.7, linewidth=0.5,
                   label=f"{station_code} ({station_name[:30]})")
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('PM10 (µg/m³)', fontsize=12)
    ax.set_title('APPA PM10 Curated Dataset - All Stations Overview\n(Actual Measurements Only)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / 'appa_overview_all_stations.png', dpi=300, bbox_inches='tight')
        print(f"   Saved: appa_overview_all_stations.png")
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
        station_df = df[df['station_code'] == station_code].copy()
        station_name = station_df['station_name'].iloc[0]
        
        # Separate actual and interpolated data
        if has_interpolation:
            actual_data = station_df[station_df['is_actual']]
            linear_data = station_df[station_df['is_linear']]
            copied_data = station_df[station_df['is_copied']]
            ml_data = station_df[station_df['is_ml']]
        else:
            actual_data = station_df[station_df['pm10'].notna()]
            linear_data = pd.DataFrame()
            copied_data = pd.DataFrame()
            ml_data = pd.DataFrame()
        
        # Plot actual measurements
        if len(actual_data) > 0:
            ax.plot(actual_data['datetime'], actual_data['pm10'], 
                   color='blue', alpha=0.6, linewidth=0.5, label='Actual')
        
        # Plot interpolated values
        if len(linear_data) > 0:
            ax.scatter(linear_data['datetime'], linear_data['pm10'], 
                      color='green', alpha=0.4, s=2, label='Linear interp.')
        
        if len(copied_data) > 0:
            ax.scatter(copied_data['datetime'], copied_data['pm10'],
                      color='purple', alpha=0.4, s=2, label='Nearest copy')
        
        if len(ml_data) > 0:
            ax.scatter(ml_data['datetime'], ml_data['pm10'],
                      color='orange', alpha=0.5, s=2, label='ML predicted')
        
        # Formatting
        ax.set_title(f"{station_code}\n{station_name[:40]}", fontsize=9, fontweight='bold')
        ax.set_xlabel('Date', fontsize=8)
        ax.set_ylabel('PM10 (µg/m³)', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='upper right')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.tick_params(axis='x', labelsize=7, rotation=45)
        ax.tick_params(axis='y', labelsize=7)
        
        # Statistics
        total_points = len(station_df)
        actual_points = len(actual_data)
        linear_points = len(linear_data)
        copied_points = len(copied_data)
        ml_points = len(ml_data)
        interp_points = linear_points + copied_points + ml_points
        interp_pct = (interp_points / total_points * 100) if total_points > 0 else 0
        
        # Add text box with statistics
        textstr = f'Total: {total_points:,}\nActual: {actual_points:,}'
        if has_interpolation:
            textstr += f'\nLinear: {linear_points:,}'
            textstr += f'\nCopied: {copied_points:,}'
            textstr += f'\nML: {ml_points:,}'
            textstr += f'\nInterp%: {interp_pct:.1f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=6,
               verticalalignment='top', bbox=props)
    
    # Hide unused subplots
    for i in range(n_stations, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle('APPA PM10 Curated Dataset - Individual Stations', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / 'appa_individual_stations.png', dpi=300, bbox_inches='tight')
        print(f"   Saved: appa_individual_stations.png")
    else:
        plt.show()
    
    plt.close()
    
    # ========================================================================
    # Plot 3: Monthly averages for all stations
    # ========================================================================
    print(f"\n5. Creating monthly averages plot...")
    
    # Calculate monthly averages
    df['year_month'] = df['datetime'].dt.to_period('M').dt.to_timestamp()
    monthly_avg = df.groupby(['station_code', 'station_name', 'year_month'])['pm10'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for i, station_code in enumerate(stations):
        station_data = monthly_avg[monthly_avg['station_code'] == station_code]
        station_name = station_data['station_name'].iloc[0]
        
        ax.plot(station_data['year_month'], station_data['pm10'], 
               color=colors[i], alpha=0.8, linewidth=1.5,
               label=f"{station_code} ({station_name[:30]})", marker='o', markersize=2)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('PM10 Monthly Average (µg/m³)', fontsize=12)
    ax.set_title('APPA PM10 Curated Dataset - Monthly Averages by Station', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / 'appa_monthly_averages.png', dpi=300, bbox_inches='tight')
        print(f"   Saved: appa_monthly_averages.png")
    else:
        plt.show()
    
    plt.close()
    
    # ========================================================================
    # Plot 4: Interpolation statistics (if available)
    # ========================================================================
    if has_interpolation:
        print(f"\n6. Creating interpolation statistics plot...")
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        
        # Plot 4a: Interpolation breakdown per station (stacked)
        ax = axes[0]
        
        breakdown = []
        for station_code in stations:
            station_data = df[df['station_code'] == station_code]
            station_name = station_data['station_name'].iloc[0]
            total = len(station_data)
            breakdown.append({
                'station_code': station_code,
                'station_name': station_name,
                'actual': station_data['is_actual'].sum(),
                'linear': station_data['is_linear'].sum(),
                'copied': station_data['is_copied'].sum(),
                'ml': station_data['is_ml'].sum(),
                'total': total
            })
        
        breakdown_df = pd.DataFrame(breakdown)
        breakdown_df = breakdown_df.sort_values('ml', ascending=False)
        
        indices = np.arange(len(breakdown_df))
        width = 0.6
        
        bottoms = np.zeros(len(breakdown_df))
        colors_map = {
            'actual': '#1f77b4',
            'linear': '#2ca02c',
            'copied': '#9467bd',
            'ml': '#ff7f0e'
        }
        
        for key in ['actual', 'linear', 'copied', 'ml']:
            values = breakdown_df[key] / breakdown_df['total'] * 100
            ax.bar(indices, values, width, bottom=bottoms, color=colors_map[key], label=key.capitalize())
            bottoms += values
        
        ax.set_xticks(indices)
        ax.set_xticklabels([f"{row['station_code']}\n{row['station_name'][:20]}" 
                            for _, row in breakdown_df.iterrows()],
                           rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Percentage of measurements (%)', fontsize=12)
        ax.set_title('Measurement breakdown by method per station', fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(loc='upper right')
        
        # Plot 4b: Yearly breakdown for linear & ML
        ax = axes[1]
        df['year'] = df['datetime'].dt.year
        yearly = df.groupby('year').agg({
            'is_linear': 'sum',
            'is_copied': 'sum',
            'is_ml': 'sum',
            'pm10': 'count'
        }).reset_index()
        yearly['linear_pct'] = yearly['is_linear'] / yearly['pm10'] * 100
        yearly['copied_pct'] = yearly['is_copied'] / yearly['pm10'] * 100
        yearly['ml_pct'] = yearly['is_ml'] / yearly['pm10'] * 100
        
        ax.plot(yearly['year'], yearly['linear_pct'], marker='o', linewidth=2, color=colors_map['linear'], label='Linear')
        ax.plot(yearly['year'], yearly['copied_pct'], marker='o', linewidth=2, color=colors_map['copied'], label='Copied')
        ax.plot(yearly['year'], yearly['ml_pct'], marker='o', linewidth=2, color=colors_map['ml'], label='ML')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Percentage of measurements (%)', fontsize=12)
        ax.set_title('Interpolation percentage over time', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        for _, row in yearly.iterrows():
            ax.text(row['year'], row['ml_pct'], f"{row['ml_pct']:.1f}%", fontsize=8, color=colors_map['ml'], ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_dir / 'appa_interpolation_statistics.png', dpi=300, bbox_inches='tight')
            print(f"   Saved: appa_interpolation_statistics.png")
        else:
            plt.show()
        
        plt.close()
    
    # Plot ML summary if file provided
    if ml_summary is not None:
        print(f"\n7. Creating ML summary plot...")
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        ax1.bar(ml_summary['station_code'], ml_summary['n_predicted'], color='#ff7f0e', alpha=0.7)
        ax1.set_ylabel('Predicted values (count)', color='#ff7f0e')
        ax1.tick_params(axis='y', labelcolor='#ff7f0e')
        ax1.set_xlabel('Station')
        ax1.set_title('ML Gap Filling Summary')
        
        ax2 = ax1.twinx()
        ax2.plot(ml_summary['station_code'], ml_summary['confidence'], color='black', marker='o', label='Confidence')
        ax2.set_ylabel('Confidence', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_ylim(0, 1.05)
        
        fig.tight_layout()
        if output_dir:
            plt.savefig(output_dir / 'appa_ml_gap_summary.png', dpi=300, bbox_inches='tight')
            print(f"   Saved: appa_ml_gap_summary.png")
        else:
            plt.show()
        plt.close()
    
    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot APPA curated dataset with interpolation/ML breakdown.")
    workspace_default = Path(__file__).parent.parent
    parser.add_argument(
        "--data-file",
        type=Path,
        default=workspace_default / "data" / "appa-data" / "merged_pm10_hourly_curated.csv",
        help="Curated dataset CSV (with interpolation metadata).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace_default / "output" / "appa_plots",
        help="Directory to store generated plots.",
    )
    parser.add_argument(
        "--ml-summary-file",
        type=Path,
        default=workspace_default / "output" / "appa_ml_gap_filling" / "ml_gap_filling_summary.csv",
        help="Optional ML summary CSV (ml_gap_filling_summary.csv).",
    )
    args = parser.parse_args()

    plot_appa_curated_data(
        data_file=args.data_file,
        output_dir=args.output_dir,
        ml_summary_file=args.ml_summary_file,
    )

