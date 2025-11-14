#!/usr/bin/env python3
"""
Plot ARPAL PM10 Curated Dataset

This script creates visualizations of the curated ARPAL PM10 dataset.
Generates both individual station plots and an overview plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import numpy as np

def plot_arpal_curated_data(data_file, output_dir=None):
    """
    Create visualizations for the ARPAL curated PM10 dataset.
    
    Args:
        data_file: Path to the curated PM10 CSV file
        output_dir: Optional directory to save plots (if None, displays interactively)
    """
    print("=" * 80)
    print("ARPAL PM10 Curated Dataset Visualization")
    print("=" * 80)
    
    # Load data
    print(f"\n1. Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['station_code', 'datetime'])
    
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
    stations = sorted(df['station_code'].unique())
    
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
        actual_data = station_df[station_df['interpolation_method'] == 'actual']
        if len(actual_data) > 0:
            ax.plot(actual_data['datetime'], actual_data['pm10'], 
                   color=colors[i], alpha=0.7, linewidth=0.5,
                   label=f"{station_code} ({station_name[:30]})")
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('PM10 (µg/m³)', fontsize=12)
    ax.set_title('ARPAL PM10 Curated Dataset - All Stations Overview\n(Actual Measurements Only)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / 'arpal_overview_all_stations.png', dpi=300, bbox_inches='tight')
        print(f"   Saved: arpal_overview_all_stations.png")
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
        actual_data = station_df[station_df['interpolation_method'] == 'actual']
        interpolated_data = station_df[station_df['interpolation_method'] != 'actual']
        
        # Plot actual measurements
        if len(actual_data) > 0:
            ax.plot(actual_data['datetime'], actual_data['pm10'], 
                   color='blue', alpha=0.6, linewidth=0.5, label='Actual')
        
        # Plot interpolated values with different colors based on method
        if len(interpolated_data) > 0:
            # Linear interpolation
            linear_data = interpolated_data[interpolated_data['interpolation_method'] == 'linear']
            if len(linear_data) > 0:
                ax.scatter(linear_data['datetime'], linear_data['pm10'], 
                          color='green', alpha=0.3, s=1, label='Linear interp.')
            
            # Distance-weighted interpolation
            dist_data = interpolated_data[interpolated_data['interpolation_method'] == 'distance_weighted']
            if len(dist_data) > 0:
                ax.scatter(dist_data['datetime'], dist_data['pm10'], 
                          color='red', alpha=0.3, s=1, label='Dist. weighted')
        
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
        interp_points = len(interpolated_data)
        interp_pct = (interp_points / total_points * 100) if total_points > 0 else 0
        
        # Add text box with statistics
        textstr = f'Total: {total_points:,}\nActual: {actual_points:,}\nInterp: {interp_points:,} ({interp_pct:.1f}%)'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=6,
               verticalalignment='top', bbox=props)
    
    # Hide unused subplots
    for i in range(n_stations, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle('ARPAL PM10 Curated Dataset - Individual Stations', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / 'arpal_individual_stations.png', dpi=300, bbox_inches='tight')
        print(f"   Saved: arpal_individual_stations.png")
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
    ax.set_title('ARPAL PM10 Curated Dataset - Monthly Averages by Station', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / 'arpal_monthly_averages.png', dpi=300, bbox_inches='tight')
        print(f"   Saved: arpal_monthly_averages.png")
    else:
        plt.show()
    
    plt.close()
    
    # ========================================================================
    # Plot 4: Interpolation confidence distribution
    # ========================================================================
    if 'interpolation_confidence' in df.columns:
        print(f"\n6. Creating interpolation confidence plot...")
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Plot 4a: Confidence histogram
        ax = axes[0]
        interp_data = df[df['interpolation_confidence'].notna()]
        
        if len(interp_data) > 0:
            ax.hist(interp_data['interpolation_confidence'], bins=50, 
                   color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Interpolation Confidence', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of Interpolation Confidence Values', 
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add statistics
            mean_conf = interp_data['interpolation_confidence'].mean()
            median_conf = interp_data['interpolation_confidence'].median()
            ax.axvline(mean_conf, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_conf:.3f}')
            ax.axvline(median_conf, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_conf:.3f}')
            ax.legend(fontsize=10)
        
        # Plot 4b: Confidence per station (boxplot)
        ax = axes[1]
        
        # Prepare data for boxplot
        conf_by_station = []
        station_labels = []
        
        for station_code in stations:
            station_interp = df[(df['station_code'] == station_code) & 
                               (df['interpolation_confidence'].notna())]
            if len(station_interp) > 0:
                conf_by_station.append(station_interp['interpolation_confidence'].values)
                station_labels.append(station_code)
        
        if conf_by_station:
            bp = ax.boxplot(conf_by_station, labels=station_labels, patch_artist=True)
            
            # Color boxplots
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            ax.set_xlabel('Station Code', fontsize=12)
            ax.set_ylabel('Interpolation Confidence', fontsize=12)
            ax.set_title('Interpolation Confidence by Station', 
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_dir / 'arpal_interpolation_confidence.png', dpi=300, bbox_inches='tight')
            print(f"   Saved: arpal_interpolation_confidence.png")
        else:
            plt.show()
        
        plt.close()
    
    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Paths
    workspace = Path(__file__).parent.parent
    data_file = workspace / "data" / "arpal" / "PM10" / "merged_pm10_hourly_curated.csv"
    output_dir = workspace / "output" / "arpal_plots"
    
    # Create plots
    plot_arpal_curated_data(data_file, output_dir)

