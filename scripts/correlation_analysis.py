#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
APPA Aria Monthly Correlation Analysis Script

This script analyzes correlations between air quality measurements across different
weather stations in 30-day intervals and plots monthly correlation series for each pollutant.

Usage:
    python correlation_analysis.py [--data-folder DATA_FOLDER] [--pollutant POLLUTANT]
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data(data_folder: Path) -> pd.DataFrame:
    """Load the merged CSV data with proper encoding handling."""
    merged_file = data_folder / "merged_data.csv"
    
    if not merged_file.exists():
        raise FileNotFoundError(f"Merged data file not found: {merged_file}")
    
    # Try different encodings
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            df = pd.read_csv(merged_file, encoding=encoding)
            print(f"Successfully loaded data with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Could not read the CSV file with any encoding")
    
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data for correlation analysis."""
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Remove rows with missing values
    df = df.dropna(subset=['Valore'])
    
    # Convert Valore to numeric
    df['Valore'] = pd.to_numeric(df['Valore'], errors='coerce')
    df = df.dropna(subset=['Valore'])
    
    # Handle hour 24 (convert to 00:00 of next day)
    df['Ora'] = df['Ora'].astype(int)
    
    # Create a copy to track original hour 24 values
    df['OriginalHour'] = df['Ora'].copy()
    
    # Convert hour 24 to 0 and add 1 day to the date
    hour_24_mask = df['Ora'] == 24
    df.loc[hour_24_mask, 'Ora'] = 0
    
    # Convert date and time columns
    df['DateTime'] = pd.to_datetime(df['Data'] + ' ' + df['Ora'].astype(str).str.zfill(2) + ':00:00')
    
    # Add 1 day to rows that were originally hour 24
    df.loc[hour_24_mask, 'DateTime'] = df.loc[hour_24_mask, 'DateTime'] + pd.Timedelta(days=1)
    
    # Drop the temporary column
    df = df.drop('OriginalHour', axis=1)
    
    # Create a pivot table for correlation analysis
    # Each row represents a timestamp, columns are station-pollutant combinations
    df_pivot = df.pivot_table(
        index='DateTime',
        columns=['Stazione', 'Inquinante'],
        values='Valore',
        aggfunc='mean'
    )
    
    return df, df_pivot

def create_monthly_intervals(df: pd.DataFrame) -> list:
    """Create 30-day intervals for the data."""
    start_date = df['DateTime'].min().date()
    end_date = df['DateTime'].max().date()
    
    intervals = []
    current_date = start_date
    
    while current_date < end_date:
        interval_end = min(current_date + timedelta(days=30), end_date)
        intervals.append((current_date, interval_end))
        current_date = interval_end
    
    return intervals

def calculate_monthly_correlations(df_pivot: pd.DataFrame, intervals: list) -> pd.DataFrame:
    """Calculate correlations for each 30-day interval."""
    correlation_results = []
    
    for start_date, end_date in intervals:
        # Filter data for this interval
        mask = (df_pivot.index.date >= start_date) & (df_pivot.index.date <= end_date)
        interval_data = df_pivot[mask]
        
        if len(interval_data) < 10:  # Skip intervals with too few data points
            continue
        
        # Calculate correlation matrix
        corr_matrix = interval_data.corr()
        
        # Extract correlations between different stations for each pollutant
        pollutants = set()
        stations = set()
        
        for col in corr_matrix.columns:
            station, pollutant = col
            pollutants.add(pollutant)
            stations.add(station)
        
        # Calculate average correlation for each pollutant
        for pollutant in pollutants:
            pollutant_cols = [col for col in corr_matrix.columns if col[1] == pollutant]
            
            if len(pollutant_cols) > 1:
                # Get correlations between different stations for this pollutant
                pollutant_corr = corr_matrix.loc[pollutant_cols, pollutant_cols]
                
                # Get upper triangle (excluding diagonal)
                upper_triangle = np.triu(pollutant_corr.values, k=1)
                correlations = upper_triangle[upper_triangle != 0]
                
                if len(correlations) > 0:
                    avg_correlation = np.mean(correlations)
                    correlation_results.append({
                        'Date': start_date,
                        'Pollutant': pollutant,
                        'Avg_Correlation': avg_correlation,
                        'N_Stations': len(pollutant_cols),
                        'N_Correlations': len(correlations)
                    })
    
    return pd.DataFrame(correlation_results)

def plot_correlation_series(corr_df: pd.DataFrame, save_path: Path = None):
    """Plot monthly correlation series for each pollutant."""
    pollutants = corr_df['Pollutant'].unique()
    n_pollutants = len(pollutants)
    
    fig, axes = plt.subplots(n_pollutants, 1, figsize=(15, 5 * n_pollutants))
    if n_pollutants == 1:
        axes = [axes]
    
    for i, pollutant in enumerate(pollutants):
        ax = axes[i]
        pol_data = corr_df[corr_df['Pollutant'] == pollutant].copy()
        
        if not pol_data.empty:
            # Sort by date
            pol_data = pol_data.sort_values('Date')
            
            # Plot correlation series
            ax.plot(pol_data['Date'], pol_data['Avg_Correlation'], 
                   marker='o', linewidth=2, markersize=6)
            
            # Add trend line
            if len(pol_data) > 1:
                z = np.polyfit(range(len(pol_data)), pol_data['Avg_Correlation'], 1)
                p = np.poly1d(z)
                ax.plot(pol_data['Date'], p(range(len(pol_data))), 
                       '--', alpha=0.7, color='red', linewidth=1)
            
            ax.set_title(f'Monthly Correlation Series - {pollutant}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Average Correlation')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1, 1)
            
            # Add horizontal line at 0
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Rotate x-axis labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Add statistics text
            mean_corr = pol_data['Avg_Correlation'].mean()
            std_corr = pol_data['Avg_Correlation'].std()
            ax.text(0.02, 0.98, f'Mean: {mean_corr:.3f}\nStd: {std_corr:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation series plot saved to: {save_path}")
    
    plt.show()

def plot_correlation_heatmap(corr_df: pd.DataFrame, save_path: Path = None):
    """Create a heatmap of correlations over time."""
    # Pivot the data for heatmap
    heatmap_data = corr_df.pivot(index='Date', columns='Pollutant', values='Avg_Correlation')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data.T, annot=True, cmap='RdBu_r', center=0, 
                fmt='.3f', cbar_kws={'label': 'Average Correlation'})
    plt.title('Monthly Correlation Heatmap Across Pollutants')
    plt.xlabel('Date')
    plt.ylabel('Pollutant')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to: {save_path}")
    
    plt.show()

def plot_correlation_distribution(corr_df: pd.DataFrame, save_path: Path = None):
    """Plot distribution of correlations for each pollutant."""
    pollutants = corr_df['Pollutant'].unique()
    n_pollutants = len(pollutants)
    
    fig, axes = plt.subplots(1, n_pollutants, figsize=(5 * n_pollutants, 5))
    if n_pollutants == 1:
        axes = [axes]
    
    for i, pollutant in enumerate(pollutants):
        ax = axes[i]
        pol_data = corr_df[corr_df['Pollutant'] == pollutant]['Avg_Correlation']
        
        # Create histogram
        ax.hist(pol_data, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(pol_data.mean(), color='red', linestyle='--', 
                  label=f'Mean: {pol_data.mean():.3f}')
        ax.set_title(f'Correlation Distribution - {pollutant}')
        ax.set_xlabel('Average Correlation')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation distribution plot saved to: {save_path}")
    
    plt.show()

def plot_station_pairwise_correlations(df_pivot: pd.DataFrame, intervals: list, save_path: Path = None):
    """Plot pairwise correlations between stations for each pollutant over time."""
    # Get unique stations and pollutants
    stations = set()
    pollutants = set()
    
    for col in df_pivot.columns:
        station, pollutant = col
        stations.add(station)
        pollutants.add(pollutant)
    
    stations = sorted(list(stations))
    pollutants = sorted(list(pollutants))
    
    # Create a consistent color map for stations
    station_colors = {}
    colors = plt.cm.Set1(np.linspace(0, 1, len(stations)))
    for i, station in enumerate(stations):
        station_colors[station] = colors[i]
    
    # Create subplots for each pollutant
    n_pollutants = len(pollutants)
    fig, axes = plt.subplots(n_pollutants, 1, figsize=(20, 5 * n_pollutants))
    if n_pollutants == 1:
        axes = [axes]
    
    for i, pollutant in enumerate(pollutants):
        ax = axes[i]
        
        # Get columns for this pollutant
        pollutant_cols = [col for col in df_pivot.columns if col[1] == pollutant]
        
        if len(pollutant_cols) < 2:
            ax.text(0.5, 0.5, f'Insufficient stations for {pollutant}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'Station Pairwise Correlations - {pollutant}')
            continue
        
        # Calculate correlations for each time interval
        correlation_data = []
        
        for start_date, end_date in intervals:
            # Filter data for this interval
            mask = (df_pivot.index.date >= start_date) & (df_pivot.index.date <= end_date)
            interval_data = df_pivot[mask]
            
            if len(interval_data) < 10:  # Skip intervals with too few data points
                continue
            
            # Get data for this pollutant
            pol_data = interval_data[pollutant_cols]
            
            # Calculate correlation matrix
            corr_matrix = pol_data.corr()
            
            # Extract pairwise correlations
            for j in range(len(pollutant_cols)):
                for k in range(j + 1, len(pollutant_cols)):
                    station1 = pollutant_cols[j][0]
                    station2 = pollutant_cols[k][0]
                    correlation = corr_matrix.iloc[j, k]
                    
                    if not pd.isna(correlation):
                        correlation_data.append({
                            'Date': start_date,
                            'Station1': station1,
                            'Station2': station2,
                            'Correlation': correlation
                        })
        
        if not correlation_data:
            ax.text(0.5, 0.5, f'No correlation data for {pollutant}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'Station Pairwise Correlations - {pollutant}')
            continue
        
        # Convert to DataFrame
        corr_df = pd.DataFrame(correlation_data)
        
        # Plot each station pair with consistent colors
        station_pairs = corr_df.groupby(['Station1', 'Station2'])
        
        for (station1, station2), group in station_pairs:
            group = group.sort_values('Date')
            # Use a color that represents both stations (mix or use one station's color)
            color = station_colors[station1]
            ax.plot(group['Date'], group['Correlation'], 
                   marker='o', linewidth=2, markersize=4, alpha=0.8,
                   color=color, label=f'{station1} - {station2}')
        
        ax.set_title(f'Station Pairwise Correlations - {pollutant}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Correlation', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1, 1)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add legend with all station pairs
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, 
                 frameon=True, fancybox=True, shadow=True)
        
        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Add a global legend showing station colors
    if len(stations) > 0:
        # Create a separate legend for station colors
        legend_elements = [plt.Line2D([0], [0], color=station_colors[station], 
                                     label=station, linewidth=3) for station in stations]
        
        # Add the station color legend to the last subplot
        last_ax = axes[-1]
        station_legend = last_ax.legend(handles=legend_elements, 
                                       title='Station Colors', 
                                       bbox_to_anchor=(1.05, 0), 
                                       loc='lower left', 
                                       fontsize=10,
                                       frameon=True, 
                                       fancybox=True, 
                                       shadow=True)
        station_legend.get_title().set_fontweight('bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Station pairwise correlations plot saved to: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze monthly correlations in APPA air quality data")
    parser.add_argument("--data-folder", default="appa-data", 
                       help="Path to the data folder containing merged_data.csv")
    parser.add_argument("--pollutant", default=None,
                       help="Specific pollutant to analyze (e.g., PM10, NO2)")
    parser.add_argument("--output-dir", default=None,
                       help="Directory to save plots (default: auto-generated based on data)")
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    data_folder = project_root / args.data_folder
    
    # Load data first to determine default output directory
    print("Loading data...")
    df = load_data(data_folder)
    df, df_pivot = preprocess_data(df)
    
    # Generate default output directory name
    if args.output_dir is None:
        # Extract date range from data
        start_date = df['DateTime'].min().strftime('%Y-%m-%d')
        end_date = df['DateTime'].max().strftime('%Y-%m-%d')
        
        # Get pollutants for this analysis
        if args.pollutant:
            pollutants_str = args.pollutant
        else:
            pollutants = sorted(df['Inquinante'].unique())
            pollutants_str = '_'.join(pollutants)
        
        # Create descriptive folder name within plots directory
        default_output_name = f"correlations_{start_date}_to_{end_date}_{pollutants_str}"
        output_dir = project_root / "plots" / default_output_name
    else:
        output_dir = project_root / "plots" / args.output_dir
    
    output_dir.mkdir(exist_ok=True)
    
    try:
        print(f"Loaded {len(df)} records")
        print(f"After preprocessing: {len(df)} records")
        print(f"Pivot table shape: {df_pivot.shape}")
        print(f"Output directory: {output_dir}")
        
        # Create monthly intervals
        print("Creating monthly intervals...")
        intervals = create_monthly_intervals(df)
        print(f"Created {len(intervals)} monthly intervals")
        
        # Calculate correlations
        print("Calculating monthly correlations...")
        corr_df = calculate_monthly_correlations(df_pivot, intervals)
        print(f"Calculated correlations for {len(corr_df)} intervals")
        
        if corr_df.empty:
            print("No correlation data calculated. Check if you have multiple stations for the same pollutant.")
            return 1
        
        # Filter by pollutant if specified
        if args.pollutant:
            corr_df = corr_df[corr_df['Pollutant'] == args.pollutant]
            if corr_df.empty:
                print(f"No data found for pollutant: {args.pollutant}")
                return 1
        
        # Print summary statistics
        print("\nCorrelation Summary:")
        for pollutant in corr_df['Pollutant'].unique():
            pol_data = corr_df[corr_df['Pollutant'] == pollutant]
            print(f"{pollutant}: Mean={pol_data['Avg_Correlation'].mean():.3f}, "
                  f"Std={pol_data['Avg_Correlation'].std():.3f}, "
                  f"N={len(pol_data)}")
        
        # Create visualizations
        print("\nCreating correlation series plots...")
        plot_correlation_series(corr_df, 
                               output_dir / f"correlation_series_{args.pollutant or 'all'}.png")
        
        print("Creating correlation heatmap...")
        plot_correlation_heatmap(corr_df, output_dir / "correlation_heatmap.png")
        
        print("Creating correlation distribution plots...")
        plot_correlation_distribution(corr_df, output_dir / "correlation_distributions.png")
        
        print("Creating station pairwise correlation plots...")
        plot_station_pairwise_correlations(df_pivot, intervals, output_dir / "station_pairwise_correlations.png")
        
        # Save correlation data
        corr_df.to_csv(output_dir / "monthly_correlations.csv", index=False)
        print(f"Correlation data saved to: {output_dir / 'monthly_correlations.csv'}")
        
        print(f"\nAll plots saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
