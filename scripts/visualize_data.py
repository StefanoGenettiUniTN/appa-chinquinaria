#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
APPA Aria Data Visualization Script

This script visualizes the downloaded APPA air quality data as time series.
It reads the merged CSV file and creates interactive plots for different pollutants.

Usage:
    python visualize_data.py [--data-folder DATA_FOLDER] [--pollutant POLLUTANT]
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime
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
    """Preprocess the data for visualization."""
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
    
    return df

def plot_time_series(df: pd.DataFrame, pollutant: str = None, save_path: Path = None):
    """Create time series plots for air quality data."""
    
    if pollutant:
        df_filtered = df[df['Inquinante'] == pollutant].copy()
        title_suffix = f" - {pollutant}"
    else:
        df_filtered = df.copy()
        title_suffix = " - All Pollutants"
    
    # Get unique stations and pollutants
    stations = df_filtered['Stazione'].unique()
    pollutants = df_filtered['Inquinante'].unique()
    
    # Create subplots
    n_pollutants = len(pollutants)
    n_stations = len(stations)
    
    fig, axes = plt.subplots(n_pollutants, 1, figsize=(15, 5 * n_pollutants))
    if n_pollutants == 1:
        axes = [axes]
    
    for i, pol in enumerate(pollutants):
        ax = axes[i]
        pol_data = df_filtered[df_filtered['Inquinante'] == pol]
        
        # Plot each station
        for station in stations:
            station_data = pol_data[pol_data['Stazione'] == station]
            if not station_data.empty:
                # Group by date and calculate daily mean
                daily_means = station_data.groupby(station_data['DateTime'].dt.date)['Valore'].mean()
                ax.plot(daily_means.index, daily_means.values, label=station, alpha=0.7, linewidth=1)
        
        ax.set_title(f'Time Series - {pol}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Concentration')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def plot_station_comparison(df: pd.DataFrame, pollutant: str, save_path: Path = None):
    """Create comparison plots for different stations."""
    
    df_filtered = df[df['Inquinante'] == pollutant].copy()
    stations = df_filtered['Stazione'].unique()
    
    # Create subplots for each station
    n_stations = len(stations)
    fig, axes = plt.subplots(n_stations, 1, figsize=(15, 4 * n_stations))
    if n_stations == 1:
        axes = [axes]
    
    for i, station in enumerate(stations):
        ax = axes[i]
        station_data = df_filtered[df_filtered['Stazione'] == station]
        
        if not station_data.empty:
            # Group by date and calculate daily statistics
            daily_stats = station_data.groupby(station_data['DateTime'].dt.date)['Valore'].agg(['mean', 'min', 'max'])
            
            ax.fill_between(daily_stats.index, daily_stats['min'], daily_stats['max'], 
                           alpha=0.3, label='Min-Max Range')
            ax.plot(daily_stats.index, daily_stats['mean'], label='Daily Mean', linewidth=2)
            
            ax.set_title(f'{station} - {pollutant}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Concentration')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Station comparison plot saved to: {save_path}")
    
    plt.show()

def plot_pollutant_distribution(df: pd.DataFrame, save_path: Path = None):
    """Create distribution plots for different pollutants."""
    
    pollutants = df['Inquinante'].unique()
    n_pollutants = len(pollutants)
    
    fig, axes = plt.subplots(2, (n_pollutants + 1) // 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, pol in enumerate(pollutants):
        ax = axes[i]
        pol_data = df[df['Inquinante'] == pol]['Valore']
        
        # Create histogram
        ax.hist(pol_data, bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'Distribution - {pol}')
        ax.set_xlabel('Concentration')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_pollutants, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize APPA air quality data")
    parser.add_argument("--data-folder", default="appa-data", 
                       help="Path to the data folder containing merged_data.csv")
    parser.add_argument("--pollutant", default=None,
                       help="Specific pollutant to visualize (e.g., PM10, NO2)")
    parser.add_argument("--output-dir", default=None,
                       help="Directory to save plots (default: auto-generated based on data)")
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    data_folder = project_root / args.data_folder
    
    # Load data first to determine default output directory
    print("Loading data...")
    df = load_data(data_folder)
    df = preprocess_data(df)
    
    # Generate default output directory name
    if args.output_dir is None:
        # Extract date range from data
        start_date = df['DateTime'].min().strftime('%Y-%m-%d')
        end_date = df['DateTime'].max().strftime('%Y-%m-%d')
        
        # Get pollutants for this visualization
        if args.pollutant:
            pollutants_str = args.pollutant
        else:
            pollutants = sorted(df['Inquinante'].unique())
            pollutants_str = '_'.join(pollutants)
        
        # Create descriptive folder name within plots directory
        default_output_name = f"{start_date}_to_{end_date}_{pollutants_str}"
        output_dir = project_root / "plots" / default_output_name
    else:
        output_dir = project_root / "plots" / args.output_dir
    
    output_dir.mkdir(exist_ok=True)
    
    try:
        print(f"Loaded {len(df)} records")
        print(f"After preprocessing: {len(df)} records")
        
        # Print data summary
        print("\nData Summary:")
        print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
        print(f"Stations: {df['Stazione'].unique()}")
        print(f"Pollutants: {df['Inquinante'].unique()}")
        print(f"Output directory: {output_dir}")
        
        # Create visualizations
        print("\nCreating time series plots...")
        plot_time_series(df, args.pollutant, 
                        output_dir / f"time_series_{args.pollutant or 'all'}.png")
        
        if args.pollutant:
            print(f"Creating station comparison for {args.pollutant}...")
            plot_station_comparison(df, args.pollutant,
                                  output_dir / f"station_comparison_{args.pollutant}.png")
        
        print("Creating distribution plots...")
        plot_pollutant_distribution(df, output_dir / "pollutant_distributions.png")
        
        print(f"\nAll plots saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
