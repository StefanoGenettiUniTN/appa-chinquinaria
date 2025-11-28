"""
Script to analyze PM10 hourly measurement data from ARPAV stations.

This script performs the following analyses:
1. Min, mean, max statistics for PM10 measurements (to detect invalid values)
2. Missing values per year for all stations
3. Distribution of contiguous missing value periods
4. Distance matrix among stations in kilometers
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import from chinquinaria if needed
sys.path.insert(0, str(Path(__file__).parent.parent))


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth using Haversine formula.
    
    Args:
        lat1, lon1: Latitude and longitude of first point in degrees
        lat2, lon2: Latitude and longitude of second point in degrees
    
    Returns:
        Distance in kilometers
    """
    # Earth radius in kilometers
    R = 6371.0
    
    # Convert degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c
    return distance


def calculate_distance_matrix(stations_df):
    """
    Calculate distance matrix between stations in kilometers.
    
    Args:
        stations_df: DataFrame with columns [station_code, station_name, latitude, longitude]
    
    Returns:
        DataFrame with distance matrix in km
    """
    n_stations = len(stations_df)
    distance_matrix = np.zeros((n_stations, n_stations))
    station_codes = stations_df['station_code'].values
    
    for i in range(n_stations):
        for j in range(n_stations):
            if i == j:
                distance_matrix[i, j] = 0.0
            else:
                lat1 = stations_df.iloc[i]['latitude']
                lon1 = stations_df.iloc[i]['longitude']
                lat2 = stations_df.iloc[j]['latitude']
                lon2 = stations_df.iloc[j]['longitude']
                distance_matrix[i, j] = haversine_distance(lat1, lon1, lat2, lon2)
    
    # Create DataFrame with station codes as index and columns
    distance_df = pd.DataFrame(
        distance_matrix,
        index=station_codes,
        columns=station_codes
    )
    
    return distance_df


def find_contiguous_missing_periods(station_df, expected_times):
    """
    Find all contiguous missing value periods for a specific station.
    
    Missing values are identified by comparing expected hourly timestamps
    against actual measurements. Gaps in the time series represent missing periods.
    
    Args:
        station_df: DataFrame with datetime and pm10 columns for one station
        expected_times: Series of expected hourly timestamps
    
    Returns:
        List of lengths of contiguous missing periods (in hours)
    """
    # Get actual measurement times
    actual_times = pd.to_datetime(station_df['datetime']).sort_values().unique()
    actual_times_set = set(actual_times)
    
    # Find missing timestamps
    missing_times = [t for t in expected_times if t not in actual_times_set]
    
    if not missing_times:
        return []
    
    # Sort missing times and find contiguous periods
    missing_times = sorted(missing_times)
    missing_lengths = []
    current_length = 1
    
    for i in range(1, len(missing_times)):
        # Check if this timestamp is exactly 1 hour after the previous one
        time_diff = (missing_times[i] - missing_times[i-1]).total_seconds() / 3600
        if time_diff == 1.0:
            # Contiguous
            current_length += 1
        else:
            # Gap found, save current period
            if current_length > 0:
                missing_lengths.append(current_length)
            current_length = 1
    
    # Don't forget the last period
    if current_length > 0:
        missing_lengths.append(current_length)
    
    return missing_lengths


def calculate_missing_values_for_station(station_df, global_start=None, global_end=None):
    """
    Calculate missing values for a station by comparing expected vs actual measurements.
    
    Args:
        station_df: DataFrame with datetime and pm10 for one station
        global_start: Optional global start date (if None, uses station's first measurement)
        global_end: Optional global end date (if None, uses station's last measurement)
    
    Returns:
        dict with missing value statistics
    """
    if station_df.empty:
        return {
            'expected_count': 0,
            'actual_count': 0,
            'missing_count': 0,
            'missing_percentage': 0.0,
            'start_date': None,
            'end_date': None
        }
    
    actual_times = pd.to_datetime(station_df['datetime']).sort_values()
    station_start = actual_times.min()
    station_end = actual_times.max()
    
    # Use global dates if provided, otherwise use station-specific dates
    start_date = global_start if global_start else station_start
    end_date = global_end if global_end else station_end
    
    # Create expected hourly time series
    expected_times = pd.date_range(start=start_date, end=end_date, freq='h')
    expected_count = len(expected_times)
    
    # Count actual measurements
    actual_times_set = set(actual_times)
    actual_count = len([t for t in expected_times if t in actual_times_set])
    missing_count = expected_count - actual_count
    missing_percentage = (missing_count / expected_count * 100) if expected_count > 0 else 0.0
    
    return {
        'expected_count': expected_count,
        'actual_count': actual_count,
        'missing_count': missing_count,
        'missing_percentage': missing_percentage,
        'start_date': start_date,
        'end_date': end_date,
        'expected_times': expected_times
    }


def analyze_pm10_data(data_file, stations_file, output_dir=None):
    """
    Main analysis function.
    
    Args:
        data_file: Path to merged_pm10_hourly.csv
        stations_file: Path to pm10_stations.csv
        output_dir: Optional directory to save results
    """
    print("=" * 80)
    print("PM10 Data Analysis")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    print(f"   Loading PM10 data from: {data_file}")
    df = pd.read_csv(data_file, parse_dates=['datetime'])
    print(f"   Loaded {len(df):,} rows")
    
    print(f"   Loading station coordinates from: {stations_file}")
    stations_df = pd.read_csv(stations_file)
    print(f"   Loaded {len(stations_df)} stations")
    
    # Convert datetime to datetime if not already
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    
    # Get unique stations
    unique_stations = df['station_code'].unique()
    print(f"\n   Found {len(unique_stations)} unique stations in data")
    
    # ============================================================================
    # 1. Min, Mean, Max statistics
    # ============================================================================
    print("\n" + "=" * 80)
    print("2. PM10 Measurement Statistics (Min, Mean, Max)")
    print("=" * 80)
    
    stats = df.groupby('station_code')['pm10'].agg(['min', 'mean', 'max', 'count', 'std'])
    stats = stats.merge(stations_df[['station_code', 'station_name']], on='station_code', how='left')
    stats = stats[['station_code', 'station_name', 'count', 'min', 'mean', 'max', 'std']]
    
    print("\nStatistics per station:")
    print(stats.to_string(index=False))
    
    # Overall statistics
    print("\nOverall statistics (all stations combined):")
    overall_stats = {
        'min': df['pm10'].min(),
        'mean': df['pm10'].mean(),
        'max': df['pm10'].max(),
        'std': df['pm10'].std(),
        'count': df['pm10'].count()
    }
    print(f"  Count: {overall_stats['count']:,}")
    print(f"  Min: {overall_stats['min']:.2f}")
    print(f"  Mean: {overall_stats['mean']:.2f}")
    print(f"  Max: {overall_stats['max']:.2f}")
    print(f"  Std: {overall_stats['std']:.2f}")
    print(f"\n  Note: Missing values are represented by absent rows, not NaN values.")
    print(f"  See 'Missing Values per Year' section for detailed missing value analysis.")
    
    # Check for potentially invalid values (negative or extremely high)
    negative_count = (df['pm10'] < 0).sum()
    very_high_count = (df['pm10'] > 1000).sum()  # PM10 typically < 1000 µg/m³
    
    if negative_count > 0:
        print(f"\n  ⚠️  Warning: Found {negative_count} negative PM10 values (invalid)")
    if very_high_count > 0:
        print(f"  ⚠️  Warning: Found {very_high_count} PM10 values > 1000 µg/m³ (suspicious)")
    
    # ============================================================================
    # 2. Missing values per year
    # ============================================================================
    print("\n" + "=" * 80)
    print("3. Missing Values per Year")
    print("=" * 80)
    
    # Define global period: 2010-01-01 to 2025-12-31 (as per merge report)
    global_start = pd.Timestamp("2010-01-01 00:00:00")
    global_end = pd.Timestamp("2025-12-31 23:00:00")
    global_expected_hours = len(pd.date_range(start=global_start, end=global_end, freq='h'))
    
    print(f"\nGlobal expected period: {global_start.date()} to {global_end.date()}")
    print(f"Global expected hourly measurements: {global_expected_hours:,}")
    
    # Calculate missing values per year per station
    # For each station and year, calculate expected vs actual
    missing_per_year_rows = []
    
    for station_code in unique_stations:
        station_df = df[df['station_code'] == station_code].copy()
        station_name = stations_df[stations_df['station_code'] == station_code]['station_name'].values[0]
        
        # Get actual measurements per year
        station_df['year'] = pd.to_datetime(station_df['datetime']).dt.year
        actual_per_year = station_df.groupby('year').size().to_dict()
        
        # For each year, calculate expected vs actual
        for year in range(2010, 2026):
            year_start = pd.Timestamp(f"{year}-01-01 00:00:00")
            year_end = pd.Timestamp(f"{year}-12-31 23:00:00")
            expected_hours = len(pd.date_range(start=year_start, end=year_end, freq='h'))
            actual_hours = actual_per_year.get(year, 0)
            missing_hours = expected_hours - actual_hours
            missing_pct = (missing_hours / expected_hours * 100) if expected_hours > 0 else 0.0
            
            missing_per_year_rows.append({
                'year': year,
                'station_code': station_code,
                'station_name': station_name,
                'expected_hours': expected_hours,
                'actual_hours': actual_hours,
                'missing_hours': missing_hours,
                'missing_percentage': missing_pct
            })
    
    missing_per_year = pd.DataFrame(missing_per_year_rows)
    
    # Pivot for better visualization
    missing_pivot = missing_per_year.pivot_table(
        index=['station_code', 'station_name'],
        columns='year',
        values='missing_hours',
        fill_value=0
    )
    
    print("\nMissing hours per year per station:")
    print(missing_pivot.to_string())
    
    # Summary by year (all stations combined)
    print("\nMissing values summary by year (all stations combined):")
    yearly_summary = missing_per_year.groupby('year').agg({
        'expected_hours': 'sum',
        'actual_hours': 'sum',
        'missing_hours': 'sum'
    })
    yearly_summary['missing_percentage'] = (yearly_summary['missing_hours'] / 
                                           yearly_summary['expected_hours'] * 100)
    yearly_summary.columns = ['expected_hours', 'actual_hours', 'missing_hours', 'missing_percentage']
    print(yearly_summary.to_string())
    
    # Per-station overall summary
    print("\nPer-station missing values summary (global period 2010-2025):")
    station_summaries = []
    for station_code in unique_stations:
        station_df = df[df['station_code'] == station_code].copy()
        station_name = stations_df[stations_df['station_code'] == station_code]['station_name'].values[0]
        station_stats = calculate_missing_values_for_station(station_df, global_start, global_end)
        station_summaries.append({
            'station_code': station_code,
            'station_name': station_name,
            'expected_hours': station_stats['expected_count'],
            'actual_hours': station_stats['actual_count'],
            'missing_hours': station_stats['missing_count'],
            'missing_percentage': station_stats['missing_percentage']
        })
    
    station_summary_df = pd.DataFrame(station_summaries)
    print(station_summary_df.to_string(index=False))
    
    # ============================================================================
    # 3. Distribution of contiguous missing value periods
    # ============================================================================
    print("\n" + "=" * 80)
    print("4. Distribution of Contiguous Missing Value Periods")
    print("=" * 80)
    
    all_missing_lengths = []
    station_missing_lengths = {}
    
    for station_code in unique_stations:
        station_df = df[df['station_code'] == station_code].copy()
        
        # Calculate expected time range for this station
        # Use station's actual date range (from first to last measurement)
        if not station_df.empty:
            actual_times = pd.to_datetime(station_df['datetime']).sort_values()
            station_start = actual_times.min()
            station_end = actual_times.max()
            expected_times = pd.date_range(start=station_start, end=station_end, freq='h')
            
            # Find contiguous missing periods
            lengths = find_contiguous_missing_periods(station_df, expected_times)
            station_missing_lengths[station_code] = lengths
            all_missing_lengths.extend(lengths)
        else:
            station_missing_lengths[station_code] = []
    
    if all_missing_lengths:
        # Create distribution
        length_counts = pd.Series(all_missing_lengths).value_counts().sort_index()
        
        print("\nDistribution of contiguous missing period lengths (all stations):")
        print(f"{'Length (hours)':<20} {'Frequency':<15} {'Percentage':<15}")
        print("-" * 50)
        total_periods = len(all_missing_lengths)
        for length, count in length_counts.items():
            percentage = (count / total_periods * 100) if total_periods > 0 else 0
            print(f"{length:<20} {count:<15} {percentage:.2f}%")
        
        print(f"\nTotal contiguous missing periods: {total_periods}")
        print(f"Longest contiguous missing period: {max(all_missing_lengths)} hours ({max(all_missing_lengths)/24:.1f} days)")
        print(f"Shortest contiguous missing period: {min(all_missing_lengths)} hours")
        print(f"Mean contiguous missing period length: {np.mean(all_missing_lengths):.2f} hours ({np.mean(all_missing_lengths)/24:.2f} days)")
        print(f"Median contiguous missing period length: {np.median(all_missing_lengths):.2f} hours ({np.median(all_missing_lengths)/24:.2f} days)")
        
        # Per station summary
        print("\nContiguous missing periods per station:")
        print(f"{'Station Code':<15} {'Station Name':<25} {'# Periods':<12} {'Max Length':<12} {'Mean Length':<12}")
        print("-" * 80)
        for station_code in unique_stations:
            lengths = station_missing_lengths[station_code]
            station_name = stations_df[stations_df['station_code'] == station_code]['station_name'].values[0]
            if lengths:
                print(f"{station_code:<15} {station_name:<25} {len(lengths):<12} {max(lengths):<12} {np.mean(lengths):.2f}")
            else:
                print(f"{station_code:<15} {station_name:<25} {0:<12} {'N/A':<12} {'N/A':<12}")
    else:
        print("\nNo missing values found in the dataset.")
    
    # ============================================================================
    # 4. Distance matrix
    # ============================================================================
    print("\n" + "=" * 80)
    print("5. Distance Matrix Among Stations")
    print("=" * 80)
    
    distance_matrix = calculate_distance_matrix(stations_df)
    
    print("\nDistance matrix (kilometers):")
    print(distance_matrix.to_string())
    
    print("\nSummary:")
    print(f"  Minimum distance between stations: {distance_matrix[distance_matrix > 0].min().min():.2f} km")
    print(f"  Maximum distance between stations: {distance_matrix.max().max():.2f} km")
    print(f"  Mean distance between stations: {distance_matrix[distance_matrix > 0].mean().mean():.2f} km")
    
    # ============================================================================
    # Save results if output directory is provided
    # ============================================================================
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n" + "=" * 80)
        print("Saving results to files...")
        print("=" * 80)
        
        # Save statistics
        stats.to_csv(output_path / 'pm10_statistics_per_station.csv', index=False)
        print(f"  Saved: pm10_statistics_per_station.csv")
        
        # Save missing values per year
        missing_per_year.to_csv(output_path / 'missing_values_per_year.csv', index=False)
        print(f"  Saved: missing_values_per_year.csv")
        
        # Save per-station summary
        station_summary_df.to_csv(output_path / 'missing_values_per_station_summary.csv', index=False)
        print(f"  Saved: missing_values_per_station_summary.csv")
        
        # Save missing periods distribution
        if all_missing_lengths:
            length_distribution = pd.DataFrame({
                'length_hours': length_counts.index,
                'frequency': length_counts.values,
                'percentage': length_counts.values / total_periods * 100
            })
            length_distribution.to_csv(output_path / 'contiguous_missing_periods_distribution.csv', index=False)
            print(f"  Saved: contiguous_missing_periods_distribution.csv")
        
        # Save distance matrix
        distance_matrix.to_csv(output_path / 'station_distance_matrix.csv')
        print(f"  Saved: station_distance_matrix.csv")
        
        print("\nAll results saved successfully!")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # data_file = project_root / "data" / "arpav" / "PM10" / "merged_pm10_hourly.csv"
    data_file = project_root / "data" / "arpav" / "PM10" / "merged_pm10_hourly_curated.csv"
    stations_file = project_root / "data" / "arpav" / "PM10" / "pm10_stations.csv"
    output_dir = project_root / "data" / "arpav" / "PM10" / "analysis_results"
    
    # Run analysis
    analyze_pm10_data(data_file, stations_file, output_dir=output_dir)

