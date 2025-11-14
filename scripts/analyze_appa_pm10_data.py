"""
Script to analyze PM10 hourly measurement data from APPA (Trento) stations.

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
    """Calculate the great circle distance between two points on Earth using Haversine formula."""
    R = 6371.0  # Earth radius in kilometers
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def calculate_distance_matrix(stations_df):
    """Calculate distance matrix between stations in kilometers."""
    n_stations = len(stations_df)
    distance_matrix = {}
    station_codes = stations_df['station_code'].values
    
    for i in range(n_stations):
        for j in range(i, n_stations):
            lat1 = stations_df.iloc[i]['latitude']
            lon1 = stations_df.iloc[i]['longitude']
            lat2 = stations_df.iloc[j]['latitude']
            lon2 = stations_df.iloc[j]['longitude']
            distance = haversine_distance(lat1, lon1, lat2, lon2)
            code1 = station_codes[i]
            code2 = station_codes[j]
            distance_matrix[(code1, code2)] = distance
            if i != j:
                distance_matrix[(code2, code1)] = distance
    
    return distance_matrix


def find_contiguous_missing_periods(actual_times_set, expected_times):
    """Find all contiguous missing value periods."""
    missing_times = [t for t in expected_times if t not in actual_times_set]
    if not missing_times:
        return []
    missing_times = sorted(missing_times)
    periods = []
    current_start = missing_times[0]
    current_length = 1
    for i in range(1, len(missing_times)):
        time_diff = (missing_times[i] - missing_times[i-1]).total_seconds() / 3600
        if time_diff == 1.0:
            current_length += 1
        else:
            periods.append((current_start, missing_times[i-1], current_length))
            current_start = missing_times[i]
            current_length = 1
    periods.append((current_start, missing_times[-1], current_length))
    return periods


def calculate_missing_values_for_station(station_df, global_start=None, global_end=None):
    """Calculate missing values for a station by comparing expected vs actual measurements."""
    if station_df.empty:
        return {'expected_count': 0, 'actual_count': 0, 'missing_count': 0, 'missing_percentage': 0.0, 'start_date': None, 'end_date': None}
    actual_times = pd.to_datetime(station_df['datetime']).sort_values()
    station_start = actual_times.min()
    station_end = actual_times.max()
    start_date = global_start if global_start else station_start
    end_date = global_end if global_end else station_end
    expected_times = pd.date_range(start=start_date, end=end_date, freq='h')
    expected_count = len(expected_times)
    actual_times_set = set(actual_times)
    actual_count = len([t for t in expected_times if t in actual_times_set])
    missing_count = expected_count - actual_count
    missing_percentage = (missing_count / expected_count * 100) if expected_count > 0 else 0.0
    return {'expected_count': expected_count, 'actual_count': actual_count, 'missing_count': missing_count, 'missing_percentage': missing_percentage, 'start_date': start_date, 'end_date': end_date, 'expected_times': expected_times}


def analyze_appa_pm10_data(data_file, stations_file, output_dir=None):
    """
    Main analysis function for APPA data.
    
    Args:
        data_file: Path to merged_data.csv (raw) or curated dataset CSV
        stations_file: Path to appa_monitoring_stations.csv (metadata) or curated dataset CSV
        output_dir: Optional directory to save results
    """
    print("=" * 80)
    print("APPA PM10 Data Analysis")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    print(f"   Loading PM10 data from: {data_file}")
    df = pd.read_csv(data_file)
    
    # Detect if this is a curated dataset or raw data
    is_curated_data = 'datetime' in df.columns and 'station_code' in df.columns and 'pm10' in df.columns
    is_raw_data = 'Inquinante' in df.columns and 'Data' in df.columns and 'Ora' in df.columns
    
    if is_curated_data:
        print("   Detected curated dataset format")
        # Curated dataset already has the right format
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(['station_code', 'datetime'])
        print(f"   Loaded {len(df):,} rows")
        print(f"   Found {df['station_code'].nunique()} unique stations")
        print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    elif is_raw_data:
        print("   Detected raw data format")
        # Filter for PM10 only
        df = df[df['Inquinante'] == 'PM10'].copy()
        print(f"   Loaded {len(df):,} rows")
        
        # Create datetime column
        df['datetime'] = pd.to_datetime(df['Data']) + pd.to_timedelta(df['Ora'] - 1, unit='h')
        df = df.sort_values(['Stazione', 'datetime'])
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'Stazione': 'station_name',
            'Valore': 'pm10'
        })
        
        # Convert pm10 to numeric, handling any non-numeric values
        df['pm10'] = pd.to_numeric(df['pm10'], errors='coerce')
        
        # Drop rows with invalid pm10 values
        invalid_count = df['pm10'].isna().sum()
        if invalid_count > 0:
            print(f"   Dropped {invalid_count:,} rows with invalid PM10 values")
            df = df[df['pm10'].notna()].copy()
        
        print(f"   Found {df['station_name'].nunique()} unique stations")
        print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    else:
        raise ValueError("Unknown data format. Expected either curated dataset (with 'datetime', 'station_code', 'pm10') or raw data (with 'Inquinante', 'Data', 'Ora').")
    
    # Load station metadata
    print(f"\n   Loading station coordinates from: {stations_file}")
    stations_df = pd.read_csv(stations_file)
    print(f"   Loaded {len(stations_df)} rows")
    
    # Detect if stations_file is curated dataset or metadata file
    is_curated_stations = 'datetime' in stations_df.columns
    is_metadata_file = 'Posizione' in stations_df.columns
    
    if is_curated_stations:
        print("   Detected curated dataset format for stations - extracting unique stations")
        # Extract unique stations from curated dataset
        # But we still need coordinates, so try to load the original metadata file
        project_root = Path(data_file).parent.parent.parent
        metadata_file = project_root / "data" / "appa-data" / "appa_monitoring_stations.csv"
        
        if metadata_file.exists():
            print(f"   Loading original metadata from: {metadata_file}")
            stations_metadata = pd.read_csv(metadata_file)
        else:
            print("   Warning: Original metadata file not found. Cannot extract coordinates.")
            print("   Will attempt to use station codes from curated dataset.")
            stations_metadata = None
    else:
        stations_metadata = stations_df
    
    # Parse coordinates from Posizione column (if metadata file)
    def parse_coordinates(pos_str):
        """Parse lat,lon from Posizione string."""
        try:
            # Remove quotes and spaces, split by comma
            pos_str = str(pos_str).strip().strip('"').strip("'")
            parts = [p.strip() for p in pos_str.split(',')]
            lat = float(parts[0])
            lon = float(parts[1])
            return lat, lon
        except:
            return None, None
    
    if stations_metadata is not None and 'Posizione' in stations_metadata.columns:
        stations_metadata['latitude'] = stations_metadata['Posizione'].apply(lambda x: parse_coordinates(x)[0])
        stations_metadata['longitude'] = stations_metadata['Posizione'].apply(lambda x: parse_coordinates(x)[1])
        
        # Create station_code from IT codice italiano or use station name
        stations_metadata['station_code'] = stations_metadata['IT - codice italiano'].fillna(
            stations_metadata['Nome stazione'].apply(lambda x: x.replace(' ', '_').upper())
        )
        stations_metadata['station_name_meta'] = stations_metadata['Nome stazione']
    else:
        # If we don't have metadata, create a minimal stations_metadata from curated data
        if is_curated_stations:
            stations_metadata = stations_df.groupby(['station_code', 'station_name']).first().reset_index()
            stations_metadata['latitude'] = None
            stations_metadata['longitude'] = None
            stations_metadata['station_name_meta'] = stations_metadata['station_name']
        else:
            raise ValueError("Cannot parse station coordinates. Please provide appa_monitoring_stations.csv")
    
    # Handle station code assignment for raw data
    if is_raw_data and 'station_code' not in df.columns:
        # Match station names between data and metadata
        # Create a mapping dictionary with better matching logic
        station_name_mapping = {}
        
        # Create a mapping from metadata names to data names
        # Known mappings based on the data
        known_mappings = {
            'TRENTO PSC': 'Parco S. Chiara',
            'TRENTO VBZ': 'Via Bolzano',
            'PIANA ROTALIANA': 'Piana Rotaliana',
            'RIVA GAR': 'Riva del Garda',
            'MONTE GAZA': 'Monte Gazza',  # Fixed typo
            'ROVERETO LGP': 'Rovereto',
            'AVIO A22': 'A22 (Avio)',
            'BORGO VAL': 'Borgo Valsugana'
        }
        
        for _, row in stations_metadata.iterrows():
            meta_name = row.get('Nome stazione', row.get('station_name_meta', ''))
            station_code = row['station_code']
            
            # Try known mapping first
            data_name = known_mappings.get(meta_name)
            
            if data_name is None:
                # Try partial match
                meta_keywords = meta_name.split()
                for data_station in df['station_name'].unique():
                    if any(keyword.lower() in data_station.lower() for keyword in meta_keywords):
                        data_name = data_station
                        break
            
            if data_name and data_name in df['station_name'].values:
                station_name_mapping[data_name] = {
                    'station_code': station_code,
                    'station_name': meta_name,
                    'latitude': row.get('latitude'),
                    'longitude': row.get('longitude')
                }
        
        # Add station_code to data
        df['station_code'] = df['station_name'].map(lambda x: station_name_mapping.get(x, {}).get('station_code', x.replace(' ', '_').upper()))
    
    # Create stations dataframe for analysis
    analysis_stations = []
    for station_code in df['station_code'].unique():
        station_data = df[df['station_code'] == station_code]
        station_name = station_data['station_name'].iloc[0] if 'station_name' in station_data.columns else station_code
        
        # Try to find coordinates in metadata
        meta_row = stations_metadata[stations_metadata['station_code'] == station_code]
        if len(meta_row) > 0:
            meta = meta_row.iloc[0]
            analysis_stations.append({
                'station_code': station_code,
                'station_name': meta.get('station_name_meta', meta.get('Nome stazione', station_name)),
                'latitude': meta.get('latitude'),
                'longitude': meta.get('longitude')
            })
        else:
            # If no metadata found, use what we have from the data
            print(f"   Warning: No metadata found for station {station_code} ({station_name})")
            analysis_stations.append({
                'station_code': station_code,
                'station_name': station_name,
                'latitude': None,
                'longitude': None
            })
    
    stations_df_clean = pd.DataFrame(analysis_stations)
    
    # Filter out stations without coordinates (they can't be used for distance matrix)
    stations_with_coords = stations_df_clean[stations_df_clean['latitude'].notna() & stations_df_clean['longitude'].notna()]
    if len(stations_with_coords) < len(stations_df_clean):
        print(f"   Warning: {len(stations_df_clean) - len(stations_with_coords)} stations missing coordinates")
    
    if len(stations_with_coords) == 0:
        print("   Error: No stations found with valid coordinates!")
        return
    
    print(f"   Found {len(stations_with_coords)} stations with valid coordinates")
    
    # Use stations_with_coords for distance calculations
    stations_df_clean = stations_with_coords
    
    # Add year column
    df['year'] = pd.to_datetime(df['datetime']).dt.year
    
    # ============================================================================
    # 2. PM10 Measurement Statistics
    # ============================================================================
    print("\n" + "=" * 80)
    print("2. PM10 Measurement Statistics (Min, Mean, Max)")
    print("=" * 80)
    
    # Statistics per station
    # Ensure station_code is string for consistent sorting
    df['station_code'] = df['station_code'].astype(str)
    
    # Group by station_code and station_name (if available)
    groupby_cols = ['station_code']
    if 'station_name' in df.columns:
        groupby_cols.append('station_name')
    
    stats = df.groupby(groupby_cols).agg({
        'pm10': ['count', 'min', 'mean', 'max', 'std']
    }).reset_index()
    
    # Set column names based on what we grouped by
    if 'station_name' in groupby_cols:
        stats.columns = ['station_code', 'station_name', 'count', 'min', 'mean', 'max', 'std']
    else:
        stats.columns = ['station_code', 'count', 'min', 'mean', 'max', 'std']
        # Add station_name from stations_df_clean
        station_name_map = dict(zip(stations_df_clean['station_code'].astype(str), stations_df_clean['station_name']))
        stats['station_name'] = stats['station_code'].map(station_name_map)
        # Reorder columns
        stats = stats[['station_code', 'station_name', 'count', 'min', 'mean', 'max', 'std']]
    
    stats = stats.sort_values('station_code')
    
    print("\nStatistics per station:")
    print(stats.to_string(index=False))
    
    # Overall statistics
    overall_stats = {
        'count': df['pm10'].count(),
        'min': df['pm10'].min(),
        'mean': df['pm10'].mean(),
        'max': df['pm10'].max(),
        'std': df['pm10'].std()
    }
    print(f"\nOverall statistics (all stations combined):")
    print(f"  Count: {overall_stats['count']:,}")
    print(f"  Min: {overall_stats['min']:.2f}")
    print(f"  Mean: {overall_stats['mean']:.2f}")
    print(f"  Max: {overall_stats['max']:.2f}")
    print(f"  Std: {overall_stats['std']:.2f}")
    
    print(f"\n  Note: Missing values are represented by absent rows, not NaN values.")
    print(f"  See 'Missing Values per Year' section for detailed missing value analysis.")
    
    # Check for potentially invalid values
    negative_count = (df['pm10'] < 0).sum()
    very_high_count = (df['pm10'] > 1000).sum()
    
    if negative_count > 0:
        print(f"\n  ⚠️  Warning: Found {negative_count} negative PM10 values (invalid)")
    if very_high_count > 0:
        print(f"  ⚠️  Warning: Found {very_high_count} PM10 values > 1000 µg/m³ (suspicious)")
    
    # ============================================================================
    # 3. Missing values per year
    # ============================================================================
    print("\n" + "=" * 80)
    print("3. Missing Values per Year")
    print("=" * 80)
    
    # Define global period: use actual date range from data (not full years)
    year_min = df['year'].min()
    year_max = df['year'].max()
    # Use actual min/max datetime from data, not full year boundaries
    global_start = pd.to_datetime(df['datetime']).min()
    global_end = pd.to_datetime(df['datetime']).max()
    
    print(f"\nGlobal expected period: {global_start} to {global_end}")
    
    # Calculate missing values per year per station
    missing_per_year_rows = []
    
    for station_code in sorted(df['station_code'].unique()):
        station_df = df[df['station_code'] == station_code]
        station_name = station_df['station_name'].iloc[0] if 'station_name' in station_df.columns else station_code
        
        for year in range(year_min, year_max + 1):
            # Use actual date range for the year (not full year if data doesn't cover it)
            year_start = pd.Timestamp(f"{year}-01-01 00:00:00")
            year_end = pd.Timestamp(f"{year}-12-31 23:00:00")
            # Clamp to actual data range
            if year_start < global_start:
                year_start = global_start
            if year_end > global_end:
                year_end = global_end
            expected_hours = len(pd.date_range(start=year_start, end=year_end, freq='h'))
            
            year_data = station_df[station_df['year'] == year]
            # Count actual hours with valid (non-NaN) PM10 values
            actual_hours = year_data['pm10'].notna().sum()
            missing_hours = expected_hours - actual_hours
            
            missing_per_year_rows.append({
                'year': year,
                'station_code': station_code,
                'station_name': station_name,
                'expected_hours': expected_hours,
                'actual_hours': actual_hours,
                'missing_hours': missing_hours
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
    
    # Summary by year
    # Recalculate actual_hours correctly (counting non-NaN values, not rows)
    yearly_summary_recalc = []
    for year in range(year_min, year_max + 1):
        year_data = df[df['year'] == year]
        expected_hours = len(pd.date_range(start=pd.Timestamp(f"{year}-01-01 00:00:00"), 
                                          end=pd.Timestamp(f"{year}-12-31 23:00:00"), freq='h')) * len(df['station_code'].unique())
        actual_hours = year_data['pm10'].notna().sum()
        missing_hours = expected_hours - actual_hours
        yearly_summary_recalc.append({
            'year': year,
            'expected_hours': expected_hours,
            'actual_hours': actual_hours,
            'missing_hours': missing_hours,
            'missing_percentage': (missing_hours / expected_hours * 100) if expected_hours > 0 else 0.0
        })
    yearly_summary = pd.DataFrame(yearly_summary_recalc).set_index('year')
    
    print("\nMissing values summary by year (all stations combined):")
    print(yearly_summary.to_string())
    
    # Per-station summary
    # Recalculate per-station summary correctly (counting non-NaN values, not rows)
    station_summary_rows = []
    for station_code in sorted(df['station_code'].unique()):
        station_df = df[df['station_code'] == station_code]
        station_name = station_df['station_name'].iloc[0] if 'station_name' in station_df.columns else station_code
        
        # Expected hours: from first year to last year, all hours
        expected_hours = len(pd.date_range(start=global_start, end=global_end, freq='h'))
        # Actual hours: count non-NaN PM10 values
        actual_hours = station_df['pm10'].notna().sum()
        missing_hours = expected_hours - actual_hours
        
        station_summary_rows.append({
            'station_code': station_code,
            'station_name': station_name,
            'expected_hours': expected_hours,
            'actual_hours': actual_hours,
            'missing_hours': missing_hours,
            'missing_percentage': (missing_hours / expected_hours * 100) if expected_hours > 0 else 0.0
        })
    station_summary = pd.DataFrame(station_summary_rows)
    
    print("\nPer-station missing values summary (global period):")
    print(station_summary.to_string(index=False))
    
    # ============================================================================
    # 4. Distribution of contiguous missing periods
    # ============================================================================
    print("\n" + "=" * 80)
    print("4. Distribution of Contiguous Missing Value Periods")
    print("=" * 80)
    
    all_missing_lengths = []
    station_missing_info = []
    
    for station_code in sorted(df['station_code'].unique()):
        station_df = df[df['station_code'] == station_code].copy()
        station_name = station_df['station_name'].iloc[0] if 'station_name' in station_df.columns else station_code
        
        # Calculate expected times for this station
        station_start = pd.to_datetime(station_df['datetime']).min()
        station_end = pd.to_datetime(station_df['datetime']).max()
        expected_times = pd.date_range(start=station_start, end=station_end, freq='h')
        
        # Find missing periods: both missing rows AND NaN values
        # First, get times with valid (non-NaN) PM10 values
        valid_data = station_df[station_df['pm10'].notna()]
        valid_times_set = set(pd.to_datetime(valid_data['datetime']).sort_values().unique())
        
        # Find contiguous missing periods (missing rows OR NaN values)
        periods = find_contiguous_missing_periods(valid_times_set, expected_times)
        missing_lengths = [p[2] for p in periods]
        all_missing_lengths.extend(missing_lengths)
        
        if missing_lengths:
            station_missing_info.append({
                'station_code': station_code,
                'station_name': station_name,
                'n_periods': len(missing_lengths),
                'max_length': max(missing_lengths),
                'mean_length': np.mean(missing_lengths)
            })
    
    if all_missing_lengths:
        # Distribution of missing period lengths
        length_counts = pd.Series(all_missing_lengths).value_counts().sort_index()
        
        print("\nDistribution of contiguous missing period lengths (all stations):")
        print(f"{'Length (hours)':<20} {'Frequency':<20} {'Percentage':<20}")
        print("-" * 60)
        
        total_periods = len(all_missing_lengths)
        for length, count in length_counts.items():
            percentage = (count / total_periods * 100) if total_periods > 0 else 0
            print(f"{length:<20} {count:<20} {percentage:<20.2f}%")
        
        print(f"\nTotal contiguous missing periods: {total_periods}")
        print(f"Longest contiguous missing period: {max(all_missing_lengths)} hours ({max(all_missing_lengths)/24:.1f} days)")
        print(f"Shortest contiguous missing period: {min(all_missing_lengths)} hours ({min(all_missing_lengths)/24:.1f} days)")
        print(f"Mean contiguous missing period length: {np.mean(all_missing_lengths):.2f} hours ({np.mean(all_missing_lengths)/24:.2f} days)")
        print(f"Median contiguous missing period length: {np.median(all_missing_lengths):.2f} hours ({np.median(all_missing_lengths)/24:.2f} days)")
        
        # Per-station summary
        if station_missing_info:
            station_missing_df = pd.DataFrame(station_missing_info)
            print("\nContiguous missing periods per station:")
            print(f"{'Station Code':<20} {'Station Name':<40} {'# Periods':<15} {'Max Length':<15} {'Mean Length':<15}")
            print("-" * 105)
            for _, row in station_missing_df.iterrows():
                print(f"{row['station_code']:<20} {row['station_name'][:38]:<40} {row['n_periods']:<15} {row['max_length']:<15} {row['mean_length']:<15.2f}")
    else:
        print("\nNo missing periods found (all data is complete)")
    
    # ============================================================================
    # 5. Distance matrix
    # ============================================================================
    print("\n" + "=" * 80)
    print("5. Distance Matrix Among Stations")
    print("=" * 80)
    
    distance_matrix = calculate_distance_matrix(stations_df_clean)
    
    # Create a DataFrame for display
    station_codes = sorted(stations_df_clean['station_code'].unique())
    distance_df = pd.DataFrame(index=station_codes, columns=station_codes)
    
    for code1 in station_codes:
        for code2 in station_codes:
            if (code1, code2) in distance_matrix:
                distance_df.loc[code1, code2] = distance_matrix[(code1, code2)]
            else:
                distance_df.loc[code1, code2] = 0.0
    
    distance_df = distance_df.astype(float)
    print("\nDistance matrix (kilometers):")
    print(distance_df.to_string())
    
    # Summary statistics
    distances_only = [d for d in distance_matrix.values() if d > 0]
    if distances_only:
        print(f"\nSummary:")
        print(f"  Minimum distance between stations: {min(distances_only):.2f} km")
        print(f"  Maximum distance between stations: {max(distances_only):.2f} km")
        print(f"  Mean distance between stations: {np.mean(distances_only):.2f} km")
    
    # ============================================================================
    # Save results
    # ============================================================================
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 80)
        print("Saving results to files...")
        print("=" * 80)
        
        stats.to_csv(output_dir / 'pm10_statistics_per_station.csv', index=False)
        missing_per_year.to_csv(output_dir / 'missing_values_per_year.csv', index=False)
        station_summary.to_csv(output_dir / 'missing_values_per_station_summary.csv', index=False)
        
        if all_missing_lengths:
            length_distribution = pd.DataFrame({
                'length_hours': length_counts.index,
                'frequency': length_counts.values,
                'percentage': length_counts.values / total_periods * 100
            })
            length_distribution.to_csv(output_dir / 'contiguous_missing_periods_distribution.csv', index=False)
        
        distance_df.to_csv(output_dir / 'station_distance_matrix.csv')
        
        print("  Saved: pm10_statistics_per_station.csv")
        print("  Saved: missing_values_per_year.csv")
        print("  Saved: missing_values_per_station_summary.csv")
        if all_missing_lengths:
            print("  Saved: contiguous_missing_periods_distribution.csv")
        print("  Saved: station_distance_matrix.csv")
        
        print("\nAll results saved successfully!")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Use curated dataset if available, otherwise use raw data
    curated_data_file = project_root / "data" / "appa-data" / "merged_pm10_hourly_curated_no_interp_metadata.csv"
    raw_data_file = project_root / "data" / "appa-data" / "merged_data.csv"
    metadata_file = project_root / "data" / "appa-data" / "appa_monitoring_stations.csv"
    
    # Check which files exist and use appropriate ones
    if curated_data_file.exists():
        print("Using curated dataset for analysis")
        data_file = curated_data_file
        # For curated data, we still need metadata for coordinates
        stations_file = metadata_file if metadata_file.exists() else curated_data_file
    else:
        print("Using raw dataset for analysis")
        data_file = raw_data_file
        stations_file = metadata_file if metadata_file.exists() else curated_data_file
    
    output_dir = project_root / "output" / "appa_analysis"
    
    analyze_appa_pm10_data(data_file, stations_file, output_dir)

