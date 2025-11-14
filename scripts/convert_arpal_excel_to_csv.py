"""
Script to convert ARPAL PM10 Excel file to CSV format.

This script:
1. Reads the Excel file with PM10 hourly data
2. Converts it to a long format CSV (datetime, station_code, station_name, pm10)
3. Extracts station information to a separate CSV file
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def utm_to_latlon(easting, northing, zone=32, northern=True):
    """
    Convert UTM coordinates to latitude/longitude.
    
    Args:
        easting: UTM Easting coordinate
        northing: UTM Northing coordinate
        zone: UTM zone number (default: 32 for Italy)
        northern: Whether in northern hemisphere (default: True)
    
    Returns:
        Tuple: (latitude, longitude)
    """
    try:
        from pyproj import Transformer
        
        # Create transformer from UTM to WGS84
        transformer = Transformer.from_crs(
            f"EPSG:326{zone}" if northern else f"EPSG:327{zone}",
            "EPSG:4326",
            always_xy=True
        )
        
        lon, lat = transformer.transform(easting, northing)
        return lat, lon
    except ImportError:
        # Fallback: simple approximation (not accurate but works)
        # This is a rough approximation - pyproj is recommended
        print("Warning: pyproj not available, using approximation")
        # For zone 32N (Italy), rough conversion
        lat = (northing / 111320.0) - 0.5  # Rough approximation
        lon = (easting / (111320.0 * np.cos(np.radians(lat)))) + 9.0  # Rough approximation
        return lat, lon


def convert_arpal_excel_to_csv(excel_file, output_dir=None):
    """
    Convert ARPAL Excel file to CSV format.
    
    Args:
        excel_file: Path to PM10_Orari_Lombardia.xlsx
        output_dir: Optional output directory (default: same as Excel file)
    """
    print("=" * 80)
    print("ARPAL PM10 Excel to CSV Conversion")
    print("=" * 80)
    
    excel_path = Path(excel_file)
    if output_dir is None:
        output_dir = excel_path.parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # Step 1: Read and process station information
    # ============================================================================
    print(f"\n1. Reading station information from sheet 2...")
    
    stations_df = pd.read_excel(excel_path, sheet_name=1)
    
    # Clean up station names and extract info
    # First column contains station names
    station_names = stations_df.iloc[:, 0].dropna().tolist()
    
    # Extract coordinates and altitude
    stations_info = []
    for idx, row in stations_df.iterrows():
        station_name = row.iloc[0]
        if pd.isna(station_name) or station_name == 'Unnamed: 0':
            continue
        
        # Get UTM coordinates
        utm_nord = row['UTM nord'] if 'UTM nord' in stations_df.columns else None
        utm_est = row['UTM Est'] if 'UTM Est' in stations_df.columns else None
        altitude = row['Quota'] if 'Quota' in stations_df.columns else None
        
        if pd.isna(utm_nord) or pd.isna(utm_est):
            continue
        
        # Convert UTM to lat/lon
        try:
            latitude, longitude = utm_to_latlon(float(utm_est), float(utm_nord), zone=32, northern=True)
        except Exception as e:
            print(f"   Warning: Could not convert coordinates for {station_name}: {e}")
            continue
        
        # Create station code (use index or hash of name)
        station_code = f"ARPAL_{idx+1:03d}"
        
        stations_info.append({
            'station_code': station_code,
            'station_name': str(station_name).strip(),
            'latitude': latitude,
            'longitude': longitude,
            'altitude': float(altitude) if not pd.isna(altitude) else None,
            'utm_nord': float(utm_nord),
            'utm_est': float(utm_est)
        })
    
    stations_df_clean = pd.DataFrame(stations_info)
    print(f"   Found {len(stations_df_clean)} stations")
    
    # Save stations CSV
    stations_csv = output_dir / "arpal_pm10_stations.csv"
    stations_df_clean.to_csv(stations_csv, index=False)
    print(f"   Saved stations to: {stations_csv}")
    
    # ============================================================================
    # Step 2: Read and process PM10 data
    # ============================================================================
    print(f"\n2. Reading PM10 data from sheet 1...")
    
    # Read Excel file - skip first 3 rows (header rows)
    df = pd.read_excel(excel_path, sheet_name=0, skiprows=3)
    
    # First column should be datetime
    datetime_col = df.columns[0]
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    
    # Remove rows with invalid datetime
    df = df[df[datetime_col].notna()].copy()
    
    print(f"   Loaded {len(df):,} rows")
    print(f"   Date range: {df[datetime_col].min()} to {df[datetime_col].max()}")
    print(f"   Stations (columns): {len(df.columns) - 1}")
    
    # ============================================================================
    # Step 3: Convert to long format
    # ============================================================================
    print(f"\n3. Converting to long format...")
    
    # Rename datetime column
    df = df.rename(columns={datetime_col: 'datetime'})
    
    # Create mapping from column names to station codes
    # First, read the original Excel to get proper column names (before skiprows)
    df_header = pd.read_excel(excel_path, sheet_name=0, nrows=0)
    original_columns = list(df_header.columns)
    
    column_to_station = {}
    for col_idx, col in enumerate(df.columns):
        if col == 'datetime':
            continue
        
        # Get the original column name from the header
        if col_idx < len(original_columns):
            original_col_name = str(original_columns[col_idx]).strip()
        else:
            original_col_name = str(col).strip()
        
        # Extract station name from column header (format: " Station Name PM10 ...")
        station_name_raw = original_col_name.split(' PM10')[0].strip()
        
        # Try to match with stations_df by name
        matching_station = stations_df_clean[
            stations_df_clean['station_name'].str.contains(
                station_name_raw, 
                case=False, 
                na=False,
                regex=False
            )
        ]
        
        if len(matching_station) > 0:
            station_code = matching_station.iloc[0]['station_code']
            station_name_clean = matching_station.iloc[0]['station_name']
        else:
            # Try reverse matching - check if station name contains column name parts
            station_name_parts = station_name_raw.split(' - ')
            if len(station_name_parts) > 0:
                main_name = station_name_parts[-1].strip()
                matching_station = stations_df_clean[
                    stations_df_clean['station_name'].str.contains(
                        main_name, 
                        case=False, 
                        na=False,
                        regex=False
                    )
                ]
                if len(matching_station) > 0:
                    station_code = matching_station.iloc[0]['station_code']
                    station_name_clean = matching_station.iloc[0]['station_name']
                else:
                    # Fallback: use index-based code
                    station_code = f"ARPAL_{col_idx:03d}"
                    station_name_clean = station_name_raw
            else:
                station_code = f"ARPAL_{col_idx:03d}"
                station_name_clean = station_name_raw
        
        column_to_station[col] = (station_code, station_name_clean)
        print(f"   Mapped column '{col[:50]}...' -> {station_code} ({station_name_clean})")
    
    # Convert to long format
    long_format_rows = []
    for idx, row in df.iterrows():
        datetime_val = row['datetime']
        for col in df.columns:
            if col == 'datetime':
                continue
            
            pm10_value = row[col]
            station_code, station_name = column_to_station[col]
            
            # Skip NaN values (they'll be represented as missing rows)
            if pd.isna(pm10_value):
                continue
            
            long_format_rows.append({
                'datetime': datetime_val,
                'station_code': station_code,
                'station_name': station_name,
                'pm10': float(pm10_value)
            })
    
    result_df = pd.DataFrame(long_format_rows)
    result_df = result_df.sort_values(['station_code', 'datetime']).reset_index(drop=True)
    
    print(f"   Converted to {len(result_df):,} rows in long format")
    
    # ============================================================================
    # Step 4: Save to CSV
    # ============================================================================
    print(f"\n4. Saving to CSV...")
    
    output_csv = output_dir / "merged_pm10_hourly.csv"
    result_df.to_csv(output_csv, index=False)
    print(f"   Saved to: {output_csv}")
    
    # Summary
    print(f"\n5. Summary:")
    print(f"   Stations: {len(stations_df_clean)}")
    print(f"   Total measurements: {len(result_df):,}")
    print(f"   Date range: {result_df['datetime'].min()} to {result_df['datetime'].max()}")
    print(f"   Unique stations in data: {result_df['station_code'].nunique()}")
    
    print("\n" + "=" * 80)
    print("Conversion complete!")
    print("=" * 80)
    
    return result_df, stations_df_clean


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    excel_file = project_root / "data" / "arpal" / "PM10" / "PM10_Orari_Lombardia.xlsx"
    
    convert_arpal_excel_to_csv(excel_file)

