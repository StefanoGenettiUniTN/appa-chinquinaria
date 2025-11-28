#!/usr/bin/env python3
"""
Script to merge all CSV files for each station, grouping by timestamp.
Each station will have all measurements (temperature, precipitation, wind, etc.) 
merged into a single CSV with timestamps aligned.
"""

import csv
import os
import glob
from datetime import datetime
from collections import defaultdict

# Directory containing the CSV files
data_dir = "/home/user/Documents/unitn/public-ai-challenge/appa-chinquinaria/data/meteo-trentino/SampleEttore"

# Get all unique station codes
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
stations = set()
for csv_file in csv_files:
    if "_merged.csv" in csv_file:
        continue  # Skip already merged files
    filename = os.path.basename(csv_file)
    station_code = filename.split('_')[0]  # Extract station code (e.g., T0010)
    stations.add(station_code)

print(f"Found {len(stations)} stations: {sorted(stations)}")

def parse_timestamp(timestamp_str):
    """Parse timestamp string to datetime object"""
    try:
        return datetime.strptime(timestamp_str.strip(), '%H:%M:%S %d/%m/%Y')
    except:
        return None

# Process each station
for station in sorted(stations):
    print(f"\n{'='*60}")
    print(f"Processing station: {station}")
    print(f"{'='*60}")
    
    # Find all CSV files for this station
    station_files = glob.glob(os.path.join(data_dir, f"{station}_*.csv"))
    station_files = [f for f in station_files if "_merged.csv" not in f]
    print(f"Found {len(station_files)} measurement files for {station}")
    
    # Dictionary to store all data by timestamp
    # Structure: {timestamp_str: {datetime_obj: dt, measurement_type: {value: x, quality: y}}}
    all_data = defaultdict(lambda: {'datetime': None, 'measurements': {}})
    measurement_types = []
    
    # Read each measurement file
    for file_path in station_files:
        filename = os.path.basename(file_path)
        # Extract measurement type (everything after station code and underscore)
        measurement_type = filename.replace(f"{station}_", "").replace(".csv", "")
        measurement_types.append(measurement_type)
        print(f"  Reading: {measurement_type}")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                
                # Skip first 4 header rows
                for _ in range(4):
                    next(reader)
                
                record_count = 0
                # Read data rows
                for row in reader:
                    if len(row) < 3:
                        continue
                    
                    timestamp_str = row[0].strip()
                    value = row[1].strip()
                    quality = row[2].strip()
                    
                    # Parse timestamp
                    dt = parse_timestamp(timestamp_str)
                    if dt is None:
                        continue
                    
                    # Store data
                    if all_data[timestamp_str]['datetime'] is None:
                        all_data[timestamp_str]['datetime'] = dt
                    
                    all_data[timestamp_str]['measurements'][measurement_type] = {
                        'value': value,
                        'quality': quality
                    }
                    record_count += 1
                
                print(f"    Loaded {record_count} records")
                
        except Exception as e:
            print(f"    ERROR reading {filename}: {e}")
            continue
    
    # Check if we have data
    if len(all_data) == 0:
        print(f"  No valid data for station {station}, skipping...")
        continue
    
    print(f"\n  Merging {len(measurement_types)} measurement types...")
    print(f"  Total unique timestamps: {len(all_data)}")
    
    # Sort by datetime
    sorted_timestamps = sorted(all_data.items(), key=lambda x: x[1]['datetime'])
    
    # Prepare output CSV
    output_file = os.path.join(data_dir, f"{station}_merged.csv")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        header = ['datetime', 'timestamp']
        for mtype in sorted(measurement_types):
            header.append(f'{mtype}_value')
            header.append(f'{mtype}_quality')
        writer.writerow(header)
        
        # Write data rows
        for timestamp_str, data in sorted_timestamps:
            row = [
                data['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                timestamp_str
            ]
            
            # Add values for each measurement type
            for mtype in sorted(measurement_types):
                if mtype in data['measurements']:
                    row.append(data['measurements'][mtype]['value'])
                    row.append(data['measurements'][mtype]['quality'])
                else:
                    row.append('')  # Missing value
                    row.append('')  # Missing quality
            
            writer.writerow(row)
    
    # Calculate statistics
    min_dt = min(d['datetime'] for d in all_data.values())
    max_dt = max(d['datetime'] for d in all_data.values())
    
    print(f"\n  ✓ Saved merged CSV: {output_file}")
    print(f"    Total records: {len(all_data)}")
    print(f"    Date range: {min_dt} to {max_dt}")
    print(f"    Measurements included: {', '.join(sorted(measurement_types))}")

print(f"\n{'='*60}")
print("✓ All stations processed successfully!")
print(f"{'='*60}")
