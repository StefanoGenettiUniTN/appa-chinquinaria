import pandas as pd
import argparse
from pathlib import Path
import sys

def parse_column(col_name, regions):
    if col_name == 'datetime':
        return None, None, None
    
    # Iterate over regions to find the split point
    for region in regions:
        if f"_{region}_" in col_name:
            parts = col_name.split(f"_{region}_")
            variable = parts[0]
            station_code = parts[1]
            return variable, region, station_code
    return None, None, None

def main():
    parser = argparse.ArgumentParser(description="Convert hourly wide dataset to daily-like long format.")
    parser.add_argument("--hourly", required=True, help="Path to hourly dataset CSV")
    parser.add_argument("--mapping", required=True, help="Path to station mapping CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument("--sample", type=int, default=None, help="Number of rows to process (for testing)")
    
    args = parser.parse_args()
    
    hourly_path = Path(args.hourly)
    mapping_path = Path(args.mapping)
    output_path = Path(args.output)
    
    if not hourly_path.exists():
        print(f"Error: Hourly file not found at {hourly_path}")
        sys.exit(1)
        
    if not mapping_path.exists():
        print(f"Error: Mapping file not found at {mapping_path}")
        sys.exit(1)
        
    print("Loading mapping table...")
    conversion_df = pd.read_csv(mapping_path)
    conversion_df['station_code'] = conversion_df['station_code'].astype(str)
    
    regions = conversion_df['region'].unique()
    
    station_mapping = {}
    for _, row in conversion_df.iterrows():
        key = (row['region'], row['station_code'])
        station_mapping[key] = row['station_name']
        
    print(f"Loaded {len(station_mapping)} stations.")
    
    print("Loading hourly dataset...")
    if args.sample:
        df = pd.read_csv(hourly_path, nrows=args.sample)
    else:
        df = pd.read_csv(hourly_path)
        
    print(f"Dataset shape: {df.shape}")
    
    # Create column mapping
    print("Parsing columns...")
    col_mapping = {}
    for col in df.columns:
        if col != 'datetime':
            col_mapping[col] = parse_column(col, regions)
            
    # Melt
    print("Melting dataframe...")
    melted = df.melt(id_vars=['datetime'], var_name='original_col', value_name='value')
    
    # Extract metadata
    print("Extracting metadata...")
    meta_data = melted['original_col'].map(col_mapping)
    
    # Filter out columns that didn't parse (if any)
    mask = meta_data.notna()
    melted = melted[mask]
    meta_data = meta_data[mask]
    
    melted['variable'] = meta_data.apply(lambda x: x[0])
    melted['region'] = meta_data.apply(lambda x: x[1])
    melted['station_code'] = meta_data.apply(lambda x: x[2])
    
    # Pivot
    print("Pivoting dataframe...")
    pivoted = melted.pivot_table(index=['datetime', 'region', 'station_code'], 
                                 columns='variable', 
                                 values='value', 
                                 aggfunc='first').reset_index()
    
    # Map station names
    print("Mapping station names...")
    pivoted['station_key'] = list(zip(pivoted['region'], pivoted['station_code']))
    pivoted['Stazione_APPA'] = pivoted['station_key'].map(station_mapping)
    
    # Rename columns
    rename_dict = {
        'datetime': 'Data',
        'pm10': 'PM10_(ug.m-3)',
        'temperature_2m': 'Temperatura_(°C)',
        'total_precipitation': 'Precipitazione_(mm)',
    }
    pivoted = pivoted.rename(columns=rename_dict)
    
    # Unit conversions
    print("Converting units...")
    if 'Temperatura_(°C)' in pivoted.columns:
        pivoted['Temperatura_(°C)'] = pivoted['Temperatura_(°C)'] - 273.15
        
    if 'Precipitazione_(mm)' in pivoted.columns:
        pivoted['Precipitazione_(mm)'] = pivoted['Precipitazione_(mm)'] * 1000
        pivoted['Precipitazione_(mm)'] = pivoted['Precipitazione_(mm)'].clip(lower=0)
        
    # Select columns
    target_cols = ['Data', 'Stazione_APPA', 'PM10_(ug.m-3)', 'Temperatura_(°C)', 'Precipitazione_(mm)']
    other_cols = [c for c in pivoted.columns if c not in target_cols and c not in ['region', 'station_code', 'station_key']]
    final_cols = target_cols + other_cols
    final_cols = [c for c in final_cols if c in pivoted.columns]
    
    final_df = pivoted[final_cols]
    
    print(f"Saving to {output_path}...")
    final_df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
