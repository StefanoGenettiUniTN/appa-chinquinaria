import pandas as pd
import numpy as np

# Read both datasets
print("Loading datasets...")
appa = pd.read_csv('./output/historical_weather_airPM_trentino.csv')
eea = pd.read_csv('./output/eea_data_aggregated.csv')

# Convert 'Data' column to datetime for both datasets
appa['Data'] = pd.to_datetime(appa['Data'])
eea['Data'] = pd.to_datetime(eea['Data'])

print(f"APPA dataset shape: {appa.shape}")
print(f"EEA dataset shape: {eea.shape}")

# Create a pivot structure for EEA data
# For each unique Region_microarea_id, create columns for each attribute
print("\nPivoting EEA dataset...")

# Get all unique regions
unique_regions = eea['Region_microarea_id'].unique()
print(f"Unique regions in EEA: {len(unique_regions)}")

# Prepare EEA data for merging
# We'll create a wide format where each region's attributes become separate columns
eea_wide = eea.copy()

# Pivot the EEA data: create columns like Region_MA_0_Valore, Region_MA_0_Latitudine, etc.
eea_pivot = pd.DataFrame()
eea_pivot['Data'] = eea['Data'].unique()

for region in unique_regions:
    region_data = eea[eea['Region_microarea_id'] == region].copy()
    region_data = region_data[['Data', 'Valore', 'Latitudine', 'Longitudine', 'Unità di misura']]
    region_data.columns = [f'{region}_Data', f'{region}_Valore', f'{region}_Latitudine', 
                           f'{region}_Longitudine', f'{region}_Unità di misura']
    
    # Merge on Data
    if eea_pivot.shape[0] == 0:
        eea_pivot = region_data.rename(columns={f'{region}_Data': 'Data'})
    else:
        region_data_merged = region_data.rename(columns={f'{region}_Data': 'Data'})
        eea_pivot = eea_pivot.merge(region_data_merged, on='Data', how='outer')

print(f"EEA pivoted shape: {eea_pivot.shape}")

# Merge APPA with pivoted EEA on Data
print("\nMerging APPA with pivoted EEA...")
result = appa.merge(eea_pivot, on='Data', how='left')

print(f"Final merged dataset shape: {result.shape}")
print(f"Final merged dataset columns ({len(result.columns)}): {result.columns.tolist()}")

# Display first few rows
print("\nFirst few rows of merged dataset:")
print(result.head())

# Save the result
output_path = './output/merged_appa_eea.csv'
result.to_csv(output_path, index=False)
print(f"\nMerged dataset saved to: {output_path}")

# Display basic statistics
print(f"\nMerged dataset info:")
print(f"Total rows: {len(result)}")
print(f"Total columns: {len(result.columns)}")
print(f"Memory usage: {result.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
