"""
Merged Dataset Post-Processing
APPA-EEA Data Quality Analysis and Cleaning

This script performs comprehensive data quality analysis and cleaning on the merged
APPA-EEA dataset by proximity, with a focus on identifying and handling missing data
(NaN values) and problematic stations.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/merged_data_post_processing.log'),
        logging.StreamHandler()
    ]
)


def load_merged_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the merged APPA-EEA dataset.
    
    Args:
        filepath: Path to the merged dataset CSV file
    
    Returns:
        Loaded DataFrame
    """
    logging.info(f"Loading merged dataset from: {filepath}")
    df = pd.read_csv(filepath)
    logging.info(f"Dataset loaded successfully! Shape: {df.shape}")
    return df


def display_dataset_overview(df: pd.DataFrame) -> None:
    """
    Display comprehensive dataset overview including shape, size, and basic statistics.
    
    Args:
        df: Input DataFrame
    """
    print("=" * 80)
    print("DATASET SHAPE AND SIZE")
    print("=" * 80)
    print(f"Total rows: {df.shape[0]:,}")
    print(f"Total columns: {df.shape[1]}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n" + "=" * 80)
    print("FIRST 5 ROWS")
    print("=" * 80)
    print(df.head())
    
    print("\n" + "=" * 80)
    print("LAST 5 ROWS")
    print("=" * 80)
    print(df.tail())
    
    print("\n" + "=" * 80)
    print("BASIC STATISTICS")
    print("=" * 80)
    print(df.describe())
    
    print("\n" + "=" * 80)
    print("DATA TYPES")
    print("=" * 80)
    print(df.dtypes)


def analyze_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, float]:
    """
    Perform comprehensive missing values analysis.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Tuple of (missing_data DataFrame, total_missing count, total_missing_pct)
    """
    print("\n" + "=" * 80)
    print("MISSING VALUES SUMMARY")
    print("=" * 80)
    
    # Count and percentage of NaN values
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percent': (df.isnull().sum() / len(df) * 100).round(2),
        'Data_Type': df.dtypes
    })
    
    # Sort by missing percentage (descending)
    missing_data = missing_data.sort_values('Missing_Percent', ascending=False)
    
    print(missing_data.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()
    total_missing_pct = (total_missing / total_cells * 100)
    
    print(f"Total cells in dataset: {total_cells:,}")
    print(f"Total missing cells: {total_missing:,}")
    print(f"Overall missing percentage: {total_missing_pct:.2f}%")
    print(f"\nColumns with NO missing values: {(missing_data['Missing_Count'] == 0).sum()}")
    print(f"Columns with SOME missing values: {(missing_data['Missing_Count'] > 0).sum()}")
    print(f"Columns with ALL missing values: {(missing_data['Missing_Percent'] == 100).sum()}")
    
    return missing_data, total_missing, total_missing_pct


def detect_problematic_columns(missing_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Identify problematic columns based on missing data percentages.
    
    Args:
        missing_data: DataFrame with missing data statistics
    
    Returns:
        Tuple of (high_missing, all_missing, some_missing) DataFrames
    """
    print("\n" + "=" * 80)
    print("PROBLEMATIC COLUMNS ANALYSIS")
    print("=" * 80)
    
    # Columns with >50% missing
    high_missing = missing_data[missing_data['Missing_Percent'] > 50]
    print(f"\n⚠️  Columns with >50% missing data ({len(high_missing)}):")
    if len(high_missing) > 0:
        print(high_missing[['Column', 'Missing_Percent']].to_string(index=False))
    else:
        print("  None")
    
    # Columns with 100% missing (completely empty)
    all_missing = missing_data[missing_data['Missing_Percent'] == 100]
    print(f"\n❌ Columns with 100% missing data ({len(all_missing)}):")
    if len(all_missing) > 0:
        print(all_missing[['Column', 'Missing_Percent']].to_string(index=False))
        print("\n  Recommendation: Consider dropping these columns")
    else:
        print("  None")
    
    # Columns with some missing (0-50%)
    some_missing = missing_data[(missing_data['Missing_Percent'] > 0) & (missing_data['Missing_Percent'] <= 50)]
    print(f"\n⚡ Columns with some missing data (0-50%) ({len(some_missing)}):")
    if len(some_missing) > 0:
        print(some_missing[['Column', 'Missing_Percent']].to_string(index=False))
    else:
        print("  None")
    
    return high_missing, all_missing, some_missing


def drop_columns_with_high_nan(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """
    Drop columns that have more than a specified percentage of NaN values.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        threshold (float): Maximum allowed fraction of NaN values (default 0.05 = 5%)
    
    Returns:
        pd.DataFrame: DataFrame with high-NaN columns removed
    """
    logging.info(f"Checking for columns with more than {threshold*100}% NaN values")
    
    initial_columns = df.shape[1]
    nan_percentages = df.isnull().sum() / len(df)
    
    # Find columns to drop
    columns_to_drop = nan_percentages[nan_percentages > threshold].index.tolist()
    
    if columns_to_drop:
        logging.warning(f"Dropping {len(columns_to_drop)} columns with >{threshold*100}% NaN values:")
        for col in columns_to_drop:
            nan_pct = nan_percentages[col] * 100
            logging.warning(f"  - '{col}': {nan_pct:.2f}% NaN")
        
        df = df.drop(columns=columns_to_drop)
        logging.info(f"Columns reduced from {initial_columns} to {df.shape[1]}")
    else:
        logging.info("No columns found with excessive NaN values")
    
    return df

def drop_stations_with_many_negatives(df: pd.DataFrame, 
                                           max_negatives: int = 300) -> pd.DataFrame:
    """
    Drop columns starting with 'SP' that have more than a specified number of negative values.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        max_negatives (int): Maximum allowed negative values per column (default 300)
    
    Returns:
        pd.DataFrame: DataFrame with problematic SP columns removed
    """
    logging.info(f"Checking SP columns for those with more than {max_negatives} negative values")
    
    # Find all columns starting with 'SP'
    sp_columns = [col for col in df.columns if col.startswith('SP')]
    logging.info(f"Found {len(sp_columns)} columns starting with 'SP'")
    
    initial_columns = df.shape[1]
    columns_to_drop = []
    
    # Check each SP column for negative values
    for col in sp_columns:
        # Convert to numeric, coercing errors to NaN
        col_numeric = pd.to_numeric(df[col], errors='coerce')
        
        # Count negative values (excluding NaN)
        negative_count = (col_numeric < 0).sum()
        
        if negative_count > max_negatives:
            columns_to_drop.append(col)
            logging.warning(f"  - '{col}': {negative_count} negative values")
    
    if columns_to_drop:
        logging.warning(f"Dropping {len(columns_to_drop)} SP columns with >{max_negatives} negative values")
        df = df.drop(columns=columns_to_drop)
        logging.info(f"Columns reduced from {initial_columns} to {df.shape[1]}")
    else:
        logging.info("No SP columns found with excessive negative values")
    
    return df

def analyze_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze negative values in all numeric columns.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with negative value statistics per column
    """
    print("\n" + "=" * 80)
    print("NEGATIVE VALUES ANALYSIS")
    print("=" * 80)
    
    negative_stats = []
    
    for col in df.columns:
        # Convert to numeric, coercing errors to NaN
        col_numeric = pd.to_numeric(df[col], errors='coerce')
        
        # Count negative values (excluding NaN)
        negative_count = (col_numeric < 0).sum()
        
        if negative_count > 0:
            negative_pct = (negative_count / len(df) * 100)
            negative_stats.append({
                'Column': col,
                'Negative_Count': negative_count,
                'Negative_Percent': round(negative_pct, 2),
                'Data_Type': df[col].dtype
            })
    
    if negative_stats:
        negative_df = pd.DataFrame(negative_stats)
        negative_df = negative_df.sort_values('Negative_Count', ascending=False)
        
        print(f"\nFound {len(negative_df)} columns with negative values:\n")
        print(negative_df.to_string(index=False))
        
        print(f"\nTotal columns with negatives: {len(negative_df)}")
        print(f"Total negative values across all columns: {negative_df['Negative_Count'].sum():,}")
        
        return negative_df
    else:
        print("\n✅ No negative values found in any column")
        return pd.DataFrame()


def drop_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Drop duplicate rows based on specified columns or all columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (List[str], optional): Column labels to consider for identifying duplicates.
                                     If None, uses all columns.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    logging.info("Checking for duplicate rows")
    
    initial_rows = len(df)
    
    if subset:
        logging.info(f"Checking duplicates based on columns: {subset}")
        duplicates_count = df.duplicated(subset=subset).sum()
    else:
        logging.info("Checking duplicates based on all columns")
        duplicates_count = df.duplicated().sum()
    
    if duplicates_count > 0:
        logging.warning(f"Found {duplicates_count:,} duplicate rows ({duplicates_count/initial_rows*100:.2f}%)")
        
        if subset:
            df = df.drop_duplicates(subset=subset, keep='first')
        else:
            df = df.drop_duplicates(keep='first')
        
        final_rows = len(df)
        logging.info(f"Rows reduced from {initial_rows:,} to {final_rows:,}")
        logging.info(f"Removed {initial_rows - final_rows:,} duplicate rows")
    else:
        logging.info("No duplicate rows found")
    
    return df.reset_index(drop=True)


def save_cleaned_dataset(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the cleaned dataset to CSV.
    
    Args:
        df: Cleaned DataFrame
        output_path: Path to save the cleaned dataset
    """
    logging.info(f"Saving cleaned dataset to: {output_path}")
    df.to_csv(output_path, index=False)
    logging.info(f"Cleaned dataset saved successfully! Shape: {df.shape}")


def main() -> None:
    """
    Main function to run the complete post-processing pipeline.
    """
    print("=" * 80)
    print("MERGED DATASET POST-PROCESSING PIPELINE")
    print("=" * 80)
    print()
    
    # Configuration
    input_path = 'output/merged_appa_eea_by_proximity.csv'
    output_path = 'output/merged_appa_eea_cleaned.csv'
    nan_threshold = 0.01  # 1% threshold for dropping columns
    max_negatives = 300   # Maximum negative values per station
    
    # Load dataset
    df = load_merged_dataset(input_path)
    
    # Display overview
    display_dataset_overview(df)
    
    # Analyze missing values (before cleaning)
    missing_data, total_missing, total_missing_pct = analyze_missing_values(df)
    
    # Detect problematic columns
    high_missing, all_missing, some_missing = detect_problematic_columns(missing_data)
    
    # Apply cleaning operations
    print("\n" + "=" * 80)
    print("APPLYING DATA CLEANING")
    print("=" * 80)
    
    df = drop_columns_with_high_nan(df, threshold=nan_threshold)
    df = drop_stations_with_many_negatives(df, max_negatives=max_negatives)
    
    # Drop duplicates based on Date and all features
    df = drop_duplicates(df)
    
    # Re-analyze after cleaning
    print("\n" + "=" * 80)
    print("POST-CLEANING ANALYSIS")
    print("=" * 80)
    missing_data_clean, total_missing_clean, total_missing_pct_clean = analyze_missing_values(df)
    
    # Analyze negative values
    negative_data = analyze_negative_values(df)
    
    # Save cleaned dataset
    save_cleaned_dataset(df, output_path)
    
    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Final dataset shape: {df.shape}")
    print(f"Final completeness: {100 - total_missing_pct_clean:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
