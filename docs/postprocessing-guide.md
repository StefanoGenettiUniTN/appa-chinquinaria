# PostProcessing.py Guide

## Purpose
Post-processes merged APPA-EEA datasets, focusing on data quality analysis and cleaning.

## Main Features
- Missing value analysis and reporting
- Detection and removal of problematic columns and stations
- Handling and analysis of negative values
- Duplicate removal
- Saving cleaned datasets for downstream analysis

## Usage
Run this script after merging datasets to ensure high data quality for further analysis and modeling. The script provides detailed logging and summary statistics for each cleaning step.

## Inputs & Outputs
- **Input:** Merged APPA-EEA dataset (CSV)
- **Output:** Cleaned dataset (CSV), log file with cleaning summary

## Typical Workflow
1. Load merged dataset
2. Display overview and statistics
3. Analyze and clean missing/negative values
4. Remove duplicates
5. Save cleaned dataset
