# merge_datasets_by_proximity.py Guide

## Purpose
Merges APPA and EEA datasets by proximity, pivoting EEA data by individual stations for comprehensive analysis.

## Main Features
- Loads and preprocesses APPA and EEA datasets
- Normalizes schema and converts date columns
- Pivots EEA data by individual stations
- Merges datasets on date
- Saves merged dataset and reports statistics

## Usage
Use this script to create a comprehensive merged dataset for analysis, ensuring schema consistency and spatial alignment.

## Inputs & Outputs
- **Input:** Preprocessed APPA and filtered EEA datasets (CSV)
- **Output:** Merged dataset (CSV)

## Typical Workflow
1. Load and preprocess datasets
2. Normalize schema and pivot EEA data
3. Merge datasets on date
4. Save and report statistics
