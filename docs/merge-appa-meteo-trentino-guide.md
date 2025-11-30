# merge_appa_meteo_trentino.py Guide

## Purpose
Merges APPA air quality measurements with MeteoTrentino weather data for the Trentino region.

## Main Features
- Downloads and preprocesses APPA and MeteoTrentino datasets
- Matches stations by location and aggregates measurements to daily
- Merges datasets on station and date
- Cleans and standardizes columns
- Saves the final integrated dataset

## Usage
Run this script to create a unified dataset for Trentino, combining pollution and weather data for analysis and modeling.

## Inputs & Outputs
- **Input:** Raw APPA and MeteoTrentino data (CSV)
- **Output:** Merged dataset (CSV)

## Typical Workflow
1. Download and preprocess APPA and MeteoTrentino data
2. Match stations and aggregate measurements
3. Merge datasets on station and date
4. Save the final dataset
