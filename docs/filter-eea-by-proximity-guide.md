# filter_eea_by_proximity.py Guide

## Purpose
Filters EEA air quality stations based on proximity to Trentino weather stations using Haversine distance calculations.

## Main Features
- Downloads and preprocesses EEA air quality data
- Computes pairwise distances between Trentino and EEA stations
- Selects closest EEA stations for each Trentino station
- Filters EEA dataset to selected stations
- Prevents data leakage by removing problematic stations
- Saves filtered dataset and proximity mapping

## Usage
Use this script to select relevant EEA stations for integration with Trentino data, ensuring spatial relevance and preventing data leakage.

## Inputs & Outputs
- **Input:** Raw EEA air quality data, Trentino weather station data
- **Output:** Filtered EEA dataset (CSV), proximity mapping (CSV)

## Typical Workflow
1. Download and preprocess EEA data
2. Extract unique Trentino and EEA stations
3. Compute distances and select closest stations
4. Filter EEA dataset
5. Save results
