#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for Alto Adige API connection and data download.

This script tests:
1. Fetching all stations
2. Fetching sensors for a specific station
3. Downloading timeseries data for a specific sensor
"""

import sys
import requests
import pandas as pd
from io import StringIO

# API configuration
BASE_URL = "http://daten.buergernetz.bz.it/services/meteo/v1"
STATIONS_ENDPOINT = f"{BASE_URL}/stations"
SENSORS_ENDPOINT = f"{BASE_URL}/sensors"
TIMESERIES_ENDPOINT = f"{BASE_URL}/timeseries"

TIMEOUT = (20, 120)


def test_fetch_stations():
    """Test fetching all stations."""
    print("=" * 60)
    print("TEST 1: Fetching All Stations")
    print("=" * 60)
    
    try:
        params = {"output_format": "CSV"}
        response = requests.get(STATIONS_ENDPOINT, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        
        print(f"✓ Successfully fetched {len(df)} stations")
        print("\nFirst 5 stations:")
        print(df.head())
        
        print("\nColumns:", list(df.columns))
        
        return df
        
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return None


def test_fetch_sensors(station_code: str):
    """Test fetching sensors for a specific station."""
    print("\n" + "=" * 60)
    print(f"TEST 2: Fetching Sensors for Station {station_code}")
    print("=" * 60)
    
    try:
        params = {
            "station_code": station_code,
            "output_format": "JSON"
        }
        response = requests.get(SENSORS_ENDPOINT, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        
        sensors = response.json()
        
        print(f"✓ Successfully fetched {len(sensors)} sensors")
        print("\nAvailable sensors:")
        
        for sensor in sensors:
            print(f"  - {sensor.get('TYPE', 'N/A'):10} {sensor.get('DESC_I', 'N/A'):30} "
                  f"Unit: {sensor.get('UNIT', 'N/A'):10} "
                  f"Last value: {sensor.get('VALUE', 'N/A')} at {sensor.get('DATE', 'N/A')}")
        
        return sensors
        
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return None


def test_fetch_timeseries(station_code: str, sensor_code: str, year: int = 2024):
    """Test fetching timeseries data for a specific sensor."""
    print("\n" + "=" * 60)
    print(f"TEST 3: Fetching Timeseries Data")
    print(f"Station: {station_code}, Sensor: {sensor_code}, Year: {year}")
    print("=" * 60)
    
    try:
        params = {
            "station_code": station_code,
            "sensor_code": sensor_code,
            "date_from": f"{year}0101",
            "date_to": f"{year}0131",  # Just January for testing
            "output_format": "CSV"
        }
        
        response = requests.get(TIMESERIES_ENDPOINT, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        
        print(f"✓ Successfully fetched {len(df)} records")
        
        if len(df) > 0:
            print("\nFirst 10 records:")
            print(df.head(10))
            print("\nData summary:")
            print(df.describe())
        else:
            print("⚠ No data returned (may be normal if station has no data for this period)")
        
        return df
        
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return None


def test_api_limits():
    """Test API record limit (120,000 records)."""
    print("\n" + "=" * 60)
    print("TEST 4: Testing API Record Limits")
    print("=" * 60)
    
    try:
        # Try to fetch a full year from a station that likely has hourly data
        # This should have ~8760 records (365 days * 24 hours)
        station_code = "19850PG"  # Example from documentation
        sensor_code = "Q"  # Portata (flow rate)
        year = 2024
        
        params = {
            "station_code": station_code,
            "sensor_code": sensor_code,
            "date_from": f"{year}0101",
            "date_to": f"{year}1231",
            "output_format": "CSV"
        }
        
        response = requests.get(TIMESERIES_ENDPOINT, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        
        print(f"✓ Successfully fetched {len(df)} records for full year {year}")
        
        if len(df) >= 120000:
            print("⚠ WARNING: Response may be truncated (hit 120,000 record limit)")
            print("  Consider splitting requests into smaller date ranges")
        else:
            print(f"✓ Response is within limit ({len(df)} < 120,000 records)")
        
        return df
        
    except Exception as e:
        print(f"✗ Error (this may be expected if station doesn't exist): {e}")
        return None


def main():
    """Run all tests."""
    print("Alto Adige API Connection Tests")
    print("=" * 60)
    print("Testing connection to Alto Adige Open Meteo Data V1 API")
    print("=" * 60)
    
    # Test 1: Fetch all stations
    stations_df = test_fetch_stations()
    
    if stations_df is None or len(stations_df) == 0:
        print("\n✗ Failed to fetch stations. Cannot continue with other tests.")
        sys.exit(1)
    
    # Pick a station for further testing
    test_station = stations_df.iloc[0]['SCODE']
    print(f"\nUsing station '{test_station}' for further tests...")
    
    # Test 2: Fetch sensors for a station
    sensors = test_fetch_sensors(test_station)
    
    if sensors and len(sensors) > 0:
        # Test 3: Fetch timeseries for the first available sensor
        test_sensor = sensors[0].get('TYPE')
        if test_sensor:
            test_fetch_timeseries(test_station, test_sensor, year=2024)
    
    # Test 4: API limits
    test_api_limits()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("✓ All basic connectivity tests completed")
    print("You can now run the bulk downloader:")
    print("  python scripts/bulk_download_altoadige.py --start 2023 --end 2024")
    print("=" * 60)


if __name__ == "__main__":
    main()

