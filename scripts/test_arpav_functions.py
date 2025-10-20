#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for ARPAV downloader functions.
Tests the key parsing and fetching functions before running full download.
"""

import sys
from pathlib import Path

# Add parent directory to path to import the main script
sys.path.insert(0, str(Path(__file__).parent))

from bulk_download_arpav import (
    create_session,
    fetch_stations_for_year_sensor,
    download_sensor_year_html,
    parse_html_table
)


def test_fetch_stations():
    """Test fetching station metadata from XML endpoint."""
    print("=" * 60)
    print("TEST 1: Fetch Station Metadata (XML Parsing)")
    print("=" * 60)
    
    session = create_session(verify_ssl=False)
    
    # Test with year 2024 and TEMPMIN sensor
    print("\nFetching stations for year=2024, sensor=TEMPMIN...")
    stations = fetch_stations_for_year_sensor(session, 2024, "TEMPMIN")
    
    if not stations:
        print("❌ FAILED: No stations returned")
        return False
    
    print(f"✓ SUCCESS: Found {len(stations)} stations")
    
    # Show first few stations
    print("\nFirst 3 stations:")
    for i, station in enumerate(stations[:3]):
        print(f"\n  Station {i+1}:")
        print(f"    ID: {station['station_id']}")
        print(f"    Name: {station['name']}")
        print(f"    Location: ({station['x']}, {station['y']})")
        print(f"    Comune: {station['comune']}")
        print(f"    Sensors: {[s['id'] for s in station['sensors']]}")
    
    return True


def test_download_and_parse():
    """Test downloading and parsing HTML data."""
    print("\n" + "=" * 60)
    print("TEST 2: Download and Parse HTML Data")
    print("=" * 60)
    
    session = create_session(verify_ssl=False)
    
    # First get a valid station ID
    print("\nFetching a station ID from 2024...")
    stations = fetch_stations_for_year_sensor(session, 2024, "TEMPMIN")
    
    if not stations:
        print("❌ FAILED: Could not get station list")
        return False
    
    test_station_id = stations[0]['station_id']
    test_station_name = stations[0]['name']
    test_sensor_id = stations[0]['sensors'][0]['id']  # Get first sensor ID
    test_year = 2024
    
    print(f"Using station: {test_station_id} ({test_station_name})")
    print(f"Using sensor ID: {test_sensor_id}")
    
    # Download HTML
    print(f"\nDownloading HTML for sensor {test_sensor_id}, year {test_year}...")
    html_content = download_sensor_year_html(session, test_sensor_id, test_year)
    
    if not html_content:
        print("❌ FAILED: Could not download HTML")
        return False
    
    print(f"✓ SUCCESS: Downloaded {len(html_content)} bytes of HTML")
    
    # Parse HTML
    print(f"\nParsing HTML table...")
    df = parse_html_table(html_content, test_station_id, test_year)
    
    if df is None:
        print("❌ FAILED: Could not parse HTML table")
        # Save HTML for debugging
        debug_file = Path("/tmp/arpav_debug.html")
        debug_file.write_text(html_content, encoding='utf-8')
        print(f"HTML saved to {debug_file} for debugging")
        return False
    
    print(f"✓ SUCCESS: Parsed table with {len(df)} rows and {len(df.columns)} columns")
    
    print("\nDataFrame info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    return True


def test_multiple_sensors():
    """Test fetching multiple sensors to verify metadata collection."""
    print("\n" + "=" * 60)
    print("TEST 3: Fetch Multiple Sensors")
    print("=" * 60)
    
    session = create_session(verify_ssl=False)
    
    test_sensors = ["TEMPMIN", "PREC", "UMID"]
    year = 2024
    
    all_stations = {}
    
    for sensor in test_sensors:
        print(f"\nFetching stations for sensor: {sensor}...")
        stations = fetch_stations_for_year_sensor(session, year, sensor)
        
        if stations:
            print(f"  ✓ Found {len(stations)} stations")
            
            # Collect unique station IDs
            for station in stations:
                station_id = station['station_id']
                if station_id not in all_stations:
                    all_stations[station_id] = {
                        'name': station['name'],
                        'sensors': []
                    }
                all_stations[station_id]['sensors'].append(sensor)
        else:
            print(f"  ⚠ No stations found for {sensor}")
    
    print(f"\n✓ Total unique stations across all sensors: {len(all_stations)}")
    
    # Show stations with multiple sensors
    multi_sensor_stations = {sid: data for sid, data in all_stations.items() if len(data['sensors']) > 1}
    print(f"✓ Stations with multiple sensors: {len(multi_sensor_stations)}")
    
    if multi_sensor_stations:
        print("\nExample multi-sensor stations:")
        for i, (sid, data) in enumerate(list(multi_sensor_stations.items())[:3]):
            print(f"  {sid} ({data['name']}): {data['sensors']}")
    
    return True


def main():
    """Run all tests."""
    print("\n🧪 ARPAV Downloader Function Tests")
    print("=" * 60)
    print("This will test key components before running the full download.")
    print("=" * 60)
    
    results = []
    
    # Test 1: XML parsing
    try:
        results.append(("Fetch Stations (XML)", test_fetch_stations()))
    except Exception as e:
        print(f"❌ EXCEPTION in test 1: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Fetch Stations (XML)", False))
    
    # Test 2: HTML download and parsing
    try:
        results.append(("Download & Parse HTML", test_download_and_parse()))
    except Exception as e:
        print(f"❌ EXCEPTION in test 2: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Download & Parse HTML", False))
    
    # Test 3: Multiple sensors
    try:
        results.append(("Multiple Sensors", test_multiple_sensors()))
    except Exception as e:
        print(f"❌ EXCEPTION in test 3: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Multiple Sensors", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Ready to run full download.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

