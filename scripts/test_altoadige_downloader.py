#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for Alto Adige bulk downloader functions.

This script tests the core functions of the bulk downloader without
performing a full download, making it suitable for quick validation.
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the downloader module
import bulk_download_altoadige as downloader


def test_session_creation():
    """Test creating a requests session."""
    print("\n" + "=" * 60)
    print("TEST 1: Session Creation")
    print("=" * 60)
    
    try:
        session = downloader.create_session()
        assert session is not None
        assert "User-Agent" in session.headers
        print("✓ Session created successfully")
        print(f"  User-Agent: {session.headers['User-Agent']}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_fetch_stations():
    """Test fetching all stations."""
    print("\n" + "=" * 60)
    print("TEST 2: Fetch Stations")
    print("=" * 60)
    
    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            session = downloader.create_session()
            
            stations_df = downloader.fetch_all_stations(session, tmp_path)
            
            assert stations_df is not None
            assert len(stations_df) > 0
            assert 'SCODE' in stations_df.columns
            
            print(f"✓ Successfully fetched {len(stations_df)} stations")
            print(f"  Columns: {list(stations_df.columns)}")
            print(f"  Sample station: {stations_df.iloc[0]['SCODE']} - {stations_df.iloc[0].get('NAME_I', 'N/A')}")
            
            return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fetch_sensors():
    """Test fetching sensors for a specific station."""
    print("\n" + "=" * 60)
    print("TEST 3: Fetch Sensors for Station")
    print("=" * 60)
    
    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            session = downloader.create_session()
            
            # Use a known station from the API docs
            test_station = "19850PG"
            
            sensors = downloader.fetch_station_sensors(session, test_station, tmp_path)
            
            if sensors and len(sensors) > 0:
                print(f"✓ Successfully fetched {len(sensors)} sensors for station {test_station}")
                print("  Available sensors:")
                for sensor in sensors[:5]:  # Show first 5
                    print(f"    - {sensor.get('TYPE', 'N/A'):10} {sensor.get('DESC_I', 'N/A')}")
                return True
            else:
                print(f"⚠ No sensors found for station {test_station} (may be normal)")
                return True  # Not necessarily an error
                
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_download_timeseries():
    """Test downloading timeseries data."""
    print("\n" + "=" * 60)
    print("TEST 4: Download Timeseries Data")
    print("=" * 60)
    
    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            session = downloader.create_session()
            
            # Use a known station and sensor from the API docs
            test_station = "19850PG"
            test_sensor = "Q"
            test_year = 2024
            
            df = downloader.download_timeseries_data(
                session, test_station, test_sensor, test_year, tmp_path
            )
            
            if df is not None and len(df) > 0:
                print(f"✓ Successfully downloaded {len(df)} records")
                print(f"  Station: {test_station}, Sensor: {test_sensor}, Year: {test_year}")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Date range: {df['DATE'].min()} to {df['DATE'].max()}")
                return True
            else:
                print(f"⚠ No data returned (may be normal if station has no data for this period)")
                return True  # Not necessarily an error
                
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_management():
    """Test state file loading and saving."""
    print("\n" + "=" * 60)
    print("TEST 5: State File Management")
    print("=" * 60)
    
    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Test saving state
            test_state = {
                "meta": {
                    "start_year": 2023,
                    "end_year": 2024,
                    "version": 1
                },
                "tasks": [
                    {
                        "station_code": "TEST001",
                        "sensor_code": "LT",
                        "year": 2023,
                        "status": "done"
                    }
                ]
            }
            
            downloader.save_state(tmp_path, test_state)
            print("✓ State saved successfully")
            
            # Test loading state
            loaded_state = downloader.load_state(tmp_path)
            assert loaded_state == test_state
            print("✓ State loaded successfully")
            
            # Verify atomic write worked
            state_path = tmp_path / downloader.STATE_FILENAME
            assert state_path.exists()
            print(f"✓ State file exists at {state_path}")
            
            return True
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_output_folder_structure():
    """Test output folder creation."""
    print("\n" + "=" * 60)
    print("TEST 6: Output Folder Structure")
    print("=" * 60)
    
    try:
        # Test default folder name
        folder = downloader.build_output_folder(None, 2023, 2024)
        expected = "altoadige_2023_2024"
        assert expected in str(folder)
        print(f"✓ Default folder name: {folder.name}")
        
        # Test custom folder name
        folder = downloader.build_output_folder("custom_test", 2023, 2024)
        assert "custom_test" in str(folder)
        print(f"✓ Custom folder name: {folder.name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all unit tests."""
    print("=" * 60)
    print("Alto Adige Bulk Downloader - Unit Tests")
    print("=" * 60)
    
    tests = [
        test_session_creation,
        test_fetch_stations,
        test_fetch_sensors,
        test_download_timeseries,
        test_state_management,
        test_output_folder_structure,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test {test_func.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

