#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal integration test for Alto Adige bulk downloader.

This script performs a minimal download to verify the complete workflow:
- Fetches station metadata
- Collects sensor information for a single station
- Downloads a small date range of data
- Verifies file structure and state management
"""

import sys
import tempfile
import shutil
from pathlib import Path
import json

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the downloader module
import bulk_download_altoadige as downloader


def test_minimal_download():
    """
    Perform a minimal download test with a single station and limited date range.
    """
    print("=" * 60)
    print("Alto Adige Minimal Integration Test")
    print("=" * 60)
    
    # Create temporary directory for this test
    test_dir = Path(__file__).parent.parent / "data" / "altoadige" / "test_minimal"
    
    # Clean up if exists
    if test_dir.exists():
        print(f"\nCleaning up existing test directory: {test_dir}")
        shutil.rmtree(test_dir)
    
    try:
        print(f"\nTest directory: {test_dir}")
        print(f"Test parameters: Station 19850PG, Sensor Q, Year 2024")
        print("=" * 60)
        
        # Create session
        session = downloader.create_session()
        downloader.ensure_folder(test_dir)
        
        # Phase 1: Fetch stations
        print("\n[1/5] Fetching all stations...")
        stations_df = downloader.fetch_all_stations(session, test_dir)
        print(f"✓ Fetched {len(stations_df)} stations")
        
        # Verify stations.csv exists
        stations_file = test_dir / "stations.csv"
        assert stations_file.exists(), "stations.csv not created"
        print(f"✓ stations.csv created: {stations_file}")
        
        # Phase 2: Collect sensors for just one station
        print("\n[2/5] Fetching sensors for test station...")
        test_station = "19850PG"
        sensors = downloader.fetch_station_sensors(session, test_station, test_dir)
        
        if not sensors or len(sensors) == 0:
            print(f"⚠ No sensors found for station {test_station}")
            return False
        
        print(f"✓ Found {len(sensors)} sensors for station {test_station}")
        
        # Verify sensor JSON exists
        sensor_file = test_dir / "sensors" / f"{test_station}.json"
        assert sensor_file.exists(), "Sensor JSON not created"
        print(f"✓ Sensor file created: {sensor_file}")
        
        # Phase 3: Download timeseries for one sensor
        print("\n[3/5] Downloading timeseries data...")
        test_sensor = "Q"  # Flow rate
        test_year = 2024
        
        df = downloader.download_timeseries_data(
            session, test_station, test_sensor, test_year, test_dir
        )
        
        if df is None or len(df) == 0:
            print(f"⚠ No data downloaded")
            return False
        
        print(f"✓ Downloaded {len(df)} records")
        
        # Verify CSV exists
        data_file = test_dir / test_sensor / str(test_year) / f"{test_station}.csv"
        assert data_file.exists(), "Data CSV not created"
        print(f"✓ Data file created: {data_file}")
        
        # Phase 4: Test state management
        print("\n[4/5] Testing state management...")
        
        # Create a minimal state
        station_sensors_map = {test_station: [test_sensor]}
        state = downloader.init_state(test_dir, station_sensors_map, test_year, test_year)
        
        assert "meta" in state, "State missing meta"
        assert "tasks" in state, "State missing tasks"
        assert len(state["tasks"]) == 1, f"Expected 1 task, got {len(state['tasks'])}"
        
        print(f"✓ State created with {len(state['tasks'])} task(s)")
        
        # Mark task as done
        state["tasks"][0]["status"] = "done"
        state["tasks"][0]["records"] = len(df)
        downloader.save_state(test_dir, state)
        
        # Verify state file exists
        state_file = test_dir / "state.json"
        assert state_file.exists(), "state.json not created"
        print(f"✓ State file created: {state_file}")
        
        # Phase 5: Verify directory structure
        print("\n[5/5] Verifying directory structure...")
        
        expected_files = [
            test_dir / "stations.csv",
            test_dir / "sensors" / f"{test_station}.json",
            test_dir / test_sensor / str(test_year) / f"{test_station}.csv",
            test_dir / "state.json"
        ]
        
        for expected_file in expected_files:
            if expected_file.exists():
                size = expected_file.stat().st_size
                print(f"✓ {expected_file.relative_to(test_dir)} ({size:,} bytes)")
            else:
                print(f"✗ Missing: {expected_file}")
                return False
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"✓ All tests passed!")
        print(f"\nDownload statistics:")
        print(f"  - Stations: {len(stations_df)}")
        print(f"  - Sensors tested: 1 (station {test_station})")
        print(f"  - Records downloaded: {len(df):,}")
        print(f"  - Files created: {len(expected_files)}")
        
        print(f"\nTest directory structure:")
        print(f"  {test_dir}/")
        for item in sorted(test_dir.rglob("*")):
            if item.is_file():
                rel_path = item.relative_to(test_dir)
                size = item.stat().st_size
                print(f"    {rel_path} ({size:,} bytes)")
        
        print(f"\n✓ Test data preserved at: {test_dir}")
        print(f"  (You can delete this directory when done)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the minimal integration test."""
    success = test_minimal_download()
    
    if success:
        print("\n" + "=" * 60)
        print("INTEGRATION TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nThe downloader is working correctly!")
        print("You can now run full downloads with:")
        print("  python scripts/bulk_download_altoadige.py --start 2023 --end 2024")
        return 0
    else:
        print("\n" + "=" * 60)
        print("INTEGRATION TEST FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

