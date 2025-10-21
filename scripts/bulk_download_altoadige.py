#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Alto Adige (Provincia Autonoma di Bolzano) Meteorological Data Bulk Downloader

Downloads weather data from Alto Adige monitoring stations using the Open Meteo Data V1 API.
Data is organized by sensor type and year, with resume capability via state tracking.

Usage examples:
    python bulk_download_altoadige.py --start 2023 --end 2024
    python bulk_download_altoadige.py --start 2020 --end 2025 --out altoadige_2020_2025
    python bulk_download_altoadige.py --start 2023 --end 2023 --sensors "LT,N,Q"

API documentation:
    - Stations: http://daten.buergernetz.bz.it/services/meteo/v1/stations
    - Sensors: http://daten.buergernetz.bz.it/services/meteo/v1/sensors
    - Timeseries: http://daten.buergernetz.bz.it/services/meteo/v1/timeseries
    
Note: The API supports a maximum of 120,000 records per request.
"""

import argparse
import datetime as dt
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

try:
    import requests
    import pandas as pd
except ImportError:
    print("This script requires 'requests' and 'pandas' packages.", file=sys.stderr)
    print("Install with: pip install requests pandas", file=sys.stderr)
    sys.exit(1)

# Optional progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False


# ---- Configuration ----
BASE_URL = "http://daten.buergernetz.bz.it/services/meteo/v1"
STATIONS_ENDPOINT = f"{BASE_URL}/stations"
SENSORS_ENDPOINT = f"{BASE_URL}/sensors"
TIMESERIES_ENDPOINT = f"{BASE_URL}/timeseries"

# All available sensor types
ALL_SENSOR_CODES = {
    "LT": "Temperatura dell'aria",
    "LF": "Umidità dell'aria",
    "N": "Pioggia",
    "WG": "Velocità media del vento",
    "WG.BOE": "Raffica vento",
    "WR": "Direzione media del vento",
    "LD.RED": "Pressione aria ridotta",
    "SD": "Tempo di soleggiamento",
    "GS": "Radiazione globale",
    "HS": "Altezza neve",
    "W": "Livello dell'acqua",
    "Q": "Portata"
}

STATE_FILENAME = "state.json"
STATIONS_FILENAME = "stations.csv"
USER_AGENT = "altoadige-bulk-downloader/1.0"

# Request timeouts
HTTP_CONNECT_TIMEOUT = 20
HTTP_READ_TIMEOUT = 120

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

# Rate limiting - random sleep between requests
RATE_LIMIT_MIN = 0.5
RATE_LIMIT_MAX = 2.0


def ensure_folder(out_dir: Path):
    """Create output directory if it doesn't exist."""
    out_dir.mkdir(parents=True, exist_ok=True)


def load_state(out_dir: Path) -> Dict:
    """Load state from JSON file."""
    state_path = out_dir / STATE_FILENAME
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(out_dir: Path, state: Dict):
    """Save state to JSON file atomically."""
    state_path = out_dir / STATE_FILENAME
    tmp = state_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True, ensure_ascii=False)
    tmp.replace(state_path)


def create_session() -> requests.Session:
    """Create and configure requests session."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    })
    return session


def rate_limit_sleep():
    """Sleep for a random interval to respect rate limits."""
    time.sleep(random.uniform(RATE_LIMIT_MIN, RATE_LIMIT_MAX))


def fetch_all_stations(session: requests.Session, out_dir: Path) -> pd.DataFrame:
    """
    Fetch all stations from the API and save to CSV.
    
    Returns DataFrame with station metadata.
    """
    print("\nFetching station metadata...")
    
    stations_path = out_dir / STATIONS_FILENAME
    
    # Check if we already have stations file
    if stations_path.exists():
        print(f"Loading existing stations file: {stations_path}")
        return pd.read_csv(stations_path)
    
    try:
        params = {"output_format": "CSV"}
        response = session.get(
            STATIONS_ENDPOINT,
            params=params,
            timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT)
        )
        response.raise_for_status()
        
        # Parse CSV response
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        
        # Save to file
        df.to_csv(stations_path, index=False, encoding='utf-8')
        
        print(f"Fetched {len(df)} stations")
        print(f"Saved to: {stations_path}")
        
        return df
        
    except Exception as e:
        print(f"Error fetching stations: {e}", file=sys.stderr)
        raise


def fetch_station_sensors(session: requests.Session, station_code: str, out_dir: Path, retries: int = MAX_RETRIES) -> Optional[List[Dict]]:
    """
    Fetch available sensors for a specific station.
    
    Returns list of sensor dictionaries or None on failure.
    """
    sensors_dir = out_dir / "sensors"
    ensure_folder(sensors_dir)
    
    sensors_path = sensors_dir / f"{station_code}.json"
    
    # Check if we already have sensors for this station
    if sensors_path.exists():
        with sensors_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    
    backoff = RETRY_BACKOFF
    
    for attempt in range(1, retries + 1):
        try:
            params = {
                "station_code": station_code,
                "output_format": "JSON"
            }
            
            response = session.get(
                SENSORS_ENDPOINT,
                params=params,
                timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT)
            )
            response.raise_for_status()
            
            sensors = response.json()
            
            # Save to file
            with sensors_path.open("w", encoding="utf-8") as f:
                json.dump(sensors, f, indent=2, ensure_ascii=False)
            
            rate_limit_sleep()
            return sensors
            
        except requests.RequestException as e:
            if attempt >= retries:
                print(f"Error fetching sensors for station {station_code} after {retries} attempts: {e}")
                return None
            
            print(f"  Retry {attempt}/{retries} for station {station_code}: {e}")
            time.sleep(backoff)
            backoff *= 1.5
    
    return None


def download_timeseries_data(
    session: requests.Session,
    station_code: str,
    sensor_code: str,
    year: int,
    out_dir: Path,
    retries: int = MAX_RETRIES
) -> Optional[pd.DataFrame]:
    """
    Download timeseries data for a specific station, sensor, and year.
    
    Returns DataFrame with the data or None on failure.
    """
    # Build output path
    sensor_dir = out_dir / sensor_code / str(year)
    ensure_folder(sensor_dir)
    
    csv_path = sensor_dir / f"{station_code}.csv"
    
    # Check if we already have this data
    if csv_path.exists():
        return pd.read_csv(csv_path)
    
    # Build date range for the year
    date_from = f"{year}0101"
    date_to = f"{year}1231"
    
    backoff = RETRY_BACKOFF
    
    for attempt in range(1, retries + 1):
        try:
            params = {
                "station_code": station_code,
                "sensor_code": sensor_code,
                "date_from": date_from,
                "date_to": date_to,
                "output_format": "CSV"
            }
            
            response = session.get(
                TIMESERIES_ENDPOINT,
                params=params,
                timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT)
            )
            response.raise_for_status()
            
            # Check if response has content
            if not response.text or len(response.text.strip()) == 0:
                # Empty response - no data available
                return None
            
            # Parse CSV response
            from io import StringIO
            try:
                df = pd.read_csv(StringIO(response.text))
            except pd.errors.EmptyDataError:
                # Empty CSV or no columns - no data available
                return None
            except Exception as e:
                print(f"  Warning: Failed to parse CSV for {station_code}/{sensor_code}/{year}: {e}")
                return None
            
            # Check if we got data
            if df.empty:
                return None
            
            # Add metadata columns
            df.insert(0, 'station_code', station_code)
            df.insert(1, 'sensor_code', sensor_code)
            df.insert(2, 'year', year)
            
            # Save to file
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            rate_limit_sleep()
            return df
            
        except requests.RequestException as e:
            if attempt >= retries:
                print(f"Error downloading {station_code}/{sensor_code}/{year} after {retries} attempts: {e}")
                return None
            
            print(f"  Retry {attempt}/{retries} for {station_code}/{sensor_code}/{year}: {e}")
            time.sleep(backoff)
            backoff *= 1.5
    
    return None


def collect_all_station_sensors(
    session: requests.Session,
    stations_df: pd.DataFrame,
    out_dir: Path,
    filter_sensors: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """
    Collect sensor information for all stations.
    
    Returns dictionary mapping station_code -> list of available sensor codes.
    """
    print("\nCollecting sensor information for all stations...")
    
    station_sensors_map = {}
    
    progress = tqdm(total=len(stations_df), desc="Fetching sensors") if TQDM_AVAILABLE else None
    
    for _, row in stations_df.iterrows():
        station_code = row['SCODE']
        station_name = row.get('NAME_I', station_code)
        
        if progress:
            progress.set_description(f"Station {station_code[:15]}")
        else:
            print(f"  Fetching sensors for {station_code} ({station_name})...")
        
        sensors = fetch_station_sensors(session, station_code, out_dir)
        
        if sensors:
            # Extract unique sensor codes
            sensor_codes = list(set(s.get('TYPE') for s in sensors if s.get('TYPE')))
            
            # Filter if requested
            if filter_sensors:
                sensor_codes = [s for s in sensor_codes if s in filter_sensors]
            
            if sensor_codes:
                station_sensors_map[station_code] = sensor_codes
        
        if progress:
            progress.update(1)
    
    if progress:
        progress.close()
    
    total_sensors = sum(len(sensors) for sensors in station_sensors_map.values())
    print(f"Found {len(station_sensors_map)} stations with {total_sensors} total sensors")
    
    return station_sensors_map


def init_state(
    out_dir: Path,
    station_sensors_map: Dict[str, List[str]],
    start_year: int,
    end_year: int
) -> Dict:
    """
    Initialize download state tracking.
    
    State structure:
    {
        "meta": {...},
        "tasks": [
            {"station_code": "19850PG", "sensor_code": "Q", "year": 2024, "status": "pending", ...}
        ]
    }
    """
    state = load_state(out_dir)
    
    state_meta = {
        "start_year": start_year,
        "end_year": end_year,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "version": 1
    }
    
    # If empty, create new state
    if not state or not state.get("tasks"):
        tasks = []
        
        for station_code, sensor_codes in station_sensors_map.items():
            for sensor_code in sensor_codes:
                for year in range(start_year, end_year + 1):
                    tasks.append({
                        "station_code": station_code,
                        "sensor_code": sensor_code,
                        "year": year,
                        "status": "pending",
                        "attempts": 0,
                        "updated_at": None,
                    })
        
        state = {"meta": state_meta, "tasks": tasks}
        save_state(out_dir, state)
        return state
    
    # If state exists, verify it matches
    same_job = (
        state.get("meta", {}).get("start_year") == state_meta["start_year"] and
        state.get("meta", {}).get("end_year") == state_meta["end_year"]
    )
    if not same_job:
        raise RuntimeError(f"Existing state.json in {out_dir} belongs to a different job. Choose a new --out folder.")
    
    return state


def build_output_folder(base_out: Optional[str], start_year: int, end_year: int) -> Path:
    """Build output folder path."""
    project_root = Path(__file__).parent.parent
    altoadige_data_dir = project_root / "data" / "altoadige"
    
    if base_out:
        out = altoadige_data_dir / base_out
    else:
        default_name = f"altoadige_{start_year}_{end_year}"
        out = altoadige_data_dir / default_name
    
    return out


def run_download(
    start_year: int,
    end_year: int,
    sensor_filter: Optional[List[str]],
    out_dir: Path
):
    """Main download orchestration."""
    ensure_folder(out_dir)
    
    session = create_session()
    
    # Phase 1: Fetch stations
    print("=" * 60)
    print("PHASE 1: Fetching Station Metadata")
    print("=" * 60)
    stations_df = fetch_all_stations(session, out_dir)
    
    # Phase 2: Collect sensors for all stations
    print("\n" + "=" * 60)
    print("PHASE 2: Collecting Sensor Information")
    print("=" * 60)
    station_sensors_map = collect_all_station_sensors(session, stations_df, out_dir, sensor_filter)
    
    if not station_sensors_map:
        print("No stations with sensors found!")
        return
    
    # Phase 3: Initialize state
    state = init_state(out_dir, station_sensors_map, start_year, end_year)
    
    # Phase 4: Download timeseries data
    print("\n" + "=" * 60)
    print("PHASE 3: Downloading Timeseries Data")
    print("=" * 60)
    
    pending_tasks = [t for t in state["tasks"] if t["status"] not in ("done", "no_data")]
    total_tasks = len(pending_tasks)
    
    if total_tasks == 0:
        print("All tasks already completed!")
        print_summary(state)
        return
    
    print(f"\nDownloading data for {total_tasks} station-sensor-year combinations...")
    
    progress = tqdm(pending_tasks, desc="Downloading") if TQDM_AVAILABLE else None
    
    for task in pending_tasks:
        station_code = task["station_code"]
        sensor_code = task["sensor_code"]
        year = task["year"]
        
        if progress:
            progress.set_description(f"{station_code}/{sensor_code}/{year}")
        else:
            print(f"  Downloading {station_code}/{sensor_code}/{year}...")
        
        # Download data
        df = download_timeseries_data(session, station_code, sensor_code, year, out_dir)
        
        task["attempts"] = task.get("attempts", 0) + 1
        task["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
        
        if df is not None and len(df) > 0:
            task["status"] = "done"
            task["records"] = len(df)
        else:
            # No data available for this combination (not necessarily an error)
            task["status"] = "no_data"
            task["records"] = 0
        
        # Save state after each task
        save_state(out_dir, state)
        
        if progress:
            progress.update(1)
    
    if progress:
        progress.close()
    
    # Final summary
    print_summary(state)


def print_summary(state: Dict):
    """Print download summary statistics."""
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    
    failed_tasks = [t for t in state["tasks"] if t["status"] == "failed"]
    done_tasks = [t for t in state["tasks"] if t["status"] == "done"]
    no_data_tasks = [t for t in state["tasks"] if t["status"] == "no_data"]
    
    total_records = sum(t.get("records", 0) for t in done_tasks)
    
    print(f"Successful downloads: {len(done_tasks)}")
    print(f"No data available: {len(no_data_tasks)}")
    print(f"Failed downloads: {len(failed_tasks)}")
    print(f"Total records downloaded: {total_records:,}")
    
    # Break down by sensor type
    sensor_stats = {}
    for task in done_tasks:
        sensor_code = task["sensor_code"]
        if sensor_code not in sensor_stats:
            sensor_stats[sensor_code] = {"count": 0, "records": 0}
        sensor_stats[sensor_code]["count"] += 1
        sensor_stats[sensor_code]["records"] += task.get("records", 0)
    
    if sensor_stats:
        print("\nBreakdown by sensor type:")
        for sensor_code in sorted(sensor_stats.keys()):
            stats = sensor_stats[sensor_code]
            sensor_name = ALL_SENSOR_CODES.get(sensor_code, "Unknown")
            print(f"  {sensor_code} ({sensor_name}): {stats['count']} datasets, {stats['records']:,} records")
    
    if failed_tasks:
        print("\nYou can re-run the same command to retry failed downloads.")


def main():
    parser = argparse.ArgumentParser(
        description="Bulk download Alto Adige meteorological data with resume capability.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download data for 2023-2024
  python bulk_download_altoadige.py --start 2023 --end 2024
  
  # Download specific sensors only
  python bulk_download_altoadige.py --start 2023 --end 2023 --sensors "LT,N,Q"
  
  # Custom output folder
  python bulk_download_altoadige.py --start 2020 --end 2025 --out my_data

Available sensor codes:
  LT - Temperatura dell'aria (°C)
  LF - Umidità dell'aria (%)
  N - Pioggia (mm)
  WG - Velocità media del vento (m/s)
  WG.BOE - Raffica vento (m/s)
  WR - Direzione media del vento (gradi)
  LD.RED - Pressione aria ridotta (hPa)
  SD - Tempo di soleggiamento (secondi)
  GS - Radiazione globale (W/m²)
  HS - Altezza neve (cm)
  W - Livello dell'acqua (cm)
  Q - Portata (m³/s)
        """
    )
    
    parser.add_argument("--start", type=int, required=True, help="Start year (e.g., 2020)")
    parser.add_argument("--end", type=int, required=True, help="End year (e.g., 2025)")
    parser.add_argument("--sensors", default=None,
                        help=f"Comma-separated sensor codes (default: all available sensors)")
    parser.add_argument("--out", default=None, help="Output folder name (default: altoadige_STARTYEAR_ENDYEAR)")
    
    args = parser.parse_args()
    
    # Validate years
    if args.end < args.start:
        print("Error: --end must be >= --start", file=sys.stderr)
        sys.exit(2)
    
    current_year = dt.datetime.now().year
    if args.end > current_year:
        print(f"Warning: End year {args.end} is in the future. Setting to {current_year}.")
        args.end = current_year
    
    # Parse sensors
    sensor_filter = None
    if args.sensors:
        sensor_filter = [s.strip() for s in args.sensors.split(",") if s.strip()]
        
        # Validate sensors
        invalid_sensors = [s for s in sensor_filter if s not in ALL_SENSOR_CODES]
        if invalid_sensors:
            print(f"Error: Invalid sensors: {invalid_sensors}", file=sys.stderr)
            print(f"Valid sensors: {list(ALL_SENSOR_CODES.keys())}", file=sys.stderr)
            sys.exit(2)
    
    out_dir = build_output_folder(args.out, args.start, args.end)
    
    print("Alto Adige Bulk Downloader")
    print("=" * 60)
    print(f"Output folder: {out_dir}")
    print(f"Year range: {args.start} - {args.end}")
    print(f"Sensors: {', '.join(sensor_filter) if sensor_filter else 'all available'}")
    print("=" * 60)
    
    run_download(args.start, args.end, sensor_filter, out_dir)


if __name__ == "__main__":
    main()

