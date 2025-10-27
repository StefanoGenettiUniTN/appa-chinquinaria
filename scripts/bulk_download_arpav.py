#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ARPAV (Veneto Region) Meteorological Data Bulk Downloader

Downloads weather data from ARPAV monitoring stations for specified year ranges.
Data is organized by year with all stations and sensors merged into single CSV files.

Usage examples:
    python bulk_download_arpav.py --start-year 2023 --end-year 2024
    python bulk_download_arpav.py --start-year 2020 --end-year 2023 --out arpav_2020_2023
    python bulk_download_arpav.py --start-year 2023 --end-year 2023 --sensors "TEMPMIN,PREC"

Endpoint documentation:
    - Station metadata: https://www.ambienteveneto.it/datiorari/getXmlStazioniOrari.php
    - Station data: https://www.ambienteveneto.it/datiorari/datiSensOrari.php
"""

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from urllib.parse import parse_qsl

try:
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    # Disable SSL warnings when verification is disabled
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    print("This script requires 'requests', 'beautifulsoup4', and 'pandas' packages.", file=sys.stderr)
    print("Install with: pip install requests beautifulsoup4 pandas", file=sys.stderr)
    sys.exit(1)

# Optional progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False


# ---- Configuration ----
METADATA_ENDPOINT = "https://www.ambienteveneto.it/datiorari/getXmlStazioniOrari.php"
DATA_ENDPOINT = "https://www.ambienteveneto.it/datiorari/datiSensOrari.php"

# All available sensors we care about
ALL_SENSORS = ["TEMPMIN", "PREC", "UMID", "RADSOL", "VVENTOMEDIO", "LIVIDRO", "PORT", "PRESSMARE"]

STATE_FILENAME = "state.json"
METADATA_FILENAME = "metadata.json"
USER_AGENT = "arpav-bulk-downloader/1.0"

# Request timeouts
HTTP_CONNECT_TIMEOUT = 20
HTTP_READ_TIMEOUT = 120

# Retry configuration
MAX_RETRIES = 5
RETRY_BACKOFF = 2.0


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


def load_metadata(out_dir: Path) -> Optional[Dict]:
    """Load metadata from JSON file."""
    metadata_path = out_dir / METADATA_FILENAME
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_metadata(out_dir: Path, metadata: Dict):
    """Save metadata to JSON file atomically."""
    metadata_path = out_dir / METADATA_FILENAME
    tmp = metadata_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True, ensure_ascii=False)
    tmp.replace(metadata_path)


def create_session(verify_ssl: bool = True) -> requests.Session:
    """Create and configure requests session."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.ambienteveneto.it/datiorari/",
    })
    session.verify = verify_ssl
    return session


def fetch_stations_for_year_sensor(session: requests.Session, year: int, sensor: str) -> List[Dict]:
    """
    Fetch station metadata for a specific year and sensor type.
    
    Returns list of stations with their metadata and sensor IDs.
    """
    params = {
        "anno": year,
        "sensore": sensor,
        "_": int(time.time() * 1000)  # Timestamp parameter
    }
    
    headers = {
        "X-Requested-With": "XMLHttpRequest",
    }
    
    try:
        response = session.get(
            METADATA_ENDPOINT,
            params=params,
            headers=headers,
            timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT)
        )
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        
        stations = []
        for stazione in root.findall('.//STAZIONE'):
            station_id_elem = stazione.find('IDSTAZ')
            nome_elem = stazione.find('NOME')
            x_elem = stazione.find('X')
            y_elem = stazione.find('Y')
            quota_elem = stazione.find('QUOTA')
            provincia_elem = stazione.find('PROVINCIA')
            comune_elem = stazione.find('COMUNE')
            
            if station_id_elem is None:
                continue
                
            station_id = station_id_elem.text
            
            # Extract sensor information
            sensors = []
            for sensore in stazione.findall('.//SENSORE'):
                sensor_id_elem = sensore.find('ID')
                param_nm_elem = sensore.find('PARAMNM')
                type_elem = sensore.find('TYPE')
                unit_nm_elem = sensore.find('UNITNM')
                
                if sensor_id_elem is not None:
                    sensors.append({
                        "id": sensor_id_elem.text,
                        "param_name": param_nm_elem.text if param_nm_elem is not None else "",
                        "type": type_elem.text if type_elem is not None else "",
                        "unit": unit_nm_elem.text if unit_nm_elem is not None else "",
                    })
            
            station_info = {
                "station_id": station_id,
                "name": nome_elem.text if nome_elem is not None else "",
                "x": x_elem.text if x_elem is not None else "",
                "y": y_elem.text if y_elem is not None else "",
                "quota": quota_elem.text if quota_elem is not None else "",
                "provincia": provincia_elem.text if provincia_elem is not None else "",
                "comune": comune_elem.text if comune_elem is not None else "",
                "sensor_type": sensor,
                "sensors": sensors
            }
            
            stations.append(station_info)
        
        return stations
        
    except Exception as e:
        print(f"Warning: Failed to fetch stations for year {year}, sensor {sensor}: {e}")
        return []


def collect_all_station_metadata(session: requests.Session, start_year: int, end_year: int, sensors: List[str]) -> Dict:
    """
    Collect comprehensive station metadata for all years and sensors.
    
    Returns structure:
    {
        "years": {
            "2023": {
                "stations": {
                    "123": {
                        "name": "Station Name",
                        "x": "12.03",
                        "y": "46.27",
                        ...
                        "sensors": ["1786", "1787"]
                    }
                }
            }
        }
    }
    """
    print(f"\nCollecting station metadata for years {start_year}-{end_year}...")
    
    metadata = {
        "collected_at": dt.datetime.now().isoformat(timespec="seconds"),
        "start_year": start_year,
        "end_year": end_year,
        "sensor_types": sensors,
        "years": {}
    }
    
    total_requests = (end_year - start_year + 1) * len(sensors)
    progress = tqdm(total=total_requests, desc="Fetching metadata") if TQDM_AVAILABLE else None
    
    for year in range(start_year, end_year + 1):
        year_data = {"stations": {}}
        
        for sensor in sensors:
            if progress:
                progress.set_description(f"Fetching {year}/{sensor}")
            else:
                print(f"  Fetching {year}/{sensor}...")
            
            stations = fetch_stations_for_year_sensor(session, year, sensor)
            
            # Merge station data (station may appear in multiple sensor queries)
            for station_info in stations:
                station_id = station_info["station_id"]
                
                if station_id not in year_data["stations"]:
                    year_data["stations"][station_id] = {
                        "name": station_info["name"],
                        "x": station_info["x"],
                        "y": station_info["y"],
                        "quota": station_info["quota"],
                        "provincia": station_info["provincia"],
                        "comune": station_info["comune"],
                        "sensor_ids": []
                    }
                
                # Add sensor IDs
                for sensor in station_info["sensors"]:
                    sensor_id = sensor["id"]
                    if sensor_id not in year_data["stations"][station_id]["sensor_ids"]:
                        year_data["stations"][station_id]["sensor_ids"].append(sensor_id)
            
            if progress:
                progress.update(1)
            
            # Be polite to the server
            time.sleep(0.5)
        
        metadata["years"][str(year)] = year_data
        
    if progress:
        progress.close()
    
    # Summary
    total_stations = sum(len(metadata["years"][str(y)]["stations"]) for y in range(start_year, end_year + 1))
    print(f"Collected metadata for {total_stations} station-year combinations")
    
    return metadata


def download_sensor_year_html(
    session: requests.Session,
    sensor_id: str,
    year: int,
    retries: int = MAX_RETRIES,
    extra_params: Optional[Dict[str, str]] = None,
    debug: bool = False,
    debug_dir: Optional[Path] = None,
    can_save_html: bool = False,
    # POST-after-GET settings
    post_after_get: bool = True,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    connect_timeout: Optional[int] = None,
    read_timeout: Optional[int] = None,
) -> Optional[str]:
    """
    Download HTML data for a specific sensor ID and year.
    
    Args:
        sensor_id: The sensor ID (from SENSORE/ID in XML), not station ID
        year: Year to download
    
    Returns HTML content or None on failure.
    """
    params = {
        "cd": sensor_id,
        "an": year
    }
    if extra_params:
        # Do not override required keys if present in extra params
        for k, v in extra_params.items():
            if k not in ("cd", "an"):
                params[k] = v
    
    backoff = RETRY_BACKOFF
    ct = HTTP_CONNECT_TIMEOUT if connect_timeout is None else connect_timeout
    rt = HTTP_READ_TIMEOUT if read_timeout is None else read_timeout
    
    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                DATA_ENDPOINT,
                params=params,
                timeout=(ct, rt)
            )
            response.raise_for_status()
            if debug:
                url_used = response.url
                size = len(response.text)
                print(f"    GET {url_used} => {size} chars")
            if can_save_html and debug_dir is not None:
                try:
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    fname = debug_dir / f"sensor_{sensor_id}_{year}_get.html"
                    with fname.open("w", encoding="utf-8") as f:
                        f.write(response.text)
                except Exception as e:
                    print(f"    Warning: failed saving raw HTML for {sensor_id}/{year}: {e}")
            # If needed, follow with POST to set date range and retrieve full period
            if post_after_get:
                post_data = {}
                if date_start:
                    post_data["dinizio"] = date_start
                if date_end:
                    post_data["dfine"] = date_end
                try:
                    response_post = session.post(
                        DATA_ENDPOINT,
                        params=params,
                        data=post_data,
                        timeout=(ct, rt)
                    )
                    response_post.raise_for_status()
                    if debug:
                        url_used = response_post.url
                        size = len(response_post.text)
                        print(f"    POST {url_used} body={post_data} => {size} chars")
                    if can_save_html and debug_dir is not None:
                        try:
                            fname = debug_dir / f"sensor_{sensor_id}_{year}_post.html"
                            with fname.open("w", encoding="utf-8") as f:
                                f.write(response_post.text)
                        except Exception as e:
                            print(f"    Warning: failed saving POST HTML for {sensor_id}/{year}: {e}")
                    return response_post.text
                except requests.RequestException as e:
                    if attempt >= retries:
                        print(f"Error POSTing sensor {sensor_id}/{year} after GET: {e}")
                        return None
                    print(f"  Retry POST {attempt}/{retries} for sensor {sensor_id}/{year}: {e}")
                    time.sleep(backoff)
                    backoff *= 1.5
                    continue
            else:
                return response.text
            
        except requests.RequestException as e:
            if attempt >= retries:
                print(f"Error downloading sensor {sensor_id}/{year} after {retries} attempts: {e}")
                return None
            
            print(f"  Retry {attempt}/{retries} for sensor {sensor_id}/{year}: {e}")
            time.sleep(backoff)
            backoff *= 1.5
    
    return None


def parse_html_table(html_content: str, station_id: str, year: int) -> Optional[pd.DataFrame]:
    """
    Parse HTML table containing station data.
    
    Returns DataFrame with columns: timestamp, station_id, year, and measurement columns.
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find the data table
        table = soup.find('table', class_='table table-bordered table-hover')
        if not table:
            print(f"Warning: No data table found for station {station_id}/{year}")
            return None
        
        rows = table.find_all('tr')
        if len(rows) < 3:  # Need at least header rows + 1 data row
            print(f"Warning: Insufficient data rows for station {station_id}/{year}")
            return None
        
        # Parse headers (can be multi-row)
        # First row has main column headers, second row may have sub-headers
        header_row_1 = rows[0].find_all(['th', 'td'])
        header_row_2 = rows[1].find_all(['th', 'td']) if len(rows) > 1 else []
        
        # Build column names by combining headers
        columns = []
        col_idx = 0
        
        for i, cell in enumerate(header_row_1):
            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))
            
            cell_text = cell.get_text(strip=True).replace('\n', ' ').replace('  ', ' ')
            
            if rowspan == 2 or not header_row_2:
                # Single column or spans both rows
                columns.append(cell_text)
            else:
                # Has sub-columns in row 2
                for j in range(colspan):
                    if col_idx < len(header_row_2):
                        sub_text = header_row_2[col_idx].get_text(strip=True)
                        columns.append(f"{cell_text}_{sub_text}" if cell_text else sub_text)
                        col_idx += 1
                    else:
                        columns.append(cell_text)
        
        # Ensure first column is "timestamp" or similar
        if columns:
            columns[0] = "timestamp"
        
        # Parse data rows (start from row 2, after headers)
        data_rows = []
        start_row = 2 if len(rows) > 2 else 1
        
        for row in rows[start_row:]:
            cells = row.find_all(['th', 'td'])
            if not cells:
                continue
            
            row_data = [cell.get_text(strip=True) for cell in cells]
            
            # Skip empty rows
            if all(not val or val == '' for val in row_data):
                continue
            
            data_rows.append(row_data)
        
        if not data_rows:
            print(f"Warning: No data rows found for station {station_id}/{year}")
            return None
        
        # Create DataFrame
        # Handle case where columns and data might not align perfectly
        max_cols = max(len(columns), max(len(row) for row in data_rows))
        
        while len(columns) < max_cols:
            columns.append(f"col_{len(columns)}")
        
        # Pad rows if needed
        for row in data_rows:
            while len(row) < len(columns):
                row.append('')
        
        df = pd.DataFrame(data_rows, columns=columns[:len(data_rows[0])])
        
        # Add metadata columns
        df.insert(0, 'station_id', station_id)
        df.insert(1, 'year', year)
        
        # Try to parse timestamp column into proper datetime
        if 'timestamp' in df.columns:
            try:
                # Format is typically "dd/mm/yyyy HH"
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H', errors='coerce')
            except:
                # Keep as string if parsing fails
                pass
        
        return df
        
    except Exception as e:
        print(f"Error parsing HTML for station {station_id}/{year}: {e}")
        return None


def init_state(out_dir: Path, metadata: Dict, start_year: int, end_year: int) -> Dict:
    """
    Initialize download state tracking.
    
    State structure:
    {
        "meta": {...},
        "tasks": [
            {"station_id": "123", "sensor_id": "1786", "year": 2023, "status": "pending", ...}
        ]
    }
    
    Note: We track sensor IDs since that's what the data endpoint requires.
    """
    state = load_state(out_dir)
    
    state_meta = {
        "start_year": start_year,
        "end_year": end_year,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "version": 1
    }
    
    # If empty, create new state
    if not state:
        tasks = []
        
        for year in range(start_year, end_year + 1):
            year_str = str(year)
            if year_str not in metadata.get("years", {}):
                continue
            
            stations = metadata["years"][year_str]["stations"]
            
            for station_id, station_data in stations.items():
                # Create a task for each sensor ID
                for sensor_id in station_data.get("sensor_ids", []):
                    tasks.append({
                        "station_id": station_id,
                        "station_name": station_data.get("name", ""),
                        "sensor_id": sensor_id,
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


def merge_station_data_to_yearly_csv(out_dir: Path, year: int, dataframes: List[pd.DataFrame]) -> Optional[Path]:
    """
    Merge all station DataFrames for a specific year into a single CSV.
    
    Returns path to merged CSV file.
    """
    if not dataframes:
        print(f"Warning: No dataframes to merge for year {year}")
        return None
    
    try:
        print(f"\nMerging {len(dataframes)} stations for year {year}...")
        
        # Concatenate all dataframes
        merged_df = pd.concat(dataframes, ignore_index=True)
        
        # Sort by timestamp and station_id
        if 'timestamp' in merged_df.columns:
            merged_df = merged_df.sort_values(['timestamp', 'station_id'])
        else:
            merged_df = merged_df.sort_values(['station_id'])
        
        # Save to CSV
        csv_path = out_dir / f"arpav_{year}.csv"
        merged_df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"Saved {csv_path} ({len(merged_df)} records)")
        return csv_path
        
    except Exception as e:
        print(f"Error merging data for year {year}: {e}")
        return None


def build_output_folder(base_out: Optional[str], start_year: int, end_year: int) -> Path:
    """Build output folder path."""
    project_root = Path(__file__).parent.parent
    arpav_data_dir = project_root / "data" / "arpav"
    
    if base_out:
        out = arpav_data_dir / base_out
    else:
        default_name = f"arpav_{start_year}_{end_year}"
        out = arpav_data_dir / default_name
    
    return out


def run_download(
    start_year: int,
    end_year: int,
    sensors: List[str],
    out_dir: Path,
    verify_ssl: bool = False,
    *,
    extra_params: Optional[Dict[str, str]] = None,
    debug: bool = False,
    save_html: bool = False,
    save_html_limit: int = 0,
    limit_per_year: Optional[int] = None,
    post_range: bool = True,
    date_start_override: Optional[str] = None,
    date_end_override: Optional[str] = None,
    connect_timeout: Optional[int] = None,
    read_timeout: Optional[int] = None,
):
    """Main download orchestration."""
    ensure_folder(out_dir)
    
    session = create_session(verify_ssl=verify_ssl)
    
    # Phase 1: Collect metadata (or load if exists)
    metadata = load_metadata(out_dir)
    
    if metadata is None:
        print("=" * 60)
        print("PHASE 1: Collecting Station Metadata")
        print("=" * 60)
        metadata = collect_all_station_metadata(session, start_year, end_year, sensors)
        save_metadata(out_dir, metadata)
    else:
        print("Using existing metadata file")
    
    # Phase 2: Initialize state
    state = init_state(out_dir, metadata, start_year, end_year)
    
    # Phase 3: Download station data
    print("\n" + "=" * 60)
    print("PHASE 2: Downloading Station Data")
    print("=" * 60)
    if debug:
        print(f"Debug mode: on | save_html={save_html} limit={save_html_limit}")
        if extra_params:
            print(f"Extra query params: {extra_params}")
        print(f"POST range enabled: {post_range}")
    
    pending_tasks = [t for t in state["tasks"] if t["status"] != "done"]
    total_tasks = len(pending_tasks)
    
    if total_tasks == 0:
        print("All tasks already completed!")
        return
    
    print(f"\nDownloading data for {total_tasks} station-year combinations...")
    
    # Group tasks by year for efficient merging
    tasks_by_year = {}
    for task in state["tasks"]:
        year = task["year"]
        if year not in tasks_by_year:
            tasks_by_year[year] = []
        tasks_by_year[year].append(task)
    
    # Process each year
    for year in sorted(tasks_by_year.keys()):
        year_tasks = tasks_by_year[year]
        pending_year_tasks = [t for t in year_tasks if t["status"] != "done"]
        if limit_per_year is not None:
            pending_year_tasks = pending_year_tasks[:limit_per_year]
        
        if not pending_year_tasks:
            print(f"\nYear {year}: Already complete")
            continue
        
        print(f"\nYear {year}: {len(pending_year_tasks)} stations to download")
        
        year_dataframes = []
        saved_html_count = 0
        debug_dir = (out_dir / "debug_html" / str(year)) if (debug or save_html) else None
        # Build default date range for the year (dd/mm/yyyy)
        default_start = f"01/01/{year:04d}"
        default_end = f"31/12/{year:04d}"
        year_date_start = date_start_override or default_start
        year_date_end = date_end_override or default_end
        
        progress = tqdm(pending_year_tasks, desc=f"Year {year}") if TQDM_AVAILABLE else None
        
        for task in pending_year_tasks:
            station_id = task["station_id"]
            sensor_id = task["sensor_id"]
            station_name = task.get("station_name", station_id)
            
            if progress:
                progress.set_description(f"{station_name[:20]} (sensor {sensor_id})")
            else:
                print(f"  Downloading {station_name} (sensor {sensor_id})...")
            
            # Download HTML using sensor ID
            can_save_html = save_html and (saved_html_count < save_html_limit if save_html_limit else True)
            html_content = download_sensor_year_html(
                session,
                sensor_id,
                year,
                extra_params=extra_params,
                debug=debug,
                debug_dir=debug_dir,
                can_save_html=can_save_html,
                post_after_get=post_range,
                date_start=year_date_start,
                date_end=year_date_end,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )
            
            task["attempts"] = task.get("attempts", 0) + 1
            task["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
            
            if html_content:
                # Parse HTML to DataFrame
                df = parse_html_table(html_content, station_id, year)
                
                if df is not None and len(df) > 0:
                    # Add sensor ID to the dataframe for tracking
                    df.insert(2, 'sensor_id', sensor_id)
                    year_dataframes.append(df)
                    if can_save_html:
                        saved_html_count += 1
                    if debug and 'timestamp' in df.columns:
                        try:
                            ts = df['timestamp']
                            min_ts = ts.min()
                            max_ts = ts.max()
                            non_null = ts.notna().sum()
                            print(f"    Coverage {station_name} ({sensor_id}) {year}: {non_null} rows, min={min_ts}, max={max_ts}")
                        except Exception:
                            pass
                    task["status"] = "done"
                else:
                    task["status"] = "failed"
                    print(f"  Warning: Failed to parse data for {station_name} (sensor {sensor_id})/{year}")
            else:
                task["status"] = "failed"
            
            # Save state after each task
            save_state(out_dir, state)
            
            if progress:
                progress.update(1)
            
            # Be polite to the server
            time.sleep(0.5)
        
        if progress:
            progress.close()
        
        # Merge year data into CSV
        if year_dataframes:
            merge_station_data_to_yearly_csv(out_dir, year, year_dataframes)
            if debug:
                try:
                    import pandas as pd  # already imported above
                    merged = pd.concat(year_dataframes, ignore_index=True)
                    if 'timestamp' in merged.columns:
                        ts = merged['timestamp']
                        print(f"  Year {year} merged coverage: rows={len(merged)}, min={ts.min()}, max={ts.max()}")
                except Exception:
                    pass
    
    # Final summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    
    failed_tasks = [t for t in state["tasks"] if t["status"] == "failed"]
    done_tasks = [t for t in state["tasks"] if t["status"] == "done"]
    
    print(f"Successful: {len(done_tasks)}")
    print(f"Failed: {len(failed_tasks)}")
    
    if failed_tasks:
        print("\nYou can re-run the same command to retry failed downloads.")


def main():
    parser = argparse.ArgumentParser(
        description="Bulk download ARPAV meteorological data with resume capability.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download data for 2023-2024
  python bulk_download_arpav.py --start-year 2023 --end-year 2024
  
  # Download specific sensors only
  python bulk_download_arpav.py --start-year 2023 --end-year 2023 --sensors "TEMPMIN,PREC"
  
  # Custom output folder
  python bulk_download_arpav.py --start-year 2020 --end-year 2023 --out my_data
        """
    )
    
    parser.add_argument("--start-year", type=int, required=True, help="Start year (e.g., 2020)")
    parser.add_argument("--end-year", type=int, required=True, help="End year (e.g., 2023)")
    parser.add_argument("--sensors", default=",".join(ALL_SENSORS),
                        help=f"Comma-separated sensor types (default: all sensors)")
    parser.add_argument("--out", default=None, help="Output folder name (default: arpav_STARTYEAR_ENDYEAR)")
    parser.add_argument("--verify-ssl", action="store_true", help="Verify SSL certificates (default: disabled)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging and coverage output")
    parser.add_argument("--save-html", action="store_true", help="Save raw HTML responses to disk (implies --debug)")
    parser.add_argument("--save-html-limit", type=int, default=50, help="Max number of raw HTML files to save per year (0 = no limit)")
    parser.add_argument("--extra-params", default="", help="Extra query params for data endpoint, e.g. 'pg=1&foo=bar'")
    parser.add_argument("--limit-per-year", type=int, default=None, help="Limit number of station sensors processed per year (for debugging)")
    parser.add_argument("--no-post-range", dest="post_range", action="store_false", help="Disable POST-after-GET date range request")
    parser.set_defaults(post_range=True)
    parser.add_argument("--date-start", default=None, help="Override start date in dd/mm/yyyy (default 01/01/<year>)")
    parser.add_argument("--date-end", default=None, help="Override end date in dd/mm/yyyy (default 31/12/<year>)")
    parser.add_argument("--connect-timeout", type=int, default=None, help=f"HTTP connect timeout seconds (default {HTTP_CONNECT_TIMEOUT})")
    parser.add_argument("--read-timeout", type=int, default=None, help=f"HTTP read timeout seconds (default {HTTP_READ_TIMEOUT})")
    
    args = parser.parse_args()
    
    # Validate years
    if args.end_year < args.start_year:
        print("Error: --end-year must be >= --start-year", file=sys.stderr)
        sys.exit(2)
    
    # Parse sensors
    sensors = [s.strip() for s in args.sensors.split(",") if s.strip()]
    if not sensors:
        sensors = ALL_SENSORS
    
    # Validate sensors
    invalid_sensors = [s for s in sensors if s not in ALL_SENSORS]
    if invalid_sensors:
        print(f"Error: Invalid sensors: {invalid_sensors}", file=sys.stderr)
        print(f"Valid sensors: {ALL_SENSORS}", file=sys.stderr)
        sys.exit(2)
    
    out_dir = build_output_folder(args.out, args.start_year, args.end_year)
    
    print("ARPAV Bulk Downloader")
    print("=" * 60)
    print(f"Output folder: {out_dir}")
    print(f"Year range: {args.start_year} - {args.end_year}")
    print(f"Sensors: {', '.join(sensors)}")
    print(f"SSL Verification: {'enabled' if args.verify_ssl else 'disabled'}")
    print("=" * 60)
    
    extra_params = dict(parse_qsl(args.extra_params)) if args.extra_params else None
    run_download(
        args.start_year,
        args.end_year,
        sensors,
        out_dir,
        verify_ssl=args.verify_ssl,
        extra_params=extra_params,
        debug=(args.debug or args.save_html),
        save_html=args.save_html,
        save_html_limit=args.save_html_limit,
        limit_per_year=args.limit_per_year,
        post_range=args.post_range,
        date_start_override=args.date_start,
        date_end_override=args.date_end,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
    )


if __name__ == "__main__":
    main()

