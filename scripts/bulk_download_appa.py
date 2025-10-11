#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
APPA Aria bulk downloader (resumable, chunked by <=90 days)

Usage examples:
    python appa_bulk_downloader.py --start 2025-08-01 --end 2025-10-01
    python appa_bulk_downloader.py --start 2024-01-01 --end 2025-09-30 --format csv \
        --stations "2,4,6,8,9,15,22,23" --out ./downloads

Station/pollutant syntax (optional):
    STATIONS is a comma list like: 2,4,6,8
    To restrict pollutants per station: 2[48,53],4,6
    (IDs are documented on the APPA page.)

Endpoint doc:
    https://bollettino.appa.tn.it/aria/opendata/{FORMAT}/{DATE}/{STATIONS}
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd

try:
    import requests
except ImportError:
    print("This script requires the 'requests' package. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)

# Optional nice progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False


# ---- Config ----
OPEN_DATA_BASE = "https://bollettino.appa.tn.it/aria/opendata"
DEFAULT_FORMAT = "csv"
# From the APPA page (all stations)
DEFAULT_STATIONS = "2,4,6,8,9,15,22,23"
MAX_DAYS_PER_CHUNK = 90  # APPA allows "max 90 giorni" for date intervals
CHUNK_FILENAME_PATTERN = "{fmt}_{start}_{end}.data"  # extension decided by fmt
STATE_FILENAME = "state.json"
USER_AGENT = "appa-bulk-downloader/1.0 (+https://bollettino.appa.tn.it/)"


def parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def date_chunks(start: dt.date, end: dt.date, max_days: int) -> List[Tuple[dt.date, dt.date]]:
    """
    Inclusive chunks: [start, chunk_end], each covering <= max_days days.
    """
    if end < start:
        raise ValueError("end date must be >= start date")
    chunks = []
    cur = start
    while cur <= end:
        # chunk_end inclusive; ensure total days <= max_days
        chunk_end = min(end, cur + dt.timedelta(days=max_days - 1))
        chunks.append((cur, chunk_end))
        cur = chunk_end + dt.timedelta(days=1)
    return chunks


def ensure_folder(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def load_state(out_dir: Path) -> Dict:
    state_path = out_dir / STATE_FILENAME
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(out_dir: Path, state: Dict):
    state_path = out_dir / STATE_FILENAME
    tmp = state_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True, ensure_ascii=False)
    tmp.replace(state_path)


def guess_extension(fmt: str) -> str:
    fmt = fmt.lower()
    if fmt in ("csv", "json", "xml"):
        return "." + fmt
    return ".dat"


def build_url(fmt: str, start: dt.date, end: dt.date, stations: Optional[str]) -> str:
    # DATE segment supports YYYY-MM-DD,YYYY-MM-DD
    date_seg = f"{start.isoformat()},{end.isoformat()}"
    if stations:
        return f"{OPEN_DATA_BASE}/{fmt}/{date_seg}/{stations}"
    else:
        # STATIONS is optional per docs; omit the trailing segment
        return f"{OPEN_DATA_BASE}/{fmt}/{date_seg}"


def sanitize_filename(s: str) -> str:
    # Keep it simple; only allow basic chars
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", s)


def build_output_folder(base_out: Optional[str], start: dt.date, end: dt.date, fmt: str, stations: str) -> Path:
    # Always save in data/appa-data folder in project root
    project_root = Path(__file__).parent.parent.parent
    appa_data_dir = project_root / "data" / "appa-data"
    
    if base_out:
        # If custom output specified, use it but still under data/appa-data
        out = appa_data_dir / base_out
    else:
        # Default naming under data/appa-data
        default_name = f"appa-aria_{start.isoformat()}_{end.isoformat()}_{fmt.lower()}"
        out = appa_data_dir / default_name
    
    return out


def init_state(out_dir: Path, fmt: str, start: dt.date, end: dt.date, stations: str, chunks: List[Tuple[dt.date, dt.date]]) -> Dict:
    """
    Initialize the state manifest if not present. If present, reconcile with files and chunks.
    """
    state = load_state(out_dir)
    meta = {
        "format": fmt.lower(),
        "global_start": start.isoformat(),
        "global_end": end.isoformat(),
        "stations": stations,
        "max_days_per_chunk": MAX_DAYS_PER_CHUNK,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "version": 1
    }
    # If empty, create new state
    if not state:
        entries = []
        for s, e in chunks:
            entries.append({
                "start": s.isoformat(),
                "end": e.isoformat(),
                "filename": CHUNK_FILENAME_PATTERN.format(
                    fmt=fmt.lower(),
                    start=s.isoformat(),
                    end=e.isoformat()
                ) + guess_extension(fmt),
                "status": "pending",
                "attempts": 0,
                "updated_at": None,
            })
        state = {"meta": meta, "chunks": entries}
        save_state(out_dir, state)
        return state

    # If state exists, ensure it matches current request; otherwise, error out to avoid mixing jobs
    same_job = (
        state.get("meta", {}).get("format") == meta["format"] and
        state.get("meta", {}).get("global_start") == meta["global_start"] and
        state.get("meta", {}).get("global_end") == meta["global_end"] and
        state.get("meta", {}).get("stations") == meta["stations"]
    )
    if not same_job:
        raise RuntimeError(f"Existing state.json in {out_dir} belongs to a different job. Choose a new --out folder.")

    # Reconcile file existence (mark done if file present)
    for entry in state.get("chunks", []):
        file_path = out_dir / entry["filename"]
        part_path = file_path.with_suffix(file_path.suffix + ".part")
        if file_path.exists():
            entry["status"] = "done"
        elif part_path.exists():
            # Keep as pending/failed; we'll retry
            if entry["status"] == "done":
                entry["status"] = "pending"
        else:
            # leave as-is
            pass

    save_state(out_dir, state)
    return state


def human_sleep(seconds: float):
    # Small helper to avoid busy loops
    time.sleep(seconds)


def stream_download(session: requests.Session, url: str, dst: Path, total_pbar=None, perfile_pbar=None, retries: int = 5) -> bool:
    """
    Stream download to dst (write to dst.part then rename). Returns True on success.
    Retries with exponential backoff on transient errors.
    """
    tmp = dst.with_suffix(dst.suffix + ".part")
    backoff = 2.0

    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=(10, 120)) as r:
                r.raise_for_status()

                # Content length for progress, if available
                total_bytes = int(r.headers.get("Content-Length", "0")) or None
                # Prepare write
                with tmp.open("wb") as f:
                    if TQDM_AVAILABLE and perfile_pbar is not None:
                        if total_bytes:
                            perfile_pbar.reset(total=total_bytes)
                        else:
                            perfile_pbar.reset(total=None)  # indeterminate
                        perfile_pbar.set_description("Chunk")
                    # iterate chunks
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        if not chunk:
                            continue
                        f.write(chunk)
                        if perfile_pbar is not None:
                            # If Content-Length known, update by bytes; else show spinner steps
                            perfile_pbar.update(len(chunk) if total_bytes else 1)
                tmp.replace(dst)
                return True

        except requests.RequestException as ex:
            if attempt >= retries:
                return False
            # backoff and retry
            if total_pbar is not None:
                total_pbar.write(f"Request failed (attempt {attempt}/{retries}) for {dst.name}: {ex}. Retrying...")
            human_sleep(backoff)
            backoff *= 1.8
        except Exception as ex:
            if attempt >= retries:
                return False
            if total_pbar is not None:
                total_pbar.write(f"Error (attempt {attempt}/{retries}) for {dst.name}: {ex}. Retrying...")
            human_sleep(backoff)
            backoff *= 1.8

    return False


def merge_csv_files(out_dir: Path, merged_filename: str = "merged_data.csv") -> bool:
    """
    Merge all CSV files in the output directory into a single CSV file.
    Returns True if successful, False otherwise.
    """
    try:
        csv_files = list(out_dir.glob("*.csv"))
        if not csv_files:
            print("No CSV files found to merge.")
            return False
        
        print(f"Merging {len(csv_files)} CSV files...")
        
        # Read and concatenate all CSV files
        dataframes = []
        for csv_file in csv_files:
            try:
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        df = pd.read_csv(csv_file, encoding=encoding)
                        dataframes.append(df)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    print(f"Warning: Could not read {csv_file} with any encoding")
                    continue
            except Exception as e:
                print(f"Warning: Could not read {csv_file}: {e}")
                continue
        
        if not dataframes:
            print("No valid CSV files could be read.")
            return False
        
        # Concatenate all dataframes
        merged_df = pd.concat(dataframes, ignore_index=True)
        
        # Save merged file
        merged_path = out_dir / merged_filename
        merged_df.to_csv(merged_path, index=False)
        
        print(f"Merged CSV saved as: {merged_path}")
        print(f"Total records: {len(merged_df)}")
        return True
        
    except Exception as e:
        print(f"Error merging CSV files: {e}")
        return False


def run_download(fmt: str, start: dt.date, end: dt.date, stations: str, out_dir: Path):
    ensure_folder(out_dir)
    chunks = date_chunks(start, end, MAX_DAYS_PER_CHUNK)

    state = init_state(out_dir, fmt, start, end, stations, chunks)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})

    # Prepare progress bars
    total_tasks = sum(1 for c in state["chunks"] if c["status"] != "done")
    total_bar = None
    perfile_bar = None

    try:
        if TQDM_AVAILABLE:
            total_bar = tqdm(total=total_tasks, desc="Chunks", unit="chunk")
            perfile_bar = tqdm(total=None, desc="Chunk", unit="B", leave=False)
        else:
            print(f"Downloading {total_tasks} chunk(s)...")

        for entry in state["chunks"]:
            if entry["status"] == "done":
                continue

            s = dt.date.fromisoformat(entry["start"])
            e = dt.date.fromisoformat(entry["end"])
            url = build_url(fmt, s, e, stations)
            filename = sanitize_filename(entry["filename"])
            dst = out_dir / filename

            # Reset per-file bar
            if TQDM_AVAILABLE:
                perfile_bar.reset()
                perfile_bar.set_description(f"{s} → {e}")

            ok = stream_download(session, url, dst, total_bar, perfile_bar, retries=5)

            # Update state
            entry["attempts"] = entry.get("attempts", 0) + 1
            entry["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
            if ok and dst.exists():
                entry["status"] = "done"
                if total_bar:
                    total_bar.update(1)
            else:
                entry["status"] = "failed"

            save_state(out_dir, state)

        # Done summary
        failed = [c for c in state["chunks"] if c["status"] != "done"]
        if failed:
            msg = f"Completed with {len(failed)} failed chunk(s). You can re-run the same command to retry."
        else:
            msg = "All chunks downloaded successfully."
            # Merge CSV files if format is CSV and all downloads succeeded
            if fmt.lower() == "csv":
                print("\nMerging CSV files...")
                merge_success = merge_csv_files(out_dir)
                if merge_success:
                    msg += " CSV files merged successfully."
                else:
                    msg += " Warning: CSV merge failed."
        
        if total_bar:
            total_bar.write(msg)
        else:
            print(msg)

    finally:
        if TQDM_AVAILABLE:
            if perfile_bar is not None:
                perfile_bar.close()
            if total_bar is not None:
                total_bar.close()


def main():
    parser = argparse.ArgumentParser(description="Bulk download APPA Aria data with automatic 90-day chunking and resume.")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--format", default=DEFAULT_FORMAT, choices=["csv", "json", "xml"], help="Output format")
    parser.add_argument("--stations", default=DEFAULT_STATIONS,
                        help=("Stations list (e.g., '2,4,6' or with pollutants '2[48,53],4'). "
                              "Default: all stations: " + DEFAULT_STATIONS))
    parser.add_argument("--out", default=None, help="Output folder (default: appa-aria_{start}_{end}_{format})")

    args = parser.parse_args()

    start = parse_date(args.start)
    end = parse_date(args.end)
    if (end - start).days < 0:
        print("Error: --end must be on/after --start", file=sys.stderr)
        sys.exit(2)

    out_dir = build_output_folder(args.out, start, end, args.format, args.stations)

    print(f"Output folder: {out_dir}")
    print(f"Range: {start} → {end} (will split into <= {MAX_DAYS_PER_CHUNK}-day chunks)")
    print(f"Format: {args.format}")
    print(f"Stations: {args.stations}")

    run_download(args.format, start, end, args.stations, out_dir)


if __name__ == "__main__":
    main()