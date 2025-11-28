#!/usr/bin/env python3
"""
Merge ARPAV PM10 hourly Excel files into a single CSV.

Input:
  - Directory containing .xlsx files like:
      H_YYYY_<station_code>_<station_name>.xlsx
    Example:
      H_2010_502604_Conegliano.xlsx

Output:
  - A CSV with columns:
      datetime,station_code,station_name,pm10
    Sorted by station_code, datetime, with duplicates removed.

Notes:
  - The script is resilient to common schema variations:
      - datetime may appear as a single column (e.g., 'DATA ORA', 'DataOra', 'datetime')
      - or as separate 'DATA' (date) and 'ORA' (hour) columns
      - value column prefers ones containing 'PM10', else falls back to common names
  - Datetime is parsed with dayfirst=True and left timezone-naive (local time).
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd


# Aggregated counters collected during processing
GLOBAL_STATS = {
	"rows_input": 0,
	"rows_after_drop": 0,
	"invalid_datetime": 0,
	"empty_pm10": 0,
}


def configure_logging(verbose: bool) -> None:
	logging.basicConfig(
		level=logging.DEBUG if verbose else logging.INFO,
		format="%(asctime)s %(levelname)s %(message)s",
	)


def find_column_case_insensitive(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
	"""
	Return the first column name from `columns` that matches any of `candidates` case-insensitively.
	"""
	lower_to_actual = {str(c).strip().lower(): c for c in columns}
	for candidate in candidates:
		key = candidate.strip().lower()
		if key in lower_to_actual:
			return lower_to_actual[key]
	# Also try contains matching for robustness (e.g., "DATA ORA" vs "data_ora")
	for candidate in candidates:
		cand = candidate.strip().lower()
		for col in columns:
			if cand in str(col).strip().lower():
				return col
	return None


def extract_station_from_filename(path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
	"""
	Parse filename like H_2010_502604_Conegliano.xlsx -> (station_code, station_name, year)
	Returns (None, None, None) if pattern does not match.
	"""
	match = re.match(r"^H_(\d{4})_(\d{6})_(.+)\.xlsx$", path.name)
	if not match:
		return None, None, None
	year, code, name = match.groups()
	# Normalize station name (replace underscores with spaces if any linger)
	name = name.replace("_", " ").strip()
	return code, name, year


def coerce_hour_to_string(hour_series: pd.Series) -> pd.Series:
	"""
	Coerce an hour-like series to 'HH:MM' strings.
	Handles integers (0-23), floats, or strings like '0', '00', '0:00', '00:00'.
	"""
	def to_hhmm(x) -> Optional[str]:
		if pd.isna(x):
			return None
		# Try numeric
		try:
			val = float(x)
			if pd.isna(val):
				return None
			h = int(val)
			if 0 <= h <= 23:
				return f"{h:02d}:00"
		except Exception:
			pass
		# String cases
		s = str(x).strip()
		# Common patterns
		if re.fullmatch(r"\d{1,2}", s):
			h = int(s)
			if 0 <= h <= 23:
				return f"{h:02d}:00"
		if re.fullmatch(r"\d{1,2}[:.]\d{2}", s):
			parts = re.split(r"[:.]", s)
			h = int(parts[0])
			m = int(parts[1])
			if 0 <= h <= 23 and 0 <= m <= 59:
				return f"{h:02d}:{m:02d}"
		# Last resort: return original string if reasonable
		return s

	return hour_series.map(to_hhmm)


def parse_datetime(df: pd.DataFrame) -> pd.Series:
	"""
	Attempt to parse a datetime series from `df` trying multiple common ARPAV schema variants.
	"""
	# 1) Combined datetime column
	combined_candidates = [
		"DATA ORA",
		"DATA_ORA",
		"DataOra",
		"DateTime",
		"datetime",
		"date time",
		"dataora",
		"istante",
		"time",
	]
	col = find_column_case_insensitive(df.columns, combined_candidates)
	if col:
		dt = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
		return dt

	# 2) Separate date and hour columns
	date_candidates = ["DATA", "Data", "date", "giorno", "Giorno"]
	hour_candidates = ["ORA", "Ora", "hour", "HH", "H", "ora"]
	date_col = find_column_case_insensitive(df.columns, date_candidates)
	hour_col = find_column_case_insensitive(df.columns, hour_candidates)
	if date_col and hour_col:
		date_str = df[date_col].astype(str).str.strip()
		hour_str = coerce_hour_to_string(df[hour_col])
		combined = (date_str + " " + hour_str.fillna("")).str.strip()
		dt = pd.to_datetime(combined, errors="coerce", dayfirst=True)
		return dt

	# 3) Fallback: attempt to parse any column that looks datetime-like
	for candidate in list(df.columns):
		series = pd.to_datetime(df[candidate], errors="coerce", dayfirst=True)
		if series.notna().sum() >= max(3, int(0.2 * len(series))):  # heuristic: at least some valid datetimes
			return series

	# Nothing worked
	return pd.to_datetime(pd.Series([pd.NA] * len(df)))


def select_pm10_column(df: pd.DataFrame, datetime_colnames: Iterable[str]) -> Optional[str]:
	"""
	Select the PM10 value column from df.
	Priority:
	  1) any column containing 'pm10' (case-insensitive)
	  2) common names: 'VALORE', 'Valore', 'media_oraria', 'media oraria', 'value'
	  3) best numeric column by non-null count excluding datetime and common metadata
	"""
	exclude = set(name.strip().lower() for name in datetime_colnames)
	exclude.update(
		[
			"stazione", "nome stazione", "station", "nome", "name",
			"codice", "codice stazione", "station_code",
			"flag", "quality", "quality_flag", "qc", "qcflag",
			"unit", "unita", "unità",
			"anno", "year", "mese", "mese_anno", "giorno",
			"provincia", "comune",
		]
	)
	# 1) contains 'pm10'
	for col in df.columns:
		if "pm10" in str(col).strip().lower():
			return col

	# 2) common names
	for cand in ["VALORE", "Valore", "media_oraria", "media oraria", "value", "valore"]:
		col = find_column_case_insensitive(df.columns, [cand])
		if col:
			return col

	# 3) best numeric column by non-null count, excluding metadata
	best_col = None
	best_non_null = -1
	for col in df.columns:
		col_lower = str(col).strip().lower()
		if col_lower in exclude:
			continue
		series = pd.to_numeric(df[col], errors="coerce")
		non_null = series.notna().sum()
		if non_null > best_non_null:
			best_non_null = non_null
			best_col = col
	return best_col


def process_excel_file(path: Path) -> Optional[pd.DataFrame]:
	"""
	Read one Excel file and return a DataFrame with columns:
	  station_code, station_name, datetime, pm10
	Returns None if parsing fails meaningfully.
	"""
	station_code, station_name, year = extract_station_from_filename(path)
	if not station_code:
		logging.warning("Skipping file with unexpected name pattern: %s", path.name)
		return None

	try:
		df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
	except Exception as exc:
		logging.error("Failed reading %s: %s", path.name, exc)
		return None
	GLOBAL_STATS["rows_input"] += len(df)

	# Parse datetime
	datetime_series = parse_datetime(df)
	GLOBAL_STATS["invalid_datetime"] += int(datetime_series.isna().sum())
	if datetime_series.notna().sum() == 0:
		logging.warning("No valid datetime parsed in %s; skipping.", path.name)
		return None

	# Pick pm10 value column
	pm10_col = select_pm10_column(df, datetime_colnames=["datetime"])
	if pm10_col is None:
		logging.warning("No PM10-like column found in %s; skipping.", path.name)
		return None

	pm10_series = pd.to_numeric(df[pm10_col], errors="coerce")
	GLOBAL_STATS["empty_pm10"] += int(pm10_series.isna().sum())

	result = pd.DataFrame(
		{
			"station_code": station_code,
			"station_name": station_name,
			"datetime": datetime_series,
			"pm10": pm10_series,
		}
	)

	# Clean
	before = len(result)
	result = result.dropna(subset=["datetime", "pm10"])
	if len(result) < before:
		logging.debug("Dropped %d rows with missing datetime/pm10 in %s", before - len(result), path.name)
	GLOBAL_STATS["rows_after_drop"] += len(result)

	return result


def merge_directory(input_dir: Path) -> pd.DataFrame:
	"""
	Process all .xlsx files in input_dir and concatenate results.
	"""
	files = sorted(input_dir.glob("*.xlsx"))
	if not files:
		logging.warning("No .xlsx files found under: %s", input_dir)
		return pd.DataFrame(columns=["station_code", "station_name", "datetime", "pm10"])

	collected: list[pd.DataFrame] = []
	for f in files:
		logging.info("Processing %s", f.name)
		one = process_excel_file(f)
		if one is not None and not one.empty:
			collected.append(one)
		else:
			logging.warning("No usable data in %s", f.name)

	if not collected:
		return pd.DataFrame(columns=["station_code", "station_name", "datetime", "pm10"])

	df = pd.concat(collected, ignore_index=True)
	# Normalize datetime to second resolution and sort
	df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", dayfirst=True)
	df = df.dropna(subset=["datetime"])
	df = df.sort_values(["station_code", "datetime"]).reset_index(drop=True)
	# Drop duplicates by station_code+datetime keeping first non-null pm10
	df = df.drop_duplicates(subset=["station_code", "datetime"], keep="first")
	# Reorder columns as requested: datetime first
	df = df[["datetime", "station_code", "station_name", "pm10"]]
	return df


def default_paths() -> Tuple[Path, Path]:
	"""
	Compute default input and output paths based on repository layout.
	"""
	# scripts/ -> repo root is parent
	root = Path(__file__).resolve().parents[1]
	input_dir = root / "data" / "arpav" / "PM10" / "orari"
	output_csv = root / "data" / "arpav" / "PM10" / "merged_pm10_hourly.csv"
	return input_dir, output_csv


def parse_args() -> argparse.Namespace:
	in_default, out_default = default_paths()
	parser = argparse.ArgumentParser(description="Merge ARPAV PM10 hourly Excel files into one CSV.")
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=in_default,
		help=f"Directory containing ARPAV PM10 hourly .xlsx files (default: {in_default})",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=out_default,
		help=f"Output CSV path (default: {out_default})",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Enable verbose logging",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	configure_logging(args.verbose)

	input_dir: Path = args.input_dir
	output_csv: Path = args.output

	if not input_dir.exists() or not input_dir.is_dir():
		logging.error("Input directory does not exist or is not a directory: %s", input_dir)
		raise SystemExit(2)

	df = merge_directory(input_dir)
	if df.empty:
		logging.warning("No data collected; nothing to write.")
	else:
		output_csv.parent.mkdir(parents=True, exist_ok=True)
		df.to_csv(output_csv, index=False)
		logging.info("Wrote %d rows to %s", len(df), output_csv)

		# Log aggregated missing/empty metrics
		total_input = GLOBAL_STATS["rows_input"]
		total_after = GLOBAL_STATS["rows_after_drop"]
		total_invalid_dt = GLOBAL_STATS["invalid_datetime"]
		total_empty_pm10 = GLOBAL_STATS["empty_pm10"]
		logging.info(
			"Aggregate counts — input rows: %d, parsed rows: %d, invalid datetimes: %d, empty PM10: %d",
			total_input,
			total_after,
			total_invalid_dt,
			total_empty_pm10,
		)

		# Per-station coverage: compute percentage of missing hourly timestamps within observed span
		# And also coverage over a fixed global period
		global_start = pd.Timestamp("2010-01-01 00:00:00")
		global_end = pd.Timestamp("2025-12-31 23:00:00")
		global_expected = len(pd.date_range(start=global_start, end=global_end, freq="H"))

		report_rows = []
		for (station_code, station_name), g in df.groupby(["station_code", "station_name"], dropna=False):
			if g.empty:
				continue
			unique_times = pd.to_datetime(g["datetime"]).dropna().sort_values().unique()
			start = pd.Timestamp(unique_times[0])
			end = pd.Timestamp(unique_times[-1])
			full_range = pd.date_range(start=start, end=end, freq="H")
			expected = len(full_range)
			present = int(pd.Series(unique_times).nunique())
			missing = max(0, expected - present)
			missing_pct = (missing / expected) if expected > 0 else 0.0

			# Global period coverage (fixed)
			g_times = pd.to_datetime(g["datetime"])
			present_global = int(g_times[(g_times >= global_start) & (g_times <= global_end)].nunique())
			missing_global = max(0, global_expected - present_global)
			missing_pct_global = (missing_global / global_expected) if global_expected > 0 else 0.0

			report_rows.append(
				{
					"station_code": station_code,
					"station_name": station_name,
					"start_datetime": start,
					"end_datetime": end,
					"expected_hours": expected,
					"present_hours": present,
					"missing_hours": missing,
					"missing_pct": missing_pct,
					"global_start_datetime": global_start,
					"global_end_datetime": global_end,
					"global_expected_hours": global_expected,
					"global_present_hours": present_global,
					"global_missing_hours": missing_global,
					"global_missing_pct": missing_pct_global,
				}
			)
			logging.info(
				"Station %s (%s): observed-span expected %d, present %d, missing %d (%.2f%%) | global expected %d, present %d, missing %d (%.2f%%)",
				station_code,
				station_name,
				expected,
				present,
				missing,
				missing_pct * 100.0,
				global_expected,
				present_global,
				missing_global,
				missing_pct_global * 100.0,
			)

		if report_rows:
			report_df = pd.DataFrame(report_rows).sort_values(["station_code"])
			report_path = output_csv.with_name("merged_pm10_hourly_missing_report.csv")
			report_df.to_csv(report_path, index=False)
			logging.info("Wrote per-station missing-hours report to %s", report_path)


if __name__ == "__main__":
	main()


