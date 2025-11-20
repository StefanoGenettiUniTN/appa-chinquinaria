#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive matching helper between APPA air quality stations and
Meteo Trentino weather stations.

For each APPA station and for each weather variable, this script:

1. Finds the closest N (default: 3) Meteo Trentino stations that have
   ANY available data for that variable (using the coverage cache built
   by the weather coverage analysis utilities).
2. Saves time-series comparison plots for the closest stations so you
   can visually inspect them and decide which one to use.
3. Exposes a hard-coded selection structure where you can manually
   specify the preferred Meteo Trentino station for each APPA station
   and variable.
4. Creates a map plot (similar to the one in
   `analyze_weather_station_coverage.py`) to visualize the final
   APPA–Meteo Trentino matching defined in the selection structure.

This script is intentionally simple and focused on interactive/manual
matching rather than automatic optimization logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import folium  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    folium = None  # type: ignore

# Reuse utilities from the existing analysis script (same directory)
from analyze_weather_station_coverage import (  # type: ignore
    build_or_load_coverage_cache,
    create_merged_csvs_per_variable,
    extract_all_zip_files,
    find_variable_file,
    get_variable_definitions,
    load_appa_stations,
    load_meteo_trentino_stations,
    load_or_compute_distance_matrix,
    load_weather_csv,
)


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


# ---------------------------------------------------------------------------
# Hard-coded manual selection structure
# ---------------------------------------------------------------------------

# All keys (variables) should be taken from `get_variable_definitions()`.

APPA_METEO_SELECTION: Dict[str, Dict[str, str]] = {
    "402212": { # Piana Rotaliana
        "humidity": "T0135", # Trento (Roncafort), 11.3km
        "pressure": "T0118", # Cembra, 8.6km
        "radiation": "T0118", # Cembra, 8.6km
        "rain": "T0408", # Mezzolombardo (Maso Delle Part), 1.3km
        "temperature": "T0408", # Mezzolombardo (Maso Delle Part), 1.3km
        "wind_direction": "T0129", # Trento (Laste), 14km
        "wind_speed": "T0129", # Trento (Laste), 14km
    },
    "402204": { # Riva del Garda
        "humidity": "T0354", # Tremalzo, 13.6km / Rovereto, 15.4km
        "pressure": "T0354", # Tremalzo, 13.6km / Rovereto, 15.4km
        "radiation": "T0147", # Rovereto, 15.4km
        "rain": "T0193", # Torbole (Belvedere), 3.5km
        "temperature": "T0193", # Torbole (Belvedere), 3.5km
        "wind_direction": "T0354", # Tremalzo, 13.6km / Rovereto, 15.4km
        "wind_speed": "T0354", # Tremalzo, 13.6km / Rovereto, 15.4km
    },  # Riva is closer to Tremalzo which is however located in the mountains. Rovereto may be more useful.
    "402203": { # Monte Gazza
        "humidity": "T0135", # Trento (Roncafort), 11.1km
        "pressure": "T0368", # Monte Bondone, 10.7km
        "radiation": "T0368", # Monte Bondone, 10.7km
        "rain": "T0189", # Santa Massenza (Centrale), 2.6km
        "temperature": "T0189", # Santa Massenza (Centrale), 2.6km
        "wind_direction": "T0368", # Monte Bondone, 10.7km
        "wind_speed": "T0368", # Monte Bondone, 10.7km
    },
    "402209": { # Parco S. Chiara
        "humidity": "T0129", # Trento (Laste), 1.2km
        "pressure": "T0129", # Trento (Laste), 1.2km
        "radiation": "T0129", # Trento (Laste), 1.2km
        "rain": "T0454", # Trento (Liceo Galilei), 0.9km
        "temperature": "T0454", # Trento (Liceo Galilei), 0.9km
        "wind_direction": "T0129", # Trento (Laste), 1.2km
        "wind_speed": "T0129", # Trento (Laste), 1.2km
    },
    "402206": { # Rovereto
        "humidity": "T0147", # Rovereto, 0.6km
        "pressure": "T0147", # Rovereto, 0.6km
        "radiation": "T0147", # Rovereto, 0.6km
        "rain": "T0147", # Rovereto, 0.6km
        "temperature": "T0147", # Rovereto, 0.6km
        "wind_direction": "T0147", # Rovereto, 0.6km
        "wind_speed": "T0147", # Rovereto, 0.6km
    },
    "402211": { # Trento via Bolzano
        "humidity": "T0135", # Trento (Roncafort), 1.2km
        "pressure": "T0135", # Trento (Roncafort), 1.2km
        "radiation": "T0135", # Trento (Roncafort), 1.2km
        "rain": "T0135", # Trento (Roncafort), 1.2km
        "temperature": "T0135", # Trento (Roncafort), 1.2km
        "wind_direction": "T0129", # Trento (Laste), 4.1km
        "wind_speed": "T0129", # Trento (Laste), 4.1km
    },
    "402213": { # Avio
        "humidity": "T0147", # Rovereto, 18.1km
        "pressure": "T0153", # Ala (Ronchi), 7.4km
        "radiation": "T0147", # Rovereto, 18.1km
        "rain": "T0405", # Ala (Maso Le Pozze), 6.1km
        "temperature": "T0405", # Ala (Maso Le Pozze), 6.1km
        "wind_direction": "T0147", # Rovereto, 18.1km
        "wind_speed": "T0147", # Rovereto, 18.1km
    },
    "402201": { # Borgo Valsugana
        "humidity": "T0369", # Passo Sommo, 24.2 km
        "pressure": "T0010", # Levico (Terme), 12.4km
        "radiation": "T0469", # Castello Tesino (Le Parti), 13.6km
        "rain": "T0392", # Telve, 1.9km
        "temperature": "T0392", # Telve, 1.9km
        "wind_direction": "T0369", # Passo Sommo, 24.2 km
        "wind_speed": "T0369", # Passo Sommo, 24.2 km
    },
}


def ensure_per_station_csvs_from_merged(
    var_dirs: Dict[str, Path],
    merged_csvs: Dict[str, Optional[Path]],
) -> None:
    """
    Ensure that per-station CSVs exist for each variable.

    For this interactive matching script, reading small per-station CSVs is
    much faster than repeatedly scanning the large merged CSVs.

    Strategy:
      - For each variable:
        - If the per-station directory already contains CSVs, do nothing.
        - Else, if a merged CSV exists, split it into one CSV per station
          under the corresponding var_dir.

    The per-station files keep the merged CSV structure (datetime, value,
    quality, station_code) so `load_weather_csv` can still read them.
    """
    for var_name, var_dir in var_dirs.items():
        merged_path = merged_csvs.get(var_name)
        if merged_path is None or not Path(merged_path).exists():
            continue

        var_dir = Path(var_dir)
        var_dir.mkdir(parents=True, exist_ok=True)

        # If there are already CSVs, assume the split was done (or data came
        # from the original ZIP extraction) and skip.
        if any(var_dir.glob("*.csv")):
            continue

        print(
            f"  Splitting merged CSV for '{var_name}' into per-station files "
            f"under {var_dir}..."
        )

        # Stream the merged CSV in chunks and write per-station files.
        chunks = pd.read_csv(
            merged_path,
            encoding="latin-1",
            dtype={"station_code": str},
            chunksize=200_000,
            low_memory=False,
        )

        for chunk in chunks:
            if "station_code" not in chunk.columns:
                continue

            for station_code, group in chunk.groupby("station_code"):
                if not station_code or group.empty:
                    continue

                out_path = var_dir / f"{station_code}.csv"
                # Append if exists, otherwise create new with header.
                write_header = not out_path.exists()
                group.to_csv(
                    out_path,
                    index=False,
                    mode="a",
                    header=write_header,
                )

        print(f"  ✓ Finished splitting merged CSV for '{var_name}'.")


def get_variable_display_names() -> Dict[str, str]:
    """Human-friendly names and units for variables (for plotting labels)."""
    return {
        "temperature": "Temperature (°C)",
        "rain": "Rain (mm)",
        "wind_speed": "Wind speed (m/s)",
        "wind_direction": "Wind direction (°)",
        "pressure": "Pressure (hPa)",
        "radiation": "Radiation (W/m²)",
        "humidity": "Humidity (%)",
    }


# Shared styling for variables when drawing edges on maps
VARIABLE_STYLES: Dict[str, Dict[str, object]] = {
    # Use very bright, high-contrast colors that stand out clearly
    # both on plain backgrounds and on map tiles.
    "temperature": {"color": "#ff0000", "linestyle": "-", "offset_index": 0},  # bright red
    "rain": {"color": "#00bfff", "linestyle": "--", "offset_index": 1},  # deep sky blue
    "wind_speed": {"color": "#ff00ff", "linestyle": ":", "offset_index": -1},  # magenta
    "wind_direction": {
        "color": "#ff8c00",
        "linestyle": "-.",
        "offset_index": 2,
    },  # vivid orange
    "pressure": {
        "color": "#000000",
        "linestyle": (0, (3, 1)),
        "offset_index": -2,
    },  # black, dense dashes
    "radiation": {
        "color": "#ffff00",
        "linestyle": (0, (5, 2)),
        "offset_index": 3,
    },  # bright yellow, longer dashes
    "humidity": {
        "color": "#00ff00",
        "linestyle": (0, (1, 2)),
        "offset_index": -3,
    },  # lime green, fine dots
}

DEFAULT_VARIABLE_STYLE: Dict[str, object] = {
    "color": "#8c564b",
    "linestyle": "-",
    "offset_index": 0,
}


def build_station_variable_usage(
    selection: Dict[str, Dict[str, str]]
) -> Dict[str, Set[str]]:
    """
    From the APPA→Meteo selection mapping, build a mapping:

        meteo_station_code -> set(variables used at that station)
    """
    usage: Dict[str, Set[str]] = {}
    for _, var_map in selection.items():
        for var_name, meteo_code in var_map.items():
            code_str = str(meteo_code)
            if code_str not in usage:
                usage[code_str] = set()
            usage[code_str].add(var_name)
    return usage


def find_top_k_candidates(
    distance_matrix: pd.DataFrame,
    coverage_cache: pd.DataFrame,
    k: int = 3,
) -> pd.DataFrame:
    """
    For each APPA station and each variable, find the closest K
    Meteo Trentino stations that have ANY available data.

    Uses the pre-computed coverage_cache from the analysis utilities.

    Returns a DataFrame with:
        appa_station, variable, rank, meteo_station, distance_km,
        available, valid_percent, coverage_percent
    """
    print("Finding closest candidate stations per APPA station and variable...")

    variables = list(get_variable_definitions().keys())

    # Ensure correct dtypes
    coverage_cache = coverage_cache.copy()
    coverage_cache["meteo_station"] = coverage_cache["meteo_station"].astype(str)
    coverage_cache["variable"] = coverage_cache["variable"].astype(str)

    records: List[Dict] = []

    for appa_station in distance_matrix.index:
        distances = distance_matrix.loc[appa_station].sort_values()

        for var_name in variables:
            # Filter coverage to stations that have this variable available
            var_cache = coverage_cache[
                (coverage_cache["variable"] == var_name)
                & (coverage_cache["available"] == True)  # noqa: E712
            ].set_index("meteo_station")

            candidates: List[Dict] = []
            for meteo_station in distances.index:
                if meteo_station not in var_cache.index:
                    continue

                row = var_cache.loc[meteo_station]
                # Accept any station with >0 valid coverage
                valid_pct = float(row.get("valid_percent", 0.0))
                if valid_pct <= 0.0:
                    continue

                candidates.append(
                    {
                        "appa_station": appa_station,
                        "variable": var_name,
                        "meteo_station": meteo_station,
                        "distance_km": float(distances.loc[meteo_station]),
                        "available": bool(row["available"]),
                        "valid_percent": valid_pct,
                        "coverage_percent": float(row.get("coverage_percent", np.nan)),
                    }
                )

                if len(candidates) >= k:
                    break

            if not candidates:
                # Still emit a row to make it clear that nothing was found
                records.append(
                    {
                        "appa_station": appa_station,
                        "variable": var_name,
                        "rank": np.nan,
                        "meteo_station": None,
                        "distance_km": np.nan,
                        "available": False,
                        "valid_percent": 0.0,
                        "coverage_percent": 0.0,
                    }
                )
            else:
                for rank, cand in enumerate(candidates, start=1):
                    cand["rank"] = rank
                    records.append(cand)

    return pd.DataFrame(records)


def plot_top_k_timeseries(
    candidates_df: pd.DataFrame,
    appa_stations: pd.DataFrame,
    meteo_stations: pd.DataFrame,
    var_dirs: Dict[str, Path],
    start_date: str,
    end_date: str,
    output_dir: Path,
) -> None:
    """
    For each APPA station and variable, create a time-series figure of the
    closest K Meteo Trentino candidate stations.

    Performance notes:
    - We aggressively downsample by resampling to daily means and then
      sub-sampling, which reduces points per series by O(10–100x).
    - Each candidate station gets its OWN subplot in the same figure so
      that lines do not overlap visually.
    """
    if candidates_df.empty:
        print("No candidates found, skipping time-series plots.")
        return

    plots_dir = output_dir / "plots" / "candidates"
    plots_dir.mkdir(parents=True, exist_ok=True)

    var_display = get_variable_display_names()

    # Pre-compute time window once
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Group by APPA station and variable
    grouped = candidates_df[candidates_df["meteo_station"].notna()].groupby(
        ["appa_station", "variable"]
    )

    for (appa_station, var_name), group in grouped:
        appa_row = appa_stations[appa_stations["station_code"] == appa_station]
        appa_label = (
            f"{appa_station}"
            if appa_row.empty
            else f"{appa_station} - {appa_row.iloc[0]['station_name']}"
        )

        # One subplot per candidate station so that lines do not overlap
        group_sorted = group.sort_values("rank")
        n_candidates = len(group_sorted)
        fig, axes = plt.subplots(
            n_candidates,
            1,
            figsize=(14, max(3 * n_candidates, 4)),
            sharex=True,
        )
        if n_candidates == 1:
            axes = [axes]

        for ax, (_, row) in zip(axes, group_sorted.iterrows()):
            station_code = str(row["meteo_station"])
            distance_km = row["distance_km"]
            valid_pct = row.get("valid_percent", np.nan)

            # For this script we deliberately bypass merged CSVs and use
            # per-station CSVs only (much faster to read).
            csv_file = find_variable_file(
                station_code, var_name, var_dirs, merged_csvs=None
            )
            if csv_file is None or not csv_file.exists():
                ax.text(
                    0.5,
                    0.5,
                    f"No data file for {station_code}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
                continue

            df = load_weather_csv(
                csv_file,
            )
            if df is None or len(df) == 0:
                ax.text(
                    0.5,
                    0.5,
                    f"No data for {station_code}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
                continue

            # Filter by date range
            df = df[(df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)].copy()
            if len(df) == 0:
                ax.text(
                    0.5,
                    0.5,
                    f"No data in range for {station_code}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
                continue

            # Use only valid data
            df_valid = df[(df["quality"] == 1) & df["value"].notna()]
            if len(df_valid) == 0:
                ax.text(
                    0.5,
                    0.5,
                    f"No valid data for {station_code}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
                continue

            # Aggressive downsampling:
            # 1) resample to daily means, 2) sub-sample if still too dense.
            df_valid = (
                df_valid.set_index("datetime")
                .resample("D")  # daily mean
                .mean(numeric_only=True)
            )
            df_valid = df_valid[df_valid["value"].notna()].reset_index()

            max_points = 2000  # keep plots light-weight
            if len(df_valid) > max_points:
                step = max(1, len(df_valid) // max_points)
                df_valid = df_valid.iloc[::step].copy()

            station_info = meteo_stations[meteo_stations["code"] == station_code]
            station_name = (
                station_code
                if station_info.empty
                else station_info.iloc[0]["name"][:40]
            )

            # Plot single series per subplot (no overlapping lines)
            ax.plot(
                df_valid["datetime"],
                df_valid["value"],
                linewidth=0.7,
                color="C0",
            )

            ax.set_ylabel(var_display.get(var_name, var_name), fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title(
                f"{station_code} ({station_name}) – {distance_km:.1f} km, "
                f"valid {float(valid_pct):.1f}%",
                fontsize=9,
            )
            ax.tick_params(axis="y", labelsize=7)

        # Shared x-axis formatting on the bottom subplot
        axes[-1].set_xlabel("Date")
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.tick_params(axis="x", labelsize=7, rotation=45)

        fig.suptitle(
            f"{appa_label}\nClosest stations for "
            f"{var_name.replace('_', ' ').title()}",
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()

        safe_var = var_name.replace(" ", "_")
        out_file = plots_dir / f"{appa_station}__{safe_var}.png"
        plt.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"✓ Saved candidate time-series plots (downsampled) to {out_file}")

    print(f"✓ Saved candidate time-series plots (downsampled) to {plots_dir}")


def create_selection_map_plot(
    appa_stations: pd.DataFrame,
    meteo_stations: pd.DataFrame,
    selection: Dict[str, Dict[str, str]],
    map_variable: str,
    output_dir: Path,
) -> None:
    """
    Create a map plot showing the final APPA→Meteo Trentino matching
    defined in `selection` for a single variable (e.g., 'temperature').
    """
    if not selection:
        print("Selection mapping is empty, skipping map plot.")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot Meteo Trentino stations
    ax.scatter(
        meteo_stations["longitude"],
        meteo_stations["latitude"],
        c="green",
        marker="^",
        s=80,
        alpha=0.6,
        label="Meteo Trentino stations",
        zorder=3,
    )

    # Plot APPA stations
    ax.scatter(
        appa_stations["longitude"],
        appa_stations["latitude"],
        c="blue",
        marker="o",
        s=120,
        alpha=0.8,
        label="APPA stations",
        zorder=4,
        edgecolors="black",
        linewidths=1.2,
    )

    # Add labels for stations (offset slightly so text is not covered by markers)
    for _, row in meteo_stations.iterrows():
        ax.annotate(
            row["code"],
            (row["longitude"], row["latitude"]),
            fontsize=6,
            alpha=0.7,
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )

    for _, row in appa_stations.iterrows():
        ax.annotate(
            row["station_code"],
            (row["longitude"], row["latitude"]),
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="top",
            color="blue",
            xytext=(0, -6),
            textcoords="offset points",
        )

    # Draw connections for the selected variable
    drawn_pairs = 0
    for appa_code, var_map in selection.items():
        if map_variable not in var_map:
            continue
        meteo_code = var_map[map_variable]

        # Cast to string on both sides to avoid type mismatches (e.g. numeric vs string codes)
        appa_row = appa_stations[
            appa_stations["station_code"].astype(str) == str(appa_code)
        ]
        meteo_row = meteo_stations[
            meteo_stations["code"].astype(str) == str(meteo_code)
        ]

        if appa_row.empty or meteo_row.empty:
            continue

        ax.plot(
            [appa_row.iloc[0]["longitude"], meteo_row.iloc[0]["longitude"]],
            [appa_row.iloc[0]["latitude"], meteo_row.iloc[0]["latitude"]],
            "r--",
            alpha=0.5,
            linewidth=1.0,
            zorder=2,
        )
        drawn_pairs += 1

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"APPA – Meteo Trentino matching for variable: {map_variable}",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_aspect("equal", adjustable="box")

    # Simple scale bar (km)
    try:
        all_lats = pd.concat(
            [appa_stations["latitude"], meteo_stations["latitude"]]
        )
        all_lons = pd.concat(
            [appa_stations["longitude"], meteo_stations["longitude"]]
        )
        mean_lat = all_lats.mean()
        min_lon = all_lons.min()
        min_lat = all_lats.min()

        km_length = 10
        km_per_deg_lon = 111.0 * np.cos(np.deg2rad(mean_lat))
        if km_per_deg_lon > 0:
            deg_length = km_length / km_per_deg_lon
            x0 = min_lon + 0.01
            y0 = min_lat + 0.01
            ax.plot([x0, x0 + deg_length], [y0, y0], color="k", linewidth=2, zorder=5)
            ax.text(
                x0 + deg_length / 2,
                y0 + 0.01,
                f"{km_length} km",
                ha="center",
                va="bottom",
                fontsize=9,
                color="k",
            )
    except Exception:
        pass

    plt.tight_layout()
    out_file = plots_dir / f"station_map_selection__{map_variable}.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()

    print(
        f"✓ Saved selection map ({drawn_pairs} matched pairs) to {out_file}"
    )


def create_multi_variable_selection_map_plot(
    appa_stations: pd.DataFrame,
    meteo_stations: pd.DataFrame,
    selection: Dict[str, Dict[str, str]],
    output_dir: Path,
) -> None:
    """
    Create a single map plot that shows all APPA→Meteo Trentino matches
    for all variables at once.

    - APPA stations are plotted as blue circles.
    - Matched Meteo Trentino stations are plotted as green triangles.
    - Edges between stations are colored by variable, so you can visually
      distinguish which variable each connection refers to.
    """
    if not selection:
        print("Selection mapping is empty, skipping multi-variable map plot.")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Determine which variables are actually used in the selection.
    used_variables = sorted(
        {
            var_name
            for var_map in selection.values()
            for var_name in var_map.keys()
        }
    )
    if not used_variables:
        print("No variables found in selection, skipping multi-variable map plot.")
        return

    # Restrict Meteo stations to those that are actually used in the selection.
    matched_meteo_codes = {
        str(meteo_code)
        for var_map in selection.values()
        for meteo_code in var_map.values()
    }
    meteo_matched = meteo_stations[
        meteo_stations["code"].astype(str).isin(matched_meteo_codes)
    ]

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot matched Meteo Trentino stations
    meteo_scatter = ax.scatter(
        meteo_matched["longitude"],
        meteo_matched["latitude"],
        c="green",
        marker="^",
        s=80,
        alpha=0.7,
        label="Meteo Trentino (matched)",
        zorder=3,
    )

    # Plot all APPA stations
    appa_scatter = ax.scatter(
        appa_stations["longitude"],
        appa_stations["latitude"],
        c="blue",
        marker="o",
        s=120,
        alpha=0.9,
        label="APPA stations",
        zorder=4,
        edgecolors="black",
        linewidths=1.2,
    )

    # Add labels for stations (offset slightly so text is not covered by markers).
    for _, row in meteo_matched.iterrows():
        ax.annotate(
            row["code"],
            (row["longitude"], row["latitude"]),
            fontsize=6,
            alpha=0.7,
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )

    for _, row in appa_stations.iterrows():
        ax.annotate(
            row["station_code"],
            (row["longitude"], row["latitude"]),
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="top",
            color="blue",
            xytext=(0, -6),
            textcoords="offset points",
        )

    # Draw connections for all variables, using color, line style, and a small
    # perpendicular offset per variable to improve readability.
    base_offset_deg = 0.003  # ~0.3 km at Trentino latitudes (roughly)
    drawn_pairs = 0
    for appa_code, var_map in selection.items():
        appa_row = appa_stations[
            appa_stations["station_code"].astype(str) == str(appa_code)
        ]
        if appa_row.empty:
            continue

        for var_name, meteo_code in var_map.items():
            meteo_row = meteo_matched[
                meteo_matched["code"].astype(str) == str(meteo_code)
            ]
            if meteo_row.empty:
                continue

            style = VARIABLE_STYLES.get(var_name, DEFAULT_VARIABLE_STYLE)
            color = style["color"]
            linestyle = style["linestyle"]
            offset_index = style["offset_index"]

            x1 = float(appa_row.iloc[0]["longitude"])
            y1 = float(appa_row.iloc[0]["latitude"])
            x2 = float(meteo_row.iloc[0]["longitude"])
            y2 = float(meteo_row.iloc[0]["latitude"])

            dx = x2 - x1
            dy = y2 - y1
            seg_len = np.hypot(dx, dy)

            if seg_len > 0:
                # Perpendicular unit vector to (dx, dy)
                nx = -dy / seg_len
                ny = dx / seg_len
            else:
                # Degenerate case: coincide; nudge vertically
                nx, ny = 0.0, 1.0

            offset = base_offset_deg * offset_index
            x1o = x1 + nx * offset
            y1o = y1 + ny * offset
            x2o = x2 + nx * offset
            y2o = y2 + ny * offset

            ax.plot(
                [x1o, x2o],
                [y1o, y2o],
                color=color,
                linestyle=linestyle,
                alpha=0.75,
                linewidth=1.6,
                zorder=2,
            )
            drawn_pairs += 1

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        "APPA – Meteo Trentino matching for weather variables",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    # Legend: first for station types, then for variables (edge colors).
    from matplotlib.lines import Line2D  # local import to avoid cluttering top-level

    station_legend = ax.legend(
        handles=[appa_scatter, meteo_scatter],
        loc="upper right",
        fontsize=9,
        title="Stations",
    )
    ax.add_artist(station_legend)

    var_display = get_variable_display_names()
    variable_handles = []
    for var_name in used_variables:
        style = VARIABLE_STYLES.get(var_name, DEFAULT_VARIABLE_STYLE)
        color = style["color"]
        linestyle = style["linestyle"]
        label = var_display.get(var_name, var_name.replace("_", " ").title())
        variable_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=linestyle,
                lw=2.0,
                alpha=0.9,
                label=label,
            )
        )

    ax.legend(
        handles=variable_handles,
        loc="lower left",
        fontsize=8,
        title="Variables",
    )

    # Simple scale bar (km), reusing the logic from the single-variable map.
    try:
        all_lats = pd.concat(
            [appa_stations["latitude"], meteo_matched["latitude"]]
        )
        all_lons = pd.concat(
            [appa_stations["longitude"], meteo_matched["longitude"]]
        )
        mean_lat = all_lats.mean()
        min_lon = all_lons.min()
        min_lat = all_lats.min()

        km_length = 10
        km_per_deg_lon = 111.0 * np.cos(np.deg2rad(mean_lat))
        if km_per_deg_lon > 0:
            deg_length = km_length / km_per_deg_lon
            x0 = min_lon + 0.01
            y0 = min_lat + 0.01
            ax.plot([x0, x0 + deg_length], [y0, y0], color="k", linewidth=2, zorder=5)
            ax.text(
                x0 + deg_length / 2,
                y0 + 0.01,
                f"{km_length} km",
                ha="center",
                va="bottom",
                fontsize=9,
                color="k",
            )
    except Exception:
        pass

    plt.tight_layout()
    out_file = plots_dir / "station_map_selection__all_variables.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()

    print(
        f"✓ Saved multi-variable selection map "
        f"({drawn_pairs} matched variable-level connections) to {out_file}"
    )


def create_multi_variable_selection_map_interactive(
    appa_stations: pd.DataFrame,
    meteo_stations: pd.DataFrame,
    selection: Dict[str, Dict[str, str]],
    output_dir: Path,
) -> None:
    """
    Create an interactive web map (HTML) with a real basemap background
    for all APPA→Meteo matches and variables, using folium/Leaflet.

    - Uses OpenStreetMap tiles (via folium) as a geographic basemap.
    - APPA stations are shown as blue markers.
    - Matched Meteo Trentino stations are shown as green markers.
    - Connections are drawn as colored polylines per variable.
    """
    if folium is None:
        print(
            "folium is not installed – skipping interactive basemap map. "
            "Install 'folium' to enable this feature."
        )
        return

    if not selection:
        print("Selection mapping is empty, skipping interactive basemap map.")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Determine which variables are actually used in the selection.
    used_variables = sorted(
        {
            var_name
            for var_map in selection.values()
            for var_name in var_map.keys()
        }
    )
    if not used_variables:
        print("No variables found in selection, skipping interactive basemap map.")
        return

    # Restrict Meteo stations to those that are actually used in the selection.
    matched_meteo_codes = {
        str(meteo_code)
        for var_map in selection.values()
        for meteo_code in var_map.values()
    }
    meteo_matched = meteo_stations[
        meteo_stations["code"].astype(str).isin(matched_meteo_codes)
    ].copy()

    # Compute bounding box and center to initialize the map over Trentino region.
    all_lats = pd.concat(
        [appa_stations["latitude"], meteo_matched["latitude"]]
    )
    all_lons = pd.concat(
        [appa_stations["longitude"], meteo_matched["longitude"]]
    )
    center_lat = float(all_lats.mean())
    center_lon = float(all_lons.mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Add APPA stations (bright yellow markers with black edge)
    for _, row in appa_stations.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        code = str(row["station_code"])
        name = str(row.get("station_name", code))
        popup_html = f"<b>APPA {code}</b><br/>{name}"
        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color="#000000",
            fill=True,
            fill_color="#ffff00",
            fill_opacity=0.95,
            weight=2,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"APPA {code}",
        ).add_to(m)

    # Add matched Meteo Trentino stations (bright magenta markers with dark edge)
    for _, row in meteo_matched.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        code = str(row["code"])
        name = str(row.get("name", code))
        popup_html = f"<b>Meteo {code}</b><br/>{name}"
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color="#330033",
            fill=True,
            fill_color="#ff00ff",
            fill_opacity=0.9,
            weight=2,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"Meteo {code}",
        ).add_to(m)

    # Draw connections per variable as separate layers so you can toggle them on/off.
    var_display = get_variable_display_names()
    for var_name in used_variables:
        style = VARIABLE_STYLES.get(var_name, DEFAULT_VARIABLE_STYLE)
        color = str(style["color"])
        layer_name = var_display.get(
            var_name, var_name.replace("_", " ").title()
        )
        feature_group = folium.FeatureGroup(name=layer_name, show=True)

        for appa_code, var_map in selection.items():
            if var_name not in var_map:
                continue

            meteo_code = var_map[var_name]

            appa_row = appa_stations[
                appa_stations["station_code"].astype(str) == str(appa_code)
            ]
            meteo_row = meteo_matched[
                meteo_matched["code"].astype(str) == str(meteo_code)
            ]
            if appa_row.empty or meteo_row.empty:
                continue

            lat1 = float(appa_row.iloc[0]["latitude"])
            lon1 = float(appa_row.iloc[0]["longitude"])
            lat2 = float(meteo_row.iloc[0]["latitude"])
            lon2 = float(meteo_row.iloc[0]["longitude"])

            # Simple straight line for interactive map (no offset – you can zoom in).
            folium.PolyLine(
                locations=[[lat1, lon1], [lat2, lon2]],
                color=color,
                weight=5,
                opacity=0.95,
            ).add_to(feature_group)

        feature_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    out_file = plots_dir / "station_map_selection__all_variables_basemap.html"
    m.save(str(out_file))
    print(
        f"✓ Saved interactive basemap map with all variables to {out_file}"
    )


def create_selected_meteo_dataset(
    meteo_stations: pd.DataFrame,
    selection: Dict[str, Dict[str, str]],
    var_dirs: Dict[str, Path],
    merged_csvs: Dict[str, Optional[Path]],
    start_date: str,
    end_date: str,
    output_dir: Path,
) -> None:
    """
    Build a joint dataset with all data from the selected Meteo Trentino
    stations (no APPA data), using the manual APPA→Meteo selection.

    Output schema (wide format, keyed only by datetime):
        datetime,
        <APPA>_<Meteo>_<variable>,
        ...

    Example column names:
        402212_T0135_humidity
        402204_T0354_pressure
        ...

    Notes:
    - Only variables/stations that actually appear in APPA_METEO_SELECTION
      are included.
    - Only records with quality == 1 and non-null values are kept.
    - Time range is limited to [start_date, end_date].
    """
    if not selection:
        print("Selection mapping is empty, skipping selected meteo dataset.")
        return

    print("Step 10: Building joint dataset for selected Meteo Trentino stations...")

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    frames: List[pd.DataFrame] = []

    # Traverse the manual APPA → (variable → Meteo) mapping directly so that
    # we keep track of which APPA station each weather series belongs to.
    for appa_code, var_map in sorted(selection.items()):
        for var_name, meteo_code in sorted(var_map.items()):
            # Prefer the per-station CSVs in var_dirs (fast, small files),
            # rather than the large merged CSVs.
            var_dir = var_dirs.get(var_name)
            if var_dir is None:
                continue

            csv_file = Path(var_dir) / f"{meteo_code}.csv"
            if not csv_file.exists():
                continue

            df = load_weather_csv(
                csv_file,
                variable_name=None,
                station_code=None,
            )
            if df is None or len(df) == 0:
                continue

            # Filter by date range and keep only valid data.
            df = df[(df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)].copy()
            if len(df) == 0:
                continue

            df_valid = df[(df["quality"] == 1) & df["value"].notna()].copy()
            if len(df_valid) == 0:
                continue

            # Keep only datetime/value and rename value column to the
            # APPA–Meteo–variable composite identifier.
            df_pair = df_valid[["datetime", "value"]].copy()
            col_name = f"{appa_code}_{meteo_code}_{var_name}"
            df_pair = df_pair.rename(columns={"value": col_name})
            # Avoid duplicate datetimes within the same pair (keep first).
            df_pair = df_pair.sort_values("datetime").drop_duplicates(
                subset="datetime", keep="first"
            )
            frames.append(df_pair)

    if not frames:
        print("  No valid records found for selected stations; skipping dataset.")
        return

    # Outer-join all series on datetime to obtain a single wide dataframe.
    wide: Optional[pd.DataFrame] = None
    for df_pair in frames:
        if wide is None:
            wide = df_pair.set_index("datetime")
        else:
            wide = wide.join(df_pair.set_index("datetime"), how="outer")

    if wide is None:
        print("  No data after joining; skipping dataset creation.")
        return

    wide = wide.sort_index().reset_index()

    # Columns: datetime + one column per APPA–Meteo–variable combination.
    variable_cols = [c for c in wide.columns if c != "datetime"]

    out_path = output_dir / "selected_meteo_trentino_weather_dataset.csv"
    wide.to_csv(out_path, index=False)

    print(
        "  ✓ Saved selected Meteo Trentino weather dataset to "
        f"{out_path} ({len(wide):,} rows, {len(variable_cols)} variables)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive helper to match APPA stations with Meteo Trentino "
            "stations using closest K candidates and manual selection."
        )
    )
    parser.add_argument(
        "--appa-stations",
        type=Path,
        default=Path("data/appa-data/appa_monitoring_stations.csv"),
        help="Path to APPA stations CSV file.",
    )
    parser.add_argument(
        "--meteo-stations-xml",
        type=Path,
        default=Path("data/meteo-trentino/stations.xml"),
        help="Path to Meteo Trentino stations XML file.",
    )
    parser.add_argument(
        "--temp-rain-dir",
        type=Path,
        default=Path(
            "data/meteo-trentino/"
            "meteo-trentino-storico-completo-temperatura-pioggia"
        ),
        help="Directory with temperature and rain ZIP/CSV files.",
    )
    parser.add_argument(
        "--wind-pressure-dir",
        type=Path,
        default=Path(
            "data/meteo-trentino/"
            "meteo-trentino-storico-completo-vento-pressione-radiazione-umidità"
        ),
        help="Directory with wind, pressure, radiation, humidity ZIP/CSV files.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2014-01-01",
        help="Start date for time-series plots (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-12-31",
        help="End date for time-series plots (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of closest Meteo Trentino stations to consider per variable.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for coverage cache building (-1 = all cores).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/meteo-trentino-appa-matching"),
        help="Output directory for CSVs and plots.",
    )
    parser.add_argument(
        "--map-variable",
        type=str,
        default="wind_speed",
        help=(
            "Variable to use when drawing the selection map "
            "(e.g., 'temperature', 'rain', ...)."
        ),
    )
    parser.add_argument(
        "--force-recompute-coverage",
        action="store_true",
        help="Force recomputation of the coverage cache.",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print("APPA – Meteo Trentino matching helper")
    print("=" * 80)
    print(f"Time period: {args.start_date} → {args.end_date}")
    print(f"Output dir : {args.output_dir}")
    print("=" * 80)
    print()

    # 1. Load station metadata
    print("Step 1: Loading station metadata...")
    appa_stations = load_appa_stations(args.appa_stations)
    meteo_stations = load_meteo_trentino_stations(args.meteo_stations_xml)
    print(
        f"  ✓ Loaded {len(appa_stations)} APPA stations and "
        f"{len(meteo_stations)} Meteo Trentino stations."
    )
    print()

    # 2. Distance matrix
    print("Step 2: Loading/computing distance matrix...")
    distance_matrix_file = args.output_dir / "distance_matrix.csv"
    distance_matrix = load_or_compute_distance_matrix(
        appa_stations,
        meteo_stations,
        distance_matrix_file,
        force_recompute=False,
    )
    print("  ✓ Distance matrix ready.")
    print()

    # 3. Extract ZIPs and prepare per-variable directories
    print("Step 3: Preparing per-variable data directories...")
    var_dirs = extract_all_zip_files(
        args.temp_rain_dir,
        args.wind_pressure_dir,
        extract_dir=None,
        force_reextract=False,
    )

    print("Step 3.5: Creating merged CSVs per variable (if not already present)...")
    merged_csvs = create_merged_csvs_per_variable(
        var_dirs,
        args.output_dir,
        force_recreate=False,
    )

    # 3.7. Ensure per-station CSVs exist (split merged if needed), so that
    # plotting uses small, fast-to-read files instead of large merged ones.
    print("Step 3.7: Ensuring per-station CSVs exist (splitting merged if needed)...")
    ensure_per_station_csvs_from_merged(var_dirs, merged_csvs)

    # 4. Build or load coverage cache
    print("Step 4: Building/loading coverage cache...")
    coverage_cache = build_or_load_coverage_cache(
        var_dirs,
        meteo_stations,
        args.start_date,
        args.end_date,
        args.output_dir,
        n_jobs=args.n_jobs,
        merged_csvs=merged_csvs,
        force_recompute=args.force_recompute_coverage,
    )

    # 5. Find top-k candidate stations
    print("Step 5: Finding closest candidate stations per variable...")
    candidates_df = find_top_k_candidates(
        distance_matrix,
        coverage_cache,
        k=args.top_k,
    )
    candidates_file = args.output_dir / "candidate_matches_top_k.csv"
    candidates_df.to_csv(candidates_file, index=False)
    print(f"  ✓ Saved candidate matches to {candidates_file}")
    print()

    # 6. Time-series plots for candidates
    # print("Step 6: Creating time-series plots for closest candidates...")
    # plot_top_k_timeseries(
    #     candidates_df,
    #     appa_stations,
    #     meteo_stations,
    #     var_dirs,
    #     args.start_date,
    #     args.end_date,
    #     args.output_dir,
    # )
    # print()

    # 7. Map plot based on manual selection
    print("Step 7: Creating map plot for current manual selection (single variable)...")
    create_selection_map_plot(
        appa_stations,
        meteo_stations,
        APPA_METEO_SELECTION,
        args.map_variable,
        args.output_dir,
    )
    print()

    print("Step 8: Creating map plot for current manual selection (all variables)...")
    create_multi_variable_selection_map_plot(
        appa_stations,
        meteo_stations,
        APPA_METEO_SELECTION,
        args.output_dir,
    )
    print()

    print(
        "Step 9: Creating interactive basemap map for current manual "
        "selection (all variables)..."
    )
    create_multi_variable_selection_map_interactive(
        appa_stations,
        meteo_stations,
        APPA_METEO_SELECTION,
        args.output_dir,
    )
    print()

    # 10. Joint dataset for selected Meteo Trentino stations (no APPA data)
    create_selected_meteo_dataset(
        meteo_stations,
        APPA_METEO_SELECTION,
        var_dirs,
        merged_csvs,
        args.start_date,
        args.end_date,
        args.output_dir,
    )
    print()

    print("=" * 80)
    print("Done. You can now:")
    print("  1) Inspect the candidate plots under 'plots/candidates/'")
    print("  2) Edit APPA_METEO_SELECTION in this script to pick your final matches")
    print(
        "  3) Re-run the script to regenerate the selection map "
        f"(and keep re-using the candidate plots)."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()


