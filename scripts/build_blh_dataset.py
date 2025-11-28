
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build BLH dataset from ERA5 ZIPs.

- Legge tutti gli ZIP in --in-dir (default: data_blh)
- Rileva variabile BLH (blh o avg_ibld)
- Calcola cella griglia più vicina per ogni stazione
- Estrae serie orarie e giornaliere su tutti gli ZIP
- Scrive Parquet e, opzionalmente, CSV in --out-dir (default: data_blh/processed)

Requisiti: xarray, netCDF4, pandas, pyarrow
"""

import argparse
import zipfile
import shutil
from pathlib import Path
import gc
import numpy as np
import pandas as pd
import xarray as xr

# ---- Stazioni (nome -> (lat, lon)) ----
STATIONS = {
    "Monte Gaza": (46.08253, 10.95804),
    "Piana Rotaliana": (46.19683, 11.11343),
    "Borgo Valsugana": (46.05184, 11.45389),
    "Rovereto LGP": (45.89243, 11.03941),
    "Riva del Garda": (45.89146, 10.84448),
    "Avio A22": (45.74215, 10.97043),
    "Trento PSC": (46.06292, 11.12620),
    "Trento VBZ": (46.10433, 11.11022),
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def open_first_nc_from_zip(zip_path: Path, tmp_nc: Path) -> xr.Dataset:
    with zipfile.ZipFile(zip_path) as z:
        ncs = [n for n in z.namelist() if n.endswith(".nc")]
        if not ncs:
            raise RuntimeError(f"Nessun .nc in {zip_path.name}")
        with z.open(ncs[0]) as src, open(tmp_nc, "wb") as dst:
            shutil.copyfileobj(src, dst)
    ds = xr.open_dataset(tmp_nc)
    tmp_nc.unlink(missing_ok=True)
    return ds

def detect_dims_and_var(ds: xr.Dataset, prefer_var: str | None = None):
    if prefer_var and prefer_var in ds.data_vars:
        var = prefer_var
    else:
        var = "blh" if "blh" in ds.data_vars else ("avg_ibld" if "avg_ibld" in ds.data_vars else None)
    if not var:
        raise RuntimeError(f"Variabile BLH non trovata. Presenti: {list(ds.data_vars)}")
    t_dim = "time" if "time" in ds[var].dims else ("valid_time" if "valid_time" in ds[var].dims else None)
    if not t_dim:
        raise RuntimeError("Dimensione temporale non trovata (time/valid_time).")
    lat_name, lon_name = "latitude", "longitude"
    for n in (lat_name, lon_name):
        if n not in ds.coords:
            raise RuntimeError(f"Coordinata mancante: {n}")
    return var, t_dim, lat_name, lon_name

def build_station_mapping(ds: xr.Dataset, var: str, lat_name: str, lon_name: str):
    mapping = {}
    for name, (lat, lon) in STATIONS.items():
        sel = ds[var].sel({lat_name: lat, lon_name: lon}, method="nearest")
        glat = float(sel[lat_name].values)
        glon = float(sel[lon_name].values)
        dist = float(haversine(lat, lon, glat, glon))
        mapping[name] = {"grid_lat": glat, "grid_lon": glon, "dist_km": dist}
    return mapping

def extract_from_zip(zip_path: Path, tmp_nc: Path, var: str, t_dim: str,
                     lat_name: str, lon_name: str, mapping: dict):
    ds = open_first_nc_from_zip(zip_path, tmp_nc)
    frames_h, frames_d = [], []
    try:
        for name, info in mapping.items():
            s = ds[var].sel({lat_name: info["grid_lat"], lon_name: info["grid_lon"]})

            # hourly
            h = (
                s.to_series()
                 .rename("blh_m")
                 .reset_index()
                 .rename(columns={t_dim: "time"})
            )
            h["station"] = name
            h["grid_lat"] = info["grid_lat"]
            h["grid_lon"] = info["grid_lon"]
            h["dist_km"] = info["dist_km"]
            # colonne cella
            cell_lat = round(info["grid_lat"], 2)
            cell_lon = round(info["grid_lon"], 2)
            cell = (cell_lat, cell_lon)
            h["cell_lat"] = cell_lat
            h["cell_lon"] = cell_lon
            h["cell"] = [cell] * len(h)
            frames_h.append(h)

            # daily mean
            d = (
                s.resample({t_dim: "1D"}).mean()
                 .to_series()
                 .rename("blh_m")
                 .reset_index()
                 .rename(columns={t_dim: "time"})
            )
            d["station"] = name
            d["grid_lat"] = info["grid_lat"]
            d["grid_lon"] = info["grid_lon"]
            d["dist_km"] = info["dist_km"]
            d["cell_lat"] = cell_lat
            d["cell_lon"] = cell_lon
            d["cell"] = [cell] * len(d)
            frames_d.append(d)
    finally:
        ds.close(); del ds; gc.collect()
    return frames_h, frames_d

def main():
    ap = argparse.ArgumentParser(description="Crea dataset BLH da ZIP ERA5.")
    ap.add_argument("--in-dir", default="data_blh", help="Cartella input con ZIP (default: data_blh)")
    ap.add_argument("--out-dir", default=None, help="Cartella output (default: <in-dir>/processed)")
    ap.add_argument("--var", default=None, help="Nome variabile BLH se nota (es. blh)")
    ap.add_argument("--csv", action="store_true", help="Scrivi anche CSV oltre ai Parquet")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (in_dir / "processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_nc = out_dir / "_tmp.nc"

    zips = sorted(in_dir.rglob("*.zip"))
    if not zips:
        raise SystemExit(f"Nessuno ZIP trovato in {in_dir}")

    ds0 = open_first_nc_from_zip(zips[0], tmp_nc)
    var, t_dim, lat_name, lon_name = detect_dims_and_var(ds0, prefer_var=args.var)
    mapping = build_station_mapping(ds0, var, lat_name, lon_name)
    ds0.close(); del ds0; gc.collect()

    all_h, all_d = [], []
    for zp in zips:
        fh, fd = extract_from_zip(zp, tmp_nc, var, t_dim, lat_name, lon_name, mapping)
        all_h.extend(fh); all_d.extend(fd)

    df_hourly = pd.concat(all_h, ignore_index=True).sort_values(["station", "time"])
    df_daily  = pd.concat(all_d,  ignore_index=True).sort_values(["station", "time"])

    out_hourly_parq = out_dir / "blh_hourly_all_stations.parquet"
    out_daily_parq  = out_dir / "blh_daily_all_stations.parquet"
    df_hourly.to_parquet(out_hourly_parq, index=False, compression="snappy", engine="pyarrow")
    df_daily.to_parquet(out_daily_parq,  index=False, compression="snappy", engine="pyarrow")
    print(f"Scritto Parquet: {out_hourly_parq}")
    print(f"Scritto Parquet: {out_daily_parq}")

    if args.csv:
        out_hourly_csv = out_dir / "blh_hourly_all_stations.csv"
        out_daily_csv  = out_dir / "blh_daily_all_stations.csv"
        df_hourly.to_csv(out_hourly_csv, index=False)
        df_daily.to_csv(out_daily_csv, index=False)
        print(f"Scritto CSV: {out_hourly_csv}")
        print(f"Scritto CSV: {out_daily_csv}")

if __name__ == "__main__":
    main()
