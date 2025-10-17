#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ERA5 BLH bulk downloader (resumable, chunked by N anni) — ZIP per chunk

Esempi:
  python download_blh.py --start-year 2008 --end-year 2025
  python download_blh.py --start-year 2024 --end-year 2025 --chunk-years 1 \
      --area "47.67,4.61,43.54,16.12" --variable boundary_layer_height --out ./downloads --pause 2

Note:
- Richiede credenziali CDS (~/.cdsapirc).
- Per il CDS-Beta si possono usare "data_format" e "download_format" (zip).
- Salvataggio stato e ripresa chunk incompleti.
"""

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# progress bar opzionale
try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

try:
    import cdsapi
except ImportError:
    print("Serve 'cdsapi'. Installa con: pip install cdsapi", file=sys.stderr)
    sys.exit(1)

# ---- Config ----
DATASET = "reanalysis-era5-single-levels"
DEFAULT_VARIABLE = "boundary_layer_height"
DEFAULT_AREA = "47.67,4.61,43.54,16.12"  # N,W,S,E
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS_01_31 = [f"{d:02d}" for d in range(1, 32)]
TIMES = [f"{h:02d}:00" for h in range(24)]
STATE_FILENAME = "state.json"
DEFAULT_OUT_BASENAME = "era5_blh"
PROJECT_DATA_SUBDIR = Path("data_blh")


def parse_area(area_str: str) -> List[float]:
    try:
        n, w, s, e = [float(x.strip()) for x in area_str.split(",")]
        return [n, w, s, e]
    except Exception:
        raise argparse.ArgumentTypeError("Area deve essere 'N,W,S,E' con numeri float")


def year_chunks(start_y: int, end_y: int, span: int) -> List[Tuple[int, int]]:
    if end_y < start_y:
        raise ValueError("end-year deve essere >= start-year")
    chunks = []
    y = start_y
    while y <= end_y:
        y_end = min(y + span - 1, end_y)
        chunks.append((y, y_end))
        y = y_end + 1
    return chunks


def ensure_folder(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_state(out_dir: Path) -> Dict:
    p = out_dir / STATE_FILENAME
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(out_dir: Path, state: Dict):
    p = out_dir / STATE_FILENAME
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, sort_keys=True)
    tmp.replace(p)


def build_out_dir(base_out: Optional[str], start_y: int, end_y: int) -> Path:
    # salva sotto data/era5-data per coerenza di progetto
    project_root = Path(__file__).resolve().parent
    base = project_root / PROJECT_DATA_SUBDIR
    if base_out:
        return base / base_out
    return base / f"{DEFAULT_OUT_BASENAME}_{start_y}_{end_y}"


def init_state(out_dir: Path, variable: str, area: List[float],
               start_y: int, end_y: int, chunk_years: int,
               chunks: List[Tuple[int,int]]) -> Dict:
    state = load_state(out_dir)
    meta = {
        "dataset": DATASET,
        "variable": variable,
        "area": area,
        "global_start_year": start_y,
        "global_end_year": end_y,
        "chunk_years": chunk_years,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "version": 1
    }
    if not state:
        entries = []
        for y0, y1 in chunks:
            entries.append({
                "start_year": y0,
                "end_year": y1,
                "target": f"era5_blh_{y0}_{y1}.zip",  # ZIP per chunk
                "status": "pending",
                "attempts": 0,
                "updated_at": None
            })
        state = {"meta": meta, "chunks": entries}
        save_state(out_dir, state)
        return state

    same = (
        state.get("meta", {}).get("dataset") == meta["dataset"] and
        state["meta"].get("variable") == meta["variable"] and
        state["meta"].get("global_start_year") == meta["global_start_year"] and
        state["meta"].get("global_end_year") == meta["global_end_year"] and
        state["meta"].get("chunk_years") == meta["chunk_years"] and
        state["meta"].get("area") == meta["area"]
    )
    if not same:
        raise RuntimeError(f"Esiste già uno state.json incompatibile in {out_dir}. Scegli un --out diverso.")

    # riconcilia file esistenti
    for entry in state.get("chunks", []):
        dst = out_dir / entry["target"]
        part = dst.with_suffix(dst.suffix + ".part")
        if dst.exists():
            entry["status"] = "done"
        elif part.exists() and entry["status"] == "done":
            entry["status"] = "pending"

    save_state(out_dir, state)
    return state


def retrieve_chunk(c: "cdsapi.Client", req: Dict, target: Path, retries: int = 5, bar=None) -> bool:
    backoff = 5.0
    for attempt in range(1, retries + 1):
        try:
            # scarica in .part e poi rinomina
            c.retrieve(DATASET, req, str(target.with_suffix(target.suffix + ".part")))
            part = target.with_suffix(target.suffix + ".part")
            if part.exists():
                part.replace(target)
            return True
        except Exception as ex:
            if attempt >= retries:
                if bar:
                    bar.write(f"Fallito: {target.name} dopo {attempt} tentativi: {ex}")
                return False
            if bar:
                bar.write(f"Tentativo {attempt}/{retries} fallito per {target.name}: {ex}. Retry...")
            time.sleep(backoff)
            backoff *= 1.8
    return False


def run(variable: str, area: List[float], start_y: int, end_y: int,
        chunk_years: int, pause: float, out_dir: Path):
    ensure_folder(out_dir)
    chunks = year_chunks(start_y, end_y, chunk_years)
    state = init_state(out_dir, variable, area, start_y, end_y, chunk_years, chunks)

    c = cdsapi.Client()

    todo = [e for e in state["chunks"] if e["status"] != "done"]
    total_bar = tqdm(total=len(todo), desc="Chunks", unit="chunk") if TQDM else None
    if not TQDM:
        print(f"Chunk da scaricare: {len(todo)}")

    for entry in state["chunks"]:
        if entry["status"] == "done":
            continue

        y0, y1 = entry["start_year"], entry["end_year"]
        target = out_dir / entry["target"]

        years = [f"{y}" for y in range(y0, y1 + 1)]
        req = {
            "product_type": "reanalysis",
            "variable": [variable],
            "year": years,
            "month": MONTHS,
            "day": DAYS_01_31,
            "time": TIMES,
            "area": area,
            # CDS-Beta: zip con netcdf interni
            "data_format": "netcdf",
            "download_format": "zip"
        }

        if total_bar:
            total_bar.write(f"[INFO] {y0}-{y1} → {target.name}")

        ok = retrieve_chunk(c, req, target, retries=5, bar=total_bar)
        entry["attempts"] += 1
        entry["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
        entry["status"] = "done" if ok and target.exists() else "failed"
        save_state(out_dir, state)

        if total_bar and entry["status"] == "done":
            total_bar.update(1)

        time.sleep(pause)

    failed = [e for e in state["chunks"] if e["status"] != "done"]
    msg = ("OK: tutti i chunk completati."
           if not failed else
           f"Completato con {len(failed)} chunk falliti. Rilancia per riprendere.")
    if total_bar:
        total_bar.write(msg)
        total_bar.close()
    else:
        print(msg)


def main():
    ap = argparse.ArgumentParser(description="ERA5 BLH downloader con resume e chunking per anni.")
    ap.add_argument("--start-year", type=int, required=True, help="Anno iniziale (YYYY)")
    ap.add_argument("--end-year", type=int, required=True, help="Anno finale (YYYY)")
    ap.add_argument("--chunk-years", type=int, default=2, help="Ampiezza chunk in anni (default: 2)")
    ap.add_argument("--variable", default=DEFAULT_VARIABLE, help="Variabile ERA5 (default: boundary_layer_height)")
    ap.add_argument("--area", type=parse_area, default=parse_area(DEFAULT_AREA),
                    help="Area N,W,S,E (default: 47.67,4.61,43.54,16.12)")
    ap.add_argument("--out", default=None, help="Cartella output sotto data/era5-data (default: auto)")
    ap.add_argument("--pause", type=float, default=2.0, help="Pausa tra chunk in secondi (default: 2)")
    args = ap.parse_args()

    if args.end_year < args.start_year:
        print("Errore: --end-year deve essere >= --start-year", file=sys.stderr)
        sys.exit(2)

    out_dir = build_out_dir(args.out, args.start_year, args.end_year)
    print(f"Output: {out_dir}")
    print(f"Range: {args.start_year} → {args.end_year} | chunk={args.chunk_years} anno/i")
    print(f"Area: {args.area} | Var: {args.variable}")

    run(variable=args.variable,
        area=args.area,
        start_y=args.start_year,
        end_y=args.end_year,
        chunk_years=args.chunk_years,
        pause=args.pause,
        out_dir=out_dir)


if __name__ == "__main__":
    main()
