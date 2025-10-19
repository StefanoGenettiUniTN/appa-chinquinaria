# ERA5 Boundary Layer Height (BLH) Downloader & Visualizer

Repository per l’estrazione, costruzione e analisi dei dati di **Boundary Layer Height (BLH)** (o **Mixing Layer Height**) dal dataset **ERA5 – reanalysis single levels**.  
Il progetto automatizza il download da Copernicus CDS, la conversione in dataset strutturato e la visualizzazione per un sample delle stazioni APPA Trentino.

---

## Descrizione

La **mixing layer height (MLH)**, o **planetary boundary layer height (PBLH)**, rappresenta lo spessore dello strato atmosferico entro cui avviene il rimescolamento dell’aria con la superficie.

- **Valori bassi (100–300 m)** → atmosfera stabile, accumulo di inquinanti.  
- **Valori alti (1000–2000 m)** → forte rimescolamento e dispersione efficiente.

È una variabile chiave per correlare concentrazioni di **PM10 / NO₂** con condizioni meteorologiche e per individuare episodi di **inversione termica** o **stagnazione**.

---

## 📂 Contenuti principali
| File / Cartella | Descrizione |
|-----------------|-------------|
| `download_blh.py` | Script per il download automatico da ERA5 (CDS API). |
| `data_blh` | Cartella in cui vengono salvati i file scaricati (.zip / .nc) e i dataset derivati (CSV orari e giornalieri) |
| `build_blh_dataset.py` | Script di unione dei file NetCDF/ZIP e creazione del dataset finale. |
| `sample_blh_hourly_stations.csv` | Dataset di esempio già pronto per il notebook.. |
| `README.md` | Documento informativo sulla branch. |
| `.gitignore` | File che esclude i dataset grezzi (`*.nc`, `*.zip`, ecc.) dal versionamento. |
| `requirements.txt` | |

---

## Origine dei dati

- **Dataset**: [ERA5 Single Levels – Reanalysis](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)
- **Variabile principale**: `boundary_layer_height`  
- **Unità**: metri (m)  
- **Risoluzione spaziale**: 0.25° × 0.25° (~25 km)  
- **Risoluzione temporale**: 1 ora  
- **Area coperta**: Nord Italia – `[N=47.67, W=4.61, S=43.54, E=16.12]`  
- **Periodo analizzato**: 2008–2025 (richieste biennali automatizzate)

---
## ⚙️ Download automatico

### Configurazione API
Per scaricare i dati ERA5 è necessario configurare il file `~/.cdsapirc` con la propria chiave utente Copernicus:

```text
url: https://cds.climate.copernicus.eu/api
key: <API_KEY>
```
Esempio di richiesta automatizzata:

```
import cdsapi
c = cdsapi.Client()
c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": "boundary_layer_height",
        "year": ["2024", "2025"],
        "month": [f"{m:02d}" for m in range(1,13)],
        "day": [f"{d:02d}" for d in range(1,32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [47.67, 4.61, 43.54, 16.12],
        "format": "netcdf"
    },
    "era5_blh_2024_2025.zip"
)

```

Dipendenze principali:

```
cdsapi
xarray
netCDF4
pandas
matplotlib
numpy
tqdm
```

Installa con:

```
pip install -r requirements.txt
