#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Meteo Trentino bulk downloader (browser-based approach)

Usage examples:
    python bulk_download_meteo_trentino.py --stations "T0038,T0129" --variables "Pioggia,Temperatura aria"
    python bulk_download_meteo_trentino.py --all-stations --all-variables
    python bulk_download_meteo_trentino.py --stations "T0038" --out ./downloads

Station/variable syntax:
    STATIONS is a comma list like: T0038,T0129,T0139
    VARIABLES is a comma list like: Pioggia,Temperatura aria,Umid.relativa aria

This script uses the browser-based download approach by mimicking the web interface.
"""

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import xml.etree.ElementTree as ET

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("This script requires the 'requests' and 'beautifulsoup4' packages. Install with: pip install requests beautifulsoup4", file=sys.stderr)
    sys.exit(1)

# Optional nice progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False


# ---- Config ----
BASE_URL = "http://storico.meteotrentino.it"
DEFAULT_STATIONS = "T0038,T0129,T0139"  # Sample stations
DEFAULT_VARIABLES = "Pioggia,Temperatura aria,Umid.relativa aria,Direzione vento media,Veloc. vento media,Pressione atmosferica,Radiazione solare totale"
STATE_FILENAME = "state.json"
USER_AGENT = "meteo-trentino-bulk-downloader/1.0"

# Wanted variables (exclude "Annale Idrologico" variants)
WANTED_VARIABLES = {
    "Pioggia": "Pioggia",
    "Temperatura aria": "Temperatura aria",
    "Umid.relativa aria": "Umid.relativa aria",
    "Direzione vento media": "Direzione vento media",
    "Veloc. vento media": "Veloc. vento media",
    "Pressione atmosferica": "Pressione atmosferica",
    "Radiazione solare totale": "Radiazione solare totale"
}


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


def sanitize_filename(s: str) -> str:
    """Sanitize string for use in filename."""
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", s)


def build_output_folder(base_out: Optional[str]) -> Path:
    """Build output folder path."""
    # Always save in data/meteo-trentino folder in project root
    project_root = Path(__file__).parent.parent
    meteo_data_dir = project_root / "data" / "meteo-trentino"
    
    if base_out:
        # If custom output specified, use it but still under data/meteo-trentino
        out = meteo_data_dir / base_out
    else:
        # Default naming under data/meteo-trentino
        default_name = f"meteo-trentino_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out = meteo_data_dir / default_name
    
    return out


def create_session() -> requests.Session:
    """Create and configure requests session with proper cookies."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    })
    
    # Set cookies like the browser
    session.cookies.update({
        "bandwidth": "high",
        "username": "webuser",
        "userclass": "anon",
        "is_admin": "0",
        "fontsize": "80.01",
        "plotsize": "normal",
        "menuwidth": "20"
    })
    
    # Add timeout adapter
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


def scrape_stations_from_website(session: requests.Session) -> List[Dict[str, str]]:
    """Scrape station list from the Meteo Trentino website."""
    try:
        # Try to get stations from the main page or stations page
        # This is a placeholder - would need to be implemented based on the actual website structure
        response = session.get(f"{BASE_URL}/")
        if response.status_code != 200:
            return []
        
        # Parse HTML to extract station information
        soup = BeautifulSoup(response.text, 'html.parser')
        stations = []
        
        # Look for station links or dropdowns
        # This would need to be customized based on the actual HTML structure
        station_links = soup.find_all('a', href=re.compile(r'T\d{4}'))
        for link in station_links:
            href = link.get('href', '')
            match = re.search(r'T(\d{4})', href)
            if match:
                station_code = f"T{match.group(1)}"
                station_name = link.get_text().strip()
                if station_name:
                    stations.append({"code": station_code, "name": station_name})
        
        return stations
        
    except Exception as e:
        print(f"Warning: Could not scrape stations from website: {e}")
        return []


def get_all_stations(session: requests.Session) -> List[Dict[str, str]]:
    """Get list of all available stations from the website."""
    try:
        # Try to scrape stations from the website first
        scraped_stations = scrape_stations_from_website(session)
        if scraped_stations:
            print(f"Successfully scraped {len(scraped_stations)} stations from website")
            return scraped_stations
        
        # Fallback to hardcoded list if scraping fails
        print("Using hardcoded station list (scraping failed)")
        stations = [
            {'code': 'T0038', 'name': 'San Michele Alladige'},
            {'code': 'T0101', 'name': 'Zambana (Idrovora)'},
            {'code': 'T0129', 'name': 'Trento (Laste)'},
            {'code': 'T0135', 'name': 'Trento (Roncafort)'},
            {'code': 'T0136', 'name': 'Trento (Ufficio)'},
            {'code': 'T0137', 'name': 'Trento (Piazza Vittoria)'},
            {'code': 'T0139', 'name': 'Santorsola Terme'},
            {'code': 'T0140', 'name': 'Piazze Di Pine'},
            {'code': 'T0141', 'name': 'Sternigo'},
            {'code': 'T0142', 'name': 'Povo'},
            {'code': 'T0144', 'name': 'Monte Bondone'},
            {'code': 'T0146', 'name': 'Aldeno (San Zeno)'},
            {'code': 'T0147', 'name': 'Rovereto'},
            {'code': 'T0148', 'name': 'Terragnolo (Piazza)'},
            {'code': 'T0149', 'name': 'Vallarsa (Diga Di Speccheri)'},
            {'code': 'T0150', 'name': 'Vallarsa (Foxi)'},
            {'code': 'T0151', 'name': 'Mori (Loppio)'},
            {'code': 'T0152', 'name': 'Brentonico'},
            {'code': 'T0153', 'name': 'Ala (Ronchi)'},
            {'code': 'T0154', 'name': 'Ala (Convento)'},
            {'code': 'T0155', 'name': 'Brentonico (Diga Di Pra Da Stua)'},
            {'code': 'T0209', 'name': 'Lago Delle Piazze (Diga)'},
            {'code': 'T0210', 'name': 'Folgaria'},
            {'code': 'T0211', 'name': 'Ronzo'},
            {'code': 'T0326', 'name': 'Vigolo Vattaro (Frana)'},
            {'code': 'T0327', 'name': 'Monte Bondone (Giardino Botanico)'},
            {'code': 'T0356', 'name': 'Trento (Aeroporto)'},
            {'code': 'T0363', 'name': 'Vallarsa (Malga Boffetal)'},
            {'code': 'T0368', 'name': 'Monte Bondone (Viote)'},
            {'code': 'T0369', 'name': 'Passo Sommo'},
            {'code': 'T0374', 'name': 'Rovereto (Malga Zugna)'},
            {'code': 'T0381', 'name': 'Vallarsa (Parrocchia)'},
            {'code': 'T0405', 'name': 'Ala (Maso Le Pozze)'},
            {'code': 'T0408', 'name': 'Mezzolombardo (Maso Delle Part)'},
            {'code': 'T0409', 'name': 'Pergine Valsugana'},
            {'code': 'T0425', 'name': 'Passo Pian Delle Fugazze'},
            {'code': 'T0443', 'name': 'Brentonico (Santa Caterina)'},
            {'code': 'T0454', 'name': 'Trento (Liceo Galilei)'},
            {'code': 'T0008', 'name': 'Paneveggio (Campo Neve)'},
            {'code': 'T0059', 'name': 'Ziano Di Fiemme (Malga Sadole)'},
            {'code': 'T0092', 'name': 'Pian Fedaia (Diga)'},
            {'code': 'T0094', 'name': 'Passo Costalunga'},
            {'code': 'T0096', 'name': 'Moena (Diga Pezze)'},
            {'code': 'T0098', 'name': 'Moena'},
            {'code': 'T0102', 'name': 'Predazzo (Centrale)'},
            {'code': 'T0103', 'name': 'Passo Rolle'},
            {'code': 'T0104', 'name': 'Passo Valles'},
            {'code': 'T0105', 'name': 'Forte Buso (Diga)'},
            {'code': 'T0107', 'name': 'Cavalese (Convento)'},
            {'code': 'T0109', 'name': 'Val Cadino (Segheria Canton)'},
            {'code': 'T0110', 'name': 'Stramentizzo (Diga)'},
            {'code': 'T0113', 'name': 'Grumes'},
            {'code': 'T0114', 'name': 'Valda'},
            {'code': 'T0115', 'name': 'Segonzano (Gresta)'},
            {'code': 'T0116', 'name': 'Segonzano (Scancio)'},
            {'code': 'T0117', 'name': 'Albiano (Cave Di Porfido)'},
            {'code': 'T0118', 'name': 'Cembra'},
            {'code': 'T0119', 'name': 'Lisignago'},
            {'code': 'T0120', 'name': 'Pozzolago (Centrale)'},
            {'code': 'T0121', 'name': 'Lavis'},
            {'code': 'T0226', 'name': 'Monte Ruioch (Rifugio Tonini)'},
            {'code': 'T0227', 'name': 'Cermis (Casere)'},
            {'code': 'T0228', 'name': 'Vigo Di Fassa (Stalon De Vael)'},
            {'code': 'T0229', 'name': 'Campitello (Malga Do Col Daura)'},
            {'code': 'T0231', 'name': 'Mazzin (Campestrin)'},
            {'code': 'T0237', 'name': 'Mazzin (Fontanazzo)'},
            {'code': 'T0265', 'name': 'Mazzin'},
            {'code': 'T0267', 'name': 'Paneveggio (Bellamonte)'},
            {'code': 'T0367', 'name': 'Cavalese'},
            {'code': 'T0371', 'name': 'Lases (Frana)'},
            {'code': 'T0375', 'name': 'Marmolada (Pian Dei Fiacconi)'},
            {'code': 'T0376', 'name': 'Tesero (Pala De Santa)'},
            {'code': 'T0384', 'name': 'Passo Manghen'},
            {'code': 'T0389', 'name': 'Predazzo'},
            {'code': 'T0403', 'name': 'Canazei (Ciampac)'},
            {'code': 'T0404', 'name': 'Marmolada (Sas Del Mul)'},
            {'code': 'T0431', 'name': 'Capriana'},
            {'code': 'T0437', 'name': 'Canazei (Gries)'},
            {'code': 'T0445', 'name': 'Canazei (Coi De Paussa)'},
            {'code': 'T0003', 'name': 'Tenna'},
            {'code': 'T0009', 'name': 'Centa San Nicolo'},
            {'code': 'T0010', 'name': 'Levico (Terme)'},
            {'code': 'T0014', 'name': 'Telve (Pontarso)'},
            {'code': 'T0015', 'name': 'Bieno'},
            {'code': 'T0017', 'name': 'Costabrunella (Diga)'},
            {'code': 'T0018', 'name': 'Pieve Tesino (O.P. Enel)'},
            {'code': 'T0032', 'name': 'Lavarone (Chiesa)'},
            {'code': 'T0222', 'name': 'Borgo Valsugana'},
            {'code': 'T0243', 'name': 'Vetriolo'},
            {'code': 'T0245', 'name': 'Levico'},
            {'code': 'T0355', 'name': 'Passo Brocon'},
            {'code': 'T0392', 'name': 'Telve'},
            {'code': 'T0407', 'name': 'Grigno (Barricata)'},
            {'code': 'T0422', 'name': 'Pieve Tesino (Malga Sorgazza)'},
            {'code': 'T0423', 'name': 'Grigno'},
            {'code': 'T0424', 'name': 'Ronchi Valsugana (Malga Casapinello)'},
            {'code': 'T0432', 'name': 'Val Sella (Montagnola)'},
            {'code': 'T0469', 'name': 'Castello Tesino (Le Parti)'},
            {'code': 'T0156', 'name': 'Daone (Diga Di Malga Bissina)'},
            {'code': 'T0157', 'name': 'Daone (Diga Di Malga Boazzo)'},
            {'code': 'T0158', 'name': 'Daone (Diga Di Ponte Morandin)'},
            {'code': 'T0160', 'name': 'Cimego (Centrale)'},
            {'code': 'T0163', 'name': 'Storo (Centrale)'},
            {'code': 'T0203', 'name': 'Forte Dampola'},
            {'code': 'T0324', 'name': 'Prezzo (Frana)'},
            {'code': 'T0354', 'name': 'Tremalzo'},
            {'code': 'T0370', 'name': 'Storo (Lodrone)'},
            {'code': 'T0373', 'name': 'Daone (Malga Bissina)'},
            {'code': 'T0393', 'name': 'Storo'},
            {'code': 'T0410', 'name': 'Daone (Pracul)'},
            {'code': 'T0428', 'name': 'Pieve Di Bono'},
            {'code': 'T0021', 'name': 'San Martino Di Castrozza'},
            {'code': 'T0024', 'name': 'Passo Cereda'},
            {'code': 'T0026', 'name': 'Tonadico'},
            {'code': 'T0027', 'name': 'Val Noana (Diga)'},
            {'code': 'T0242', 'name': 'San Silvestro (Centrale)'},
            {'code': 'T0377', 'name': 'Cima Rosetta'},
            {'code': 'T0419', 'name': 'Tonadico (Castelpietra)'},
            {'code': 'T0420', 'name': 'Mezzano'},
            {'code': 'T0444', 'name': 'Ghiacciaio Di Fradusta'},
            {'code': 'T0450', 'name': 'San Martino Di Castrozza'},
            {'code': 'T0063', 'name': 'Pian Palu (Diga)'},
            {'code': 'T0064', 'name': 'Peio'},
            {'code': 'T0065', 'name': 'Careser (Diga)'},
            {'code': 'T0066', 'name': 'Cima Cavaion'},
            {'code': 'T0068', 'name': 'Cogolo Pont (Centrale)'},
            {'code': 'T0069', 'name': 'Passo Tonale'},
            {'code': 'T0071', 'name': 'Mezzana'},
            {'code': 'T0074', 'name': 'Male'},
            {'code': 'T0075', 'name': 'Rabbi (Somrabbi)'},
            {'code': 'T0076', 'name': 'Rabbi (San Bernardo)'},
            {'code': 'T0080', 'name': 'Fondo'},
            {'code': 'T0082', 'name': 'Passo Mendola'},
            {'code': 'T0083', 'name': 'Cles (Convento)'},
            {'code': 'T0084', 'name': 'Santa Giustina (Diga)'},
            {'code': 'T0086', 'name': 'Denno'},
            {'code': 'T0088', 'name': 'Tres'},
            {'code': 'T0090', 'name': 'Mezzolombardo (Convento)'},
            {'code': 'T0167', 'name': 'Pradalago (Rifugio Viviani)'},
            {'code': 'T0169', 'name': 'Monte Groste (Rifugio Graffer)'},
            {'code': 'T0212', 'name': 'Spormaggiore'},
            {'code': 'T0236', 'name': 'Romeno'},
            {'code': 'T0238', 'name': 'Malga Mare (Centrale)'},
            {'code': 'T0240', 'name': 'Rabbi (Tasse)'},
            {'code': 'T0241', 'name': 'Rabbi (Piazzola)'},
            {'code': 'T0308', 'name': 'Careser Alla Baia'},
            {'code': 'T0323', 'name': 'Campodenno (Frana)'},
            {'code': 'T0357', 'name': 'Male (Bivacco Marinelli)'},
            {'code': 'T0360', 'name': 'Passo Tonale'},
            {'code': 'T0364', 'name': 'Vermiglio (Capanna Presena)'},
            {'code': 'T0365', 'name': 'Cima Presena'},
            {'code': 'T0366', 'name': 'Peio'},
            {'code': 'T0372', 'name': 'Peio (Crozzi Taviela)'},
            {'code': 'T0380', 'name': 'Pian Palu (Malga Giumella)'},
            {'code': 'T0397', 'name': 'Cles (Maso Maiano)'},
            {'code': 'T0399', 'name': 'Fondo'},
            {'code': 'T0415', 'name': 'Bresimo (Malga Bordolona)'},
            {'code': 'T0416', 'name': 'Vermiglio (Masi Di Palu)'},
            {'code': 'T0417', 'name': 'Rumo (Lanza)'},
            {'code': 'T0418', 'name': 'Castelfondo (Malga Castrin)'},
            {'code': 'T0427', 'name': 'Folgarida Alta'},
            {'code': 'T0439', 'name': 'Ghiacciaio Presena'},
            {'code': 'T0442', 'name': 'Ghiacciaio Presena (Passo Paradiso)'},
            {'code': 'T0473', 'name': 'Ghiacciaio Del Careser'},
            {'code': 'T0994', 'name': 'Folgarida Bassa'},
            {'code': 'T0099', 'name': 'Cima Paganella'},
            {'code': 'T0166', 'name': 'Val Di Genova (O.P. Enel)'},
            {'code': 'T0168', 'name': 'Passo Campo Carlo Magno'},
            {'code': 'T0172', 'name': 'Santantonio Di Mavignola'},
            {'code': 'T0175', 'name': 'Pinzolo'},
            {'code': 'T0177', 'name': 'Val Di Breguzzo (Ponte Arno)'},
            {'code': 'T0178', 'name': 'La Rocca (Centrale)'},
            {'code': 'T0179', 'name': 'Tione'},
            {'code': 'T0182', 'name': 'Montagne (Larzana)'},
            {'code': 'T0183', 'name': 'Stenico'},
            {'code': 'T0184', 'name': 'San Lorenzo In Banale'},
            {'code': 'T0186', 'name': 'Nembia (Centrale)'},
            {'code': 'T0189', 'name': 'Santa Massenza (Centrale)'},
            {'code': 'T0190', 'name': 'Lago Di Cavedine'},
            {'code': 'T0193', 'name': 'Torbole (Belvedere)'},
            {'code': 'T0200', 'name': 'Tenno'},
            {'code': 'T0204', 'name': 'Bezzecca'},
            {'code': 'T0233', 'name': 'Dare'},
            {'code': 'T0239', 'name': 'Pinzolo (Ponte Plaza)'},
            {'code': 'T0286', 'name': 'Madonna Di Campiglio'},
            {'code': 'T0298', 'name': 'Riva'},
            {'code': 'T0322', 'name': 'Arco (Arboreto)'},
            {'code': 'T0325', 'name': 'Villa Rendena (Frana)'},
            {'code': 'T0379', 'name': 'Dro (Marocche)'},
            {'code': 'T0382', 'name': 'Dos Del Sabion (Monte Grual)'},
            {'code': 'T0383', 'name': 'Molveno'},
            {'code': 'T0396', 'name': 'Sarca Di Val Genova Al Ghiacciaio Del Mandrone'},
            {'code': 'T0401', 'name': 'Arco (Bruttagosto)'},
            {'code': 'T0402', 'name': 'Bezzecca (Spessa)'},
            {'code': 'T0406', 'name': 'Paganella (Malga Terlago)'},
            {'code': 'T0411', 'name': 'Villa Rendena (Rifugio Gork)'},
            {'code': 'T0412', 'name': 'Zuclo (Malga Casinot)'},
            {'code': 'T0413', 'name': 'Val Dambiez'},
            {'code': 'T0414', 'name': 'San Lorenzo In Banale (Pergoletti)'},
            {'code': 'T0426', 'name': 'Giustino (Frana)'},
            {'code': 'T0433', 'name': 'Val Genova (Malga Caret)'},
            {'code': 'T0435', 'name': 'Pinzolo (Malga Zeledria)'},
            {'code': 'T0436', 'name': 'Ragoli (Rifugio Alimonta)'},
            {'code': 'T0029', 'name': 'Caoria (Centrale)'},
            {'code': 'T0030', 'name': 'Canal San Bovo'},
            {'code': 'T0421', 'name': 'Caoria'},
            {'code': 'T0429', 'name': 'Lago Di Calaita'},
        ]
        return stations
    except Exception as e:
        print(f"Warning: Could not fetch station list: {e}")
        return []


def get_station_variables(session: requests.Session, station_code: str) -> List[Dict[str, str]]:
    """Get available variables for a specific station."""
    try:
        url = f"{BASE_URL}/wgen/cache/anon/cf{station_code}.xml"
        print(f"Fetching variables for {station_code}...")
        response = session.get(url, timeout=(10, 30))
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.text)
        variables = []
        
        for var_elem in root.findall(".//variable"):
            name = var_elem.attrib.get("name", "")
            subdesc = var_elem.attrib.get("subdesc", "")
            var_id = var_elem.attrib.get("var", "")
            unit = var_elem.attrib.get("varunits", "")
            period = var_elem.attrib.get("varperiod", "")
            
            # Skip "Annale Idrologico" variants
            if "Annale" in subdesc:
                continue
                
            # Only include wanted variables
            if name in WANTED_VARIABLES:
                variables.append({
                    "var": var_id,
                    "name": name,
                    "unit": unit,
                    "period": period,
                    "subdesc": subdesc
                })
        
        return variables
        
    except Exception as e:
        print(f"Warning: Could not fetch variables for station {station_code}: {e}")
        return []


def get_station_metadata(session: requests.Session, station_code: str) -> Dict[str, str]:
    """Get metadata for a specific station."""
    try:
        url = f"{BASE_URL}/cgi/webhyd.pl?df={station_code}&cat=rs&lvl=1"
        response = session.get(url)
        response.raise_for_status()
        
        # Parse HTML to extract metadata
        soup = BeautifulSoup(response.text, 'html.parser')
        metadata = {"station_code": station_code}
        
        # This would need to be implemented based on the HTML structure
        # For now, return basic info
        metadata.update({
            "name": f"Station {station_code}",
            "comment": "Metadata extraction not implemented yet"
        })
        
        return metadata
        
    except Exception as e:
        print(f"Warning: Could not fetch metadata for station {station_code}: {e}")
        return {"station_code": station_code, "name": f"Station {station_code}"}


def trigger_download(session: requests.Session, station_code: str, var_id: str, var_name: str) -> Optional[str]:
    """Trigger download and return ZIP URL."""
    try:
        params = {
            "co": station_code,
            "v": var_id,
            "vn": f"{var_name} ",
            "p": "Tutti i dati,01/01/1800,01/01/1800,period,1",
            "o": "Download,download",
            "i": "Tutte le misure,Point,1",
            "cat": "rs"
        }
        
        print(f"Triggering download for {station_code}/{var_name}...")
        response = session.get(f"{BASE_URL}/cgi/webhyd.pl", params=params, timeout=(10, 120))
        response.raise_for_status()
        
        # Extract ZIP link from HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup.find_all("script"):
            if "downloadlink" in script.text:
                match = re.search(r"http://storico\.meteotrentino\.it/wgen/users/.+?\.zip", script.text)
                if match:
                    return match.group(0)
        
        return None
        
    except Exception as e:
        print(f"Error triggering download for {station_code}/{var_name}: {e}")
        return None


def download_zip(session: requests.Session, zip_url: str, output_path: Path) -> bool:
    """Download ZIP file to specified path."""
    try:
        print(f"Downloading ZIP from {zip_url}...")
        response = session.get(zip_url, timeout=(10, 120))
        response.raise_for_status()
        
        with output_path.open("wb") as f:
            f.write(response.content)
        
        return True
        
    except Exception as e:
        print(f"Error downloading ZIP from {zip_url}: {e}")
        return False


def extract_csv_from_zip(zip_path: Path, output_dir: Path) -> Optional[Path]:
    """Extract CSV from ZIP file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Find CSV file in ZIP
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if not csv_files:
                return None
            
            csv_file = csv_files[0]
            csv_path = output_dir / csv_file
            
            # Extract CSV
            with zip_ref.open(csv_file) as source, csv_path.open('wb') as target:
                target.write(source.read())
            
            return csv_path
            
    except Exception as e:
        print(f"Error extracting CSV from {zip_path}: {e}")
        return None


def init_state(out_dir: Path, stations: List[str], variables: List[str]) -> Dict:
    """Initialize the state manifest."""
    state = load_state(out_dir)
    
    meta = {
        "stations": stations,
        "variables": variables,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "version": 1
    }
    
    # If empty, create new state
    if not state:
        entries = []
        for station in stations:
            for variable in variables:
                filename = f"{station}_{sanitize_filename(variable)}.zip"
                entries.append({
                    "station": station,
                    "variable": variable,
                    "filename": filename,
                    "status": "pending",
                    "attempts": 0,
                    "updated_at": None,
                })
        
        state = {"meta": meta, "downloads": entries}
        save_state(out_dir, state)
        return state

    # If state exists, ensure it matches current request
    same_job = (
        state.get("meta", {}).get("stations") == meta["stations"] and
        state.get("meta", {}).get("variables") == meta["variables"]
    )
    if not same_job:
        raise RuntimeError(f"Existing state.json in {out_dir} belongs to a different job. Choose a new --out folder.")

    # Reconcile file existence (mark done if file present)
    for entry in state.get("downloads", []):
        file_path = out_dir / entry["filename"]
        if file_path.exists():
            entry["status"] = "done"

    save_state(out_dir, state)
    return state


def run_download(stations: List[str], variables: List[str], out_dir: Path, extract_csv: bool = True):
    """Main download function."""
    ensure_folder(out_dir)
    
    # Create session
    session = create_session()
    
    # Get all stations if needed
    if "all" in stations:
        all_stations = get_all_stations(session)
        station_codes = [s["code"] for s in all_stations]
        print(f"Found {len(station_codes)} stations")
    else:
        station_codes = stations
    
    # Get all variables if needed
    if "all" in variables:
        # Get variables from first station as template
        if station_codes:
            sample_vars = get_station_variables(session, station_codes[0])
            variable_names = [v["name"] for v in sample_vars]
            print(f"Found {len(variable_names)} variables")
        else:
            variable_names = list(WANTED_VARIABLES.keys())
    else:
        variable_names = variables
    
    # Initialize state
    state = init_state(out_dir, station_codes, variable_names)
    
    # Prepare progress bars
    total_tasks = sum(1 for d in state["downloads"] if d["status"] != "done")
    total_bar = None

    try:
        if TQDM_AVAILABLE:
            total_bar = tqdm(total=total_tasks, desc="Downloads", unit="file")
        else:
            print(f"Downloading {total_tasks} file(s)...")

        for i, entry in enumerate(state["downloads"]):
            if entry["status"] == "done":
                continue

            station = entry["station"]
            variable = entry["variable"]
            filename = entry["filename"]

            print(f"\n[{i+1}/{total_tasks}] Processing {station}/{variable}")
            
            if total_bar:
                total_bar.set_description(f"{station}/{variable}")

            # Get station variables to find the correct var ID
            print(f"Getting variables for station {station}...")
            station_vars = get_station_variables(session, station)
            var_info = None
            for v in station_vars:
                if v["name"] == variable:
                    var_info = v
                    break
            
            if not var_info:
                print(f"Warning: Variable {variable} not found for station {station}")
                entry["status"] = "failed"
                entry["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
                save_state(out_dir, state)
                continue

            # Trigger download
            zip_url = trigger_download(session, station, var_info["var"], var_info["name"])
            
            if not zip_url:
                print(f"Warning: No ZIP URL found for {station}/{variable}")
                entry["status"] = "failed"
                entry["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
                save_state(out_dir, state)
                continue

            # Download ZIP
            zip_path = out_dir / filename
            success = download_zip(session, zip_url, zip_path)
            
            # Update state
            entry["attempts"] = entry.get("attempts", 0) + 1
            entry["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
            
            if success and zip_path.exists():
                entry["status"] = "done"
                
                # Extract CSV if requested
                if extract_csv:
                    csv_path = extract_csv_from_zip(zip_path, out_dir)
                    if csv_path:
                        entry["csv_file"] = csv_path.name
                
                if total_bar:
                    total_bar.update(1)
            else:
                entry["status"] = "failed"

            save_state(out_dir, state)
            
            # Polite delay between requests
            time.sleep(2)

        # Done summary
        failed = [d for d in state["downloads"] if d["status"] != "done"]
        if failed:
            msg = f"Completed with {len(failed)} failed download(s). You can re-run the same command to retry."
        else:
            msg = "All downloads completed successfully."
        
        if total_bar:
            total_bar.write(msg)
        else:
            print(msg)

    finally:
        if TQDM_AVAILABLE and total_bar is not None:
            total_bar.close()


def main():
    parser = argparse.ArgumentParser(description="Bulk download Meteo Trentino data using browser-based approach.")
    parser.add_argument("--stations", default=DEFAULT_STATIONS,
                        help=f"Stations list (comma-separated). Default: {DEFAULT_STATIONS}")
    parser.add_argument("--all-stations", action="store_true",
                        help="Download from all available stations")
    parser.add_argument("--variables", default=DEFAULT_VARIABLES,
                        help=f"Variables list (comma-separated). Default: {DEFAULT_VARIABLES}")
    parser.add_argument("--all-variables", action="store_true",
                        help="Download all available variables")
    parser.add_argument("--out", default=None, help="Output folder (default: meteo-trentino_TIMESTAMP)")
    parser.add_argument("--no-extract", action="store_true", help="Don't extract CSV files from ZIP archives")

    args = parser.parse_args()

    # Determine stations
    if args.all_stations:
        stations = ["all"]
    else:
        stations = [s.strip() for s in args.stations.split(",")]

    # Determine variables
    if args.all_variables:
        variables = ["all"]
    else:
        variables = [v.strip() for v in args.variables.split(",")]

    out_dir = build_output_folder(args.out)

    print(f"Output folder: {out_dir}")
    print(f"Stations: {stations}")
    print(f"Variables: {variables}")
    print(f"Extract CSV: {not args.no_extract}")

    run_download(stations, variables, out_dir, extract_csv=not args.no_extract)


if __name__ == "__main__":
    main()