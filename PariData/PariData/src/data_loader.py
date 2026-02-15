"""
data_loader.py — Collecte des données de comptage routier depuis l'Open Data Paris
"""

import requests
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# API Open Data Paris — Comptages routiers permanents
BASE_URL = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets"
DATASET_COMPTAGES = "comptages-routiers-permanents"
DATASET_LOCALISATION = "referentiel-comptages-routiers"


def fetch_traffic_data(limit: int = 10000, offset: int = 0) -> pd.DataFrame:
    """
    Récupère les données de comptage routier depuis l'API Open Data Paris.
    
    Parameters
    ----------
    limit : int
        Nombre max de lignes à récupérer (max API = 100 par appel)
    offset : int
        Décalage pour la pagination
    
    Returns
    -------
    pd.DataFrame
    """
    all_records = []
    batch_size = 100  # Max par appel API
    current_offset = offset
    
    print(f"⬇️  Téléchargement des données de trafic (limit={limit})...")
    
    while current_offset < offset + limit:
        url = (
            f"{BASE_URL}/{DATASET_COMPTAGES}/records"
            f"?limit={batch_size}&offset={current_offset}"
            f"&order_by=date_comptage DESC"
        )
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            records = data.get("results", [])
            if not records:
                break
            
            all_records.extend(records)
            current_offset += batch_size
            
            if len(records) < batch_size:
                break
                
        except requests.RequestException as e:
            print(f"❌ Erreur API : {e}")
            break
    
    if not all_records:
        print("⚠️  Aucune donnée récupérée. Génération de données de démonstration...")
        return generate_demo_data()
    
    df = pd.json_normalize(all_records)
    print(f"✅ {len(df)} enregistrements récupérés")
    return df


def fetch_counter_locations() -> pd.DataFrame:
    """
    Récupère la localisation des compteurs routiers.
    """
    url = f"{BASE_URL}/{DATASET_LOCALISATION}/records?limit=100"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        df = pd.json_normalize(data.get("results", []))
        print(f"✅ {len(df)} compteurs localisés")
        return df
    except requests.RequestException as e:
        print(f"⚠️  Erreur localisation : {e}. Utilisation des données de démo.")
        return generate_demo_locations()


def generate_demo_data(n_records: int = 10000) -> pd.DataFrame:
    """
    Génère des données de démonstration réalistes pour le trafic parisien.
    Utile si l'API est indisponible ou pour les tests.
    """
    import numpy as np
    
    np.random.seed(42)
    
    # Axes routiers parisiens réalistes
    axes = {
        "AX001": {"nom": "Boulevard Périphérique — Porte de Vincennes", "lat": 48.847, "lon": 2.410, "capacite": 6000},
        "AX002": {"nom": "Boulevard Périphérique — Porte de la Chapelle", "lat": 48.898, "lon": 2.360, "capacite": 5800},
        "AX003": {"nom": "Boulevard Périphérique — Porte d'Orléans", "lat": 48.824, "lon": 2.325, "capacite": 5500},
        "AX004": {"nom": "Boulevard Périphérique — Porte Maillot", "lat": 48.878, "lon": 2.283, "capacite": 5700},
        "AX005": {"nom": "Champs-Élysées", "lat": 48.870, "lon": 2.307, "capacite": 3200},
        "AX006": {"nom": "Rue de Rivoli", "lat": 48.860, "lon": 2.347, "capacite": 2800},
        "AX007": {"nom": "Boulevard Saint-Germain", "lat": 48.853, "lon": 2.338, "capacite": 2500},
        "AX008": {"nom": "Boulevard Haussmann", "lat": 48.874, "lon": 2.330, "capacite": 2600},
        "AX009": {"nom": "Avenue de la République", "lat": 48.867, "lon": 2.377, "capacite": 2200},
        "AX010": {"nom": "Quai de Bercy", "lat": 48.838, "lon": 2.380, "capacite": 3000},
        "AX011": {"nom": "Boulevard Voltaire", "lat": 48.862, "lon": 2.380, "capacite": 2100},
        "AX012": {"nom": "Avenue des Gobelins", "lat": 48.836, "lon": 2.352, "capacite": 1800},
        "AX013": {"nom": "Boulevard de Sébastopol", "lat": 48.863, "lon": 2.349, "capacite": 2400},
        "AX014": {"nom": "Rue Lafayette", "lat": 48.876, "lon": 2.350, "capacite": 2000},
        "AX015": {"nom": "Boulevard Magenta", "lat": 48.880, "lon": 2.357, "capacite": 2300},
    }
    
    # Matrice d'adjacence (quels axes sont proches / connectés)
    adjacency = {
        "AX001": ["AX010", "AX011", "AX009"],
        "AX002": ["AX015", "AX014", "AX013"],
        "AX003": ["AX012", "AX007"],
        "AX004": ["AX005", "AX008"],
        "AX005": ["AX004", "AX006", "AX008"],
        "AX006": ["AX005", "AX007", "AX013"],
        "AX007": ["AX006", "AX003", "AX012"],
        "AX008": ["AX005", "AX004", "AX014"],
        "AX009": ["AX001", "AX011", "AX015"],
        "AX010": ["AX001", "AX012"],
        "AX011": ["AX001", "AX009", "AX013"],
        "AX012": ["AX003", "AX007", "AX010"],
        "AX013": ["AX002", "AX006", "AX011"],
        "AX014": ["AX002", "AX008", "AX015"],
        "AX015": ["AX002", "AX009", "AX014"],
    }
    
    # Générer des timestamps sur 30 jours, horaire
    dates = pd.date_range("2025-01-01", periods=30 * 24, freq="h")
    
    records = []
    for date in dates:
        hour = date.hour
        dow = date.dayofweek  # 0=lundi
        is_weekend = dow >= 5
        
        for ax_id, ax_info in axes.items():
            # Pattern de trafic réaliste
            if is_weekend:
                base = ax_info["capacite"] * 0.4
                peak_factor = 1.0 + 0.3 * np.sin(np.pi * (hour - 12) / 12)
            else:
                base = ax_info["capacite"] * 0.5
                # Double pic : matin 8h et soir 18h
                morning_peak = np.exp(-0.5 * ((hour - 8) / 1.5) ** 2) * 0.5
                evening_peak = np.exp(-0.5 * ((hour - 18) / 1.5) ** 2) * 0.5
                night_dip = 0.15 if (hour < 6 or hour > 22) else 0
                peak_factor = 1.0 + morning_peak + evening_peak - night_dip
            
            debit = int(base * peak_factor * (1 + np.random.normal(0, 0.1)))
            debit = max(50, min(debit, ax_info["capacite"]))
            taux_occ = min(100, round(debit / ax_info["capacite"] * 100, 1))
            
            records.append({
                "id_arc": ax_id,
                "nom_compteur": ax_info["nom"],
                "date_comptage": date.isoformat(),
                "debit_horaire": debit,
                "taux_occupation": taux_occ,
                "latitude": ax_info["lat"] + np.random.normal(0, 0.001),
                "longitude": ax_info["lon"] + np.random.normal(0, 0.001),
                "capacite": ax_info["capacite"],
            })
    
    df = pd.DataFrame(records)
    print(f"✅ {len(df)} enregistrements de démo générés (15 axes × 30 jours × 24h)")
    
    # Sauvegarder l'adjacence
    adj_records = []
    for ax, neighbors in adjacency.items():
        for n in neighbors:
            adj_records.append({"source": ax, "target": n})
    
    adj_df = pd.DataFrame(adj_records)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    adj_df.to_csv(PROCESSED_DIR / "adjacency.csv", index=False)
    
    return df


def generate_demo_locations() -> pd.DataFrame:
    """Génère les localisations de compteurs pour la démo."""
    axes = {
        "AX001": {"nom": "Bd Périphérique — Pte de Vincennes", "lat": 48.847, "lon": 2.410},
        "AX002": {"nom": "Bd Périphérique — Pte de la Chapelle", "lat": 48.898, "lon": 2.360},
        "AX003": {"nom": "Bd Périphérique — Pte d'Orléans", "lat": 48.824, "lon": 2.325},
        "AX004": {"nom": "Bd Périphérique — Pte Maillot", "lat": 48.878, "lon": 2.283},
        "AX005": {"nom": "Champs-Élysées", "lat": 48.870, "lon": 2.307},
        "AX006": {"nom": "Rue de Rivoli", "lat": 48.860, "lon": 2.347},
        "AX007": {"nom": "Bd Saint-Germain", "lat": 48.853, "lon": 2.338},
        "AX008": {"nom": "Bd Haussmann", "lat": 48.874, "lon": 2.330},
        "AX009": {"nom": "Av de la République", "lat": 48.867, "lon": 2.377},
        "AX010": {"nom": "Quai de Bercy", "lat": 48.838, "lon": 2.380},
        "AX011": {"nom": "Bd Voltaire", "lat": 48.862, "lon": 2.380},
        "AX012": {"nom": "Av des Gobelins", "lat": 48.836, "lon": 2.352},
        "AX013": {"nom": "Bd de Sébastopol", "lat": 48.863, "lon": 2.349},
        "AX014": {"nom": "Rue Lafayette", "lat": 48.876, "lon": 2.350},
        "AX015": {"nom": "Bd Magenta", "lat": 48.880, "lon": 2.357},
    }
    
    records = [{"id_arc": k, "nom": v["nom"], "latitude": v["lat"], "longitude": v["lon"]} for k, v in axes.items()]
    return pd.DataFrame(records)


def load_adjacency() -> dict:
    """Charge la matrice d'adjacence des axes."""
    adj_path = PROCESSED_DIR / "adjacency.csv"
    
    if adj_path.exists():
        df = pd.read_csv(adj_path)
        adjacency = {}
        for _, row in df.iterrows():
            adjacency.setdefault(row["source"], []).append(row["target"])
        return adjacency
    
    # Par défaut
    return {
        "AX001": ["AX010", "AX011", "AX009"],
        "AX002": ["AX015", "AX014", "AX013"],
        "AX003": ["AX012", "AX007"],
        "AX004": ["AX005", "AX008"],
        "AX005": ["AX004", "AX006", "AX008"],
        "AX006": ["AX005", "AX007", "AX013"],
        "AX007": ["AX006", "AX003", "AX012"],
        "AX008": ["AX005", "AX004", "AX014"],
        "AX009": ["AX001", "AX011", "AX015"],
        "AX010": ["AX001", "AX012"],
        "AX011": ["AX001", "AX009", "AX013"],
        "AX012": ["AX003", "AX007", "AX010"],
        "AX013": ["AX002", "AX006", "AX011"],
        "AX014": ["AX002", "AX008", "AX015"],
        "AX015": ["AX002", "AX009", "AX014"],
    }
