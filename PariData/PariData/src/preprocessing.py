"""
preprocessing.py — Nettoyage et transformation des données de trafic routier
"""

import pandas as pd
import numpy as np


def clean_traffic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les données brutes de comptage routier.
    """
    df = df.copy()

    # Normaliser les noms de colonnes
    df.columns = (
        df.columns.str.lower().str.strip()
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    # Conversion date
    if "date_comptage" in df.columns:
        df["date_comptage"] = pd.to_datetime(df["date_comptage"], errors="coerce")
        df = df.dropna(subset=["date_comptage"])

    # Conversion numérique
    for col in ["debit_horaire", "taux_occupation", "capacite"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Supprimer les lignes sans débit
    if "debit_horaire" in df.columns:
        df = df.dropna(subset=["debit_horaire"])
        df = df[df["debit_horaire"] >= 0]

    # Supprimer les doublons
    df = df.drop_duplicates()

    print(f"✅ Nettoyage terminé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les features temporelles à partir de date_comptage.
    """
    df = df.copy()

    if "date_comptage" not in df.columns:
        print("⚠️  Colonne 'date_comptage' absente")
        return df

    dt = df["date_comptage"]
    df["heure"] = dt.dt.hour
    df["jour_semaine"] = dt.dt.dayofweek        # 0=Lundi
    df["jour_nom"] = dt.dt.day_name()
    df["mois"] = dt.dt.month
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["is_heure_pointe"] = df["heure"].apply(
        lambda h: 1 if (7 <= h <= 9) or (17 <= h <= 19) else 0
    )
    df["periode"] = df["heure"].apply(categorize_period)

    print(f"✅ Features temporelles ajoutées ({7} colonnes)")
    return df


def categorize_period(hour: int) -> str:
    """Catégorise l'heure en période de la journée."""
    if 6 <= hour < 10:
        return "matin_pointe"
    elif 10 <= hour < 16:
        return "journee"
    elif 16 <= hour < 20:
        return "soir_pointe"
    elif 20 <= hour < 23:
        return "soiree"
    else:
        return "nuit"


def compute_axis_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les statistiques agrégées par axe routier.
    """
    if "id_arc" not in df.columns or "debit_horaire" not in df.columns:
        print("⚠️  Colonnes requises absentes")
        return pd.DataFrame()

    stats = df.groupby("id_arc").agg(
        nom_compteur=("nom_compteur", "first"),
        debit_moyen=("debit_horaire", "mean"),
        debit_median=("debit_horaire", "median"),
        debit_max=("debit_horaire", "max"),
        debit_std=("debit_horaire", "std"),
        taux_occ_moyen=("taux_occupation", "mean"),
        nb_mesures=("debit_horaire", "count"),
    ).reset_index()

    if "capacite" in df.columns:
        cap = df.groupby("id_arc")["capacite"].first().reset_index()
        stats = stats.merge(cap, on="id_arc", how="left")
        stats["ratio_charge"] = (stats["debit_moyen"] / stats["capacite"]).round(3)

    if "latitude" in df.columns:
        coords = df.groupby("id_arc").agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
        ).reset_index()
        stats = stats.merge(coords, on="id_arc", how="left")

    stats = stats.round(1)
    print(f"✅ Stats calculées pour {len(stats)} axes")
    return stats
