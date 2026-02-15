"""
features.py — Feature engineering pour la modélisation du trafic
"""

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> tuple:
    """
    Construit la matrice de features et la variable cible pour la modélisation.

    Returns
    -------
    X : pd.DataFrame  — Features
    y : pd.Series      — Variable cible (debit_horaire)
    """
    df = df.copy()

    # Features numériques de base
    feature_cols = []

    for col in ["heure", "jour_semaine", "mois", "is_weekend", "is_heure_pointe"]:
        if col in df.columns:
            feature_cols.append(col)

    # Encoding cyclique de l'heure (sin/cos)
    if "heure" in df.columns:
        df["heure_sin"] = np.sin(2 * np.pi * df["heure"] / 24)
        df["heure_cos"] = np.cos(2 * np.pi * df["heure"] / 24)
        feature_cols.extend(["heure_sin", "heure_cos"])

    # Encoding cyclique du jour
    if "jour_semaine" in df.columns:
        df["jour_sin"] = np.sin(2 * np.pi * df["jour_semaine"] / 7)
        df["jour_cos"] = np.cos(2 * np.pi * df["jour_semaine"] / 7)
        feature_cols.extend(["jour_sin", "jour_cos"])

    # Capacité de l'axe
    if "capacite" in df.columns:
        feature_cols.append("capacite")

    # One-hot encoding de l'axe routier
    if "id_arc" in df.columns:
        dummies = pd.get_dummies(df["id_arc"], prefix="axe", drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        feature_cols.extend(dummies.columns.tolist())

    # Variable cible
    target = "debit_horaire"
    if target not in df.columns:
        raise ValueError("Colonne 'debit_horaire' absente du DataFrame")

    X = df[feature_cols].copy()
    y = df[target].copy()

    # Remplir les NaN résiduels
    X = X.fillna(0)

    print(f"✅ Matrice de features : {X.shape[0]} lignes × {X.shape[1]} features")
    print(f"   Cible '{target}' : mean={y.mean():.0f}, std={y.std():.0f}")
    return X, y


def get_peak_hours_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le profil horaire moyen du trafic par axe.
    """
    if not {"id_arc", "heure", "debit_horaire"}.issubset(df.columns):
        return pd.DataFrame()

    profile = df.groupby(["id_arc", "heure"]).agg(
        debit_moyen=("debit_horaire", "mean"),
        debit_std=("debit_horaire", "std"),
        nom=("nom_compteur", "first"),
    ).reset_index()

    return profile.round(1)
