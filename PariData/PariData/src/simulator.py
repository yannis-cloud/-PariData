"""
simulator.py — Simulation de fermeture d'axes routiers et redistribution du trafic
"""

import pandas as pd
import numpy as np


def simulate_closure(
    axis_stats: pd.DataFrame,
    adjacency: dict,
    closed_axis: str,
    redistribution_mode: str = "proportional",
) -> pd.DataFrame:
    """
    Simule la fermeture d'un axe routier et calcule la redistribution
    du trafic sur les axes adjacents.

    Parameters
    ----------
    axis_stats : pd.DataFrame
        Statistiques par axe (doit contenir : id_arc, debit_moyen, capacite)
    adjacency : dict
        Dictionnaire d'adjacence {id_arc: [voisins]}
    closed_axis : str
        ID de l'axe à fermer (ex: "AX005")
    redistribution_mode : str
        Mode de redistribution :
        - "proportional" : redistribue au prorata de la capacité restante
        - "equal" : redistribue à parts égales

    Returns
    -------
    pd.DataFrame
        Tableau avec colonnes : id_arc, nom, debit_avant, debit_apres,
        delta, surcharge_pct, status
    """
    stats = axis_stats.copy()

    if closed_axis not in stats["id_arc"].values:
        raise ValueError(f"Axe '{closed_axis}' non trouvé dans les données")

    neighbors = adjacency.get(closed_axis, [])
    if not neighbors:
        print(f"⚠️  Aucun axe adjacent connu pour {closed_axis}")
        neighbors = _find_nearest_axes(stats, closed_axis, n=3)

    # Trafic de l'axe fermé
    closed_row = stats[stats["id_arc"] == closed_axis].iloc[0]
    traffic_to_redistribute = closed_row["debit_moyen"]

    print(f"\n🚧 Fermeture de : {closed_row['nom_compteur']}")
    print(f"   Débit moyen à redistribuer : {traffic_to_redistribute:.0f} véh/h")
    print(f"   Axes adjacents : {neighbors}")

    # Préparer le résultat
    results = []

    for _, row in stats.iterrows():
        ax_id = row["id_arc"]
        result = {
            "id_arc": ax_id,
            "nom": row["nom_compteur"],
            "debit_avant": round(row["debit_moyen"], 0),
            "capacite": row.get("capacite", row["debit_moyen"] * 1.5),
            "debit_apres": round(row["debit_moyen"], 0),
            "delta": 0,
            "surcharge_pct": 0,
            "status": "normal",
        }

        if ax_id == closed_axis:
            result["debit_apres"] = 0
            result["delta"] = -result["debit_avant"]
            result["status"] = "fermé"

        results.append(result)

    results_df = pd.DataFrame(results)

    # Redistribution sur les voisins
    neighbor_mask = results_df["id_arc"].isin(neighbors)
    n_neighbors = neighbor_mask.sum()

    if n_neighbors == 0:
        print("⚠️  Pas de voisins trouvés pour la redistribution")
        return results_df

    if redistribution_mode == "proportional":
        # Redistribue au prorata de la capacité résiduelle
        neighbor_rows = results_df[neighbor_mask].copy()
        residual_cap = neighbor_rows["capacite"] - neighbor_rows["debit_avant"]
        residual_cap = residual_cap.clip(lower=1)  # Éviter division par 0
        total_residual = residual_cap.sum()

        for idx in results_df[neighbor_mask].index:
            cap_ratio = (results_df.loc[idx, "capacite"] - results_df.loc[idx, "debit_avant"]) / total_residual
            cap_ratio = max(cap_ratio, 0.05)  # Min 5% même si saturé
            added = traffic_to_redistribute * cap_ratio
            results_df.loc[idx, "debit_apres"] = round(results_df.loc[idx, "debit_avant"] + added, 0)
            results_df.loc[idx, "delta"] = round(added, 0)

    elif redistribution_mode == "equal":
        added_each = traffic_to_redistribute / n_neighbors
        for idx in results_df[neighbor_mask].index:
            results_df.loc[idx, "debit_apres"] = round(results_df.loc[idx, "debit_avant"] + added_each, 0)
            results_df.loc[idx, "delta"] = round(added_each, 0)

    # Calcul surcharge
    results_df["surcharge_pct"] = (
        (results_df["debit_apres"] / results_df["capacite"] * 100) - 
        (results_df["debit_avant"] / results_df["capacite"] * 100)
    ).round(1)

    results_df["taux_occupation_apres"] = (
        results_df["debit_apres"] / results_df["capacite"] * 100
    ).round(1)

    # Status
    results_df.loc[results_df["taux_occupation_apres"] > 90, "status"] = "🔴 congestion"
    results_df.loc[
        (results_df["taux_occupation_apres"] > 70) & 
        (results_df["taux_occupation_apres"] <= 90), "status"
    ] = "🟡 chargé"
    results_df.loc[results_df["taux_occupation_apres"] <= 70, "status"] = "🟢 fluide"
    results_df.loc[results_df["id_arc"] == closed_axis, "status"] = "⛔ fermé"

    # Résumé
    congested = (results_df["taux_occupation_apres"] > 90).sum()
    print(f"\n📊 Résultat de la simulation :")
    print(f"   Axes en congestion (>90%) : {congested}")
    print(f"   Surcharge max : +{results_df['surcharge_pct'].max():.1f}%")

    return results_df


def simulate_multiple_closures(
    axis_stats: pd.DataFrame,
    adjacency: dict,
    closed_axes: list,
) -> pd.DataFrame:
    """
    Simule la fermeture simultanée de plusieurs axes.
    """
    stats = axis_stats.copy()
    
    total_traffic = 0
    all_neighbors = set()
    
    for ax in closed_axes:
        if ax in stats["id_arc"].values:
            row = stats[stats["id_arc"] == ax].iloc[0]
            total_traffic += row["debit_moyen"]
            neighbors = adjacency.get(ax, [])
            # Les voisins ne doivent pas être des axes fermés
            all_neighbors.update([n for n in neighbors if n not in closed_axes])
    
    print(f"\n🚧 Fermeture simultanée de {len(closed_axes)} axes")
    print(f"   Trafic total à redistribuer : {total_traffic:.0f} véh/h")
    
    results = []
    
    for _, row in stats.iterrows():
        ax_id = row["id_arc"]
        result = {
            "id_arc": ax_id,
            "nom": row["nom_compteur"],
            "debit_avant": round(row["debit_moyen"], 0),
            "capacite": row.get("capacite", row["debit_moyen"] * 1.5),
            "debit_apres": round(row["debit_moyen"], 0),
            "delta": 0,
            "status": "normal",
        }
        
        if ax_id in closed_axes:
            result["debit_apres"] = 0
            result["delta"] = -result["debit_avant"]
            result["status"] = "⛔ fermé"
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Redistribution proportionnelle
    if all_neighbors:
        neighbor_mask = results_df["id_arc"].isin(all_neighbors)
        neighbor_rows = results_df[neighbor_mask]
        residual = (neighbor_rows["capacite"] - neighbor_rows["debit_avant"]).clip(lower=1)
        total_residual = residual.sum()
        
        for idx in results_df[neighbor_mask].index:
            ratio = max((results_df.loc[idx, "capacite"] - results_df.loc[idx, "debit_avant"]) / total_residual, 0.05)
            added = total_traffic * ratio
            results_df.loc[idx, "debit_apres"] = round(results_df.loc[idx, "debit_avant"] + added, 0)
            results_df.loc[idx, "delta"] = round(added, 0)
    
    results_df["taux_occupation_apres"] = (results_df["debit_apres"] / results_df["capacite"] * 100).round(1)
    results_df["surcharge_pct"] = ((results_df["debit_apres"] - results_df["debit_avant"]) / results_df["capacite"] * 100).round(1)
    
    results_df.loc[results_df["taux_occupation_apres"] > 90, "status"] = "🔴 congestion"
    results_df.loc[(results_df["taux_occupation_apres"] > 70) & (results_df["taux_occupation_apres"] <= 90), "status"] = "🟡 chargé"
    results_df.loc[results_df["taux_occupation_apres"] <= 70, "status"] = "🟢 fluide"
    for ax in closed_axes:
        results_df.loc[results_df["id_arc"] == ax, "status"] = "⛔ fermé"
    
    return results_df


def _find_nearest_axes(stats: pd.DataFrame, axis_id: str, n: int = 3) -> list:
    """Trouve les N axes les plus proches géographiquement."""
    if "latitude" not in stats.columns:
        return stats[stats["id_arc"] != axis_id]["id_arc"].head(n).tolist()

    ref = stats[stats["id_arc"] == axis_id].iloc[0]
    others = stats[stats["id_arc"] != axis_id].copy()
    others["distance"] = np.sqrt(
        (others["latitude"] - ref["latitude"]) ** 2 + 
        (others["longitude"] - ref["longitude"]) ** 2
    )
    return others.nsmallest(n, "distance")["id_arc"].tolist()
