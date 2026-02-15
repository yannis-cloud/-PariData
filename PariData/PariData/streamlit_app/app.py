"""
PariData — Dashboard Streamlit
Simulation interactive de fermeture d'axes routiers parisiens
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="PariData — Simulation Trafic Paris",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DONNÉES
# ============================================================
AXES = {
    "AX001": {"nom": "Bd Périphérique — Pte de Vincennes", "lat": 48.847, "lon": 2.410, "capacite": 6000},
    "AX002": {"nom": "Bd Périphérique — Pte de la Chapelle", "lat": 48.898, "lon": 2.360, "capacite": 5800},
    "AX003": {"nom": "Bd Périphérique — Pte d'Orléans", "lat": 48.824, "lon": 2.325, "capacite": 5500},
    "AX004": {"nom": "Bd Périphérique — Pte Maillot", "lat": 48.878, "lon": 2.283, "capacite": 5700},
    "AX005": {"nom": "Champs-Élysées", "lat": 48.870, "lon": 2.307, "capacite": 3200},
    "AX006": {"nom": "Rue de Rivoli", "lat": 48.860, "lon": 2.347, "capacite": 2800},
    "AX007": {"nom": "Boulevard Saint-Germain", "lat": 48.853, "lon": 2.338, "capacite": 2500},
    "AX008": {"nom": "Boulevard Haussmann", "lat": 48.874, "lon": 2.330, "capacite": 2600},
    "AX009": {"nom": "Avenue de la République", "lat": 48.867, "lon": 2.377, "capacite": 2200},
    "AX010": {"nom": "Quai de Bercy", "lat": 48.838, "lon": 2.380, "capacite": 3000},
    "AX011": {"nom": "Boulevard Voltaire", "lat": 48.862, "lon": 2.380, "capacite": 2100},
    "AX012": {"nom": "Avenue des Gobelins", "lat": 48.836, "lon": 2.352, "capacite": 1800},
    "AX013": {"nom": "Bd de Sébastopol", "lat": 48.863, "lon": 2.349, "capacite": 2400},
    "AX014": {"nom": "Rue Lafayette", "lat": 48.876, "lon": 2.350, "capacite": 2000},
    "AX015": {"nom": "Boulevard Magenta", "lat": 48.880, "lon": 2.357, "capacite": 2300},
}

ADJACENCY = {
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


@st.cache_data
def generate_traffic_data():
    """Génère les données de trafic simulées."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=30 * 24, freq="h")
    records = []

    for date in dates:
        hour = date.hour
        is_weekend = date.dayofweek >= 5
        for ax_id, info in AXES.items():
            if is_weekend:
                base = info["capacite"] * 0.4
                peak = 1.0 + 0.3 * np.sin(np.pi * (hour - 12) / 12)
            else:
                base = info["capacite"] * 0.5
                morning = np.exp(-0.5 * ((hour - 8) / 1.5) ** 2) * 0.5
                evening = np.exp(-0.5 * ((hour - 18) / 1.5) ** 2) * 0.5
                night = 0.15 if (hour < 6 or hour > 22) else 0
                peak = 1.0 + morning + evening - night

            debit = int(base * peak * (1 + np.random.normal(0, 0.1)))
            debit = max(50, min(debit, info["capacite"]))

            records.append({
                "id_arc": ax_id, "nom": info["nom"], "date": date,
                "debit": debit, "capacite": info["capacite"],
                "lat": info["lat"], "lon": info["lon"],
            })

    return pd.DataFrame(records)


def get_axis_stats(df):
    """Calcule les stats par axe."""
    stats = df.groupby("id_arc").agg(
        nom=("nom", "first"), debit_moyen=("debit", "mean"),
        debit_max=("debit", "max"), capacite=("capacite", "first"),
        lat=("lat", "first"), lon=("lon", "first"),
    ).reset_index()
    stats["taux_charge"] = (stats["debit_moyen"] / stats["capacite"] * 100).round(1)
    return stats.round(1)


def simulate(stats, closed_axes):
    """Simule la fermeture d'axes et redistribue le trafic."""
    res = stats.copy()
    res["debit_apres"] = res["debit_moyen"]
    res["delta"] = 0.0

    total_traffic = 0
    all_neighbors = set()

    for ax in closed_axes:
        row = res[res["id_arc"] == ax]
        if len(row) > 0:
            total_traffic += row.iloc[0]["debit_moyen"]
            for n in ADJACENCY.get(ax, []):
                if n not in closed_axes:
                    all_neighbors.add(n)

    # Mettre à 0 les axes fermés
    for ax in closed_axes:
        res.loc[res["id_arc"] == ax, "debit_apres"] = 0
        res.loc[res["id_arc"] == ax, "delta"] = -res.loc[res["id_arc"] == ax, "debit_moyen"]

    # Redistribution
    if all_neighbors and total_traffic > 0:
        n_mask = res["id_arc"].isin(all_neighbors)
        residual = (res.loc[n_mask, "capacite"] - res.loc[n_mask, "debit_moyen"]).clip(lower=1)
        total_res = residual.sum()
        for idx in res[n_mask].index:
            ratio = max((res.loc[idx, "capacite"] - res.loc[idx, "debit_moyen"]) / total_res, 0.05)
            added = total_traffic * ratio
            res.loc[idx, "debit_apres"] = round(res.loc[idx, "debit_moyen"] + added, 0)
            res.loc[idx, "delta"] = round(added, 0)

    res["taux_avant"] = (res["debit_moyen"] / res["capacite"] * 100).round(1)
    res["taux_apres"] = (res["debit_apres"] / res["capacite"] * 100).round(1)
    res["surcharge"] = (res["taux_apres"] - res["taux_avant"]).round(1)

    def status(row):
        if row["id_arc"] in closed_axes:
            return "⛔ Fermé"
        if row["taux_apres"] > 90:
            return "🔴 Congestion"
        if row["taux_apres"] > 70:
            return "🟡 Chargé"
        return "🟢 Fluide"

    res["status"] = res.apply(status, axis=1)
    return res


# ============================================================
# APP
# ============================================================
df = generate_traffic_data()
stats = get_axis_stats(df)

# --- SIDEBAR ---
st.sidebar.image("https://img.icons8.com/fluency/96/traffic-light.png", width=60)
st.sidebar.title("🚦 PariData")
st.sidebar.markdown("**Simulation de trafic routier parisien**")
st.sidebar.divider()

page = st.sidebar.radio("Navigation", ["📊 Vue d'ensemble", "🚧 Simulation", "📈 Analyse temporelle"])

st.sidebar.divider()
st.sidebar.caption("Yannis Albert — 2026")
st.sidebar.caption("[GitHub](https://github.com/yannis-cloud) · [Portfolio](https://yannis-cloud.github.io/yannis-albert-portfolio/)")

# --- PAGE 1 : VUE D'ENSEMBLE ---
if page == "📊 Vue d'ensemble":
    st.title("📊 Vue d'ensemble du trafic parisien")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Axes surveillés", f"{len(AXES)}")
    c2.metric("Débit moyen global", f"{stats['debit_moyen'].mean():,.0f} véh/h")
    c3.metric("Axe le + chargé", stats.sort_values('taux_charge', ascending=False).iloc[0]['nom'].split('—')[-1].strip()[:20])
    c4.metric("Taux charge max", f"{stats['taux_charge'].max():.0f}%")

    col1, col2 = st.columns([1, 1])

    with col1:
        fig = px.scatter_mapbox(
            stats, lat="lat", lon="lon", size="debit_moyen",
            color="taux_charge", hover_name="nom",
            hover_data={"debit_moyen": ":.0f", "capacite": True, "taux_charge": ":.1f"},
            color_continuous_scale="RdYlGn_r", size_max=25, zoom=11.5,
            mapbox_style="carto-darkmatter",
            title="Carte du trafic",
        )
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            stats.sort_values("debit_moyen", ascending=True),
            x="debit_moyen", y="nom", orientation="h",
            color="taux_charge", color_continuous_scale="RdYlGn_r",
            title="Débit moyen par axe",
            labels={"debit_moyen": "Débit (véh/h)", "nom": ""},
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2 : SIMULATION ---
elif page == "🚧 Simulation":
    st.title("🚧 Simulation de fermeture d'axes")
    st.markdown("Sélectionnez un ou plusieurs axes à fermer et observez l'impact sur le réseau.")

    axis_options = {f"{v['nom']} ({k})": k for k, v in AXES.items()}
    selected = st.multiselect(
        "🛑 Axes à fermer :",
        options=list(axis_options.keys()),
        default=["Champs-Élysées (AX005)"],
    )
    closed = [axis_options[s] for s in selected]

    if closed:
        sim = simulate(stats, closed)

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        total_redirected = sim[sim["delta"] > 0]["delta"].sum()
        congested = (sim["taux_apres"] > 90).sum()
        max_surcharge = sim["surcharge"].max()
        k1.metric("Axes fermés", len(closed))
        k2.metric("Trafic redistribué", f"{total_redirected:,.0f} véh/h")
        k3.metric("Axes en congestion", f"{congested}", delta=f"+{congested}" if congested > 0 else "0", delta_color="inverse")
        k4.metric("Surcharge max", f"+{max_surcharge:.1f}%")

        st.divider()

        col1, col2 = st.columns([1, 1])

        with col1:
            # Carte impact
            fig = px.scatter_mapbox(
                sim, lat="lat", lon="lon",
                size=sim["debit_apres"].clip(lower=100),
                color="surcharge", hover_name="nom",
                hover_data={"debit_moyen": ":.0f", "debit_apres": ":.0f", "surcharge": ":.1f", "status": True},
                color_continuous_scale="RdYlGn_r", range_color=[-5, max(30, max_surcharge)],
                size_max=25, zoom=11.5, mapbox_style="carto-darkmatter",
                title="Impact géographique",
            )
            fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Barres avant/après
            affected = sim[sim["delta"] != 0].copy()
            affected["nom_short"] = affected["nom"].apply(lambda x: x.split("—")[-1].strip()[:22] if "—" in x else x[:22])

            fig = go.Figure()
            fig.add_trace(go.Bar(name="Avant", x=affected["nom_short"], y=affected["debit_moyen"], marker_color="#4FC3F7"))
            fig.add_trace(go.Bar(name="Après", x=affected["nom_short"], y=affected["debit_apres"], marker_color="#FF8A65"))
            fig.add_trace(go.Scatter(x=affected["nom_short"], y=affected["capacite"],
                                     mode="markers+lines", name="Capacité",
                                     line=dict(color="red", dash="dash", width=2),
                                     marker=dict(size=8, symbol="diamond")))
            fig.update_layout(title="Comparaison avant / après", barmode="group", height=500,
                              xaxis_tickangle=-25, yaxis_title="Débit (véh/h)")
            st.plotly_chart(fig, use_container_width=True)

        # Tableau détaillé
        st.subheader("📋 Tableau détaillé")
        display_df = sim[sim["delta"] != 0][["nom", "debit_moyen", "debit_apres", "delta", "taux_avant", "taux_apres", "surcharge", "status"]]
        display_df.columns = ["Axe", "Débit avant", "Débit après", "Delta", "Taux avant (%)", "Taux après (%)", "Surcharge (%)", "Status"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    else:
        st.info("👆 Sélectionnez au moins un axe à fermer pour lancer la simulation.")

# --- PAGE 3 : ANALYSE TEMPORELLE ---
elif page == "📈 Analyse temporelle":
    st.title("📈 Analyse temporelle du trafic")

    df["heure"] = df["date"].dt.hour
    df["jour_semaine"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["jour_semaine"] >= 5

    # Profil horaire
    hourly = df.groupby(["heure", "is_weekend"])["debit"].mean().reset_index()
    hourly["type"] = hourly["is_weekend"].map({False: "Semaine", True: "Weekend"})

    fig = px.line(
        hourly, x="heure", y="debit", color="type",
        title="Profil horaire moyen — Semaine vs Weekend",
        labels={"heure": "Heure", "debit": "Débit moyen (véh/h)", "type": ""},
        color_discrete_map={"Semaine": "#4FC3F7", "Weekend": "#FF8A65"},
    )
    fig.add_vrect(x0=7, x1=9, fillcolor="red", opacity=0.08, annotation_text="Pointe AM")
    fig.add_vrect(x0=17, x1=19, fillcolor="red", opacity=0.08, annotation_text="Pointe PM")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    st.subheader("🗓️ Heatmap heure × jour")
    pivot = df.pivot_table(values="debit", index="jour_semaine", columns="heure", aggfunc="mean")
    jours = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

    fig = px.imshow(
        pivot, x=list(range(24)), y=jours,
        color_continuous_scale="YlOrRd", aspect="auto",
        labels={"x": "Heure", "y": "Jour", "color": "Débit"},
        title="Intensité du trafic par heure et jour",
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Sélection d'un axe
    st.subheader("🔍 Profil détaillé par axe")
    selected_ax = st.selectbox("Axe :", [f"{v['nom']} ({k})" for k, v in AXES.items()])
    ax_id = selected_ax.split("(")[-1].replace(")", "")

    ax_data = df[df["id_arc"] == ax_id]
    ax_hourly = ax_data.groupby("heure")["debit"].agg(["mean", "std"]).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ax_hourly["heure"], y=ax_hourly["mean"],
                              mode="lines+markers", name="Débit moyen",
                              line=dict(color="#4FC3F7", width=3)))
    fig.add_trace(go.Scatter(x=ax_hourly["heure"],
                              y=ax_hourly["mean"] + ax_hourly["std"],
                              mode="lines", name="+1 std", line=dict(width=0),
                              showlegend=False))
    fig.add_trace(go.Scatter(x=ax_hourly["heure"],
                              y=(ax_hourly["mean"] - ax_hourly["std"]).clip(lower=0),
                              mode="lines", name="-1 std", line=dict(width=0),
                              fill="tonexty", fillcolor="rgba(79,195,247,0.2)",
                              showlegend=False))

    cap = AXES[ax_id]["capacite"]
    fig.add_hline(y=cap, line_dash="dash", line_color="red", annotation_text=f"Capacité : {cap}")
    fig.update_layout(title=f"Profil horaire — {AXES[ax_id]['nom']}",
                      xaxis_title="Heure", yaxis_title="Débit (véh/h)", height=400)
    st.plotly_chart(fig, use_container_width=True)
