📊 PariData — Simulation de trafic routier & modélisation prédictive
     

🎯 Problématique
Quel serait l'impact sur le trafic parisien si l'on fermait certains axes routiers majeurs ? Comment modéliser et prédire les reports de charge sur les axes adjacents ?

Ce projet exploite les données ouvertes de comptage routier de la Ville de Paris pour analyser les flux de trafic, identifier les axes critiques, et simuler l'effet de fermetures de voies sur la redistribution du trafic.

📋 Table des matières
Contexte & Objectifs
Données
Architecture du projet
Méthodologie
Utilisation
Dashboard Streamlit
Résultats
Limites & Perspectives
Auteur
📖 Contexte & Objectifs
La Ville de Paris met à disposition des données de comptage routier en temps réel et historiques via sa plateforme Open Data. Ces données permettent de comprendre les dynamiques de flux de véhicules et de simuler des scénarios d'aménagement urbain.

PariData propose :

Collecter les données de comptage routier depuis l'Open Data Paris
Analyser les patterns de trafic (temporels, géographiques, saisonniers)
Modéliser les relations entre axes routiers pour prédire les reports de charge
Simuler la fermeture d'axes et estimer l'impact sur le réseau adjacent
Visualiser les résultats via un dashboard interactif Streamlit
📊 Données
Source principale
Source	Dataset	Format	Lien
Open Data Paris	Comptages routiers — Données de trafic	CSV	opendata.paris.fr
Variables clés
Variable	Description
id_compteur	Identifiant unique du compteur
nom_compteur	Nom / localisation du compteur
id_arc	Identifiant de l'arc routier
date_comptage	Date et heure du comptage
debit_horaire	Nombre de véhicules par heure
taux_occupation	Taux d'occupation de la voie (%)
coordonnees	Latitude / longitude du compteur
🏗 Architecture du projet
PariData/
├── data/
│   ├── raw/                          # Données brutes Open Data
│   └── processed/                    # Données nettoyées
├── notebooks/
│   └── PariData_Colab.ipynb          # Notebook complet (Google Colab)
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Collecte des données via API
│   ├── preprocessing.py              # Nettoyage et transformation
│   ├── features.py                   # Feature engineering
│   ├── model.py                      # Modélisation prédictive
│   └── simulator.py                  # Simulation de fermeture d'axes
├── streamlit_app/
│   └── app.py                        # Dashboard Streamlit
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
🔬 Méthodologie
1. Collecte & Ingestion
API REST Open Data Paris (téléchargement automatisé)
Données de comptage horaire sur les principaux axes parisiens
2. Analyse Exploratoire (EDA)
Distribution du trafic par heure, jour, mois
Identification des axes les plus chargés
Corrélations spatiales entre compteurs proches
3. Modélisation Prédictive
Random Forest et Gradient Boosting pour prédire le débit horaire
Features : heure, jour de semaine, mois, météo, vacances scolaires
Validation croisée et comparaison des performances
4. Simulation de fermeture d'axes
Sélection d'un axe à fermer
Redistribution proportionnelle du trafic sur les axes adjacents
Estimation du facteur de surcharge et détection de congestion
🚀 Utilisation
Option 1 — Google Colab (recommandé)
Open In Colab

Le notebook Colab contient l'intégralité du pipeline : collecte, nettoyage, EDA, modélisation et simulation.

Option 2 — Installation locale
git clone https://github.com/yannis-cloud/PariData.git
cd PariData
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Lancer le notebook
jupyter notebook notebooks/

# Ou lancer le dashboard Streamlit
streamlit run streamlit_app/app.py
📊 Dashboard Streamlit
Le dashboard interactif permet de :

Visualiser le trafic en temps réel sur une carte de Paris
Sélectionner un axe à fermer
Voir l'impact simulé sur les axes adjacents
Comparer avant/après fermeture
streamlit run streamlit_app/app.py
⚠️ Limites & Perspectives
Limites
Modèle de redistribution simplifié : redistribution proportionnelle, ne prend pas en compte la topologie complète du réseau
Données historiques : pas de données temps-réel intégrées
Facteurs externes : météo, événements, travaux non pris en compte dans la simulation
Perspectives
Intégration d'un modèle de graphe (NetworkX) pour une simulation réaliste
Données temps-réel via l'API streaming
Prise en compte de la météo et des événements parisiens
Déploiement du dashboard sur Streamlit Cloud
👤 Auteur
Yannis ALBERT

📧 yannis.albert78@gmail.com
💼 LinkedIn
🐙 GitHub
🌐 Portfolio
📄 Licence
Ce projet est sous licence MIT. Voir LICENSE.
