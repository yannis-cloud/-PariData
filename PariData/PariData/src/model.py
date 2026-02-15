"""
model.py — Modélisation prédictive du trafic routier
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

MODELS = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
}


def train_and_evaluate(X, y, model_name="random_forest", test_size=0.2, cv_folds=5):
    """
    Entraîne un modèle et retourne les résultats.
    """
    if model_name not in MODELS:
        raise ValueError(f"Modèle inconnu : {model_name}")

    model = MODELS[model_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Entraînement
    model.fit(X_train_s, y_train)

    y_pred_train = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)

    metrics = {
        "model": model_name,
        "train_rmse": round(np.sqrt(mean_squared_error(y_train, y_pred_train)), 1),
        "test_rmse": round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 1),
        "train_mae": round(mean_absolute_error(y_train, y_pred_train), 1),
        "test_mae": round(mean_absolute_error(y_test, y_pred_test), 1),
        "train_r2": round(r2_score(y_train, y_pred_train), 4),
        "test_r2": round(r2_score(y_test, y_pred_test), 4),
    }

    # Validation croisée
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=cv_folds, scoring="r2")
    metrics["cv_r2_mean"] = round(cv_scores.mean(), 4)
    metrics["cv_r2_std"] = round(cv_scores.std(), 4)

    print(f"\n📊 {model_name}")
    print(f"   Test RMSE: {metrics['test_rmse']}  |  Test R²: {metrics['test_r2']}")
    print(f"   CV R²: {metrics['cv_r2_mean']} ± {metrics['cv_r2_std']}")

    return {"model": model, "scaler": scaler, "metrics": metrics, "predictions": y_pred_test, "y_test": y_test}


def compare_models(X, y):
    """Compare tous les modèles et retourne un tableau récapitulatif."""
    results = []
    best = None

    for name in MODELS:
        result = train_and_evaluate(X, y, model_name=name)
        results.append(result["metrics"])
        if best is None or result["metrics"]["test_r2"] > best["metrics"]["test_r2"]:
            best = result

    comparison = pd.DataFrame(results).sort_values("test_r2", ascending=False)
    print("\n📋 Comparaison :")
    print(comparison[["model", "test_rmse", "test_r2", "cv_r2_mean"]].to_string(index=False))

    return comparison, best


def get_feature_importance(model, feature_names):
    """Extrait l'importance des features."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    else:
        return pd.DataFrame()

    fi = pd.DataFrame({"feature": feature_names, "importance": imp})
    fi = fi.sort_values("importance", ascending=False)
    fi["importance_pct"] = (fi["importance"] / fi["importance"].sum() * 100).round(2)
    return fi.head(15)
