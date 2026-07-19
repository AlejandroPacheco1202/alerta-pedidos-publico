"""Fase 4b — Entrenamiento y comparación de modelos.

Entrena regresión logística y gradient boosting sobre el panel, con split
temporal, y los compara contra el baseline (tasa por día de semana sola)
en las métricas del proyecto: AUC-ROC, AUC-PR y Precision@15 diaria.

Uso:
    python src/modelo.py           # entrena, evalúa, guarda el mejor modelo

Prefiere XGBoost si está instalado; si no, usa HistGradientBoosting de
scikit-learn (mismo tipo de modelo, resultados comparables).
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import config
    from features import FEATURES
except ImportError:
    from src import config
    from src.features import FEATURES

FIN_TRAIN = "2026-03-31"
FIN_VAL = "2026-05-31"
RUTA_MODELO = None  # se define en main() -> data/processed/modelo.pkl


def crear_boosting():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, learning_rate=0.08, max_depth=6,
                             scale_pos_weight=4.7, eval_metric="logloss",
                             random_state=42), "XGBoost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=6,
            class_weight="balanced", random_state=42), "HistGradBoost"


def precision_at_k(df_scores, col, k=15):
    """Promedio diario de: entre los top-k del día, % que pidió de verdad."""
    return np.mean([g.nlargest(k, col)["target"].mean()
                    for _, g in df_scores.groupby("Fecha")])


def main():
    panel = pd.read_parquet(config.DATA_PROCESSED / "panel.parquet")

    train = panel[panel["Fecha"] <= FIN_TRAIN]
    test = panel[panel["Fecha"] > FIN_VAL]
    print(f"Train: {len(train):,} filas (hasta {FIN_TRAIN})")
    print(f"Test:  {len(test):,} filas ({test['Fecha'].nunique()} días, "
          f"{test['Fecha'].min():%d/%m} a {test['Fecha'].max():%d/%m})")

    Xtr, ytr = train[FEATURES].fillna(0), train["target"]
    Xte, yte = test[FEATURES].fillna(0), test["target"]

    # Modelo lineal
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(sc.transform(Xtr), ytr)

    # Boosting
    gb, nombre_gb = crear_boosting()
    gb.fit(Xtr, ytr)

    scores = test[["Fecha", "target"]].copy()
    scores["baseline"] = Xte["tasa_dia_semana"].values
    scores["logistica"] = lr.predict_proba(sc.transform(Xte))[:, 1]
    scores["boosting"] = gb.predict_proba(Xte)[:, 1]

    print(f"\n== Comparación en test ==")
    filas = []
    for m, etiqueta in [("baseline", "Baseline (tasa día)"),
                        ("logistica", "Regresión logística"),
                        ("boosting", nombre_gb)]:
        filas.append({"modelo": etiqueta,
                      "AUC-ROC": roc_auc_score(yte, scores[m]),
                      "AUC-PR": average_precision_score(yte, scores[m]),
                      "P@15": precision_at_k(scores, m)})
    tabla = pd.DataFrame(filas).set_index("modelo").round(3)
    print(tabla.to_string())

    config.OUTPUTS.mkdir(exist_ok=True)
    tabla.to_csv(config.OUTPUTS / "comparacion_modelos.csv")

    # Persistir el modelo ganador + el scaler por si se usa la logística
    ruta = config.DATA_PROCESSED / "modelo.pkl"
    with open(ruta, "wb") as f:
        pickle.dump({"modelo": gb, "tipo": nombre_gb, "features": FEATURES,
                     "logistica": lr, "scaler": sc,
                     "entrenado_hasta": FIN_TRAIN}, f)
    print(f"\nModelo guardado: {ruta}")


if __name__ == "__main__":
    main()
