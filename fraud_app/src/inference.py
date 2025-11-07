from __future__ import annotations
from typing import Optional, List, Tuple, Dict
import joblib
import numpy as np
import pandas as pd

# ---------------- Chargement modèle disque ----------------
def load_joblib_model(path: str):
    return joblib.load(path)

# ---------------- Utils modèle ----------------
def _expected_features(model) -> Optional[List[str]]:
    for attr in ("feature_names_in_",):
        if hasattr(model, attr):
            try:
                return list(getattr(model, attr))
            except Exception:
                pass
    if hasattr(model, "named_steps"):  # Pipeline
        try:
            last_est = list(model.named_steps.values())[-1]
            if hasattr(last_est, "feature_names_in_"):
                return list(last_est.feature_names_in_)
        except Exception:
            pass
    return None

def _predict_proba_robust(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    proba = np.asarray(proba)
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1]
    return proba.reshape(-1)

# ---------------- Inférence principale ----------------
def run_inference(
    model,
    df_norm: pd.DataFrame,
    *,
    threshold: float = 0.50
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Retourne:
      df_pred: [transaction_id, fraud_probability, fraud_prediction]
      debug: infos (mode, used_columns, expected_features, missing, extra)
    """
    debug: Dict[str, object] = {"mode": None, "expected_features": None,
                                "used_columns": None, "missing": None, "extra": None}
    # 1) Essai direct (pipeline gère le brut)
    try:
        proba = _predict_proba_robust(model, df_norm)
        debug["mode"] = "direct_df"
        used_cols = list(df_norm.columns)
    except Exception:
        # 2) Essai avec expected features
        exp = _expected_features(model)
        debug["expected_features"] = exp
        if exp:
            missing = [c for c in exp if c not in df_norm.columns]
            extra = [c for c in df_norm.columns if c not in exp]
            debug["missing"] = missing
            debug["extra"] = extra
            X = df_norm.reindex(columns=exp)
            try:
                proba = _predict_proba_robust(model, X)
                debug["mode"] = "expected_cols"
                used_cols = exp
            except Exception:
                # 3) Fallback numérique
                Xnum = df_norm.select_dtypes(include=["number", "bool"])
                proba = _predict_proba_robust(model, Xnum)
                debug["mode"] = "numeric_fallback"
                used_cols = list(Xnum.columns)
        else:
            Xnum = df_norm.select_dtypes(include=["number", "bool"])
            proba = _predict_proba_robust(model, Xnum)
            debug["mode"] = "numeric_fallback"
            used_cols = list(Xnum.columns)

    debug["used_columns"] = used_cols

    out = pd.DataFrame({
        "transaction_id": df_norm["transaction_id"].astype(str).values,
        "fraud_probability": np.clip(proba, 0.0, 1.0),
    })
    out["fraud_prediction"] = (out["fraud_probability"] >= threshold).astype(int)
    return out, debug

# ---------------- Simulation (5% aléatoire, déterministe) ----------------
def simulate_predictions(
    df_norm: pd.DataFrame,
    *,
    rate: float = 0.05,
    seed: int = 2025
) -> pd.DataFrame:
    """
    Crée des scores/probas factices:
      - 'fraud_probability' ~ U(0,1)
      - 'fraud_prediction' = 1 pour ~rate des lignes (seuil trié)
    Déterministe via seed fixe (pour éviter les variations à chaque run).
    """
    n = len(df_norm)
    rng = np.random.default_rng(seed)
    probs = rng.random(n)
    # seuil tel que ~rate positifs
    if n > 0:
        k = max(1, int(round(rate * n)))
        thr_idx = np.argsort(probs)[-k]
        thr = probs[thr_idx]
    else:
        thr = 1.0
    pred = (probs >= thr).astype(int)
    out = pd.DataFrame({
        "transaction_id": df_norm["transaction_id"].astype(str).values,
        "fraud_probability": probs,
        "fraud_prediction": pred
    })
    return out