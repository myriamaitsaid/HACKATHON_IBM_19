from __future__ import annotations
from typing import Dict, Any
import pandas as pd
from .ingestion import to_datetime, parse_amount, normalize_use_chip, to_str, clean_zip

def validate_and_normalize(df_raw: pd.DataFrame, schema: Dict[str, Any]):
    """
    Valide le schéma et normalise les colonnes clés.
    Retourne (df_norm, report) où report = {errors[], warnings[], info{}}.
    """

    report = {"errors": [], "warnings": [], "info": {}}
    df = df_raw.copy()

    # 1) Colonnes obligatoires
    required = schema.get("required_columns", [])
    missing = [c for c in required if c not in df.columns]
    extra = [c for c in df.columns if c not in required]
    if missing:
        report["errors"].append(f"Colonnes manquantes : {missing}")
        return df_raw, report
    if extra:
        report["warnings"].append(f"Colonnes supplémentaires détectées : {extra} (tolérées)")

    # 2) Casts/normalisation
    for col in ["transaction_id", "client_id", "card_id", "merchant_id", "merchant_city", "merchant_state", "zip", "mcc"]:
        try:
            df[col] = to_str(df[col])
            if col == "zip":
                df[col] = clean_zip(df[col])
        except Exception as e:
            report["warnings"].append(f"{col} : cast string partiel ({e})")

    df["date"], n_bad_dates = to_datetime(df["date"])
    if n_bad_dates:
        report["warnings"].append(f"date : {n_bad_dates} valeur(s) non parsée(s)")

    df["amount"], n_unparsable_amount = parse_amount(df["amount"])
    if n_unparsable_amount:
        report["warnings"].append(f"amount : {n_unparsable_amount} montant(s) non parsé(s) (NaN)")

    df["use_chip_norm"], chip_counts = normalize_use_chip(df["use_chip"])
    report["info"]["use_chip_counts"] = chip_counts

    # 3) Nulls critiques / unicité
    nullable = set(schema.get("nullable", []))
    nulls = df[required].isna().sum().to_dict()
    hard_nulls = {k: int(v) for k, v in nulls.items() if v > 0 and k not in nullable}
    if hard_nulls:
        report["warnings"].append(f"Valeurs manquantes détectées : {hard_nulls}")

    dup = int(df["transaction_id"].duplicated().sum())
    if dup > 0:
        report["errors"].append(f"transaction_id non unique : {dup} doublon(s)")

    # 4) Période (timestamp complet)
    date_min = pd.to_datetime(df["date"], errors="coerce").min()
    date_max = pd.to_datetime(df["date"], errors="coerce").max()
    report["info"]["period"] = {
        "min": None if pd.isna(date_min) else date_min.strftime("%Y-%m-%d %H:%M:%S"),
        "max": None if pd.isna(date_max) else date_max.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 5) Stats montants
    amt_desc = df["amount"].describe(percentiles=[0.5, 0.9, 0.99]).to_dict()
    report["info"]["amount"] = {k: (None if pd.isna(v) else float(v)) for k, v in amt_desc.items()}
    report["info"]["negatives"] = int((df["amount"] < 0).sum())



    return df, report