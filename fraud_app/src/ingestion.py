from __future__ import annotations
import io
import pandas as pd
from typing import Tuple, Dict

def read_csv_safely(uploaded_file: io.BytesIO) -> pd.DataFrame:
    try:
        df = pd.read_csv(uploaded_file, engine="pyarrow")
        return df
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, low_memory=False)

def to_datetime(series: pd.Series) -> Tuple[pd.Series, int]:
    # infer_datetime_format est désormais le comportement par défaut → on l’enlève
    s = pd.to_datetime(series, errors="coerce", utc=False)
    return s, int(s.isna().sum())

def parse_amount(series: pd.Series) -> Tuple[pd.Series, int]:
    s = series.astype(str).str.strip()
    s = s.str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.replace("€", "", regex=False)
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    s = s.replace({"": None})
    f = pd.to_numeric(s, errors="coerce")
    return f, int(f.isna().sum())

def normalize_use_chip(series: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    s = series.astype(str).str.strip().str.lower()
    chip_flags = (s == "chip transaction") | (s == "chip") | (s == "1") | (s == "true")
    out = chip_flags.astype(int)
    return out, {"chip": int((out == 1).sum()), "non_chip": int((out == 0).sum())}

def to_str(series: pd.Series) -> pd.Series:
    return series.astype(str)

def clean_zip(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True)