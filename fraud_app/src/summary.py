from __future__ import annotations
import streamlit as st
import pandas as pd

def _fmt_dt(x, show_seconds: bool) -> str:
    fmt = "%Y-%m-%d %H:%M:%S" if show_seconds else "%Y-%m-%d"
    try:
        return pd.to_datetime(x).strftime(fmt)
    except Exception:
        return str(x) if x is not None else "—"

def _fmt_int(n) -> str:
    try:
        return f"{int(n):,}".replace(",", " ")
    except Exception:
        return str(n)

def _fmt_money(x: float) -> str:
    try:
        return f"{float(x):,.2f}".replace(",", " ")
    except Exception:
        return str(x)

def _widget(value: str, label: str, color_class: str) -> None:
    st.markdown(
        f"""
        <div class="widget {color_class}">
          <div class="value">{value}</div>
          <div class="label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_dataset_summary(
    df: pd.DataFrame,
    report: dict,
    *,
    show_seconds: bool = False,
    debug: bool = False
) -> None:
    info = report.get("info", {})
    period = info.get("period", {})
    amt = info.get("amount", {})
    neg_n = info.get("negatives", 0)

    rows = len(df)
    total_amt = float(df["amount"].sum(skipna=True))
    median_amt = (amt.get("50%") or 0)

    start = _fmt_dt(period.get("min"), show_seconds)
    end = _fmt_dt(period.get("max"), show_seconds)
    period_txt = f"{start} → {end}" if start != "—" and end != "—" else "—"

    # --- Widgets 2x2 (2 colonnes, plusieurs lignes) ---
    st.markdown('<div class="widgets-2col">', unsafe_allow_html=True)
    _widget(_fmt_int(rows), "Transactions", "c1")
    _widget(period_txt, "Période", "c2")
    _widget(_fmt_money(total_amt), "Montant total", "c3")
    _widget(_fmt_money(median_amt), "Montant médian", "c4")
    _widget(_fmt_int(neg_n), "Montants négatifs", "c5")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Unicités (petites cartes sobres, alignées) ---
    st.markdown('<div class="uniques">', unsafe_allow_html=True)
    st.markdown(f'<div class="unique">Clients uniques : {_fmt_int(df["client_id"].nunique())}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="unique">Cartes uniques : {_fmt_int(df["card_id"].nunique())}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="unique">Marchands uniques : {_fmt_int(df["merchant_id"].nunique())}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if debug:
        print("[SUMMARY] widgets affichés : rows/period/total/median/negatives + uniques")