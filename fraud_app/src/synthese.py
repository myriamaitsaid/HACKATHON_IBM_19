# src/synthese.py
from __future__ import annotations
from typing import Dict, Tuple, Optional, List
import math
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Plotly: style corporate dark ----------------
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}
PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_FONT = "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial"
# Palette discrète et cohérente avec l'app
ACCENT_1 = "#38BDF8"  # sky-400
ACCENT_2 = "#34D399"  # emerald-400
ACCENT_3 = "#A78BFA"  # violet-400
ACCENT_4 = "#F59E0B"  # amber-500
ACCENT_5 = "#F472B6"  # pink-400
NEUTRAL = "#94A3B8"    # slate-400
PLOTLY_MARGINS = dict(l=10, r=10, t=10, b=10)

def _style_fig(fig: go.Figure, *, height: int | None = None, x_title: str | None = None, y_title: str | None = None):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=dict(family=PLOTLY_FONT, size=13),
        margin=PLOTLY_MARGINS,
        height=height,
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    # Grilles très discrètes
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    return fig

def plot_sparkline(per_day: pd.Series):
    """Sparkline 14 jours area chart, dark premium."""
    if (not HAS_PLOTLY) or per_day is None or len(per_day) == 0:
        st.caption("—")
        return
    last_day = per_day.index.max()
    first_needed = last_day - pd.Timedelta(days=13)
    idx = pd.date_range(first_needed, last_day, freq="D")
    y = per_day.reindex(idx, fill_value=0).astype(int).values

    fig = go.Figure(go.Scatter(
        x=idx, y=y, mode="lines",
        line=dict(width=2, color=ACCENT_3),
        fill="tozeroy",
        fillcolor="rgba(167,139,250,0.25)",
        hovertemplate="%{x|%d/%m/%Y} • %{y} alerte(s)<extra></extra>",
    ))
    _style_fig(fig, height=86)
    fig.update_xaxes(title=None, tickformat="%d/%m", tickfont=dict(size=11))
    fig.update_yaxes(title=None, tickfont=dict(size=11))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

def plot_bar_top10(df: pd.DataFrame, x_col: str, y_col: str, title: str, *, height: int = 300, color: str = ACCENT_1):
    """Barres horizontales triées, top-10, lisibles."""
    if (not HAS_PLOTLY) or df is None or len(df) == 0:
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            st.dataframe(df, use_container_width=True)
        else:
            st.caption("—")
        return
    fig = go.Figure(go.Bar(
        x=df[x_col],
        y=df[y_col],
        orientation="h",
        marker=dict(color=color, line=dict(width=0)),
        hovertemplate="%{y} • %{x}<extra></extra>",
    ))
    _style_fig(fig, height=height, x_title=title, y_title=None)
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

# ---------------- Paramètres métier ----------------
EXPOSURE_ONLY_POSITIVE = True  # ignorer < 0€ pour "montant exposé"

# ---------------- Formatting FR ----------------
def fr_int(n) -> str:
    try:
        return f"{int(n):,}".replace(",", " ")
    except Exception:
        return str(n)

def fr_pct(x: float) -> str:
    try:
        v = float(x) * 100.0
        s = f"{v:.2f}"
        return s.replace(".", ",") + " %"
    except Exception:
        return "—"

def fr_eur(x: float) -> str:
    try:
        s = f"{float(x):,.2f}"
        s = s.replace(",", "X").replace(".", " ").replace("X", ",")
        return s + " €"
    except Exception:
        return "—"

def fmt_ratio(x: Optional[float], digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    try:
        return f"×{float(x):.{digits}f}"
    except Exception:
        return "—"

# ---------------- Normalisations ----------------
def _parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")

def _ensure_pred(df: pd.DataFrame, run_meta: Optional[Dict] = None) -> pd.DataFrame:
    """pred depuis fraud_prediction/target/label/is_fraud ; sinon score+threshold ; sinon 0."""
    df = df.copy()
    if "pred" in df.columns:
        return df
    for alt_name in ("fraud_prediction", "target", "label", "is_fraud"):
        if alt_name in df.columns:
            df["pred"] = df[alt_name].astype(int)
            return df
    if "score" in df.columns and run_meta and isinstance(run_meta.get("threshold"), (int, float)):
        thr = float(run_meta["threshold"])
        df["pred"] = (df["score"].astype(float) >= thr).astype(int)
        return df
    df["pred"] = 0
    return df

def _ensure_use_chip(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "use_chip_norm" in df.columns:
        df["use_chip"] = df["use_chip_norm"]
        return df
    if "use_chip" in df.columns:
        s = df["use_chip"].astype(str).str.strip().str.lower()
        chip = (s.isin(["1", "true", "chip", "chip transaction"])).astype(int)
        df["use_chip"] = chip
        return df
    df["use_chip"] = np.nan
    return df

def _exposure_series(df_alerts: pd.DataFrame) -> pd.Series:
    if "amount" not in df_alerts.columns:
        return pd.Series(dtype=float)
    s = df_alerts["amount"].astype(float)
    if EXPOSURE_ONLY_POSITIVE:
        s = s[s > 0]
    return s

# ---------------- Calculs métriques ----------------
def compute_metrics(df_view: pd.DataFrame) -> Dict[str, object]:
    total = len(df_view)
    alerts = int((df_view.get("pred", 0) == 1).sum())
    rate = alerts / total if total else 0.0

    df_alerts = df_view[df_view["pred"] == 1].copy()
    exposure = float(_exposure_series(df_alerts).sum()) if len(df_alerts) else 0.0

    mean_fraud = float(df_view.loc[df_view["pred"] == 1, "amount"].mean()) if alerts else float("nan")
    mean_ok = float(df_view.loc[df_view["pred"] == 0, "amount"].mean()) if (total - alerts) > 0 else float("nan")
    ratio_mean = (mean_fraud / mean_ok) if (not math.isnan(mean_fraud) and not math.isnan(mean_ok) and mean_ok != 0) else None

    # concentration top 10% marchands
    share_top10 = None
    if "merchant_id" in df_view.columns and alerts:
        expo_alerts = df_alerts.copy()
        if EXPOSURE_ONLY_POSITIVE:
            expo_alerts = expo_alerts[expo_alerts["amount"] > 0]
        grp = expo_alerts.groupby("merchant_id", as_index=False)["amount"].sum()
        if len(grp):
            grp = grp.sort_values("amount", ascending=False)
            m = len(grp)
            topk = max(1, int(math.ceil(0.10 * m)))
            top_sum = float(grp.head(topk)["amount"].sum())
            tot_sum = float(grp["amount"].sum())
            if tot_sum > 0:
                share_top10 = top_sum / tot_sum

    # risque puce
    chip_ratio = None
    if "use_chip" in df_view.columns:
        chip = df_view["use_chip"]
        if chip.notna().any():
            no = df_view[chip == 0]
            yes = df_view[chip == 1]
            r_no = (no["pred"] == 1).mean() if len(no) else np.nan
            r_yes = (yes["pred"] == 1).mean() if len(yes) else np.nan
            if not (np.isnan(r_no) or np.isnan(r_yes) or r_yes == 0):
                chip_ratio = float(r_no / r_yes)

    # alerts/day + spike
    df_view = df_view.copy()
    if "date" in df_view.columns:
        df_view["date_parsed"] = _parse_date(df_view["date"])
    alerts_by_day = df_view.loc[df_view["pred"] == 1].copy()
    if "date_parsed" in alerts_by_day.columns:
        alerts_by_day["day"] = alerts_by_day["date_parsed"].dt.floor("D")
        per_day = alerts_by_day.groupby("day").size().sort_index()
    else:
        per_day = pd.Series(dtype=int)
    spike = None
    if len(per_day):
        med = float(per_day.median())
        mx = float(per_day.max())
        if med > 0 and mx / med >= 3.0:
            spike_day = per_day.idxmax()
            spike = {"day": spike_day, "ratio": mx / med}

    # top-3 MCC
    top3_mcc = []
    if "mcc" in df_view.columns and alerts:
        mcc_counts = df_alerts["mcc"].astype(str).value_counts().head(3)
        top3_mcc = list(mcc_counts.index)

    return dict(
        total=total, alerts=alerts, rate=rate, exposure=exposure,
        mean_fraud=mean_fraud, mean_ok=mean_ok, ratio_mean=ratio_mean,
        share_top10=share_top10, chip_ratio=chip_ratio, per_day=per_day, spike=spike,
        top3_mcc=top3_mcc
    )

# ---------------- Filtres ----------------
def apply_filters(
    df_result: pd.DataFrame,
    *,
    period: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]],
    states: List[str],
    cities: List[str],
    mccs: List[str],
    puce_mode: str
) -> pd.DataFrame:
    df = df_result.copy()
    if "date" in df.columns:
        df["date_parsed"] = _parse_date(df["date"])
    else:
        df["date_parsed"] = pd.NaT

    mmin, mmax = period
    if mmin is not None:
        df = df[df["date_parsed"] >= mmin]
    if mmax is not None:
        df = df[df["date_parsed"] <= mmax]

    if states and "merchant_state" in df.columns:
        df = df[df["merchant_state"].astype(str).isin(states)]
    if cities and "merchant_city" in df.columns:
        df = df[df["merchant_city"].astype(str).isin(cities)]
    if mccs and "mcc" in df.columns:
        df = df[df["mcc"].astype(str).isin(mccs)]

    if puce_mode == "Avec puce":
        df = df[df["use_chip"] == 1]
    elif puce_mode == "Sans puce":
        df = df[df["use_chip"] == 0]

    return df

# ---------------- Rendu principal Page 2 ----------------
def render_synthese(df_result: pd.DataFrame, run_meta: Optional[Dict] = None) -> None:
    """
    df_result : df normalisé avec 'pred' (et 'score' facultatif)
    run_meta  : {model_version, threshold, trained_on, run_id, file_hash, exported_at}
    """
    # normalisations soft
    df = _ensure_pred(df_result, run_meta)
    df = _ensure_use_chip(df)
    if "date" in df.columns:
        df["date_parsed"] = _parse_date(df["date"])

    # ---------- Barre de filtres ----------
    st.subheader("Synthèse du risque")
    st.caption("Aide à la décision — une revue humaine reste nécessaire.")

    dmin = pd.to_datetime(df["date_parsed"]).min() if "date_parsed" in df.columns else None
    dmax = pd.to_datetime(df["date_parsed"]).max() if "date_parsed" in df.columns else None

    with st.container():
        c1, c2 = st.columns([3, 2])
        with c1:
            period = st.date_input(
                "Période",
                value=(None if pd.isna(dmin) else dmin.date(), None if pd.isna(dmax) else dmax.date()),
                min_value=None,
                max_value=None
            )
        with c2:
            puce_mode = st.selectbox("Puce", ["Tous", "Avec puce", "Sans puce"], index=0)

        c3, c4, c5 = st.columns(3)
        states_opts = sorted(df["merchant_state"].dropna().astype(str).unique().tolist()) if "merchant_state" in df.columns else []
        cities_opts = sorted(df["merchant_city"].dropna().astype(str).unique().tolist()) if "merchant_city" in df.columns else []
        mcc_opts = sorted(df["mcc"].dropna().astype(str).unique().tolist()) if "mcc" in df.columns else []

        with c3:
            states = st.multiselect("États", states_opts, default=[])
        with c4:
            cities = st.multiselect("Villes", cities_opts, default=[])
        with c5:
            mccs = st.multiselect("MCC", mcc_opts, default=[])

        cc1, cc2, _ = st.columns([1, 1, 6])
        with cc1:
            apply_btn = st.button("Appliquer", type="primary")
        with cc2:
            reset_btn = st.button("Réinitialiser")

    if reset_btn:
        states, cities, mccs = [], [], []
        puce_mode = "Tous"
        period = (None if pd.isna(dmin) else dmin.date(), None if pd.isna(dmax) else dmax.date())

    pmin = pd.to_datetime(period[0]) if isinstance(period, (tuple, list)) and period[0] else None
    pmax = pd.to_datetime(period[1]) if isinstance(period, (tuple, list)) and period[1] else None

    df_view = apply_filters(df, period=(pmin, pmax), states=states, cities=cities, mccs=mccs, puce_mode=puce_mode)
    st.caption(f"Périmètre courant : {len(df_view):,}".replace(",", " ") + " transactions")

    # ---------- KPIs rangée 1 (widgets 2x2) ----------
    met = compute_metrics(df_view)
    st.markdown('<div class="widgets-2col">', unsafe_allow_html=True)
    st.markdown(f'<div class="widget c1"><div class="value">{fr_int(met["total"])}</div><div class="label">Transactions analysées</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="widget c2"><div class="value">{fr_int(met["alerts"])}</div><div class="label">Alertes détectées</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="widget c3"><div class="value">{fr_pct(met["rate"])}</div><div class="label">Taux d’alerte</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="widget c4"><div class="value">{fr_eur(met["exposure"])}</div><div class="label">Montant exposé</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- KPIs rangée 2 (3 widgets + sparkline Plotly) ----------
    cA, cB = st.columns([2, 1])
    with cA:
        st.markdown('<div class="widgets-2col">', unsafe_allow_html=True)
        st.markdown(f'<div class="widget c5"><div class="value">{fmt_ratio(met["ratio_mean"])}</div><div class="label">Montant moyen fraude / normal</div></div>', unsafe_allow_html=True)

        share_txt = f"Top 10% → {fr_pct(met['share_top10'])}" if met["share_top10"] is not None else "—"
        st.markdown(f'<div class="widget c1"><div class="value">{share_txt}</div><div class="label">Indice de concentration</div></div>', unsafe_allow_html=True)

        chip_txt = fmt_ratio(met["chip_ratio"], 1) if met["chip_ratio"] is not None else "—"
        st.markdown(f'<div class="widget c2"><div class="value">{chip_txt}</div><div class="label">Sans puce : risque relatif</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with cB:
        st.markdown('<div class="widget c3">', unsafe_allow_html=True)
        st.markdown('<div class="label">Alertes (14 derniers jours)</div>', unsafe_allow_html=True)
        plot_sparkline(met["per_day"])
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Badge de pic ----------
    if met["spike"] is not None:
        day = met["spike"]["day"]
        ratio = met["spike"]["ratio"]
        st.info(f"Pic d’alertes le {day.strftime('%d/%m/%Y')} (×{ratio:.1f} vs médiane)")

    # ---------- Visuels principaux (Plotly) ----------
    st.markdown("---")
    m1, m2 = st.columns(2)

    # MCC — mesure
    with m1:
        st.markdown("**MCC (Top 10)**")
        measure = st.radio("Mesure", ["Nombre d’alertes", "Montant exposé"], horizontal=True, key="mcc_measure")
        if "mcc" not in df_view.columns:
            st.caption("—")
        else:
            df_alerts = df_view[df_view["pred"] == 1].copy()
            if measure == "Nombre d’alertes":
                agg = df_alerts.groupby(df_alerts["mcc"].astype(str)).size().sort_values(ascending=False).head(10)
                data = pd.DataFrame({"MCC": agg.index, "Valeur": agg.values})
                plot_bar_top10(data, x_col="Valeur", y_col="MCC", title="Nombre d’alertes", color=ACCENT_1)
            else:
                s = _exposure_series(df_alerts)
                df_expo = df_alerts.loc[s.index]
                agg = df_expo.groupby(df_expo["mcc"].astype(str))["amount"].sum().sort_values(ascending=False).head(10)
                data = pd.DataFrame({"MCC": agg.index, "Montant": agg.values})
                plot_bar_top10(data, x_col="Montant", y_col="MCC", title="Montant exposé", color=ACCENT_4)

    # Zone — état/ville par montant exposé
    with m2:
        st.markdown("**Zone (Top 10 par montant exposé)**")
        dim = st.radio("Dimension", ["État", "Ville"], horizontal=True, key="zone_dim")
        key = "merchant_state" if dim == "État" else "merchant_city"
        if key not in df_view.columns:
            st.caption("—")
        else:
            df_alerts = df_view[df_view["pred"] == 1].copy()
            s = _exposure_series(df_alerts)
            df_expo = df_alerts.loc[s.index]
            agg = df_expo.groupby(df_expo[key].astype(str))["amount"].sum().sort_values(ascending=False).head(10)
            data = pd.DataFrame({"Zone": agg.index, "Montant": agg.values})
            plot_bar_top10(data, x_col="Montant", y_col="Zone", title="Montant exposé", color=ACCENT_2)

    # ---------- Insight cards ----------
    st.markdown("---")
    top3 = ", ".join(map(str, met["top3_mcc"])) if met["top3_mcc"] else "—"
    conc = f"Top 10% marchands = {fr_pct(met['share_top10'])}" if met["share_top10"] is not None else "—"
    chip_i = f"Sans puce : {fmt_ratio(met['chip_ratio'], 1)}" if met["chip_ratio"] is not None else "Sans puce : —"

    i1, i2, i3 = st.columns(3)
    i1.info(chip_i)
    i2.info(f"Top-3 MCC : {top3}")
    i3.info(conc)

    # ---------- Narrative (3 phrases) ----------
    s1 = f"Sur le périmètre courant, {fr_int(met['total'])} transactions analysées dont {fr_int(met['alerts'])} alertes, soit {fr_pct(met['rate'])}."
    s2 = f"La concentration des risques est de {fr_pct(met['share_top10'])} pour le top 10% des marchands." if met["share_top10"] is not None else "La concentration des risques est faible ou non mesurable."
    if met["spike"] is not None:
        s3 = f"Un pic d’alertes a été détecté le {met['spike']['day'].strftime('%d/%m/%Y')} (×{met['spike']['ratio']:.1f}); exposition estimée à {fr_eur(met['exposure'])}."
    elif met["chip_ratio"] is not None:
        s3 = f"Sans puce, le risque est {fmt_ratio(met['chip_ratio'], 1)} ; exposition estimée à {fr_eur(met['exposure'])}."
    else:
        s3 = f"Exposition estimée à {fr_eur(met['exposure'])}."
    st.write(s1 + " " + s2 + " " + s3)

    # ---------- Actions ----------
    cta1, cta2 = st.columns([1, 2])
    with cta1:
        st.button("➡️ Voir les alertes détaillées", key="go_page3", help="Navigation vers Page 3 (liste & tri des alertes)")

    # Export synthèse CSV clé/valeur
    synth_rows = [
        ("Transactions analysées", fr_int(met["total"])),
        ("Alertes détectées", fr_int(met["alerts"])),
        ("Taux d’alerte", fr_pct(met["rate"])),
        ("Montant exposé", fr_eur(met["exposure"])),
        ("Ratio montant fraude/normal", fmt_ratio(met["ratio_mean"])),
        ("Indice de concentration (Top 10%)", fr_pct(met["share_top10"]) if met["share_top10"] is not None else "—"),
        ("Sans puce (risque relatif)", fmt_ratio(met["chip_ratio"], 1) if met["chip_ratio"] is not None else "—"),
        ("Top-3 MCC", top3),
    ]
    if run_meta:
        for k in ("model_version", "threshold", "trained_on", "run_id", "exported_at"):
            if k in run_meta:
                synth_rows.append((f"meta.{k}", str(run_meta[k])))

    synth_df = pd.DataFrame(synth_rows, columns=["clé", "valeur"])
    csv_bytes = synth_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("💾 Exporter la synthèse (CSV)", data=csv_bytes, file_name="synthese_risque.csv", mime="text/csv", type="primary")