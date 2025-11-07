# src/investigation.py
from __future__ import annotations
from typing import Dict, Tuple, Optional, List
import io, zipfile, math
import numpy as np
import pandas as pd
import streamlit as st

# Plotly (optionnel)
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

PLOTLY_CFG = {"displayModeBar": False, "responsive": True}
ACCENT_1 = "#38BDF8"; ACCENT_2 = "#34D399"; ACCENT_3 = "#A78BFA"

# ---------------- Formatting FR ----------------
def fr_int(n) -> str:
    try:
        return f"{int(n):,}".replace(",", " ")
    except Exception:
        return str(n)

def fr_pct(x: float) -> str:
    try:
        v = float(x) * 100.0
        return f"{v:.2f}".replace(".", ",") + " %"
    except Exception:
        return "—"

def fr_eur(x: float) -> str:
    try:
        s = f"{float(x):,.2f}".replace(",", "X").replace(".", " ").replace("X", ",")
        return s + " €"
    except Exception:
        return "—"

def _parse_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def _ensure_pred(df: pd.DataFrame, run_meta: Optional[Dict] = None) -> pd.DataFrame:
    df = df.copy()
    if "pred" in df.columns:
        return df
    for alt in ("fraud_prediction", "target", "label", "is_fraud"):
        if alt in df.columns:
            df["pred"] = df[alt].astype(int)
            return df
    if "score" in df.columns and run_meta and isinstance(run_meta.get("threshold"), (int, float)):
        thr = float(run_meta["threshold"])
        df["pred"] = (df["score"].astype(float) >= thr).astype(int)
        return df
    df["pred"] = 0
    return df

def _ensure_use_chip(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "use_chip" in df.columns:
        s = df["use_chip"].astype(str).str.strip().str.lower()
        df["use_chip"] = (s.isin(["1", "true", "chip", "chip transaction"])).astype(int)
        return df
    if "use_chip_norm" in df.columns:
        df["use_chip"] = df["use_chip_norm"].astype(int)
        return df
    df["use_chip"] = np.nan
    return df

# ---------------- Filtres (mêmes logiques que Page 2) ----------------
def apply_filters(
    df_result: pd.DataFrame,
    *,
    period: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]],
    states: List[str], cities: List[str], mccs: List[str],
    puce_mode: str
) -> pd.DataFrame:
    df = df_result.copy()
    if "date" in df.columns:
        df["date_parsed"] = _parse_date(df["date"])
    else:
        df["date_parsed"] = pd.NaT

    pmin, pmax = period
    if pmin is not None:
        df = df[df["date_parsed"] >= pmin]
    if pmax is not None:
        df = df[df["date_parsed"] <= pmax]

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

# ---------------- Calcul “risk score éco” ----------------
def compute_risk_score(df_view: pd.DataFrame) -> pd.Series:
    amt_pos = df_view["amount"].astype(float).clip(lower=0)
    denom = float(amt_pos.max()) if len(amt_pos) else 0.0
    if "score" in df_view.columns:
        base = df_view["score"].astype(float) * amt_pos
    else:
        base = amt_pos
    if denom <= 0:
        return pd.Series(np.zeros(len(df_view)), index=df_view.index)
    return (base / denom) * 100.0

# ---------------- Reason codes (déterministes) ----------------
@st.cache_data(show_spinner=False)
def _p95_by_mcc(df_view: pd.DataFrame) -> Dict[str, float]:
    if "mcc" not in df_view.columns or "amount" not in df_view.columns:
        return {}
    q = (
        df_view.assign(mcc_str=df_view["mcc"].astype(str))
        .groupby("mcc_str")["amount"].quantile(0.95)
    )
    return q.to_dict()

@st.cache_data(show_spinner=False)
def _topk_alert_merchants(df_view: pd.DataFrame, k: int = 10) -> set:
    if "merchant_id" not in df_view.columns or "pred" not in df_view.columns:
        return set()
    counts = (
        df_view[df_view["pred"] == 1]
        .groupby("merchant_id").size()
        .sort_values(ascending=False)
        .head(k)
    )
    return set(counts.index.astype(str))

@st.cache_data(show_spinner=False)
def _rare_locations(df_view: pd.DataFrame, q_low: float = 0.10) -> Dict[str, set]:
    rare_states, rare_cities = set(), set()
    if "merchant_state" in df_view.columns:
        cs = df_view["merchant_state"].astype(str).value_counts()
        thr = cs.quantile(q_low) if len(cs) else 0
        rare_states = set(cs[cs <= max(thr, 1)].index.astype(str))
    if "merchant_city" in df_view.columns:
        cc = df_view["merchant_city"].astype(str).value_counts()
        thr = cc.quantile(q_low) if len(cc) else 0
        rare_cities = set(cc[cc <= max(thr, 1)].index.astype(str))
    return {"states": rare_states, "cities": rare_cities}



# ---------------- Petits graphes optionnels (Plotly) ----------------
def plot_bar(values: Dict[str, float], title: str):
    if not HAS_PLOTLY or not values:
        st.write(pd.DataFrame({"clé": list(values.keys()), "valeur": list(values.values())}))
        return
    x = list(values.keys()); y = list(values.values())
    # Palette simple, cyclée si >3 barres
    colors = ( [ACCENT_1, ACCENT_2, ACCENT_3] * ((len(x) // 3) + 1) )[:len(x)]
    fig = go.Figure(go.Bar(x=x, y=y, marker=dict(color=colors)))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=260, title=title, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

# ---------------- Export helpers ----------------
def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

def build_investigator_zip(payloads: Dict[str, pd.DataFrame], summary_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, df in payloads.items():
            zf.writestr(path, _df_to_csv_bytes(df))
        zf.writestr("summary.txt", summary_text)
    buf.seek(0)
    return buf.read()

# ---------------- Rendu principal Page 3 ----------------
def render_investigation(df_result: pd.DataFrame, run_meta: Optional[Dict] = None) -> None:
    # Normalisations
    df = _ensure_pred(_ensure_use_chip(df_result), run_meta)
    if "date" in df.columns:
        df["date_parsed"] = _parse_date(df["date"])

    # ---------- Bandeau filtres ----------
    st.subheader("Investigation & Export")
    st.caption("Passer de la synthèse à l’action — interface métier, sobre et efficace.")

    # Dates par défaut stables
    if "date_parsed" in df.columns and df["date_parsed"].notna().any():
        dmin = pd.to_datetime(df["date_parsed"]).min().normalize()
        dmax = pd.to_datetime(df["date_parsed"]).max().normalize()
    else:
        today = pd.Timestamp.today().normalize()
        dmin, dmax = today - pd.Timedelta(days=30), today

    c1, c2 = st.columns([3, 2])
    with c1:
        period = st.date_input(
            "Période",
            value=(dmin.date(), dmax.date()),
            key="inv_period"  # <<< clé unique (évite le conflit avec la Page 2)
        )
    with c2:
        puce_mode = st.selectbox(
            "Puce",
            ["Tous", "Avec puce", "Sans puce"],
            index=0,
            key="inv_puce"  # <<< clé unique
        )

    c3, c4, c5 = st.columns(3)
    states_opts = sorted(df["merchant_state"].dropna().astype(str).unique().tolist()) if "merchant_state" in df.columns else []
    cities_opts = sorted(df["merchant_city"].dropna().astype(str).unique().tolist()) if "merchant_city" in df.columns else []
    mcc_opts    = sorted(df["mcc"].dropna().astype(str).unique().tolist()) if "mcc" in df.columns else []
    with c3:
        states = st.multiselect("États", states_opts, default=[], key="inv_states")   # <<< clé unique
    with c4:
        cities = st.multiselect("Villes", cities_opts, default=[], key="inv_cities") # <<< clé unique
    with c5:
        mccs   = st.multiselect("MCC", mcc_opts, default=[], key="inv_mccs")         # <<< clé unique

    # Périmètre
    pmin = pd.to_datetime(period[0]) if isinstance(period, (tuple, list)) and period[0] else None
    pmax = pd.to_datetime(period[1]) if isinstance(period, (tuple, list)) and period[1] else None
    df_view = apply_filters(df, period=(pmin, pmax), states=states, cities=cities, mccs=mccs, puce_mode=puce_mode)
    st.caption(f"Périmètre courant : {fr_int(len(df_view))} transactions")

    # ---------- Zone recherche & options ----------
    q = st.text_input(
        "Recherche (transaction_id, merchant_id, mcc, ville, état)",
        "",
        key="inv_search"  # <<< clé unique
    )
    col_opt1, col_opt2, col_opt3 = st.columns([1.6, 1.8, 1])
    with col_opt1:
        show_all = st.toggle("Montrer toutes les transactions", value=False, key="inv_show_all")  # <<< clé unique
    with col_opt2:
        show_sensitive = st.toggle("Colonnes sensibles (client_id, card_id, score)", value=False, key="inv_show_sensitive")  # <<< clé
    with col_opt3:
        page_size = st.selectbox("Taille", [50, 200, 1000], index=0, key="inv_page_size")  # <<< clé unique

    df_list = df_view.copy()
    if not show_all:
        df_list = df_list[df_list["pred"] == 1]

    # Recherche plein-texte
    if q.strip():
        qlow = q.strip().lower()
        cols = ["transaction_id", "merchant_id", "mcc", "merchant_city", "merchant_state"]
        cols = [c for c in cols if c in df_list.columns]
        if cols:
            mask = pd.Series(False, index=df_list.index)
            for c in cols:
                mask = mask | df_list[c].astype(str).str.lower().str.contains(qlow, na=False)
            df_list = df_list[mask]

    # Risk score éco + reason codes
    df_list = df_list.copy()
    df_list["risk_score_econ"] = compute_risk_score(df_list)

    # Tri par défaut
    sort_cols = []
    if "risk_score_econ" in df_list.columns: sort_cols.append(("risk_score_econ", False))
    if "amount" in df_list.columns:          sort_cols.append(("amount", False))
    if "date_parsed" in df_list.columns:     sort_cols.append(("date_parsed", False))
    if sort_cols:
        df_list = df_list.sort_values([c for c, _ in sort_cols], ascending=[asc for _, asc in sort_cols])

    # ---------- Table ----------
    if df_list.empty:
        st.info("Aucune alerte sur ce périmètre. Élargissez la période ou ajustez les filtres.")
    else:
        base_cols = ["transaction_id", "date", "amount", "merchant_id", "merchant_city", "merchant_state", "mcc", "use_chip"]
        sens_cols = [c for c in ["client_id", "card_id", "score", "risk_score_econ"] if c in df_list.columns]
        show_cols = [c for c in base_cols if c in df_list.columns] + (sens_cols if show_sensitive else [])

        view = df_list[show_cols].copy()
        if "date" in view.columns:
            dt = _parse_date(view["date"])
            view["date"] = dt.dt.strftime("%d/%m/%Y %H:%M").fillna("")
        if "amount" in view.columns:
            view["amount"] = view["amount"].apply(fr_eur)
        if "use_chip" in view.columns:
            view["use_chip"] = view["use_chip"].map({1: "Avec puce", 0: "Sans puce"}).fillna("—")
        if show_sensitive and "score" in view.columns:
            view["score"] = pd.to_numeric(view["score"], errors="coerce").round(4)
        if show_sensitive and "risk_score_econ" in view.columns:
            view["risk_score_econ"] = pd.to_numeric(view["risk_score_econ"], errors="coerce").round(2)

        st.dataframe(view.head(page_size), use_container_width=True)

        # ---------- Sélection d’une alerte ----------
        tx_ids = df_list["transaction_id"].astype(str).tolist()
        sel = st.selectbox("Transaction à inspecter", options=["—"] + tx_ids, index=0, key="inv_tx_select")  # <<< clé unique
        if sel != "—":
            row = df_list[df_list["transaction_id"].astype(str) == sel].head(1)
            if not row.empty:
                r = row.iloc[0]
                st.markdown("---")
                st.subheader("Focus alerte")
                cA, cB, cC, cD = st.columns(4)
                with cA: st.metric("Montant", fr_eur(r.get("amount", 0.0)))
                with cB: st.metric("MCC", str(r.get("mcc", "—")))
                with cC: st.metric("Puce", "Avec" if int(r.get("use_chip", 0)) == 1 else "Sans")
                with cD:
                    dt = _parse_date(pd.Series([r.get("date")])).iloc[0]
                    st.metric("Date", dt.strftime("%d/%m/%Y %H:%M") if pd.notna(dt) else "—")

                st.caption("Raisons")
                st.write(r.get("reason", ""))

                st.markdown("**Mini-profils**")
                c1, c2 = st.columns(2)
                if "client_id" in df_list.columns and pd.notna(r.get("client_id", np.nan)):
                    rid = r["client_id"]
                    cli = df_view[df_view["client_id"] == rid]
                    nb_tx = len(cli); nb_alert = int((cli["pred"] == 1).sum()) if "pred" in cli.columns else 0
                    tot_amt = float(cli["amount"].sum()) if "amount" in cli.columns else 0.0
                    c1.write(f"Client `{rid}` — tx: {fr_int(nb_tx)}, alertes: {fr_int(nb_alert)}, montant total: {fr_eur(tot_amt)}")
                    if "date" in cli.columns:
                        tl = cli.sort_values("date_parsed").tail(5)[["date", "amount", "use_chip"]].copy()
                        tl["date"] = _parse_date(tl["date"]).dt.strftime("%d/%m/%Y %H:%M")
                        tl["amount"] = tl["amount"].apply(fr_eur)
                        tl["use_chip"] = tl["use_chip"].map({1:"Avec",0:"Sans"})
                        c1.dataframe(tl, use_container_width=True, height=180)
                else:
                    c1.info("Aucun identifiant client disponible.")

                if "merchant_id" in df_list.columns and pd.notna(r.get("merchant_id", np.nan)):
                    mid = r["merchant_id"]
                    mch = df_view[df_view["merchant_id"] == mid]
                    nb_tx_m = len(mch)
                    nb_alert_m = int((mch["pred"] == 1).sum()) if "pred" in mch.columns else 0
                    expo_m = float(mch.loc[(mch["pred"] == 1) & (mch["amount"] > 0), "amount"].sum()) if "amount" in mch.columns and "pred" in mch.columns else 0.0
                    expo_tot = float(df_view.loc[(df_view["pred"] == 1) & (df_view["amount"] > 0), "amount"].sum()) if "amount" in df_view.columns and "pred" in df_view.columns else 0.0
                    share = (expo_m / expo_tot) if expo_tot > 0 else np.nan
                    mcc_dom = str(mch["mcc"].astype(str).value_counts().idxmax()) if "mcc" in mch.columns and len(mch) else "—"
                    c2.write(f"Marchand `{mid}` — tx: {fr_int(nb_tx_m)}, alertes: {fr_int(nb_alert_m)}, part exposée: {fr_pct(share) if not np.isnan(share) else '—'}, MCC dominant: {mcc_dom}")
                    if "date" in mch.columns:
                        tl2 = mch.sort_values("date_parsed").tail(5)[["date", "amount", "use_chip"]].copy()
                        tl2["date"] = _parse_date(tl2["date"]).dt.strftime("%d/%m/%Y %H:%M")
                        tl2["amount"] = tl2["amount"].apply(fr_eur)
                        tl2["use_chip"] = tl2["use_chip"].map({1:"Avec",0:"Sans"})
                        c2.dataframe(tl2, use_container_width=True, height=180)
                else:
                    c2.info("Aucun identifiant marchand disponible.")

                st.button("🧩 Ajouter au pack d’enquête", key="inv_add_to_pack")

    # ---------- Analyses ciblées (repliées) ----------
    with st.expander("Puce → risque (taux d’alerte)"):
        if "use_chip" in df_view.columns and "pred" in df_view.columns:
            grp = df_view.groupby("use_chip")["pred"].mean()
            vals = {"Avec puce": grp.get(1, np.nan), "Sans puce": grp.get(0, np.nan)}
            v_yes = vals.get("Avec puce", np.nan)
            v_no  = vals.get("Sans puce", np.nan)
            ratio_txt = f"×{(v_no / v_yes):.2f}" if pd.notna(v_yes) and v_yes > 0 and pd.notna(v_no) else "—"
            st.write(f"Sans puce : {ratio_txt}")
            plot_bar({k: float(v) if pd.notna(v) else 0.0 for k, v in vals.items()}, title="Taux d’alerte")
        else:
            st.caption("—")

    with st.expander("Concentration marchands (top-10 par montant exposé)"):
        if "merchant_id" in df_view.columns and "pred" in df_view.columns and "amount" in df_view.columns:
            alerts = df_view[(df_view["pred"] == 1) & (df_view["amount"] > 0)]
            top = (alerts.groupby("merchant_id")["amount"].sum().sort_values(ascending=False).head(10)).reset_index()
            top.columns = ["merchant_id", "montant_expose"]
            st.dataframe(top, use_container_width=True, height=280)
        else:
            st.caption("—")

    with st.expander("Top 5 risques économiques"):
        if "risk_score_econ" in df_list.columns:
            top5 = df_list.sort_values("risk_score_econ", ascending=False).head(5)[
                [c for c in ["transaction_id", "amount", "merchant_state", "risk_score_econ"] if c in df_list.columns]
            ].copy()
            if "amount" in top5.columns:
                top5["amount"] = top5["amount"].apply(fr_eur)
            top5["risk_score_econ"] = pd.to_numeric(top5["risk_score_econ"], errors="coerce").round(2)
            st.dataframe(top5, use_container_width=True, height=220)
        else:
            st.caption("—")

    # ---------- Exports ----------
    st.markdown("---")
    st.subheader("Exports")

    alerts_only_cols = ["transaction_id","date","amount","use_chip","merchant_id","merchant_city","merchant_state","zip","mcc","pred","reason"]
    if "score" in df_list.columns and show_sensitive: alerts_only_cols.append("score")
    if "risk_score_econ" in df_list.columns and show_sensitive: alerts_only_cols.append("risk_score_econ")
    if "client_id" in df_list.columns and show_sensitive: alerts_only_cols.append("client_id")
    if "card_id" in df_list.columns and show_sensitive: alerts_only_cols.append("card_id")
    alerts_only_cols = [c for c in alerts_only_cols if c in df_list.columns]

    alerts_only_df = df_list[df_list["pred"] == 1][alerts_only_cols].copy() if "pred" in df_list.columns else df_list[alerts_only_cols].copy()

    pred_cols = ["transaction_id","date","amount","use_chip","merchant_id","merchant_city","merchant_state","zip","mcc","pred"]
    if "score" in df_list.columns and show_sensitive: pred_cols.append("score")
    if "client_id" in df_list.columns and show_sensitive: pred_cols.append("client_id")
    if "card_id" in df_list.columns and show_sensitive: pred_cols.append("card_id")
    pred_cols = [c for c in pred_cols if c in df_view.columns]
    predictions_df = df_view[pred_cols].copy()
    st.download_button("⬇️ predictions.csv", data=_df_to_csv_bytes(predictions_df), file_name="predictions.csv", mime="text/csv", key="dl_predictions")

    if "merchant_id" in df_view.columns and "pred" in df_view.columns:
        al = df_view[df_view["pred"] == 1].copy()
        tx_tot = df_view.groupby("merchant_id").size()
        al_cnt = al.groupby("merchant_id").size()
        al_amt = al.groupby("merchant_id")["amount"].sum(min_count=1)
        mcc_major = df_view.groupby("merchant_id")["mcc"].agg(lambda x: x.astype(str).value_counts().idxmax() if len(x) else "")
        fm = pd.DataFrame({
            "nb_alertes": al_cnt,
            "montant_expose": al_amt,
            "tx_total": tx_tot,
        }).fillna(0)
        fm["taux_alerte"] = (fm["nb_alertes"] / fm["tx_total"]).replace([np.inf, np.nan], 0.0)
        fm["mcc_majoritaire"] = mcc_major
        fm = fm.sort_values(["montant_expose","nb_alertes"], ascending=False).reset_index()
        st.download_button("⬇️ flagged_merchants.csv", data=_df_to_csv_bytes(fm), file_name="flagged_merchants.csv", mime="text/csv", key="dl_flagged_merchants")

    with st.expander("Pack d’enquête (ZIP)"):
        meta = run_meta or {}
        narrative = [
            f"Périmètre: {period[0].strftime('%d/%m/%Y')} → {period[1].strftime('%d/%m/%Y')}" if isinstance(period, (tuple, list)) else "Périmètre: —",
            f"Transactions: {fr_int(len(df_view))} | Alertes: {fr_int(int((df_view['pred']==1).sum())) if 'pred' in df_view.columns else '—'}",
            f"Modèle {meta.get('model_version','—')} | Seuil {meta.get('threshold','—')} | Run {meta.get('run_id','—')} | Export: {meta.get('exported_at','—')}",
        ]
        summary_txt = "\n".join(narrative)
        payloads = {
            "alerts_only.csv": alerts_only_df,
            "predictions.csv": predictions_df,
            "flagged_merchants.csv": fm if 'fm' in locals() else pd.DataFrame(),
        }
        zip_bytes = build_investigator_zip(payloads, summary_txt)
        st.download_button("🧩 Télécharger investigator_pack.zip", data=zip_bytes, file_name="investigator_pack.zip", mime="application/zip", key="dl_investigator_zip")