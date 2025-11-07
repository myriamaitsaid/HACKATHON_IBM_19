#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json
import streamlit as st
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.validation import validate_and_normalize
from src.ingestion import read_csv_safely
from src.summary import render_dataset_summary
from src.inference import load_joblib_model, run_inference, simulate_predictions
from src.synthese import render_synthese
from src.investigation import render_investigation  # <-- import en haut avec les autres

# --- modèle en dur (tu peux adapter le nom) ---
MODEL_PATH = os.path.join(ROOT, "models", "fraud_model.joblib")
THRESHOLD = 0.50

st.set_page_config(page_title="Fraude", page_icon="🛡️", layout="wide")

# ---------- Styles (widgets & dark premium, déjà utilisés dans Page 1) ----------
st.markdown("""
<style>
:root{ --ink:#E5E7EB; --muted:#94A3B8; --border:#374151; --c1:#38BDF8; --c2:#34D399; --c3:#A78BFA; --c4:#F59E0B; --c5:#F472B6; }
.block-container{padding-top:.6rem;}
h1,h2,h3{letter-spacing:-.02em;}
.widgets-2col{ display:grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap:14px; }
@media (max-width: 900px){ .widgets-2col{ grid-template-columns: 1fr; } }
.widget{ position:relative; background: linear-gradient(180deg, rgba(148,163,184,0.06), rgba(148,163,184,0.03));
  border:1px solid var(--border); border-radius:16px; padding:16px 14px; min-height:120px;
  display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; }
.widget .value{ color:var(--ink); font-weight:800; font-size:1.6rem; line-height:1.15; word-break:break-word; white-space:normal;}
.widget .label{ color:var(--muted); font-size:.88rem; margin-top:6px; }
.widget::before{ content:""; position:absolute; left:0; right:0; top:0; height:3px; opacity:.85; border-radius:16px 16px 0 0; }
.widget.c1::before{ background:var(--c1);} .widget.c2::before{ background:var(--c2);} .widget.c3::before{ background:var(--c3);} .widget.c4::before{ background:var(--c4);} .widget.c5::before{ background:var(--c5);}
.uniques{ display:grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap:10px; margin-top:12px;}
@media (max-width: 900px){ .uniques{ grid-template-columns: 1fr; } }
.unique{ background:transparent; border:1px dashed var(--border); border-radius:12px; padding:10px 12px; text-align:center; color:var(--ink); font-weight:700;}
[data-testid="stDataFrame"] .css-1q7iqn2{font-size:.92rem;}
</style>
""", unsafe_allow_html=True)

# ---------- Cache schema ----------
@st.cache_data(show_spinner=False)
def load_schema() -> dict:
    with open(os.path.join(ROOT, "configs", "schema.json"), "r", encoding="utf-8") as f:
        return json.load(f)

SCHEMA = load_schema()

# ---------- Onglets ----------
tab1, tab2, tab3 = st.tabs(["1) Import & validation", "2) Synthèse du risque", "3) Investigation & Export"])

# ===================== TAB 1 =====================
with tab1:
    st.title("Fraude — Import & validation")


    # Upload
    st.subheader("1️⃣ Importer et valider les données")
    uploaded = st.file_uploader("Déposez votre fichier CSV de transactions", type=["csv"], key="csv_upload")

    df_raw = None
    if uploaded is not None:
        df_raw = read_csv_safely(uploaded)

    if df_raw is None:
        st.info("Déposez un CSV (ou cochez l'exemple) pour commencer.")
    else:

        st.success(f"Fichier chargé : {df_raw.shape[0]:,} lignes × {df_raw.shape[1]} colonnes")

        with st.spinner("Validation & normalisation en cours..."):
            df_norm, report = validate_and_normalize(df_raw, SCHEMA)

        errors = report.get("errors", [])
        warnings = report.get("warnings", [])

        if errors:
            st.error("Erreurs bloquantes :")
            for e in errors: st.markdown(f"- {e}")
        else:
            st.success("✅ Schéma valide (aucune erreur bloquante)")

        if warnings:
            st.warning("Points à vérifier :")
            for w in warnings: st.markdown(f"- {w}")


        with st.expander("Aperçu des données (normalisées)", expanded=False):
            st.dataframe(df_norm.head(50), use_container_width=True)

        st.subheader("Résumé du dataset")
        render_dataset_summary(df_norm, report)

        # --- Lancer l'analyse : inférence OU simulation 5% ---
        run_btn = st.button("🚀 Lancer l’analyse", type="primary", disabled=bool(errors))
        if run_btn and not errors:
            run_meta = {
                "model_version": None,
                "threshold": THRESHOLD,
                "trained_on": None,
                "run_id": None,
                "file_hash": None,
                "exported_at": None,
            }
            df_pred = None
            # essaie modèle disque, sinon simulation
            try:
                model = load_joblib_model(MODEL_PATH)
                df_pred, dbg = run_inference(model, df_norm, threshold=THRESHOLD)
                run_meta["model_version"] = "joblib:loaded"
                st.success("Inférence OK (modèle en dur).")
            except Exception as e:
                df_pred = simulate_predictions(df_norm, rate=0.05, seed=2025)
                run_meta["model_version"] = "simulation_5pct"

            # compose df_result (join pred & score)
            df_result = df_norm.copy()
            if "fraud_probability" in df_pred.columns:
                df_result["score"] = df_pred["fraud_probability"].values
            df_result["pred"] = df_pred["fraud_prediction"].values

            st.session_state["df_result"] = df_result
            st.session_state["run_meta"] = run_meta
            st.success("Synthèse prête. Ouvre l’onglet « 2) Synthèse du risque ».")

# ===================== TAB 2 =====================
with tab2:
    st.title("📊 Synthèse du risque")

    if "df_result" not in st.session_state:
        st.info("Charge et lance l’analyse dans l’onglet « 1) Import & validation ».")
        st.stop()

    df_result = st.session_state["df_result"]
    run_meta = st.session_state.get("run_meta", {"threshold": THRESHOLD})

    # rendu complet de la page 2
    render_synthese(df_result, run_meta)



# ===================== TAB 3 =====================
with tab3:
    st.title("🔎 Investigation & Export")
    if "df_result" not in st.session_state:
        st.info("Charge et lance l’analyse dans l’onglet « 1) Import & validation », puis consulte la synthèse (onglet 2).")
        st.stop()
    df_result = st.session_state["df_result"]
    run_meta = st.session_state.get("run_meta", {})
    render_investigation(df_result, run_meta)