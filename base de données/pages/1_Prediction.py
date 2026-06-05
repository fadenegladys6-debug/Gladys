import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import preprocess_input
from utils.visualization import plot_gauge
import joblib

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "fraud_model.pkl")
    return joblib.load(model_path)

def predict_fraud(input_df):
    model = load_model()
    proba = model.predict_proba(input_df)[0, 1]
    pred = model.predict(input_df)[0]
    return proba, pred

st.title("🔎 Prédiction individuelle")
st.markdown("Saisissez les informations du contribuable (année 2022) pour évaluer son risque de fraude.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Informations générales")
    personnalite = st.selectbox("Personnalité juridique", ["Personne morale", "Personne physique"])
    regime = st.selectbox("Régime fiscal 2022", ["RSI", "REEL", "HRI", "IL", "OBNL", "SALARIE"])
    profil = st.selectbox("Profil 2022", ["C", "PC"])
    centre = st.selectbox("Type de centre de rattachement 2022", ["CDI", "CIME", "CSI", "DGE"])
    cri = st.selectbox("CRI (région) 2022", ["CRIC1", "CRIC2", "CRIL1", "CRIL2", "CRIE", "CRIN", "CRINO", "CRISO", "CRIA"])
    qual_niu = st.selectbox("Qualité NIU 2022", ["Bon", "Mauvais"])
    niu_mauvais = 1 if qual_niu == "Mauvais" else 0

with col2:
    st.subheader("Données financières 2022")
    ca_total = st.number_input("Chiffre d'affaires total (FCFA)", min_value=0, value=0, step=1000000)
    ca_export = st.number_input("Chiffre d'affaires export (FCFA)", min_value=0, value=0, step=1000000)
    val_export = st.number_input("Valeur exportée selon douane (FCFA)", min_value=0, value=0, step=1000000)
    val_import = st.number_input("Valeur importée (FCFA)", min_value=0, value=0, step=1000000)
    evol_ca = st.number_input("Évolution du CA entre 2021 et 2022 (en %)", min_value=-100.0, max_value=1000.0, value=0.0, step=1.0)
    intensite_exp = st.number_input("Intensité export (CA export / CA total)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    intensite_imp = st.number_input("Intensité import (Valeur import / CA total)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

# Les valeurs manquantes (comme evol_ca, intensite_exp) peuvent être laissées à 0 par défaut.
# L'utilisateur peut les renseigner s'il les connaît.

if st.button("Évaluer le risque", type="primary"):
    with st.spinner("Analyse en cours..."):
        input_df = preprocess_input(personnalite, regime, profil, centre, cri, qual_niu,
                                    ca_total, ca_export, val_export, val_import,
                                    evol_ca, intensite_exp, intensite_imp, niu_mauvais)
        proba, pred = predict_fraud(input_df)
    
    st.markdown("---")
    st.subheader("Résultat de l'analyse")
    col_met, col_expl = st.columns([1, 2])
    with col_met:
        st.metric("Probabilité de fraude", f"{proba:.1%}")
        if pred == 1:
            st.error("⚠️ **Contribuable suspect** – Un contrôle approfondi est recommandé.")
        else:
            st.success("✅ **Contribuable en règle** – Aucune anomalie majeure détectée.")
    with col_expl:
        st.markdown("**Facteurs pris en compte :**")
        st.markdown("- Écart entre CA export déclaré et valeur douane")
        st.markdown("- Ratio export/CA et import/CA")
        st.markdown("- Évolution du CA")
        st.markdown("- Intensité export et import")
        st.markdown("- Qualité NIU")
    plot_gauge(proba)