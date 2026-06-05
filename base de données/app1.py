# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Chargement du modèle
@st.cache_resource
def load_model():
    return joblib.load('fraud_model.pkl')

def predict_fraud(input_df):
    model = load_model()
    proba = model.predict_proba(input_df)[0, 1]
    pred = model.predict(input_df)[0]
    return proba, pred

# Interface
st.set_page_config(page_title="Détection de fraude fiscale", layout="wide")
st.title("🔍 Détection de fraude fiscale - Commerce extérieur")
st.markdown("Saisissez les informations du contribuable pour évaluer son risque de fraude.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Informations générales")
    personnalite = st.selectbox("Personnalité juridique", ["Personne morale", "Personne physique"])
    regime = st.selectbox("Régime fiscal 2023", ["RSI", "REEL", "HRI", "IL", "OBNL", "SALARIE"])
    profil = st.selectbox("Profil 2023", ["C", "PC"])
    centre = st.selectbox("Type de centre de rattachement", ["CDI", "CIME", "CSI", "DGE"])
    cri = st.selectbox("CRI (région)", ["CRIC1", "CRIC2", "CRIL1", "CRIL2", "CRIE", "CRIN", "CRINO", "CRISO", "CRIA"])

with col2:
    st.subheader("Données financières 2023")
    ca_total = st.number_input("Chiffre d'affaires total (FCFA)", min_value=0, value=0, step=1000000)
    ca_export = st.number_input("Chiffre d'affaires export (FCFA)", min_value=0, value=0, step=1000000)
    val_export = st.number_input("Valeur exportée selon douane (FCFA)", min_value=0, value=0, step=1000000)
    val_import = st.number_input("Valeur importée (FCFA)", min_value=0, value=0, step=1000000)
    score_risque = st.number_input("Score de risque fichier (0-10)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    ecart = st.number_input("Écart CA export / douane (FCFA)", value=0, step=100000)

# Calcul du ratio export/CA
ratio = ca_export / (ca_total + 1)

# Création du dataframe d'entrée
input_data = pd.DataFrame([{
    'Personnalite_juridique': personnalite,
    'Regime_2023': regime,
    'Profil_en_2023': profil,
    'Type_de_centre_de_rattachement_2023': centre,
    'CRI_2023': cri,
    'CA_DSF_2023': ca_total,
    'CA_Export_DSF_2023': ca_export,
    'Valeur_export_2023': val_export,
    'Valeur_import_2023': val_import,
    'score_du_risque_fichier_en_2023': score_risque,
    'ECART_2023': ecart,
    'ratio_export_ca': ratio
}])

if st.button("Évaluer le risque", type="primary"):
    with st.spinner("Analyse en cours..."):
        proba, pred = predict_fraud(input_data)
    
    st.markdown("---")
    st.subheader("Résultat de l'analyse")
    
    # Affichage avec jauge
    col_met, col_expl = st.columns([1, 2])
    with col_met:
        st.metric("Probabilité de fraude", f"{proba:.1%}")
        if pred == 1:
            st.error("⚠️ **Contribuable suspect** – Un contrôle approfondi est recommandé.")
        else:
            st.success("✅ **Contribuable en règle** – Aucune anomalie majeure détectée.")
    
    with col_expl:
        st.markdown("**Facteurs de risque pris en compte :**")
        st.markdown("- Écart entre CA export déclaré et valeur douane")
        st.markdown("- Ratio export / CA total")
        st.markdown("- Score de risque fichier")
        st.markdown("- Type de centre de rattachement")
        st.markdown("- Régime fiscal")
    
    # Graphique de jauge simple (barre horizontale)
    st.markdown("**Niveau de risque**")
    risk_color = "red" if proba > 0.7 else "orange" if proba > 0.3 else "green"
    st.progress(proba, text=f"{proba:.0%}")
    st.caption("Seuils : <30% = faible, 30-70% = modéré, >70% = élevé")