import streamlit as st
import pandas as pd
import sys
import os
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import preprocess_batch
import joblib

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "fraud_model.pkl")
    return joblib.load(model_path)

st.title("📊 Analyse par lot")
st.markdown("Importez un fichier CSV contenant les données de plusieurs contribuables.")

uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.write("Aperçu des données importées :")
    st.dataframe(df_raw.head())
    
    # Vérification des colonnes requises
    required_cols = ['Personnalite_juridique', 'Regime_2023', 'Profil_en_2023',
                     'Type_de_centre_de_rattachement_2023', 'CRI_2023',
                     'CA_DSF_2023', 'CA_Export_DSF_2023', 'Valeur_export_2023',
                     'Valeur_import_2023', 'score_du_risque_fichier_en_2023', 'ECART_2023']
    
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        st.error(f"Colonnes manquantes dans le fichier : {missing}")
    else:
        if st.button("Lancer l'analyse"):
            with st.spinner("Analyse en cours..."):
                # Prétraitement du batch
                X = preprocess_batch(df_raw)
                model = load_model()
                probas = model.predict_proba(X)[:, 1]
                preds = model.predict(X)
            
            df_raw['probabilite_fraude'] = probas
            df_raw['prediction'] = preds
            df_raw['niveau_risque'] = df_raw['probabilite_fraude'].apply(
                lambda x: 'Élevé' if x > 0.7 else 'Modéré' if x > 0.3 else 'Faible'
            )
            
            st.subheader("Résultats")
            st.dataframe(df_raw[['Identifiant_correspondant_NIU', 'probabilite_fraude', 'niveau_risque', 'prediction']].head(20))
            
            # Téléchargement des résultats
            csv = df_raw.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Télécharger les résultats (CSV)", csv, "resultats_analyse.csv", "text/csv")
            
            # Statistiques rapides
            st.write("**Statistiques**")
            st.write(f"Nombre de contribuables suspects : {df_raw['prediction'].sum()}")
            st.write(f"Taux de suspicion : {df_raw['prediction'].mean():.2%}")