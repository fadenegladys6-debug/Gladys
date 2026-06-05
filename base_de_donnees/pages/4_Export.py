import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.report_generator import generate_pdf_report, generate_excel_report

st.title("📄 Génération de rapports")
st.markdown("Générez un rapport personnalisé à partir des résultats d'analyse.")

uploaded_file = st.file_uploader("Chargez le fichier CSV des résultats (optionnel)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Aperçu :")
    st.dataframe(df.head())
    
    if st.button("Générer rapport Excel"):
        excel_file = generate_excel_report(df)
        with open(excel_file, "rb") as f:
            st.download_button("📥 Télécharger rapport Excel", f, "rapport_fraude.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    if st.button("Générer rapport PDF (résumé)"):
        pdf_file = generate_pdf_report(df)
        with open(pdf_file, "rb") as f:
            st.download_button("📥 Télécharger rapport PDF", f, "rapport_fraude.pdf", "application/pdf")
else:
    st.info("Veuillez d'abord exporter des résultats via la page 'Analyse par lot'.")