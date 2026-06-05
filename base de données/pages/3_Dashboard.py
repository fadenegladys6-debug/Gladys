import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualization import plot_risk_distribution

st.title("📈 Tableau de bord")

# Charger les données de référence (à adapter selon vos données)
@st.cache_data
def load_reference_data():
    # Ici, vous pouvez charger un fichier de données historiques.
    # Pour l'exemple, nous allons générer des données aléatoires (à remplacer par votre vrai fichier).
    # Idéalement, utilisez un fichier CSV de synthèse ou les résultats de l'analyse par lot.
    try:
        df = pd.read_csv("data/reference_data.csv")
    except:
        # Création de données factices
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            "probabilite_fraude": np.random.beta(1, 10, n),
            "CRI_2023": np.random.choice(["CRIC1", "CRIC2", "CRIL1", "CRIL2"], n),
            "Regime_2023": np.random.choice(["RSI", "REEL", "HRI"], n)
        })
    return df

df = load_reference_data()

col1, col2 = st.columns(2)
with col1:
    st.metric("Taux de fraude moyen", f"{df['probabilite_fraude'].mean():.1%}")
with col2:
    st.metric("Nombre de suspicions (>70%)", f"{(df['probabilite_fraude'] > 0.7).sum()}")

# Distribution par région
st.subheader("Risque moyen par région (CRI)")
risk_by_cri = df.groupby('CRI_2023')['probabilite_fraude'].mean().sort_values()
fig, ax = plt.subplots()
risk_by_cri.plot(kind='barh', ax=ax, color='coral')
ax.set_xlabel("Probabilité moyenne de fraude")
st.pyplot(fig)

# Distribution par régime
st.subheader("Risque moyen par régime fiscal")
risk_by_regime = df.groupby('Regime_2023')['probabilite_fraude'].mean().sort_values()
fig2, ax2 = plt.subplots()
risk_by_regime.plot(kind='bar', ax=ax2, color='skyblue')
ax2.set_ylabel("Probabilité moyenne")
st.pyplot(fig2)

# Histogramme des probabilités
fig3, ax3 = plt.subplots()
df['probabilite_fraude'].hist(bins=30, ax=ax3, alpha=0.7, color='green')
ax3.set_xlabel("Probabilité de fraude")
ax3.set_ylabel("Nombre de contribuables")
st.pyplot(fig3)