import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def plot_gauge(probability):
    """Affiche une jauge de risque avec Matplotlib."""
    fig, ax = plt.subplots(figsize=(6, 2))
    # Barre horizontale
    colors = ['green', 'orange', 'red']
    thresholds = [0.3, 0.7]
    # Déterminer la couleur
    if probability < thresholds[0]:
        color = colors[0]
    elif probability < thresholds[1]:
        color = colors[1]
    else:
        color = colors[2]
    
    ax.barh([0], probability, color=color, height=0.4)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Niveau de risque")
    ax.axvline(x=thresholds[0], color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=thresholds[1], color='gray', linestyle='--', alpha=0.5)
    ax.text(probability + 0.02, 0, f"{probability:.0%}", va='center')
    st.pyplot(fig)

def plot_risk_distribution(probas):
    """Histogramme des probabilités."""
    fig, ax = plt.subplots()
    ax.hist(probas, bins=30, alpha=0.7, color='steelblue')
    ax.set_xlabel("Probabilité de fraude")
    ax.set_ylabel("Nombre de contribuables")
    st.pyplot(fig)