import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os

def generate_excel_report(df):
    """Génère un fichier Excel avec mise en forme basique."""
    output = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
    with pd.ExcelWriter(output.name, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Resultats', index=False)
        # Optionnel : ajouter des formats
    return output.name

def generate_pdf_report(df):
    """Génère un rapport PDF résumé."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Rapport d'analyse de risque fiscal", ln=1, align='C')
    pdf.ln(10)
    
    # Statistiques
    n_suspects = (df['prediction'] == 1).sum() if 'prediction' in df.columns else 0
    taux_suspects = n_suspects / len(df) if len(df) > 0 else 0
    pdf.cell(200, 10, txt=f"Nombre total de contribuables analysés : {len(df)}", ln=1)
    pdf.cell(200, 10, txt=f"Nombre de suspects : {n_suspects} ({taux_suspects:.2%})", ln=1)
    
    # Ajout d'un graphique simple (optionnel)
    if 'probabilite_fraude' in df.columns:
        fig, ax = plt.subplots()
        df['probabilite_fraude'].hist(ax=ax, bins=20)
        ax.set_title("Distribution des probabilités")
        fig.savefig("temp_hist.png")
        pdf.image("temp_hist.png", x=10, y=80, w=180)
        os.remove("temp_hist.png")
    
    output = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(output.name)
    return output.name