import streamlit as st
import os
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# 1. Création automatique du fichier de thème (couleurs DGI)
# ============================================================
streamlit_dir = ".streamlit"
config_file = os.path.join(streamlit_dir, "config.toml")
if not os.path.exists(config_file):
    os.makedirs(streamlit_dir, exist_ok=True)
    with open(config_file, "w") as f:
        f.write("""
[theme]
primaryColor = "#F5A623"
backgroundColor = "#FFF3E0"
secondaryBackgroundColor = "#FFE0B2"
textColor = "#2C3E50"
font = "sans serif"
""")
    print("[INFO] Fichier de thème créé :", config_file)

# ============================================================
# 2. Création d’un logo par défaut (s’il n’existe pas)
# ============================================================
assets_dir = "assets"
logo_path = os.path.join(assets_dir, "logo_dgi.png")
if not os.path.exists(logo_path):
    os.makedirs(assets_dir, exist_ok=True)
    # Créer une image simple (texte) pour éviter l’erreur
    img = Image.new('RGB', (300, 100), color='#F5A623')
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    d.text((50, 35), "DGI CAMEROUN", fill='#FFFFFF', font=font)
    img.save(logo_path)
    print("[INFO] Logo par défaut créé :", logo_path)

# ============================================================
# 3. Configuration de la page (doit être la première commande)
# ============================================================
st.set_page_config(
    page_title="DGI Cameroun - Détection de fraude fiscale",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 4. Affichage du logo dans la barre latérale
# ============================================================
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    # Correction : utiliser width ou use_column_width au lieu de use_container_width
    st.sidebar.image(logo, width=188)   # Ajustez la largeur selon votre logo
else:
    st.sidebar.warning("Logo DGI non disponible.")

# ============================================================
# 5. Contenu principal
# ============================================================
st.title("🔍 Détection de fraude fiscale - Commerce extérieur")
st.markdown(
    """
    Bienvenue dans l'application d'aide à la détection de fraude fiscale.
    
    Utilisez le menu sur la gauche pour naviguer entre les différentes fonctionnalités :
    
    - **Prédiction individuelle** : évaluez un contribuable à partir de ses données.
    - **Analyse par lot** : importez un fichier CSV pour analyser plusieurs contribuables.
    - **Tableau de bord** : visualisez les statistiques et les tendances.
    - **Export de rapports** : générez des rapports PDF/Excel.
    """
)

st.sidebar.success("Sélectionnez une page ci-dessus.")