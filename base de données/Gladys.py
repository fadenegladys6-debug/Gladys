"""
=============================================================================
PHASE 1 — CHARGEMENT, NETTOYAGE, EDA ET VARIABLE CIBLE
DGI Cameroun — Segmentation des contribuables à risque de fraude fiscale
=============================================================================

STRATÉGIE ANTI-LEAKAGE
-----------------------
Variable cible (Y) : FRAUDE_2022
  = 1 si le contribuable a présenté une anomalie documentée en 2022
    (Dissimulation CA 2022 OU Problème déclaration export 2022)

Features (X) : UNIQUEMENT des variables structurelles et financières 2021/2022
  → Aucun score 2023, aucune variable calculée à partir de la cible 2023
  → Le modèle apprend les signaux 2022 pour prédire le risque futur

Justification : Les données 2022 sont connues au moment du contrôle 2023.
Le modèle détecte les profils à risque AVANT que la fraude 2023 soit confirmée.
=============================================================================
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 80)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

# ── Répertoire de sortie ──────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("1. CHARGEMENT DES DONNÉES")
print("=" * 70)

DATA_PATH ="base de données/datafraud.xlsx"   # ← adapter si besoin
df_raw = pd.read_excel(DATA_PATH)

print(f"  Dimensions brutes : {df_raw.shape[0]:,} lignes × {df_raw.shape[1]} colonnes")
print(f"  NIU uniques       : {df_raw['Identifiant correspondant NIU'].nunique():,}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. RENOMMAGE PROPRE DES COLONNES
# ─────────────────────────────────────────────────────────────────────────────
rename_map = {
    'Identifiant correspondant NIU'                                              : 'NIU',
    'Personnalité juridique'                                                     : 'PERSO_JUR',
    'Régime 2022'                                                                : 'REGIME_2022',
    'Régime 2023'                                                                : 'REGIME_2023',
    'Profil en 2022'                                                             : 'PROFIL_2022',
    'Profil en 2023'                                                             : 'PROFIL_2023',
    'Type de centre de rattachement 2022'                                        : 'CENTRE_2022',
    'Type de centre de rattachement 2023'                                        : 'CENTRE_2023',
    'CRI 2022'                                                                   : 'CRI_2022',
    'CRI 2023'                                                                   : 'CRI_2023',
    'CA DSF 2021'                                                                : 'CA_2021',
    'CA DSF 2022'                                                                : 'CA_2022',
    'CA DSF 2023'                                                                : 'CA_2023',
    "Variation du Chiffre d'Affaire DSF 23/22"                                  : 'VAR_CA_2322',
    "Dynamique d'évolution du CA DSF entre 2022 et 2023"                        : 'DYN_CA_2322',
    'Variation du CA export DSF entre 2022 et 2023'                             : 'VAR_CA_EXP_2322',
    'CA Export DSF 2022'                                                         : 'CA_EXP_2022',
    'CA Export DSF 2023'                                                         : 'CA_EXP_2023',
    "Dynamique d'évolution du CA DSF lié à l'exportation  entre 2022 et 2023"  : 'DYN_CA_EXP_2322',
    'Cumul CA 2022'                                                              : 'CUMUL_CA_2022',
    'Cumul CA 2023'                                                              : 'CUMUL_CA_2023',
    'Valeur export 2022'                                                         : 'VAL_EXP_2022',
    'Valeur export 2023'                                                         : 'VAL_EXP_2023',
    'Valeur import 2022'                                                         : 'VAL_IMP_2022',
    'Valeur import 2023'                                                         : 'VAL_IMP_2023',
    "Problème de déclaration du Chiffre d'Affaire des exportations DE 2022"     : 'PB_DECL_EXP_2022',
    "Problème de déclaration du Chiffre d'Affaire des exportations DE 2023"     : 'PB_DECL_EXP_2023',
    "Dissimulation du Chiffre d'Affaires"                                        : 'DISSIM_CA_2022',
    "Dissimulation du Chiffre d'Affaires.1"                                      : 'DISSIM_CA_2023',
    "Probleme de déclaration du Chiffre d'Affaires"                              : 'PB_DECL_CA_2022',
    "Probleme de déclaration du Chiffre d'Affaires.1"                            : 'PB_DECL_CA_2023',
    'Qualité NIU 2022'                                                           : 'QUAL_NIU_2022',
    'Qualité NIU 2023'                                                           : 'QUAL_NIU_2023',
    'R_Régime 2022'                                                              : 'R_REGIME_2022',
    'R_Régime 2023'                                                              : 'R_REGIME_2023',
    'R_CENTRE 2022'                                                              : 'R_CENTRE_2022',
    'R_CENTRE 2023'                                                              : 'R_CENTRE_2023',
    'R_Fichier 2022'                                                             : 'R_FICH_2022',
    'R_Fichier 2023'                                                             : 'R_FICH_2023',
    "Elasticité exportation du Chiffre d'Affaire"                               : 'ELAST_EXP_CA',
    'Rapport des variables entre les exportations'                               : 'RAPP_EXP',
    'Variation des ventes pour exportation'                                      : 'VAR_VENTES_EXP',
    'score du risque fichier en 2022'                                            : 'SCORE_FICH_2022',
    'Score de déviance à la déclaration fiscale en 2022'                         : 'SCORE_DEV_2022',
    'score du risque fichier en 2023'                                            : 'SCORE_FICH_2023',
    'Score de déviance à la déclaration fiscale en 2023'                         : 'SCORE_DEV_2023',
    'Activité'                                                                   : 'ACTIVITE',
    "dissimulation du chiffre d'affaire à l'exportation au coté de la douane"   : 'DISSIM_EXP_DOUANE_2022',
    "Dissimulation du  chiffre d'affaire à l'exportation de la douane en 2023"  : 'DISSIM_EXP_DOUANE_2023',
    'risque de fraude final'                                                     : 'RISQUE_FINAL',
    'Score de comportement danomalie par rapport à la fiscalité interne en 2022': 'SCORE_ANOM_FISK_2022',
    'Score de comportement danomalie par rapport à la fiscalité interne en 2023': 'SCORE_ANOM_FISK_2023',
    "Comportement d'anomalie avec l'administration fiscale interne en 2022"     : 'ANOM_FISK_2022',
    "Comportement d'anomalie avec l'administration fiscale du chiffre d'affaire en 2023": 'ANOM_CA_2023',
    "Evolution du comportement d'anomalie entre 2022 et 2023"                   : 'EVOL_ANOM',
    "Score du risque du chiffre d'affaire en 2022"                              : 'SCORE_RCA_2022',
    "Score du risque du chiffre d'affaire en 2023"                              : 'SCORE_RCA_2023',
    "Comportement d'anomalie avec l'administration fiscale du chiffre d'affaire en 2022": 'ANOM_CA_2022',
    "Variation de comportement d'anomalie entre 2022 et 2023"                   : 'VAR_ANOM',
    'ECART_2023'                                                                 : 'ECART_2023',
}

df = df_raw.rename(columns=rename_map).copy()
print(f"\n  Colonnes renommées : {len(rename_map)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. CONSTRUCTION DE LA VARIABLE CIBLE — FRAUDE_2022
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. CONSTRUCTION DE LA VARIABLE CIBLE (sans leakage)")
print("=" * 70)

"""
Règle de construction :
  FRAUDE_2022 = 1  si le contribuable vérifie AU MOINS UNE des conditions :
    (A) Dissimulation du CA 2022 = "Oui"
    (B) Problème déclaration CA 2022 = "Oui"
    (C) Dissimulation export douane 2022 ∈ {"fraude moyenne","fraude élevée"}

Ces trois colonnes décrivent des faits 2022 — aucun lien avec les features 2023.
"""

cond_A = df['DISSIM_CA_2022'].str.strip().str.lower() == 'oui'
cond_B = df['PB_DECL_CA_2022'].str.strip().str.lower() == 'oui'
cond_C = df['DISSIM_EXP_DOUANE_2022'].str.strip().str.lower().isin(
    ['fraude moyenne', 'fraude élevée']
)

df['FRAUDE_2022'] = ((cond_A) | (cond_B) | (cond_C)).astype(int)

dist = df['FRAUDE_2022'].value_counts()
print(f"\n  FRAUDE_2022 = 0 (non-fraudeur) : {dist[0]:,}  ({dist[0]/len(df)*100:.1f}%)")
print(f"  FRAUDE_2022 = 1 (fraudeur)     : {dist[1]:,}  ({dist[1]/len(df)*100:.1f}%)")
print(f"\n  Décomposition :")
print(f"    Dissimulation CA 2022       : {cond_A.sum():,}")
print(f"    Pb déclaration CA 2022      : {cond_B.sum():,}")
print(f"    Dissimulation export douane : {cond_C.sum():,}")
print(f"    Union (FRAUDE_2022=1)       : {df['FRAUDE_2022'].sum():,}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. NETTOYAGE DES VARIABLES FINANCIÈRES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. NETTOYAGE")
print("=" * 70)

# 4.1 Valeurs aberrantes : CA export négatif → mettre à NaN
neg_exp = (df['CA_EXP_2022'] < 0).sum()
df.loc[df['CA_EXP_2022'] < 0, 'CA_EXP_2022'] = np.nan
print(f"  CA Export 2022 négatifs remis à NaN : {neg_exp}")

neg_exp23 = (df['CA_EXP_2023'] < 0).sum()
df.loc[df['CA_EXP_2023'] < 0, 'CA_EXP_2023'] = np.nan
print(f"  CA Export 2023 négatifs remis à NaN : {neg_exp23}")

# 4.2 Activité : normaliser le texte
df['ACTIVITE'] = df['ACTIVITE'].astype(str).str.strip().str.upper()
df.loc[df['ACTIVITE'].isin(['0', 'NAN', '']), 'ACTIVITE'] = 'NON_RENSEIGNE'

# 4.3 Valeurs manquantes synthèse
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
miss_df = pd.DataFrame({'manquant': missing, 'pct': missing_pct})
miss_sig = miss_df[miss_df['manquant'] > 0].sort_values('pct', ascending=False)
print(f"\n  Colonnes avec valeurs manquantes : {len(miss_sig)}")
print(miss_sig.head(15).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. FEATURE ENGINEERING")
print("=" * 70)

eps = 1  # éviter division par zéro

# Ratios export / CA (signaux de dissimulation)
df['RATIO_EXP_CA_2022']     = df['CA_EXP_2022'] / (df['CA_2022'] + eps)
df['RATIO_IMP_CA_2022']     = df['VAL_IMP_2022'] / (df['CA_2022'] + eps)

# Écart export douane vs CA export fiscal (mesure directe d'incohérence)
df['ECART_EXP_DOUANE_2022'] = df['VAL_EXP_2022'] - df['CA_EXP_2022'].fillna(0)
df['RATIO_ECART_2022']      = df['ECART_EXP_DOUANE_2022'] / (df['CA_2022'] + eps)

# Évolution CA 2021 → 2022
df['EVOL_CA_2122']          = (df['CA_2022'] - df['CA_2021'].fillna(df['CA_2022'])) / (df['CA_2021'].fillna(df['CA_2022']) + eps)

# Intensité export = part du CA réalisée à l'export
df['INTENSITE_EXP_2022']    = (df['VAL_EXP_2022'] > 0).astype(int)
df['INTENSITE_IMP_2022']    = (df['VAL_IMP_2022'] > 0).astype(int)

# Signaux qualitatifs de qualité NIU
df['NIU_MAUVAIS']           = (df['QUAL_NIU_2022'].str.strip() == 'Pas bon').astype(int)

# Log des montants (stabiliser la distribution)
for col in ['CA_2022', 'CA_EXP_2022', 'VAL_EXP_2022', 'VAL_IMP_2022']:
    df[f'LOG_{col}'] = np.log1p(df[col].clip(lower=0).fillna(0))

print("  Features créées :")
new_feats = ['RATIO_EXP_CA_2022','RATIO_IMP_CA_2022','ECART_EXP_DOUANE_2022',
             'RATIO_ECART_2022','EVOL_CA_2122','INTENSITE_EXP_2022',
             'INTENSITE_IMP_2022','NIU_MAUVAIS',
             'LOG_CA_2022','LOG_CA_EXP_2022','LOG_VAL_EXP_2022','LOG_VAL_IMP_2022']
for f in new_feats:
    print(f"    + {f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. ANALYSE EXPLORATOIRE (EDA)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("6. ANALYSE EXPLORATOIRE")
print("=" * 70)

sns.set_style("whitegrid")
sns.set_palette("Set2")

# ── 6.1 Distribution de la cible ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
labels  = ['Non fraudeur\n(0)', 'Fraudeur\n(1)']
counts  = [dist[0], dist[1]]
colors  = ['#2ecc71', '#e74c3c']
bars = ax.bar(labels, counts, color=colors, edgecolor='white', linewidth=1.5)
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f'{cnt:,}\n({cnt/len(df)*100:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title("Distribution de la variable cible FRAUDE_2022", fontsize=13, fontweight='bold')
ax.set_ylabel("Nombre de contribuables")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "01_distribution_cible.png"), dpi=150)
plt.close()
print("  [SAUVEGARDÉ] 01_distribution_cible.png")

# ── 6.2 Taux de fraude par régime fiscal ─────────────────────────────────────
taux_regime = df.groupby('REGIME_2022')['FRAUDE_2022'].agg(['mean','count']).reset_index()
taux_regime.columns = ['Régime','Taux_fraude','Effectif']
taux_regime = taux_regime.sort_values('Taux_fraude', ascending=False)

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(taux_regime['Régime'], taux_regime['Taux_fraude']*100,
              color=sns.color_palette("RdYlGn_r", len(taux_regime)), edgecolor='white')
for bar, (_, row) in zip(bars, taux_regime.iterrows()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{row['Taux_fraude']*100:.1f}%\n(n={row['Effectif']:,})",
            ha='center', fontsize=8.5)
ax.set_title("Taux de fraude par régime fiscal (2022)", fontsize=13, fontweight='bold')
ax.set_ylabel("% de fraudeurs")
ax.set_ylim(0, taux_regime['Taux_fraude'].max()*130)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "02_taux_fraude_regime.png"), dpi=150)
plt.close()
print("  [SAUVEGARDÉ] 02_taux_fraude_regime.png")

# ── 6.3 Taux de fraude par CRI (top 12) ──────────────────────────────────────
taux_cri = df.groupby('CRI_2022')['FRAUDE_2022'].agg(['mean','count']).reset_index()
taux_cri.columns = ['CRI','Taux_fraude','Effectif']
taux_cri = taux_cri[taux_cri['Effectif'] >= 20].sort_values('Taux_fraude', ascending=True).tail(12)

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(taux_cri['CRI'], taux_cri['Taux_fraude']*100,
        color=sns.color_palette("YlOrRd", len(taux_cri)))
for i, (_, row) in enumerate(taux_cri.iterrows()):
    ax.text(row['Taux_fraude']*100 + 0.2, i,
            f"{row['Taux_fraude']*100:.1f}% (n={row['Effectif']})", va='center', fontsize=8.5)
ax.set_title("Top 12 CRI selon le taux de fraude 2022", fontsize=13, fontweight='bold')
ax.set_xlabel("% de fraudeurs")
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "03_taux_fraude_CRI.png"), dpi=150)
plt.close()
print("  [SAUVEGARDÉ] 03_taux_fraude_CRI.png")

# ── 6.4 Taux de fraude par personnalité juridique ─────────────────────────────
taux_pj = df.groupby('PERSO_JUR')['FRAUDE_2022'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(taux_pj.index, taux_pj.values*100, color=['#3498db','#e67e22'], edgecolor='white')
for i, (idx, val) in enumerate(taux_pj.items()):
    ax.text(i, val*100+0.3, f"{val*100:.1f}%", ha='center', fontsize=11, fontweight='bold')
ax.set_title("Taux de fraude par personnalité juridique", fontsize=12, fontweight='bold')
ax.set_ylabel("% de fraudeurs")
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "04_taux_fraude_perso_jur.png"), dpi=150)
plt.close()
print("  [SAUVEGARDÉ] 04_taux_fraude_perso_jur.png")

# ── 6.5 Distribution CA 2022 par classe de fraude (log-échelle) ────────────
fig, ax = plt.subplots(figsize=(8, 4))
for label, color, name in [(0, '#2ecc71', 'Non-fraudeur'), (1, '#e74c3c', 'Fraudeur')]:
    vals = np.log1p(df.loc[df['FRAUDE_2022']==label, 'CA_2022'].clip(lower=0))
    ax.hist(vals, bins=40, alpha=0.55, color=color, label=name, density=True)
ax.set_title("Distribution du log(CA 2022) selon la fraude", fontsize=12, fontweight='bold')
ax.set_xlabel("log(1 + CA 2022)  [FCFA]")
ax.set_ylabel("Densité")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "05_distrib_logCA_fraude.png"), dpi=150)
plt.close()
print("  [SAUVEGARDÉ] 05_distrib_logCA_fraude.png")

# ── 6.6 Heatmap corrélations features numériques ─────────────────────────────
num_cols_corr = ['LOG_CA_2022','LOG_CA_EXP_2022','LOG_VAL_EXP_2022','LOG_VAL_IMP_2022',
                 'RATIO_EXP_CA_2022','RATIO_IMP_CA_2022','RATIO_ECART_2022',
                 'SCORE_FICH_2022','SCORE_DEV_2022','SCORE_RCA_2022','FRAUDE_2022']
corr = df[num_cols_corr].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5)
ax.set_title("Matrice de corrélations — features numériques 2022", fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "06_heatmap_correlations.png"), dpi=150)
plt.close()
print("  [SAUVEGARDÉ] 06_heatmap_correlations.png")

# ── 6.7 Taux de fraude par type de centre ────────────────────────────────────
taux_centre = df.groupby('CENTRE_2022')['FRAUDE_2022'].agg(['mean','count']).reset_index()
taux_centre.columns = ['Centre','Taux','N']
taux_centre = taux_centre.sort_values('Taux', ascending=False)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(taux_centre['Centre'], taux_centre['Taux']*100,
       color=sns.color_palette("coolwarm", len(taux_centre)), edgecolor='white')
for i, (_, row) in enumerate(taux_centre.iterrows()):
    ax.text(i, row['Taux']*100+0.3, f"{row['Taux']*100:.1f}%\n(n={row['N']})",
            ha='center', fontsize=7.5)
ax.set_title("Taux de fraude par type de centre de rattachement (2022)", fontsize=12, fontweight='bold')
ax.set_ylabel("% de fraudeurs")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "07_taux_fraude_centre.png"), dpi=150)
plt.close()
print("  [SAUVEGARDÉ] 07_taux_fraude_centre.png")

# ── 6.8 Scatter CA vs Valeur export (log), coloré par fraude ─────────────────
fig, ax = plt.subplots(figsize=(8, 5))
mask_exp = df['VAL_EXP_2022'] > 0
sc = ax.scatter(
    np.log1p(df.loc[mask_exp, 'CA_2022']),
    np.log1p(df.loc[mask_exp, 'VAL_EXP_2022']),
    c=df.loc[mask_exp, 'FRAUDE_2022'],
    cmap='RdYlGn_r', alpha=0.4, s=15, edgecolors='none'
)
plt.colorbar(sc, ax=ax, label='FRAUDE_2022')
ax.set_xlabel("log(1+CA 2022)")
ax.set_ylabel("log(1+Valeur export 2022)")
ax.set_title("CA fiscal vs Valeur export douane (2022)\ncoloré par fraude", fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "08_scatter_CA_export.png"), dpi=150)
plt.close()
print("  [SAUVEGARDÉ] 08_scatter_CA_export.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. SÉLECTION ET SAUVEGARDE DU DATASET FINAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("7. SÉLECTION DES FEATURES ET SAUVEGARDE")
print("=" * 70)

FEATURES_CAT = [
    'PERSO_JUR', 'REGIME_2022', 'PROFIL_2022', 'CENTRE_2022', 'CRI_2022',
    'QUAL_NIU_2022'
]
FEATURES_NUM = [
    'LOG_CA_2022', 'LOG_CA_EXP_2022', 'LOG_VAL_EXP_2022', 'LOG_VAL_IMP_2022',
    'RATIO_EXP_CA_2022', 'RATIO_IMP_CA_2022', 'ECART_EXP_DOUANE_2022',
    'RATIO_ECART_2022', 'EVOL_CA_2122', 'INTENSITE_EXP_2022', 'INTENSITE_IMP_2022',
    'NIU_MAUVAIS',
    'SCORE_FICH_2022', 'SCORE_DEV_2022', 'SCORE_RCA_2022',
]
TARGET = 'FRAUDE_2022'

keep_cols = ['NIU'] + FEATURES_CAT + FEATURES_NUM + [TARGET]
df_model = df[keep_cols].copy()

# Supprimer les doublons NIU (garder première occurrence)
n_before = len(df_model)
df_model = df_model.drop_duplicates(subset='NIU')
print(f"  Doublons NIU supprimés : {n_before - len(df_model)}")
print(f"  Taille finale          : {len(df_model):,} lignes × {len(df_model.columns)} colonnes")

# ── Sauvegarde ────────────────────────────────────────────────────────────────
CLEANED_PATH = os.path.join(OUTPUT_DIR, "datafraud_cleaned.csv")
df_model.to_csv(CLEANED_PATH, index=False)
print(f"\n  [SAUVEGARDÉ] {CLEANED_PATH}")

# ── Résumé statistique de la cible par segment ───────────────────────────────
print("\n  Taux de fraude par régime (2022) :")
print(df_model.groupby('REGIME_2022')[TARGET].agg(['mean','count'])
      .rename(columns={'mean':'taux','count':'n'})
      .sort_values('taux', ascending=False).to_string())

print("\n" + "=" * 70)
print("PHASE 1 TERMINÉE — Fichier prêt pour la modélisation.")
print("=" * 70)
print(f"\n  → {CLEANED_PATH}")
print(f"  → {FIGURES_DIR}/ (8 figures)")
