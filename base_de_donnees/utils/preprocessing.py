import pandas as pd
import numpy as np

def preprocess_input(personnalite, regime, profil, centre, cri, qual_niu,
                     ca_total, ca_export, val_export, val_import,
                     evol_ca, intensite_exp, intensite_imp, niu_mauvais):
    """Transforme les entrées utilisateur en DataFrame avec les 18 features attendues."""
    # Calculs
    ratio_exp_ca = ca_export / (ca_total + 1)
    ratio_imp_ca = val_import / (ca_total + 1)
    ecart_exp_douane = ca_export - val_export
    ratio_ecart = ecart_exp_douane / (ca_total + 1) if ca_total > 0 else 0

    # Logs (éviter log(0) -> remplacer 0 par 1)
    log_ca = np.log(ca_total + 1)
    log_ca_exp = np.log(ca_export + 1)
    log_val_exp = np.log(val_export + 1)
    log_val_imp = np.log(val_import + 1)

    data = {
        'PERSO_JUR': personnalite,
        'REGIME_2022': regime,
        'PROFIL_2022': profil,
        'CENTRE_2022': centre,
        'CRI_2022': cri,
        'QUAL_NIU_2022': qual_niu,
        'LOG_CA_2022': log_ca,
        'LOG_CA_EXP_2022': log_ca_exp,
        'LOG_VAL_EXP_2022': log_val_exp,
        'LOG_VAL_IMP_2022': log_val_imp,
        'RATIO_EXP_CA_2022': ratio_exp_ca,
        'RATIO_IMP_CA_2022': ratio_imp_ca,
        'ECART_EXP_DOUANE_2022': ecart_exp_douane,
        'RATIO_ECART_2022': ratio_ecart,
        'EVOL_CA_2122': evol_ca,
        'INTENSITE_EXP_2022': intensite_exp,
        'INTENSITE_IMP_2022': intensite_imp,
        'NIU_MAUVAIS': niu_mauvais
    }
    return pd.DataFrame([data])

def preprocess_batch(df):
    """Prétraite un DataFrame batch (mêmes colonnes que le modèle)."""
    # Calculs identiques à preprocess_input
    df['RATIO_EXP_CA_2022'] = df['CA_DSF_2022'] / (df['CA_DSF_2022'] + 1)
    df['RATIO_IMP_CA_2022'] = df['Valeur_import_2022'] / (df['CA_DSF_2022'] + 1)
    df['ECART_EXP_DOUANE_2022'] = df['CA_Export_DSF_2022'] - df['Valeur_export_2022']
    df['RATIO_ECART_2022'] = df['ECART_EXP_DOUANE_2022'] / (df['CA_DSF_2022'] + 1)
    df['LOG_CA_2022'] = np.log(df['CA_DSF_2022'] + 1)
    df['LOG_CA_EXP_2022'] = np.log(df['CA_Export_DSF_2022'] + 1)
    df['LOG_VAL_EXP_2022'] = np.log(df['Valeur_export_2022'] + 1)
    df['LOG_VAL_IMP_2022'] = np.log(df['Valeur_import_2022'] + 1)
    # Sélectionner uniquement les colonnes du modèle
    features = [
        'PERSO_JUR', 'REGIME_2022', 'PROFIL_2022', 'CENTRE_2022', 'CRI_2022', 'QUAL_NIU_2022',
        'LOG_CA_2022', 'LOG_CA_EXP_2022', 'LOG_VAL_EXP_2022', 'LOG_VAL_IMP_2022',
        'RATIO_EXP_CA_2022', 'RATIO_IMP_CA_2022', 'ECART_EXP_DOUANE_2022', 'RATIO_ECART_2022',
        'EVOL_CA_2122', 'INTENSITE_EXP_2022', 'INTENSITE_IMP_2022', 'NIU_MAUVAIS'
    ]
    return df[features]