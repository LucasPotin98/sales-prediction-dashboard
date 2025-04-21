import pandas as pd
import numpy as np
import os

def check_missing_values(df):
    """Affiche les colonnes avec des valeurs manquantes."""
    missing = df.isna().sum()
    missing = missing[missing > 0]
    print(missing)
    return missing

def handle_missing_values(df):
    """
    Gère les NaN : on supprime toutes les lignes contenant des NaN pour l'instant.
    """
    df = df.copy()
    df = df.dropna()
    return df


def handle_outliers(df, col="quantity", threshold=3):
    df = df.copy()
    mean = df[col].mean()
    std = df[col].std()
    z_scores = (df[col] - mean) / std
    df = df[np.abs(z_scores) <= threshold]
    return df

def clean_dataset(df):
    """
    Pipeline complète : nettoyage des NaN et des valeurs aberrantes par suppression.
    """
    check_missing_values(df)
    df = handle_missing_values(df)
    df = handle_outliers(df, col="quantity")
    return df

def run_data_cleaning(input_path="data/raw/transactions.csv", output_path="data/processed/clean_transactions.csv"):
    """
    Charge les données brutes, les nettoie et les enregistre dans le dossier processed.
    """
    df_raw = pd.read_csv(input_path)
    print(f"{len(df_raw)} lignes chargées.")

    df_clean = clean_dataset(df_raw)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)

if __name__ == "__main__":
    run_data_cleaning()