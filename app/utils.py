import pandas as pd
import os
import joblib
import streamlit as st
from prophet.serialize import model_from_json
from src.modeling import prepare_aggregated


@st.cache_data
def load_data(path="data/raw/transactions.csv"):
    """Charge les données depuis un fichier CSV avec cache."""
    return pd.read_csv(path)


def get_kpis(df):
    """Retourne les KPIs principaux à partir du DataFrame transactions."""
    kpis = {
        "transactions": len(df),
        "produits_uniques": df["product_id"].nunique(),
        "clients": df["client_id"].nunique(),
        "revenu_total": df["revenue"].sum(),
        "quantite_totale": df["quantity"].sum(),
    }
    return kpis


def load_model(model_name: str, family: str):
    model_key = f"model_{model_name.lower()}_{family.lower()}"
    model_path = os.path.join("models", f"{model_key}.pkl")

    if not os.path.exists(model_path):
        st.error(f"Modèle introuvable : {model_path}")
        st.stop()

    return joblib.load(model_path)


def load_prophet_model(family, path_dir="models"):
    """
    Charge un modèle Prophet depuis un fichier .json
    """
    filename = f"model_prophet_{family.lower()}.json"
    path = os.path.join(path_dir, filename)
    with open(path, "r") as fin:
        return model_from_json(fin.read())


def load_all_data():
    df_train_raw = load_data("data/processed/clean_transactions.csv")
    df_train = prepare_aggregated(df_train_raw)
    df_test_raw = load_data("data/processed/clean_transactions_test.csv")
    df_test = prepare_aggregated(df_test_raw)
    return df_train, df_test
