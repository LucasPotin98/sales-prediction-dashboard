import pandas as pd
import streamlit as st

@st.cache_data
def load_data(path="data/raw/transactions.csv"):
    """Charge les données depuis un fichier CSV avec cache."""
    return pd.read_csv(path)

def get_kpis(df):
    """Retourne les KPIs principaux à partir du DataFrame transactions."""
    kpis = {
        "transactions": len(df),
        "produits_uniques": df['product_id'].nunique(),
        "clients": df['client_id'].nunique(),
        "revenu_total": df['revenue'].sum(),
        "quantite_totale": df['quantity'].sum()
    }
    return kpis