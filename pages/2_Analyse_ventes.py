import streamlit as st
from app.utils import load_data
from app.figures import plot_seasonality, plot_family_distribution
from src.analysis import compute_seasonality, compute_family_distribution
st.set_page_config(page_title="Analyse des ventes", page_icon="📊")

# Chargement des données
df = load_data()

# Titre
st.title("📊 Analyse exploratoire des ventes")

# Description courte
st.markdown(
    "Cette page propose une exploration interactive des ventes par famille de produits,\n"
    "avec un focus sur la saisonnalité et la concentration des ventes.\n"
    "Vous pouvez sélectionner une ou plusieurs familles pour filtrer dynamiquement les visualisations."
)

# Choix des familles
selected_families = st.multiselect(
    "Sélectionnez une ou plusieurs familles de produits :",
    options=df['family'].unique().tolist(),
    default=df['family'].unique().tolist()
)

# Message si rien n'est sélectionné
if not selected_families:
    st.warning("Veuillez sélectionner au moins une famille pour afficher les visualisations.")
    st.stop()

# Graphe 1 : Saisonnalité
st.subheader("📅 Saisonnalité des ventes")
seasonality_df = compute_seasonality(df, selected_families)
plot_seasonality(seasonality_df)

# Graphe 2 : Concentration des ventes
st.subheader("📦 Répartition des ventes par produit")
distribution_df = compute_family_distribution(df, selected_families)
plot_family_distribution(distribution_df, selected_families)
