import json
import streamlit as st
from app.utils import load_data
from app.figures import plot_seasonality, plot_family_distribution
from src.analysis import compute_seasonality, compute_family_distribution

st.set_page_config(page_title="Analyse des ventes", page_icon="📊")


with open("commentaires/commentaires.json", "r") as f:
    comments_data = json.load(f)


st.markdown(
    """
<style>
.navbar {
    position: sticky;
    top: 0;
    z-index: 999;
    background-color: #1e1e1e;
    padding: 0.6rem 1rem;
    border-bottom: 1px solid #444;
    display: flex;
    justify-content: center;
    gap: 2rem;
    font-size: 1.1rem;
    font-weight: 500;
}

.navbar a {
    text-decoration: none;
    color: #f0f0f0;
    padding: 0.4rem 1.2rem;
    border-radius: 6px;
    transition: background-color 0.2s ease;
}

.navbar a:hover {
    background-color: #333333;
}

.navbar a.active {
    background-color: #6c63ff;
    color: white;
}
</style>

<div class="navbar">
    <a href="/" target="_self">📦 Contexte</a>
    <a href="/Analyse_ventes" class="active" target="_self">📊 Analyse</a>
    <a href="/Modelisation" target="_self">🧠 Modélisation</a>
    <a href="/Analyse_graphes" target="_self">🔗 Graphes</a>
</div>
""",
    unsafe_allow_html=True,
)


# Chargement des données
df = load_data()

# Titre
st.markdown(
    "<h2 style='margin-bottom: 1rem;'>📊 Analyse exploratoire des ventes</h2>",
    unsafe_allow_html=True,
)


# Description courte
st.markdown(
    "Cette page propose une exploration interactive des ventes par famille de produits,\n"
    "avec un focus sur la saisonnalité et la concentration des ventes.\n"
    "Vous pouvez sélectionner une ou plusieurs familles pour filtrer dynamiquement les visualisations."
)

# Choix des familles
selected_families = st.multiselect(
    "Sélectionnez une ou plusieurs familles de produits :",
    options=df["family"].unique().tolist(),
    default=df["family"].unique().tolist(),
)

# Message si rien n'est sélectionné
if not selected_families:
    st.warning(
        "Veuillez sélectionner au moins une famille pour afficher les visualisations."
    )
    st.stop()

# Graphe 1 : Saisonnalité
st.subheader("📅 Saisonnalité des ventes")
seasonality_df = compute_seasonality(df, selected_families)
plot_seasonality(seasonality_df)
st.subheader("💬 Commentaires de l'analyse saisonnière")
for family in selected_families:
    st.markdown(comments_data["analyses"]["saisonnalite"][family])

# Graphe 2 : Concentration des ventes
st.subheader("📦 Répartition des ventes par produit")
distribution_df = compute_family_distribution(df, selected_families)
plot_family_distribution(distribution_df, selected_families)
st.subheader("💬 Commentaires de la répartition des ventes")
for family in selected_families:
    st.markdown(comments_data["analyses"]["repartition_ventes"][family])
