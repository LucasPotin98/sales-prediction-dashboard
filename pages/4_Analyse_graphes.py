import streamlit as st
import pandas as pd
from app.utils import load_data
from app.figures import plot_product_graph
from src.graphes import build_graph_cooccurrence, compute_louvain_communities

st.set_page_config(page_title="🔗 Analyse Graphe", page_icon="🔗")

# Barre de navigation
st.markdown(f"""
<style>
.navbar {{
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
}}

.navbar a {{
    text-decoration: none;
    color: #f0f0f0;
    padding: 0.4rem 1.2rem;
    border-radius: 6px;
    transition: background-color 0.2s ease;
}}

.navbar a:hover {{
    background-color: #333333;
}}

.navbar a.active {{
    background-color: #6c63ff;
    color: white;
}}
</style>

<div class="navbar">
    <a href="/" target="_self">📦 Contexte</a>
    <a href="/Analyse_ventes" target="_self">📊 Analyse</a>
    <a href="/Modelisation" target="_self">🧠 Modélisation</a>
    <a href="/Analyse_graphes" class="active" target="_self">🔗 Graphes</a>
</div>
""", unsafe_allow_html=True)

# Titre
st.title("🔗 Analyse par graphes")

# Intro
st.markdown(
    "Les graphes permettent de **modéliser les relations entre produits**, ici à travers les co-achats :\n"
    "- Chaque nœud est un produit\n"
    "- Une arête relie deux produits s’ils ont été achetés ensemble plusieurs fois\n\n"
    "Cela permet de repérer des **combinaisons fréquentes**, utiles en placement produit, en recommandation ou en analyse marketing."
)

# Chargement des données
df = load_data()

nb_products = st.slider(
    "Nombre de produits à inclure dans le graphe :",
    min_value=10,
    max_value=50,
    value=30,
    step=5,
    help="Seuls les produits les plus vendus seront pris en compte."
)

# Filtrage dynamique selon le slider
top_products = df["product_label"].value_counts().head(nb_products).index.tolist()
df_top = df[df["product_label"].isin(top_products)]

# Construction du graphe
G = build_graph_cooccurrence(df_top, min_edge_weight=20)


# Titre + bouton centrés
with st.container():
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.markdown("### 🎯 Graphe de co-achats entre produits", unsafe_allow_html=True)
    detect = st.button("🧩 Détecter les communautés (Louvain)", use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)

# Application de Louvain si cliqué
color_map = None
if detect:
    communities = compute_louvain_communities(G)
    color_map = {node: i for i, com in enumerate(communities) for node in com}

# Affichage du graphe
fig = plot_product_graph(G, color_map=color_map)
st.plotly_chart(fig, use_container_width=True)

if color_map:
    st.markdown(f"**Nombre de communautés détectées : {len(set(color_map.values()))}**")

    st.markdown(
        """
        📌 La détection des communautés dans un graphe permet de repérer des groupes de produits qui sont fréquemment achetés ensemble.\n
        Cela peut être utile pour optimiser le placement des produits en magasin et comprendre les comportements d'achat des clients
        """
    )