import streamlit as st
import pandas as pd
from app.utils import load_data
from app.figures import plot_product_graph
from src.graphes import build_graph_cooccurrence, compute_louvain_communities

st.set_page_config(page_title="🔗 Analyse Graphe", page_icon="🔗")
st.title("🔗 Analyse par Graphes")

st.markdown(
    "Cette page utilise des représentations en **graphe** pour explorer les comportements d'achat :\n"
    "- Quelles familles de produits sont fréquemment achetées ensemble ?\n"
    "- Peut-on détecter des regroupements naturels dans les ventes ?"
)

# Chargement des données
df = load_data()

# Choix du type de graphe
graph_type = st.radio(
    "Type de graphe à visualiser :",
    ["Graphe de co-achats entre produits", "Réseau biparti client-produit"]
)

# === GRAPHE DE CO-ACHATS ===
if graph_type == "Graphe de co-achats entre produits":

    st.markdown("💡 Mode : **Top 30 produits les plus vendus**, arêtes pour co-achats fréquents")

    # Filtrage des produits les plus vendus
    top_products = (
        df["product_label"]
        .value_counts()
        .head(30)
        .index
        .tolist()
    )
    df_top = df[df["product_label"].isin(top_products)]

    # Construction du graphe
    G = build_graph_cooccurrence(df_top, min_edge_weight=20)

    # Bouton Louvain
    color_map = None
    if st.button("🧩 Détecter les communautés (Louvain)"):
        communities = compute_louvain_communities(G)
        color_map = {}
        for i, com in enumerate(communities):
            for node in com:
                color_map[node] = i

    # Affichage du graphe (avec ou sans couleur)
    fig = plot_product_graph(G, color_map=color_map)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "📌 *Chaque nœud représente un produit. Deux produits sont reliés s'ils ont été achetés ensemble plusieurs fois.*\n"
        "Cela permet de repérer des combinaisons fréquentes ou des logiques de placement produit."
    )
