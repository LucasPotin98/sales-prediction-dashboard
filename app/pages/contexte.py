import streamlit as st
from app.utils import load_data, get_kpis

# Chargement des données
st.cache_data

df = load_data()
kpis = get_kpis(df)

# Titre & intro
st.title(" Projet Retail - Analyse & Prédiction des ventes")

st.markdown(
    "Bienvenue sur cette application interactive de data science appliquée à un cas retail simulé.\n"
    "Ce projet vise à explorer les dynamiques d'achat, modéliser les ventes et détecter des patterns \n"
    "grâce à une approche combinant modèles temporels classiques et analyse par graphes."
)

# Aperçu des données
st.subheader("📦 Jeu de données simulé")
st.markdown(
    "Ce jeu de données couvre une période d'un an (mars 2023 - février 2024),\n"
    "avec 90 produits répartis en 3 familles, et 500 clients différents.\n"
    "Les achats sont simulés sous forme de paniers multi-produits, avec canaux et remises."
)

# KPIs clés
col1, col2, col3 = st.columns(3)
col1.metric("Transactions", f"{kpis['transactions']:,}")
col2.metric("Produits uniques", kpis['produits_uniques'])
col3.metric("Clients", kpis['clients'])

col4, col5 = st.columns(2)
col4.metric("Chiffre d'affaires total (€)", f"{kpis['revenu_total']:,.2f}")
col5.metric("Quantités vendues", f"{kpis['quantite_totale']:,}")

# Aperçu tableau
st.subheader("🧾 Exemple de données")
st.dataframe(df.head())

# Liens
st.markdown("[🔗 Code source sur GitHub](https://github.com/...)  ")
st.download_button("📥 Télécharger les données", data=df.to_csv(index=False), file_name="transactions.csv")
