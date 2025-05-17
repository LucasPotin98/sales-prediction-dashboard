import streamlit as st
from app.utils import load_data, get_kpis

st.set_page_config(page_title="Contexte & Données", page_icon="📦")

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
    <a href="/" class="active" target="_self">📦 Contexte</a>
    <a href="/Analyse_ventes" target="_self">📊 Analyse</a>
    <a href="/Modelisation" target="_self">🧠 Modélisation</a>
    <a href="/Analyse_graphes" target="_self">🔗 Graphes</a>
</div>
""",
    unsafe_allow_html=True,
)

df = load_data()
kpis = get_kpis(df)

# Titre & intro
st.title("Analyse & Prédiction des ventes")

st.markdown("""
    Dans un contexte de gestion des stocks et d’optimisation des promotions, il est nécessaire de prédire la demande future des produits \n
    Ce projet vise à explorer les dynamiques d'achat, modéliser les ventes et détecter des patterns grâce à une approche combinant modèles temporels classiques et analyse par graphes.
""")

# Aperçu des données
st.subheader("📦 Jeu de données simulé")
st.markdown("Ce jeu de données couvre une période de 2 ans (mars 2022 - février 2024).")

# KPIs clés
col1, col2, col3 = st.columns(3)
col1.metric("Transactions", f"{kpis['transactions']:,}")
col2.metric("Produits uniques", kpis["produits_uniques"])
col3.metric("Clients", kpis["clients"])

col4, col5 = st.columns(2)
col4.metric("Chiffre d'affaires total (€)", f"{kpis['revenu_total']:,.2f}")
col5.metric("Quantités vendues", f"{kpis['quantite_totale']:,}")

# Aperçu tableau
st.subheader("🧾 Exemple de données")
st.dataframe(df.head())

# Liens
st.markdown(
    "[🔗 Code source sur GitHub](https://github.com/LucasPotin98/sales-prediction-dashboard)  "
)
st.download_button(
    "📥 Télécharger les données",
    data=df.to_csv(index=False),
    file_name="sales_transactions.csv",
)
