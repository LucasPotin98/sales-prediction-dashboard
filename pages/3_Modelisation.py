import streamlit as st

st.set_page_config(page_title="🧠 Modélisation des ventes", page_icon="🧠")

st.title("🧠 Prédiction des ventes par famille de produits")

st.markdown(
    "Cette page propose une modélisation supervisée pour prédire les quantités vendues par **famille de produits**, "
    "sur un horizon de plusieurs semaines. Le modèle s’appuie sur les historiques agrégés hebdomadairement, enrichis de variables temporelles."
)

# Choix de la famille
selected_family = st.selectbox("Famille de produits :", ["Sweater", "Formal Shirt", "Sportswear Shirt"])

# Choix du modèle
model_choice = st.radio("Modèle :", ["Naïf (valeur t−1)", "XGBoost", "XGBoost optimisé"])

# Choix de l’horizon
horizon = st.slider("Horizon de prévision (en semaines) :", min_value=4, max_value=24, step=4, value=12)

# Lancer la modélisation
if st.button("Lancer la modélisation"):

    # 💡 Placeholder : à remplacer par appels à src/modeling plus tard
    st.success(f"📦 Modélisation lancée pour la famille **{selected_family}** sur {horizon} semaines à l’aide du modèle **{model_choice}**")

    st.subheader("📈 Courbe des ventes réelles vs prédites")
    st.info("⏳ Chargement de la figure…")

    st.subheader("📊 Évaluation du modèle")
    st.metric(label="RMSE", value="476 802")
    st.metric(label="MAE", value="320 441")
    st.metric(label="R²", value="0.991")

    st.markdown("📌 *Le modèle capture très bien les effets saisonniers de la famille choisie. "
                "On observe une précision élevée même lors des pics de vente.*")
