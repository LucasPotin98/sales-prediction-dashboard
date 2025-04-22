import json
import pandas as pd
import streamlit as st
from app.utils import load_data, load_model, load_prophet_model, load_all_data
from app.figures import plot_predictions_vs_truth
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.modeling import (
    predict_with_prophet,prepare_aggregated,predict_with_naive,predict_with_xgboost
)


st.set_page_config(page_title="🧠 Modélisation des ventes", page_icon="🧠")

with open("commentaires/commentaires.json", "r") as f:
    comments_data = json.load(f)


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
    <a href="/Modelisation" class="active" target="_self">🧠 Modélisation</a>
    <a href="/Analyse_graphes" target="_self">🔗 Graphes</a>
</div>
""", unsafe_allow_html=True)


st.markdown("<h2 style='margin-bottom: 1rem;'>🧠 Prédiction des ventes par famille</h2>", unsafe_allow_html=True)

st.markdown(
    "Cette page propose une modélisation supervisée pour prédire les quantités vendues par **famille de produits**, "
    "sur un horizon de plusieurs semaines. Le modèle s’appuie sur les historiques agrégés hebdomadairement, enrichis de variables temporelles."
)

# Choix de la famille
family = st.selectbox("Famille de produits :", ["Shirt", "Activewear", "Hoodie"])

# Choix du modèle
model_map = {
    "Naïf (valeur t−1)": "naive",
    "XGBoost": "xgboost",
    "Prophet": "prophet"
}

best_models_by_family = {
    "Shirt": "XGBoost",
    "Hoodie": "XGBoost",
    "Activewear": "Prophet"
}

best_model = best_models_by_family.get(family, "N/A")
st.markdown(f"🧠 **Modèle recommandé pour cette famille** : `{best_model}`")

model_choice = st.radio("Modèle :", ["Naïf (valeur t−1)", "XGBoost", "Prophet"])
model_key = model_map[model_choice]

# Affichage des commentaires avant de lancer la modélisation
st.subheader("💬 Commentaires sur la modélisation")
if model_key == "naive":
    st.markdown("### Modèle Naïf (valeur t−1)")
elif model_key == "xgboost":
    st.markdown("### Modèle XGBoost")
elif model_key == "prophet":
    st.markdown("### Modèle Prophet")

st.markdown(f"**Méthodologie** : {comments_data['modelisation'][model_key]['methodologie']}")
st.markdown(f"**Features** : {', '.join(comments_data['modelisation'][model_key]['features'])}")
st.markdown(f"**Avantages** : {comments_data['modelisation'][model_key]['avantages']}")
st.markdown(f"**Inconvénients** : {comments_data['modelisation'][model_key]['inconvenients']}")


# Load du DS de Test :
df_train, df_test = load_all_data()

# Choix de l’horizon
horizon = st.slider("Horizon de prévision (en semaines) :", min_value=4, max_value=24, step=4, value=12)

# Lancer la modélisation
if st.button("Lancer la modélisation"):
    df_train_family = df_train[df_train["family"] == family]
    df_test_family = df_test[df_test["family"] == family]

    if model_key == "naive":
        pred_df = predict_with_naive(df_train_family, periods=horizon)

    elif model_key == "xgboost":
        model = load_model("xgboost", family)
        pred_df = predict_with_xgboost(model, horizon,df_train_family,family)

    elif model_key == "prophet":
        model = load_prophet_model(family)
        pred_df = predict_with_prophet(model, periods=horizon)
    test_dates = df_test_family["date"].unique()
    pred_df = pred_df[pred_df["date"].isin(test_dates)]
    

    df_eval = df_test_family.merge(pred_df, on="date", how="inner")
    y_true = df_eval["quantity"]
    y_pred = df_eval["prediction"]
    
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    st.subheader("📈 Courbe des ventes réelles vs prédites")
    
    fig = plot_predictions_vs_truth(df_eval, family_name=family)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Évaluation du modèle")
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse:,.0f}")
    col2.metric("MAE", f"{mae:,.0f}")
    col3.metric("R²", f"{r2:.3f}")

    st.subheader("📝 Interprétation des résultats")
    commentaire_model = comments_data.get("modelisation", {}).get(model_key, {}).get("analyse_familles", {}).get(family)

    if commentaire_model:
        st.markdown(f"**{commentaire_model}**")
    else:
        st.info("Aucun commentaire spécifique n’est encore défini pour cette combinaison famille/modèle.")
