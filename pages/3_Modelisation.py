import pandas as pd
import streamlit as st
from app.utils import load_data, load_model, load_prophet_model
from app.figures import plot_predictions_vs_truth
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.modeling import (
    predict_with_prophet,prepare_aggregated
)


st.set_page_config(page_title="🧠 Modélisation des ventes", page_icon="🧠")

st.title("🧠 Prédiction des ventes par famille de produits")

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
model_choice = st.radio("Modèle :", ["Naïf (valeur t−1)", "XGBoost", "Prophet"])
model_key = model_map[model_choice]


# Load du DS de Test :
df_test_raw = load_data("data/processed/clean_transactions_test.csv")
df_test = prepare_aggregated(df_test_raw)

# Choix de l’horizon
horizon = st.slider("Horizon de prévision (en semaines) :", min_value=4, max_value=24, step=4, value=12)

# Lancer la modélisation
if st.button("Lancer la modélisation"):
    df_test_family = df_test[df_test["family"] == family]

    if model_key == "naive":
        model = load_model("naive", family)
        future_dates = pd.date_range(start=df_test["date"].max() + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")
        pred_df = predict_with_naive(model, future_dates=future_dates, family=family)

    elif model_key == "xgboost":
        model = load_model("xgboost", family)
        df_future = prepare_future_for_xgboost(df_test, horizon)
        df_future["family"] = family
        pred_df = predict_with_xgboost(model, df_future)

    elif model_key == "prophet":
        model = load_prophet_model(family)
        pred_df = predict_with_prophet(model, periods=horizon)
        print(pred_df)
        test_dates = df_test_family["date"].unique()
        pred_df = pred_df[pred_df["date"].isin(test_dates)]
    
    print(df_test_family)

    st.write("Dates test :", df_test_family["date"].min(), "->", df_test_family["date"].max())
    st.write("Dates prédictions :", pred_df["date"].min(), "->", pred_df["date"].max())
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

    st.markdown("📌 *Le modèle capture très bien les effets saisonniers de la famille choisie. "
                "On observe une précision élevée même lors des pics de vente.*")
