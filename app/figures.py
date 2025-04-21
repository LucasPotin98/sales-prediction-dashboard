import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go

def plot_seasonality(seasonality_df):
    fig = px.line(
        seasonality_df,
        x='month', y='quantity', color='family', markers=True,
        title='Évolution mensuelle des ventes par famille'
    )
    fig.update_layout(xaxis_title='Mois', yaxis_title='Quantité vendue')
    st.plotly_chart(fig, use_container_width=True)


def plot_family_distribution(grouped_df, selected_families):
    n = len(selected_families)
    cols = st.columns(n)

    if n == 1:
        top_n = 20
    elif n == 2:
        top_n = 10
    else:
        top_n = 5

    for i, fam in enumerate(selected_families):
        fam_data = grouped_df[grouped_df['family'] == fam].sort_values('quantity', ascending=False).head(top_n)

        with cols[i]:
            st.subheader(f"Répartition – {fam}")
            fig = px.bar(
                fam_data,
                x='product_label',
                y='quantity',
                title=None
            )
            fig.update_layout(
                xaxis_title=None,
                yaxis_title="Quantité",
                margin=dict(t=10)
            )
            st.plotly_chart(fig, use_container_width=True)

def plot_predictions_vs_truth(df_eval, y_col="quantity", pred_col="prediction", family_name=None):
    """
    Affiche la courbe des prédictions vs vérité terrain (valeurs réelles).
    
    Args:
        df_eval (DataFrame): Doit contenir au moins ['date', y_col, pred_col]
        y_col (str): Nom de la colonne des valeurs réelles
        pred_col (str): Nom de la colonne des prédictions
        family_name (str): Nom affiché dans le titre du graphe
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_eval["date"], y=df_eval[y_col],
        mode='lines+markers', name="Ventes réelles"
    ))
    fig.add_trace(go.Scatter(
        x=df_eval["date"], y=df_eval[pred_col],
        mode='lines+markers', name="Prévisions modèle"
    ))
    
    fig.update_layout(
        title=f"Prédictions vs Réalité – {family_name}" if family_name else "Prédictions vs Réalité",
        xaxis_title="Date",
        yaxis_title="Quantité",
        template="plotly_white"
    )
    
    return fig