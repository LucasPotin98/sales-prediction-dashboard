import pandas as pd
import plotly.express as px
import streamlit as st

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
        top_n = 12
    else:
        top_n = 7

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
