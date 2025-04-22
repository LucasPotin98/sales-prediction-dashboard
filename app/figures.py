import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import networkx as nx

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


import plotly.graph_objects as go
import networkx as nx

def plot_product_graph(G, color_map=None):
    pos = nx.spring_layout(G, seed=42)  # Disposition fixe pour reproductibilité

    # Nœuds
    node_x = []
    node_y = []
    node_text = []
    node_color = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
        if color_map:
            node_color.append(color_map.get(node, 0))  # Couleur par communauté
        else:
            node_color.append(0)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="bottom center",
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_color,
            size=20,
            colorbar=dict(
                thickness=15,
                title=dict(text='Communauté', side='right'),
                xanchor='left'
            ),
            line_width=2
        ),
        hoverinfo='text'
    )

    # Arêtes
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Figure finale
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Graphe de co-achats entre produits",
                        title_x=0.5,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    return fig



def plot_communities(G, communities):
    # Mapping noeud -> cluster id
    node_color_map = {}
    for i, community in enumerate(communities):
        for node in community:
            node_color_map[node] = i

    pos = nx.spring_layout(G, seed=42)

    edge_trace = go.Scatter(x=[], y=[], line=dict(width=0.5, color="#888"),
                            hoverinfo="none", mode="lines")
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_trace["x"] += (x0, x1, None)
        edge_trace["y"] += (y0, y1, None)

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="markers+text",
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            color=[],
            size=15,
            colorbar=dict(
                thickness=15,
                title="Cluster",
                xanchor="left",
                titleside="right"
            )
        )
    )

    for node in G.nodes():
        x, y = pos[node]
        node_trace["x"] += (x,)
        node_trace["y"] += (y,)
        node_trace["marker"]["color"] += (node_color_map[node],)
        node_trace["text"] += (node,)

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Communautés de produits (Louvain)",
                        title_x=0.5,
                        showlegend=False,
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)
                    ))
    return fig