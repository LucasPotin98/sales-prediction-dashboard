import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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

def plot_product_graph(G, color_map=None):
    pos = nx.spring_layout(G, seed=42)
    
    # Edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Nodes
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    communities = sorted(set(color_map.values())) if color_map else [0]
    
    # Using 'Set1' for distinct colors
    colormap = cm.get_cmap('Set1', len(communities))  # Using Set1 for better contrast
    color_lookup = {com: f"rgba{tuple(int(255*c) for c in colormap(i)[:3]) + (0.9,)}" for i, com in enumerate(communities)}

    # Initializing all nodes to black before assigning their community color
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
        com = color_map[node] if color_map else 0
        node_colors.append('black')  # Default color for nodes is black
    # Now assign the correct color based on the community
    if color_map:
        for i, node in enumerate(G.nodes()):
            com = color_map[node] if color_map else 0
            node_colors[i] = color_lookup[com]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        marker=dict(
            color=node_colors,
            size=15,
            line=dict(width=2, color='DarkSlateGrey')
        ),
        hoverinfo='text',
        showlegend=False  # Important : on ne montre pas cette trace dans la légende
    )

    # Ajouter les annotations manuelles pour chaque communauté
    annotations = []
    c=0
    if color_map:
        for i, (com, color) in enumerate(color_lookup.items()):
            annotations.append(
                dict(
                    x=1.02, y=1 - (i * 0.1),  # Ajuster la position de la légende
                    xref="paper", yref="paper",
                    text="Communauté " + str(c)+ "X",
                    showarrow=False,
                    font=dict(size=12, color=color),
                    bgcolor='rgba(0,0,0,0)',  # Fond transparent pour éviter une couleur indésirable
                    align='left'
                )
            )
            c+=1

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        annotations=annotations  # Ajouter les annotations ici pour les légendes
                    ))
    
    # === Légende stylée, à droite ===
    #Si plusieurs communautés, on les affiche
    if len(communities) > 1:
        annotations = []
        for i, color in color_lookup.items():
            annotations.append(dict(
                x=1.00, y=0.20 - 0.06 * i,
                xref="paper", yref="paper",
                text=f"<b>Communauté {i}</b>",
                showarrow=False,
                font=dict(size=16, color="white"),
                bgcolor=color,
                bordercolor="white",
                borderwidth=1,
                align="center",
                opacity=0.9
            ))
        fig.update_layout(annotations=annotations)
    return fig


