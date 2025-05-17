import networkx as nx
from itertools import combinations
from collections import Counter
from networkx.algorithms.community import louvain_communities


def build_graph_cooccurrence(df, min_edge_weight=2):
    """
    Construit un graphe de co-achats entre produits.
    Une arête relie deux produits achetés ensemble dans un même panier.
    """
    G = nx.Graph()

    # Étape 1 : pour chaque panier (client + date), extraire les produits
    grouped = df.groupby(["client_id", "date"])["product_label"].apply(list)

    # Étape 2 : générer les paires de produits
    edges = []
    for products in grouped:
        if len(products) >= 2:
            edges.extend(combinations(sorted(products), 2))

    # Étape 3 : compter les co-occurrences
    edge_counts = Counter(edges)

    # Étape 4 : ajouter les arêtes avec poids
    for (prod1, prod2), weight in edge_counts.items():
        if weight >= min_edge_weight:
            G.add_edge(prod1, prod2, weight=weight)

    return G


def compute_louvain_communities(G: nx.Graph):
    """
    Détection de communautés avec l’algorithme de Louvain (via NetworkX >= 3.0)

    Args:
        G (nx.Graph): graphe de co-achats ou clients

    Returns:
        List[Set[str]]: communautés détectées
    """
    return louvain_communities(G, seed=42)  # seed pour reproductibilité
