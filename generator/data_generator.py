import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
import uuid
import os

# Initialisation
fake = Faker()
random.seed(42)
np.random.seed(42)

# Paramètres
n_clients = 500
n_products_per_family = 30
families = ["Hoodie", "Shirt", "Activewear"]
channels = ["Store", "Online", "Mobile"]
dates = pd.date_range(start="2022-03-01", end="2024-02-29", freq="D")

# Génération des produits
products = []
for family in families:
    for i in range(n_products_per_family):
        product_id = f"P{len(products):04}"
        label = f"{family} {random.choice(['Z', 'X', 'M', 'A'])}{random.randint(10,99)}"
        price = round(np.random.uniform(20, 100), 2)
        products.append({
            "product_id": product_id,
            "product_label": label,
            "family": family,
            "price_initial": price
        })

products_df = pd.DataFrame(products)

# Création de segments de clients
client_profiles = {}
for i in range(1, n_clients + 1):
    profile = random.choices(
        population=["sportif", "formel", "urbain"],
        weights=[0.3, 0.3, 0.4],
        k=1
    )[0]
    client_profiles[f"C{i:04}"] = profile

# Génération des transactions sous forme de paniers
transactions = []
for _ in range(6000):
    client_id = f"C{random.randint(1, n_clients):04}"
    profile = client_profiles[client_id]
    date = random.choice(dates)
    channel = random.choice(channels)
    n_items = np.random.randint(2, 6)

    # Choix biaisé des familles selon le profil
    if profile == "sportif":
        chosen_families = random.choices(families, weights=[1, 1, 8], k=n_items)
    elif profile == "formel":
        chosen_families = random.choices(families, weights=[1, 8, 1], k=n_items)
    else:
        chosen_families = random.choices(families, weights=[3, 3, 3], k=n_items)

    for fam in chosen_families:
        product = products_df[products_df["family"] == fam].sample(1).iloc[0]
        quantity = np.random.poisson(1) + 1
        base_price = product["price_initial"]

        if channel == "Online":
            discount = round(np.random.uniform(0.1, 0.3) * base_price, 2)
        else:
            discount = round(np.random.uniform(0.0, 0.2) * base_price, 2)

        price_sold = max(0.0, base_price - discount)
        revenue = round(price_sold * quantity, 2)

        transactions.append({
            "transaction_id": str(uuid.uuid4()),
            "client_id": client_id,
            "date": date.strftime("%Y-%m-%d"),
            "channel": channel,
            "product_id": product["product_id"],
            "product_label": product["product_label"],
            "family": product["family"],
            "price_initial": base_price,
            "price_sold": price_sold,
            "discount_amount": round(discount * quantity, 2),
            "quantity": quantity,
            "revenue": revenue
        })

transactions_df = pd.DataFrame(transactions)

# Ajout de valeurs aberrantes
outlier_indices = np.random.choice(transactions_df.index, size=20, replace=False)
transactions_df.loc[outlier_indices, "quantity"] = transactions_df.loc[outlier_indices, "quantity"] * 10
transactions_df.loc[outlier_indices, "revenue"] = transactions_df.loc[outlier_indices, "price_sold"] * transactions_df.loc[outlier_indices, "quantity"]

# Ajout de valeurs manquantes (NaN)
nan_indices = np.random.choice(transactions_df.index, size=20, replace=False)
transactions_df.loc[nan_indices, "price_sold"] = np.nan

# Sauvegarde dans un CSV local
os.makedirs("data/raw", exist_ok=True)
transactions_df.to_csv("data/raw/transactions.csv", index=False)
print("Fichier généré : data/raw/transactions.csv")