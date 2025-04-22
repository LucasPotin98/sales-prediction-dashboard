import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime
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

# Dates : 24 mois train + 6 mois test
train_dates = pd.date_range(start="2022-03-01", end="2024-02-29", freq="D")
test_dates = pd.date_range(start="2024-03-01", end="2024-08-31", freq="D")

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

# Profils clients
client_profiles = {
    f"C{i:04}": random.choices(["sportif", "formel", "urbain"], weights=[0.3, 0.3, 0.4])[0]
    for i in range(1, n_clients + 1)
}

# 🎯 Saisonnalité par famille et mois
def seasonal_multiplier(family, month):
    if family == "Activewear":
        if month == 5:
            return 2.0
        elif month == 6:
            return 3.2  # pic fort début été
        elif month in [7, 8]:
            return 2.5  # plateau lissé
        else:
            return 0.6
    elif family == "Hoodie":
        return 1.1  # quasi-plat, dépend surtout des remises
    elif family == "Shirt":
        if month in [4, 5]:
            return 2.2
        elif month in [3, 9]:
            return 1.5
        else:
            return 0.9
    return 1.0

# 🔁 Modulateur hebdo doux (forme en cloche)
def weekly_modulator(week_number):
    return 1 + 0.5 * np.exp(-((week_number - 26) / 4)**2)

# 🔄 Génération des transactions
def generate_transactions(dates, n_days):
    transactions = []

    # Base stable hebdomadaire
    weekly_quantity_map = {}
    for date in dates:
        year, week = date.isocalendar().year, date.isocalendar().week
        for fam in families:
            weekly_quantity_map[(year, week, fam)] = np.random.normal(loc=5, scale=0.1)

    for _ in range(n_days):
        client_id = f"C{random.randint(1, n_clients):04}"
        profile = client_profiles[client_id]
        date = random.choice(dates)
        month = date.month
        channel = random.choice(channels)
        n_items = np.random.randint(2, 6)

        if profile == "sportif":
            chosen_families = random.choices(families, weights=[1, 1, 8], k=n_items)
        elif profile == "formel":
            chosen_families = random.choices(families, weights=[1, 8, 1], k=n_items)
        else:
            chosen_families = random.choices(families, weights=[3, 3, 3], k=n_items)

        for fam in chosen_families:
            product = products_df[products_df["family"] == fam].sample(1).iloc[0]
            base_price = product["price_initial"]

            year, week = date.isocalendar().year, date.isocalendar().week
            base_quantity = weekly_quantity_map[(year, week, fam)]
            seasonal = seasonal_multiplier(fam, month)
            modulation = weekly_modulator(week)
            quantity = int(np.clip(base_quantity * seasonal * modulation, 1, 25))

            discount = round(
                np.random.uniform(0.1, 0.3 if channel == "Online" else 0.2) * base_price, 2
            )
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

    return pd.DataFrame(transactions)

# Génération des jeux
train_df = generate_transactions(train_dates, n_days=6000)
test_df = generate_transactions(test_dates, n_days=1500)

# 💥 Anomalies et NaN dans le train
outliers = np.random.choice(train_df.index, size=20, replace=False)
train_df.loc[outliers, "quantity"] *= 10
train_df.loc[outliers, "revenue"] = train_df.loc[outliers, "price_sold"] * train_df.loc[outliers, "quantity"]

nans = np.random.choice(train_df.index, size=20, replace=False)
train_df.loc[nans, "price_sold"] = np.nan

# Sauvegarde
os.makedirs("data/raw", exist_ok=True)
train_df.to_csv("data/raw/transactions.csv", index=False)
test_df.to_csv("data/raw/transactions_test.csv", index=False)

print("✅ Données régénérées avec saisonnalité lissée, modulateur hebdo et structure réaliste.")
