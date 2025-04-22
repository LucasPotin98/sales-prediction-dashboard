import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime
import uuid
import os


def generate_discount_and_promotion_data(families, years, weeks):
    avg_discount_dict = {
        "Shirt": {y: {w: np.random.uniform(0.05, 0.20) for w in weeks} for y in years},
        "Activewear": {y: {w: np.random.uniform(0.10, 0.25) for w in weeks} for y in years},
        "Hoodie": {y: {w: np.random.uniform(0.02, 0.15) for w in weeks} for y in years}
    }

    promotion_type_dict = {
        fam: {y: {w: np.random.choice(["online", "store", "both", "none"]) for w in weeks} for y in years}
        for fam in families
    }

    discount_data = []
    promotion_data = []

    for family in families:
        for year in years:
            for week in weeks:
                avg_discount = avg_discount_dict[family][year][week]
                promotion_type = promotion_type_dict[family][year][week]
                discount_data.append([family, year, week, avg_discount])
                promotion_data.append([family, year, week, promotion_type])

    discount_df = pd.DataFrame(discount_data, columns=["family", "year", "week", "avg_discount"])
    promotion_df = pd.DataFrame(promotion_data, columns=["family", "year", "week", "promotion_type"])

    discount_df.to_csv("data/avg_discount.csv", index=False)
    promotion_df.to_csv("data/promotion_type.csv", index=False)

    return discount_df, promotion_df


def seasonal_multiplier(family, month):
    if family == "Activewear":
        if month == 5:
            return 2.0
        elif month == 6:
            return 3.2
        elif month in [7, 8]:
            return 2.5
        else:
            return 0.6
    elif family == "Hoodie":
        return 1.1
    elif family == "Shirt":
        if month in [4, 5]:
            return 2.2
        elif month in [3, 9]:
            return 1.5
        else:
            return 0.9
    return 1.0


def weekly_modulator(week_number):
    return 1 + 0.5 * np.exp(-((week_number - 26) / 4)**2)


def generate_transactions(dates, n_days):
    transactions = []

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
            seasonal = seasonal_multiplier(fam, month)
            modulation = weekly_modulator(week)

            promo_type = promotion_df.query(
                "family == @fam and year == @year and week == @week"
            )["promotion_type"].values[0] if not promotion_df.empty else "none"

            if fam == "Activewear":
                quantity = int(np.clip(5 * seasonal * modulation, 1, 25))
            elif fam == "Hoodie":
                promo_boost = {
                    "online": 2.0,
                    "store": 1.5,
                    "both": 2.5,
                    "none": 1.0
                }.get(promo_type, 1.0)
                quantity = int(np.clip(5 * promo_boost, 1, 25))
            elif fam == "Shirt":
                promo_boost = {
                    "online": 1.1,
                    "store": 1.0,
                    "both": 1.2,
                    "none": 1.0
                }.get(promo_type, 1.0)
                quantity = int(np.clip(5 * seasonal * modulation * promo_boost, 1, 25))

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


# Initialisation
fake = Faker()
random.seed(42)
np.random.seed(42)

n_clients = 500
n_products_per_family = 30
families = ["Hoodie", "Shirt", "Activewear"]
channels = ["Store", "Online"]
years = [2022, 2023, 2024]
weeks = list(range(1, 53))

train_dates = pd.date_range(start="2022-03-01", end="2024-02-29", freq="D")
test_dates = pd.date_range(start="2024-03-01", end="2024-08-31", freq="D")

discount_df, promotion_df = generate_discount_and_promotion_data(families, years, weeks)

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

client_profiles = {
    f"C{i:04}": random.choices(["sportif", "formel", "urbain"], weights=[0.3, 0.3, 0.4])[0]
    for i in range(1, n_clients + 1)
}

train_df = generate_transactions(train_dates, n_days=6000)
test_df = generate_transactions(test_dates, n_days=1500)

outliers = np.random.choice(train_df.index, size=20, replace=False)
train_df.loc[outliers, "quantity"] *= 10
train_df.loc[outliers, "revenue"] = train_df.loc[outliers, "price_sold"] * train_df.loc[outliers, "quantity"]

nans = np.random.choice(train_df.index, size=20, replace=False)
train_df.loc[nans, "price_sold"] = np.nan

os.makedirs("data/raw", exist_ok=True)
train_df.to_csv("data/raw/transactions.csv", index=False)
test_df.to_csv("data/raw/transactions_test.csv", index=False)

print("✅ Données régénérées avec comportements différenciés par famille.")
