import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json
import os
import joblib
from xgboost import XGBRegressor
import numpy as np

def load_discount_and_promo_dicts(discount_path="data/avg_discount.csv", promo_path="data/promotion_type.csv"):
    discount_df = pd.read_csv(discount_path)
    promo_df = pd.read_csv(promo_path)

    # avg_discount_dict[fam][year][week] = avg_discount
    avg_discount_dict = {}
    for _, row in discount_df.iterrows():
        fam, year, week, val = row["family"], int(row["year"]), int(row["week"]), float(row["avg_discount"])
        avg_discount_dict.setdefault(fam, {}).setdefault(year, {})[week] = val

    # promotion_type_dict[fam][year][week] = promotion_type
    promotion_type_dict = {}
    for _, row in promo_df.iterrows():
        fam, year, week, val = row["family"], int(row["year"]), int(row["week"]), row["promotion_type"]
        promotion_type_dict.setdefault(fam, {}).setdefault(year, {})[week] = val

    return avg_discount_dict, promotion_type_dict

def add_lag_features(df, lag_col='quantity', lags=[1]):
    for lag in lags:
        df[f"lag_{lag}"] = df[lag_col].shift(lag)
    return df

def add_rolling_features(df, col='quantity', windows=[3]):
    for window in windows:
        df[f"rolling_mean_{window}"] = df[col].rolling(window=window).mean().shift(1)
        df[f"ewma_{window}"] = df[col].ewm(span=window, adjust=False).mean().shift(1)
    return df


def prepare_aggregated(df, date_col='date', family_col='family', quantity_col='quantity'):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Calcul de la date de début de semaine (toujours un lundi)
    df["week_start"] = df[date_col] - pd.to_timedelta(df[date_col].dt.weekday, unit="D")

    # On extrait année, semaine et mois depuis cette date pivot
    df["year"] = df["week_start"].dt.isocalendar().year
    df["week"] = df["week_start"].dt.isocalendar().week
    df["month"] = df["week_start"].dt.month

    # Agrégation par semaine/famille
    weekly_sales = df.groupby([family_col, "year", "month", "week", "week_start"]).agg({
        "price_initial": "mean",
        "price_sold": "mean",
        "revenue": "sum",
        "discount_amount": "sum",
        quantity_col: "sum"
    }).reset_index()

    # On renomme week_start en date pour la suite (prévisions)
    weekly_sales = weekly_sales.rename(columns={"week_start": "date"})

    return weekly_sales


def add_temporal_features(df, date_col="date"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["month"] = df[date_col].dt.month
    df["year"] = df[date_col].dt.year
    df["week"] = df[date_col].dt.isocalendar().week
    df["week_start"] = df[date_col] - pd.to_timedelta(df[date_col].dt.weekday, unit='D')
    df["week_start"] = df["week_start"].dt.normalize()

    # Sorting by date
    df = df.sort_values(by=date_col)
    return df


####### Partie modèle NAIF ########
class NaiveRollingMeanModel:
    def __init__(self, forecast_df, window=3, variation_factor=0.05):
        """
        forecast_df : DataFrame contenant au moins ['date', 'family', 'quantity']
        window : Taille de la fenêtre pour la moyenne glissante (par exemple, 3 semaines)
        variation_factor : Facteur de variation ajouté pour rendre les prédictions moins statiques
        """
        self.lookup = forecast_df[["date", "quantity"]]
        self.window = window
        self.variation_factor = variation_factor

        # Pour une moyenne glissante sur X semaines, on doit garder les X dernière semaines. 

    def predict(self, horizon):
        """
        Prédit les quantités pour l'horizon donné. Utilise la moyenne glissante avec ajout de variation aléatoire.
        horizon : nombre de semaines à prédire
        family : famille de produits à prédire
        """
        predictions = []
        # Dernière date du dataset
        last_date = self.lookup["date"].max()

        # Pour chaque semaine de l'horizon, prédire la quantité en utilisant les X dernières semaines
        for i in range(horizon):
            # On calcule la date de la semaine à prédire
            next_date = last_date + pd.Timedelta(weeks=i + 1)

            # Calcul de la fenêtre (X dernières semaines)
            window_start = next_date - pd.Timedelta(weeks=self.window)
            window_data = self.lookup[(self.lookup["date"] >= window_start) & (self.lookup["date"] < next_date)]
            print(window_data)

            # Moyenne des quantités pour la fenêtre
            mean_quantity = window_data["quantity"].mean()

            # Variation aléatoire
            variation = np.random.uniform(-self.variation_factor, self.variation_factor) * mean_quantity
            prediction = mean_quantity + variation

            predictions.append({"date": next_date, "quantity": prediction})
            # On ajoute la prediction au lookup pour la prochaine itération
            self.lookup = pd.concat([self.lookup, pd.DataFrame({"date": [next_date], "quantity": [prediction]})], ignore_index=True)

        return pd.DataFrame(predictions)

def predict_with_naive(df, periods=6, freq="W-MON", date_col="date", quantity_col="quantity"):
    """
    Prédit les quantités avec le modèle Naïf (moyenne mobile sur 3 semaines) et ajout de variation
    """
    df = df.copy()
    df = add_temporal_features(df, date_col=date_col)

    # Créer une copie pour la prédiction avec une variation
    model = NaiveRollingMeanModel(df, window=3, variation_factor=0.05)
    predictions = model.predict(periods)
    #renomer quantity en prediction
    predictions = predictions.rename(columns={"quantity": "prediction"})
    
    return predictions


######## Partie XGBoost ########

def prepare_features(df, family, quantity_col="quantity"):
    df = df.copy()
    df = add_temporal_features(df)
    avg_discount_dict, promotion_type_dict = load_discount_and_promo_dicts()
    df["avg_discount"] = df.apply(
        lambda row: avg_discount_dict.get(family, {}).get(row["year"], {}).get(row["week"], 0), axis=1
    )

    # 📢 Type de promotion
    df["promotion_type"] = df.apply(
        lambda row: promotion_type_dict.get(family, {}).get(row["year"], {}).get(row["week"], "none"), axis=1
    )
    # One hot
    df["is_promo_online"] = df["promotion_type"].isin(["online", "both"]).astype(int)
    df["is_promo_store"] = df["promotion_type"].isin(["store", "both"]).astype(int)

    # Drop promotion type
    df = df.drop(columns=["promotion_type"])

    return df


def train_xgboost(X, y):
    
    # Créer le modèle XGBoost
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
    
    # Entraîner le modèle
    model.fit(X, y)
    
    return model

def predict_with_xgboost(model, horizon, X_train,family):
   
    last_date = pd.to_datetime(X_train["date"]).max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")
    avg_discount_dict, promotion_type_dict = load_discount_and_promo_dicts()
    predictions = []

    for date in future_dates:
        year = date.year
        month = date.month
        week = date.isocalendar().week

        # Récupération dans les dictionnaires
        avg_discount = avg_discount_dict.get(family, {}).get(year, {}).get(week, 0)
        promo_type = promotion_type_dict.get(family, {}).get(year, {}).get(week, "none")

        is_promo_online = int(promo_type in ["online", "both"])
        is_promo_store = int(promo_type in ["store", "both"])

        # Création du vecteur de features
        X_pred = pd.DataFrame([{
            "month": month,
            "year": year,
            "week": week,
            "avg_discount": avg_discount,
            "is_promo_online": is_promo_online,
            "is_promo_store": is_promo_store
        }])

        # Prédiction
        y_pred = model.predict(X_pred)[0]
        predictions.append({"date": date, "prediction": y_pred})
    return pd.DataFrame(predictions)


def save_model(model, model_name, family, path_dir="models"):
    # Crée le dossier si il n'existe pas
    os.makedirs(path_dir, exist_ok=True)

    # Définir le chemin du fichier pour sauvegarder
    filename = f"model_{model_name}_{family.lower()}.pkl"
    model_path = os.path.join(path_dir, filename)

    # Sauvegarde du modèle
    joblib.dump(model, model_path)

    print(f"Modèle {model_name} sauvegardé sous {model_path}")

####### Partie Prophet ########

def train_prophet_model(df, date_col="date", quantity_col="quantity"):
    df = df.copy()
    df = df.rename(columns={date_col: "ds", quantity_col: "y"})
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",           # utile si les variations saisonnières sont + fortes quand les ventes sont hautes
        changepoint_prior_scale=0.1,                 # plus de flexibilité sur les ruptures de tendance
        seasonality_prior_scale=15.0,                # on autorise de fortes variations saisonnières
        changepoint_range=0.9                        # on laisse Prophet capter des changements même en fin de période
    )
    model.fit(df)

    return model

def predict_with_prophet(model, periods=6, freq="W-MON", return_only_future=True):
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    forecast = forecast[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "prediction"})

    if return_only_future:
        last_train_date = model.history["ds"].max()
        forecast = forecast[forecast["date"] > last_train_date]

    return forecast

def save_prophet_model(model, family, path_dir="models"):
    """
    Sauvegarde un modèle Prophet sous format .json
    """
    filename = f"model_prophet_{family.lower()}.json"
    path = os.path.join(path_dir, filename)
    with open(path, "w") as fout:
        fout.write(model_to_json(model))





####### Train all models ########
def train_all_models(df):
    """
    Entraîne et sauvegarde 3 modèles (Naïf, XGBoost, Prophet) pour chaque famille.
    """
    families = df["family"].unique()

    for fam in families:
        print(f"🔁 Entraînement des modèles pour la famille : {fam}")

        # Filtrage + groupement hebdo
        df_fam = df[df["family"] == fam].copy()
        df_fam["date"] = pd.to_datetime(df_fam["date"])
        df_fam["week_start"] = df_fam["date"] - pd.to_timedelta(df_fam["date"].dt.weekday, unit='D')
        weekly = df_fam.groupby("week_start").agg({"quantity": "sum"}).reset_index()
        weekly = weekly.rename(columns={"week_start": "date"})

        ##### XGBoost #####
        xgb_df = prepare_features(weekly.copy(),fam)
        print(xgb_df)
        xgb_df = xgb_df.dropna()
        
        # Entraînement de XGBoost
        # Séparer les features et la cible
        X = xgb_df.drop(columns=["date", "quantity", "week_start"])
        print(X)
        y = xgb_df["quantity"]
        # Entraîner le modèle
        xgb_model = train_xgboost(X, y)
        save_model(xgb_model, "xgboost", family=fam)
        print(f"✅ Modèle XGBoost sauvegardé pour : {fam}")


        ##### Prophet #####
        prophet_model = train_prophet_model(weekly.copy())
        save_prophet_model(prophet_model, family=fam)

        print(f"✅ Modèles sauvegardés pour : {fam}")



if __name__ == "__main__":
    # Exemple d'utilisation
    df = pd.read_csv("data/processed/clean_transactions.csv")
    df = prepare_aggregated(df)
    train_all_models(df)