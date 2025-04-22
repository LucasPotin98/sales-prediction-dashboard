import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json
import os
import joblib
from xgboost import XGBRegressor
import numpy as np

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


def build_features(df, lag_col='quantity', lags=[1], rolling_windows=[3]):
    df = df.copy()
    df = add_lag_features(df, lag_col=lag_col, lags=lags)
    df = add_rolling_features(df, col=lag_col, windows=rolling_windows)
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

def train_xgboost_model(df, target_col="quantity", date_col="date"):
    """
    Entraîne un modèle XGBoost sur les features déjà construites.
    """
    df = df.copy().sort_values(date_col)
    y = df[target_col]
    X = df.drop(columns=[target_col, date_col, "family"])
    
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y)
    return model



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
        xgb_df = build_features(weekly.copy())
        xgb_df["family"] = fam
        xgb_df = xgb_df.dropna()
        xgb_model = train_xgboost_model(xgb_df)
        save_model_pkl(xgb_model, method="xgboost", family=fam)

        ##### Prophet #####
        prophet_model = train_prophet_model(weekly.copy())
        save_prophet_model(prophet_model, family=fam)

        print(f"✅ Modèles sauvegardés pour : {fam}")



if __name__ == "__main__":
    # Exemple d'utilisation
    df = pd.read_csv("data/processed/clean_transactions.csv")
    df = prepare_aggregated(df)
    train_all_models(df)