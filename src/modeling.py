import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json
import os
import joblib
from xgboost import XGBRegressor

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
    df = add_temporal_features(df, date_col=date_col)
    # Aggregation par semaine + année et date
    
    weekly_sales_by_family = df.groupby(['family', 'year','month', 'week','week_start']).agg(
        {
            'price_initial': 'mean',  # Average selling price
            'price_sold': 'mean',  # Average initial price
            'revenue': 'sum',
            'discount_amount': 'sum',
            'quantity': 'sum'  # Total quantity sold
        }
    ).reset_index()
    weekly_sales_by_family = weekly_sales_by_family.rename(columns={"week_start": "date"})
    return weekly_sales_by_family


def build_features(df, lag_col='quantity', lags=[1], rolling_windows=[3]):
    df = df.copy()
    df = add_lag_features(df, lag_col=lag_col, lags=lags)
    df = add_rolling_features(df, col=lag_col, windows=rolling_windows)
    return df


####### Partie modèle NAIF ########
class NaiveMeanModel:
    def __init__(self, forecast_df):
        """
        forecast_df : DataFrame contenant au moins ['date', 'family', 'rolling_mean_t1']
        """
        self.lookup = forecast_df[["date", "family", "rolling_mean_t1"]]

    def predict(self, dates, family):
        df = self.lookup.copy()
        sub = df[(df["family"] == family) & (df["date"].isin(dates))]
        return sub.sort_values("date")["rolling_mean_t1"].values


def add_temporal_features(df, date_col="date"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["month"] = df[date_col].dt.month
    df["year"] = df[date_col].dt.year
    df["week"] = df[date_col].dt.isocalendar().week
    df["week_start"] = df[date_col] - pd.to_timedelta(df[date_col].dt.weekday, unit='D')
    df["week_start"] = df["week_start"].dt.normalize()
    return df



def save_model_pkl(model, method, family, path_dir="models"):
    """
    Sauvegarde un modèle XGBoost ou Naïf sous format .pkl
    """
    filename = f"model_{method.lower()}_{family.lower()}.pkl"
    path = os.path.join(path_dir, filename)
    joblib.dump(model, path)



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
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    model.fit(df)

    return model

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

        ##### Naïf #####
        naive_df = weekly.copy()
        naive_df["rolling_mean_t1"] = naive_df["quantity"].shift(1).rolling(3).mean()
        naive_df["family"] = fam
        forecast_df = naive_df.dropna(subset=["rolling_mean_t1"])
        naive_model = NaiveMeanModel(forecast_df)
        save_model_pkl(naive_model, method="naive", family=fam)

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



def predict_with_prophet(model, periods=6, freq="W-MON", return_only_future=True):
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    forecast = forecast[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "prediction"})

    if return_only_future:
        last_train_date = model.history["ds"].max()
        forecast = forecast[forecast["date"] > last_train_date]

    return forecast

if __name__ == "__main__":
    # Exemple d'utilisation
    df = pd.read_csv("data/processed/clean_transactions.csv")
    df = prepare_aggregated(df)
    train_all_models(df)