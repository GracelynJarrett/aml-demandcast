from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

mlflow.set_tracking_uri("http://localhost:5000")
model = mlflow.pyfunc.load_model("models:/DemandCast/Production")

data_path = Path("data/features.parquet")
df = pd.read_parquet(data_path)
df["hour"] = pd.to_datetime(df["hour"], errors="coerce")

if "day_of_week" not in df.columns:
    df["day_of_week"] = df["hour"].dt.dayofweek
if "is_weekend" not in df.columns:
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
if "is_rush_hour" not in df.columns:
    is_weekday = df["day_of_week"] < 5
    is_rush = df["hour"].dt.hour.isin([7, 8, 17, 18])
    df["is_rush_hour"] = (is_weekday & is_rush).astype(int)

val_cutoff = pd.to_datetime("2025-01-22")
test_cutoff = pd.to_datetime("2025-02-01")

train = df[df["hour"] < val_cutoff].copy()
val = df[(df["hour"] >= val_cutoff) & (df["hour"] < test_cutoff)].copy()
test = df[df["hour"] >= test_cutoff].copy()

for split_df in (train, val, test):
    split_df["hour"] = split_df["hour"].dt.hour

feature_cols = [
    "PULocationID",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_rush_hour",
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_lag_168h",
]
target = "demand"


def compute_metrics(split_df: pd.DataFrame):
    x = split_df[feature_cols]
    y = split_df[target].to_numpy(dtype=float)
    preds = np.asarray(model.predict(x), dtype=float)
    mae = float(mean_absolute_error(y, preds))
    rmse = float(root_mean_squared_error(y, preds))
    r2 = float(r2_score(y, preds))
    mbe = float(np.mean(preds - y))
    nonzero_mask = y != 0
    mape = float(np.mean(np.abs((y[nonzero_mask] - preds[nonzero_mask]) / y[nonzero_mask])) * 100.0)
    return mae, rmse, r2, mape, mbe

for name, split_df in [("Validation", val), ("Test", test)]:
    mae, rmse, r2, mape, mbe = compute_metrics(split_df)
    print(f"{name} metrics:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.6f}")
    print(f"  MAPE: {mape:.4f}")
    print(f"  MBE:  {mbe:.4f}")
    print()
