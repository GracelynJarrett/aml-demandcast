"""
train.py — Model training and MLflow logging for DemandCast
============================================================
This script loads the engineered feature set, applies a temporal train/val/test
split, and trains regression models to predict hourly taxi demand per zone.
Every run is logged to MLflow — parameters, metrics, and the model artifact.

Usage (from project root with .venv active)
-------------------------------------------
    python train.py

Before running
--------------
1. MLflow UI must be running:
       mlflow ui
   Then open http://localhost:5000 in your browser.
2. features.parquet must exist in data/:
       pipelines/build_features.py produces this file.

Functions
---------
evaluate          Compute MAE, RMSE, and R² for a set of predictions.
                  Already implemented — use it, don't rewrite it.
train_and_log     Load data, split, train one model, log everything to MLflow.
                  This is your TODO.
"""

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from typing import Any

# Prefer finalized module name, but allow current scaffold module during development.
try:
    from src.features import FEATURE_COLS  # type: ignore
except ModuleNotFoundError:
    from src.features_skeleton import FEATURE_COLS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME     = "DemandCast"

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "features.parquet"

# Temporal split cutoffs — January 2025 dataset
# Train:      Jan 1  – Jan 21, 2025
# Validation: Jan 22 – Jan 31, 2025
# Test:       Feb 1  – Feb 7, 2025 (sealed until final evaluation)
VAL_CUTOFF  = "2025-01-22"
TEST_CUTOFF = "2025-02-01"
TEST_END    = "2025-02-08"

TARGET = "demand"


# ---------------------------------------------------------------------------
# evaluate() — already implemented, use it as-is
# ---------------------------------------------------------------------------

def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and R² for a set of predictions.

    This function is pre-built for you. Call it on both the validation set
    and, at the very end, the test set (once only).

    Parameters
    ----------
    y_true : pd.Series
        Ground-truth demand values.
    y_pred : np.ndarray
        Model predictions, same length as y_true.

    Returns
    -------
    dict[str, float]
        Keys: 'mae', 'rmse', 'r2'. Values are floats rounded to 4 decimal places.

    Examples
    --------
    >>> val_preds = model.predict(X_val)
    >>> metrics = evaluate(y_val, val_preds)
    >>> print(f"Val MAE: {metrics['mae']:.2f}  RMSE: {metrics['rmse']:.2f}  R²: {metrics['r2']:.3f}")
    """
    return {
        "mae":  round(mean_absolute_error(y_true, y_pred), 4),   # Average absolute difference between true and predicted values -> Lower is better
        "rmse": round(root_mean_squared_error(y_true, y_pred), 4), # Square roots the error before averagring ( 4->16) -> Lower is better
        "r2":   round(r2_score(y_true, y_pred), 4), # How much cariance is demand is explained by the modesl -> Higher is better
    }


# ---------------------------------------------------------------------------
# train_and_log() — your TODO
# ---------------------------------------------------------------------------

def train_and_log(
    model: Any,
    run_name: str,
    params: dict,
) -> str:
    """Train one regression model and log everything to MLflow.

    This function handles the full training loop for a single model run:
      1. Load data/features.parquet
      2. Apply temporal train/val/test split
      3. Separate features (X) and target (y) for each split
      4. Fit the model on the training set
      5. Evaluate on the validation set using evaluate()
      6. Log params, val metrics, and model artifact to MLflow
      7. Print a summary line and return the MLflow run ID

    The test set must NOT be touched here. Seal it until final evaluation.

    MLflow logging checklist (every run must include all three)
    ----------------------------------------------------------
    mlflow.log_params(params)             — algorithm name + hyperparameters
    mlflow.log_metrics(val_metrics)       — val_mae, val_rmse, val_r2
    mlflow.sklearn.log_model(model, "model") — the fitted model artifact

    Consistent metric naming matters: the MLflow comparison view matches runs
    by key name. Always use 'val_mae', 'val_rmse', 'val_r2' exactly.

    Parameters
    ----------
    model : sklearn estimator
        An unfitted sklearn-compatible regression model, e.g.:
            LinearRegression()
            RandomForestRegressor(n_estimators=100, random_state=42)
            Third model of your choice
    run_name : str
        Human-readable label shown in the MLflow UI, e.g. "linear_regression_baseline".
        Use snake_case. Be specific — "rf_100_estimators" beats "random_forest_v2".
    params : dict
        Dictionary of parameters to log. Must include at minimum:
            {"model": type(model).__name__, ...hyperparameters...}
        Example:
            {"model": "RandomForestRegressor", "n_estimators": 100, "max_depth": 10}

    Returns
    -------
    str
        The MLflow run ID (a hex string). Print it or save it — you can use it
        to retrieve this exact run later with mlflow.get_run(run_id).

    Raises
    ------
    FileNotFoundError
        If data/features.parquet does not exist. Run build_features.py first.
    mlflow.exceptions.MlflowException
        If the MLflow server is not reachable at MLFLOW_TRACKING_URI.
        Fix: start the server with `mlflow ui` from your project root.

    Notes
    -----
    The `hour` column is explicitly converted to datetime before splitting.
    This keeps temporal comparisons reliable even if parquet schema inference
    changes between environments.

    The test window is intentionally not used in this function. It remains
    sealed for final evaluation only.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> run_id = train_and_log(
    ...     model=LinearRegression(),
    ...     run_name="linear_regression_baseline",
    ...     params={"model": "LinearRegression"},
    ... )
    >>> print(f"Run logged: {run_id}")
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expected features file at {DATA_PATH}. Run build_features.py first."
        )

    df = pd.read_parquet(DATA_PATH)

    if "hour" not in df.columns:
        raise KeyError("Missing required column 'hour' in features dataset.")

    df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
    if df["hour"].isna().any():
        raise ValueError("Column 'hour' contains invalid timestamps after parsing.")

    split_ts = df["hour"]

    # Backfill temporal features if the feature matrix only contains timestamps.
    if "day_of_week" not in df.columns:
        df["day_of_week"] = split_ts.dt.dayofweek
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    if "is_rush_hour" not in df.columns:
        is_weekday = df["day_of_week"] < 5
        is_rush = split_ts.dt.hour.isin([7, 8, 17, 18])
        df["is_rush_hour"] = (is_weekday & is_rush).astype(int)

    # FEATURE_COLS expects hour-of-day, so convert timestamp to numeric hour for modeling.
    df["hour"] = split_ts.dt.hour

    train = df[split_ts < VAL_CUTOFF]
    val = df[(split_ts >= VAL_CUTOFF) & (split_ts < TEST_CUTOFF)]

    if train.empty:
        raise ValueError("Training split is empty. Check VAL_CUTOFF and input data range.")
    if val.empty:
        raise ValueError("Validation split is empty. Check TEST_CUTOFF and input data range.")

    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_features:
        raise KeyError(f"Missing required feature columns: {missing_features}")
    if TARGET not in df.columns:
        raise KeyError(f"Missing required target column: {TARGET}")

    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_val, y_val = val[FEATURE_COLS], val[TARGET]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        val_metrics = evaluate(y_val, val_preds)

        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.sklearn.log_model(model, "model")

        print(
            f"[{run_name}] val_mae={val_metrics['mae']:.2f}  "
            f"val_rmse={val_metrics['rmse']:.2f}  val_r2={val_metrics['r2']:.3f}"
        )
        return run.info.run_id