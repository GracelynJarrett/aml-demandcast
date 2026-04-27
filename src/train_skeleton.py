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
import os

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
EXPERIMENT_NAME     = os.getenv("MLFLOW_EXPERIMENT_NAME", "DemandCast_RandomSplits")

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "features.parquet"

# Temporal split cutoffs — January 2025 dataset
# Train:      Jan 1  – Jan 21, 2025
# Validation: Jan 22 – Jan 31, 2025
# Test:       Feb 1  – Feb 7, 2025 (sealed until final evaluation)
VAL_CUTOFF  = "2025-01-22"
TEST_CUTOFF = "2025-02-01"
TEST_END    = "2025-02-08"

# Split mode options:
#   "date" (default) — chronological split by cutoff dates
#   "percentage" — chronological percentage split (no shuffling)
#   "random" — random shuffle then split by percentages
SPLIT_METHOD = os.getenv("SPLIT_METHOD", "date").lower()
TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", "0.50"))
VAL_RATIO = float(os.getenv("VAL_RATIO", "0.30"))
TEST_RATIO = float(os.getenv("TEST_RATIO", "0.20"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

TARGET = "demand"


def _split_indices_by_ratio(n_rows: int) -> tuple[int, int]:
    """Return (train_end_idx, val_end_idx) for chronological percentage splits."""
    if n_rows < 3:
        raise ValueError("Need at least 3 rows to create train/val/test splits.")

    ratio_sum = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.4f}.")

    train_end = int(n_rows * TRAIN_RATIO)
    val_end = train_end + int(n_rows * VAL_RATIO)

    train_end = max(1, min(train_end, n_rows - 2))
    val_end = max(train_end + 1, min(val_end, n_rows - 1))
    return train_end, val_end


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


def evaluate_mape_mbe(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAPE and MBE with zero-demand-safe MAPE handling.

    MAPE excludes rows where y_true == 0 to avoid division-by-zero.
    Additional diagnostics are returned so this behavior is transparent.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.size == 0:
        raise ValueError("Cannot evaluate metrics on an empty validation set.")

    nonzero_mask = y_true_arr != 0
    included_rows = int(nonzero_mask.sum())
    excluded_rows = int((~nonzero_mask).sum())
    excluded_pct = round((excluded_rows / y_true_arr.size) * 100.0, 4)

    mbe = round(float(np.mean(y_pred_arr - y_true_arr)), 4)

    metrics: dict[str, float] = {
        "mbe": mbe,
        "mape_excluded_rows": float(excluded_rows),
        "mape_excluded_pct": excluded_pct,
    }

    if included_rows > 0:
        mape = np.mean(
            np.abs((y_true_arr[nonzero_mask] - y_pred_arr[nonzero_mask]) / y_true_arr[nonzero_mask])
        ) * 100.0
        metrics["mape"] = round(float(mape), 4)

    return metrics


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

    split_method = SPLIT_METHOD.lower()
    if split_method == "random":
        # Shuffle data randomly, then split by percentages
        df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
        train_end, val_end = _split_indices_by_ratio(len(df))
        train = df.iloc[:train_end].copy()
        val = df.iloc[train_end:val_end].copy()
    elif split_method == "percentage":
        # Chronological percentage split (no shuffling)
        sort_cols = ["hour"]
        if "PULocationID" in df.columns:
            sort_cols = ["hour", "PULocationID"]
        df = df.sort_values(sort_cols).reset_index(drop=True)

        train_end, val_end = _split_indices_by_ratio(len(df))
        train = df.iloc[:train_end].copy()
        val = df.iloc[train_end:val_end].copy()
    else:
        # Date-based split (original behavior)
        train = df[split_ts < VAL_CUTOFF].copy()
        val = df[(split_ts >= VAL_CUTOFF) & (split_ts < TEST_CUTOFF)].copy()

    # FEATURE_COLS expects hour-of-day, so convert timestamp to numeric hour for modeling.
    train["hour"] = pd.to_datetime(train["hour"], errors="coerce").dt.hour
    val["hour"] = pd.to_datetime(val["hour"], errors="coerce").dt.hour

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
        mlflow.log_param("split_method", split_method)
        if split_method == "percentage":
            mlflow.log_param("train_ratio", TRAIN_RATIO)
            mlflow.log_param("val_ratio", VAL_RATIO)
            mlflow.log_param("test_ratio", TEST_RATIO)
        else:
            mlflow.log_param("val_cutoff", VAL_CUTOFF)
            mlflow.log_param("test_cutoff", TEST_CUTOFF)
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        val_metrics = evaluate(y_val, val_preds)
        val_extra_metrics = evaluate_mape_mbe(y_val, val_preds)

        mlflow_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
        mlflow_metrics["val_mbe"] = val_extra_metrics["mbe"]
        mlflow_metrics["val_mape_excluded_rows"] = val_extra_metrics["mape_excluded_rows"]
        mlflow_metrics["val_mape_excluded_pct"] = val_extra_metrics["mape_excluded_pct"]
        if "mape" in val_extra_metrics:
            mlflow_metrics["val_mape"] = val_extra_metrics["mape"]

        mlflow.log_metrics(mlflow_metrics)
        mlflow.sklearn.log_model(model, "model")

        val_mape_display = (
            f"{val_extra_metrics['mape']:.2f}%"
            if "mape" in val_extra_metrics
            else "N/A (all val demand values were zero)"
        )

        print(
            f"[{run_name}] val_mae={val_metrics['mae']:.2f}  "
            f"val_rmse={val_metrics['rmse']:.2f}  "
            f"val_r2={val_metrics['r2']:.3f}  "
            f"val_mape={val_mape_display}  "
            f"val_mbe={val_extra_metrics['mbe']:.2f}"
        )
        return run.info.run_id