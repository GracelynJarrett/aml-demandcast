"""
tune.py — Hyperparameter tuning for DemandCast (Optuna + MLflow)
===============================================================
Runs an Optuna study to tune a RandomForestRegressor on the train/val
split. Each trial is logged to MLflow; the best run can be registered
to the MLflow Model Registry.

Run from project root with the `.venv` active:
    python tune.py
"""
from pathlib import Path
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import datetime

try:
    from src.features_skeleton import FEATURE_COLS
except ModuleNotFoundError:
    from features_skeleton import FEATURE_COLS


# ---------------------------------------------------------------------------
# Configuration — keep in sync with train.py and cv.py
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME_TUNE", os.getenv("MLFLOW_EXPERIMENT_NAME", "DemandCast_RandomSplits"))
MODEL_REGISTRY_NAME = "DemandCast"

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "features.parquet"
VAL_CUTOFF = "2025-01-22"
TEST_CUTOFF = "2025-02-01"

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

N_TRIALS = 15


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


def _split_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split input DataFrame into train/val/test windows."""
    split_method = SPLIT_METHOD.lower()
    if split_method == "random":
        # Shuffle data randomly, then split by percentages
        shuffled = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
        train_end, val_end = _split_indices_by_ratio(len(shuffled))
        train = shuffled.iloc[:train_end].copy()
        val = shuffled.iloc[train_end:val_end].copy()
        test = shuffled.iloc[val_end:].copy()
        return train, val, test
    elif split_method == "percentage":
        sort_cols = ["hour"]
        if "PULocationID" in df.columns:
            sort_cols = ["hour", "PULocationID"]
        ordered = df.sort_values(sort_cols).reset_index(drop=True)
        train_end, val_end = _split_indices_by_ratio(len(ordered))
        train = ordered.iloc[:train_end].copy()
        val = ordered.iloc[train_end:val_end].copy()
        test = ordered.iloc[val_end:].copy()
        return train, val, test

    split_ts = df["hour"]
    val_cutoff_ts = pd.to_datetime(VAL_CUTOFF)
    test_cutoff_ts = pd.to_datetime(TEST_CUTOFF)
    train = df[split_ts < val_cutoff_ts].copy()
    val = df[(split_ts >= val_cutoff_ts) & (split_ts < test_cutoff_ts)].copy()
    test = df[split_ts >= test_cutoff_ts].copy()
    return train, val, test



def compute_validation_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute validation metrics used for MLflow comparison during tuning."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(root_mean_squared_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    mbe = float(np.mean(y_pred_arr - y_true_arr))

    nonzero_mask = y_true_arr != 0
    excluded_rows = int((~nonzero_mask).sum())
    excluded_pct = float((excluded_rows / y_true_arr.size) * 100.0)

    metrics: dict[str, float] = {
        "val_mae": mae,
        "val_rmse": rmse,
        "val_r2": r2,
        "val_mbe": mbe,
        "val_mape_excluded_rows": float(excluded_rows),
        "val_mape_excluded_pct": excluded_pct,
    }
    if nonzero_mask.any():
        mape = float(
            np.mean(np.abs((y_true_arr[nonzero_mask] - y_pred_arr[nonzero_mask]) / y_true_arr[nonzero_mask]))
            * 100.0
        )
        metrics["val_mape"] = mape

    return metrics


def load_splits():
    """Load features.parquet and return train and validation splits.

    Returns
    -------
    X_train, y_train, X_val, y_val
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

    if "day_of_week" not in df.columns:
        df["day_of_week"] = split_ts.dt.dayofweek
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    if "is_rush_hour" not in df.columns:
        is_weekday = df["day_of_week"] < 5
        is_rush = split_ts.dt.hour.isin([7, 8, 17, 18])
        df["is_rush_hour"] = (is_weekday & is_rush).astype(int)

    train, val, _ = _split_df(df)

    if train.empty:
        raise ValueError("Training split is empty. Check VAL_CUTOFF and input data range.")
    if val.empty:
        raise ValueError("Validation split is empty. Check TEST_CUTOFF and input data range.")

    # Keep strict chronological order for any downstream TimeSeriesSplit usage.
    sort_cols = ["hour"]
    if "PULocationID" in train.columns:
        sort_cols = ["hour", "PULocationID"]
    train = train.sort_values(sort_cols).reset_index(drop=True)
    val = val.sort_values(sort_cols).reset_index(drop=True)

    train["hour"] = train["hour"].dt.hour
    val["hour"] = val["hour"].dt.hour

    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_features:
        raise KeyError(f"Missing required feature columns: {missing_features}")
    if TARGET not in df.columns:
        raise KeyError(f"Missing required target column: {TARGET}")

    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_val, y_val = val[FEATURE_COLS], val[TARGET]
    return X_train, y_train, X_val, y_val


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: suggest hyperparams, run TimeSeriesSplit CV on `train`,
    log per-fold metrics to MLflow, and return the mean CV MAE (minimize).
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),  # 100-500 balances ensemble stability vs runtime for 20 trials.
        "max_depth": trial.suggest_int("max_depth", 8, 28),  # Depth 8-28 allows nonlinear interactions without defaulting to unbounded deep trees.
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),  # 1-8 controls overfitting in sparse zone-hours while preserving flexibility.
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),  # 2-20 regularizes splitting aggressiveness for noisy hourly demand.
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),  # These options test conservative and moderate feature subsampling regimes.
        "random_state": 42,
        "n_jobs": -1,
    }

    X_train, y_train, X_val, y_val = load_splits()

    tscv = TimeSeriesSplit(n_splits=5)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    utc_now = datetime.datetime.now(datetime.timezone.utc)
    utc_stamp = utc_now.strftime("%Y%m%dT%H%M%SZ")
    run_name = f"optuna_trial_{trial.number}_{utc_stamp}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("logged_at_utc", utc_now.isoformat())
        mlflow.log_param("split_method", SPLIT_METHOD.lower())
        if SPLIT_METHOD.lower() == "percentage":
            mlflow.log_param("train_ratio", TRAIN_RATIO)
            mlflow.log_param("val_ratio", VAL_RATIO)
            mlflow.log_param("test_ratio", TEST_RATIO)
        else:
            mlflow.log_param("val_cutoff", VAL_CUTOFF)
            mlflow.log_param("test_cutoff", TEST_CUTOFF)
        mlflow.log_params(params)
        mlflow.log_param("objective", "tscv_train")

        fold_maes: list[float] = []
        for fold_num, (tr_idx, te_idx) in enumerate(tscv.split(X_train), start=1):
            X_fold_train, y_fold_train = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_fold_test, y_fold_test = X_train.iloc[te_idx], y_train.iloc[te_idx]

            model = RandomForestRegressor(**params)
            model.fit(X_fold_train, y_fold_train)
            fold_preds = model.predict(X_fold_test)
            fold_mae = float(mean_absolute_error(y_fold_test, fold_preds))
            fold_maes.append(fold_mae)

            mlflow.log_metric(f"fold_{fold_num}_mae", fold_mae, step=fold_num)

        mean_cv_mae = float(np.mean(fold_maes))
        mlflow.log_metric("mean_cv_mae", mean_cv_mae)

        final_model = RandomForestRegressor(**params)
        final_model.fit(X_train, y_train)
        val_preds = final_model.predict(X_val)
        val_metrics = compute_validation_metrics(y_val, val_preds)
        mlflow.log_metrics(val_metrics)
        trial.set_user_attr("val_mae", val_metrics["val_mae"])

        # Avoid uploading full model artifacts for every trial to keep Optuna runs stable.
        # The final selected model is logged and registered in retrain_and_register().

    return mean_cv_mae


def retrain_and_register(best_params: dict, stage: str = "Production") -> None:
    """Retrain the chosen hyperparameters on train+val, evaluate on test,
    log test metrics, and register the final model to the Model Registry.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expected features file at {DATA_PATH}. Run build_features.py first."
        )

    df = pd.read_parquet(DATA_PATH)
    if "hour" not in df.columns:
        raise KeyError("Missing required column 'hour' in features dataset.")
    if TARGET not in df.columns:
        raise KeyError(f"Missing required target column: {TARGET}")

    df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
    if df["hour"].isna().any():
        raise ValueError("Column 'hour' contains invalid timestamps after parsing.")

    split_ts = df["hour"]
    if "day_of_week" not in df.columns:
        df["day_of_week"] = split_ts.dt.dayofweek
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    if "is_rush_hour" not in df.columns:
        is_weekday = df["day_of_week"] < 5
        is_rush = split_ts.dt.hour.isin([7, 8, 17, 18])
        df["is_rush_hour"] = (is_weekday & is_rush).astype(int)

    train, val, test = _split_df(df)
    trainval = pd.concat([train, val], axis=0, ignore_index=True)

    if trainval.empty:
        raise ValueError("Train+validation split is empty. Check TEST_CUTOFF.")

    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_features:
        raise KeyError(f"Missing required feature columns: {missing_features}")

    trainval["hour"] = trainval["hour"].dt.hour
    X_trainval = trainval[FEATURE_COLS]
    y_trainval = trainval[TARGET]

    X_test = None
    y_test = None
    if test.empty:
        print("Warning: Test split is empty; test_mae will not be logged.")
    else:
        test["hour"] = test["hour"].dt.hour
        X_test = test[FEATURE_COLS]
        y_test = test[TARGET]

    model = RandomForestRegressor(**best_params)
    model.fit(X_trainval, y_trainval)

    utc_now = datetime.datetime.now(datetime.timezone.utc)
    utc_stamp = utc_now.strftime("%Y%m%dT%H%M%SZ")
    run_name = f"final_retrain_and_register_{utc_stamp}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("logged_at_utc", utc_now.isoformat())
        mlflow.log_param("split_method", SPLIT_METHOD.lower())
        if SPLIT_METHOD.lower() == "percentage":
            mlflow.log_param("train_ratio", TRAIN_RATIO)
            mlflow.log_param("val_ratio", VAL_RATIO)
            mlflow.log_param("test_ratio", TEST_RATIO)
        else:
            mlflow.log_param("val_cutoff", VAL_CUTOFF)
            mlflow.log_param("test_cutoff", TEST_CUTOFF)
        mlflow.log_params(best_params)

        if X_test is not None and y_test is not None:
            test_preds = model.predict(X_test)
            test_mae = float(mean_absolute_error(y_test, test_preds))
            mlflow.log_metric("test_mae", test_mae)
            print(f"Final test MAE: {test_mae:.4f}")

        mlflow.sklearn.log_model(model, "model", pip_requirements=[])

        model_uri = f"runs:/{run.info.run_id}/model"
        registered = mlflow.register_model(model_uri=model_uri, name=MODEL_REGISTRY_NAME)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=MODEL_REGISTRY_NAME,
            version=registered.version,
            stage=stage,
            archive_existing_versions=False,
        )

        print(
            f"Registered model '{MODEL_REGISTRY_NAME}' version {registered.version} "
            f"to stage '{stage}' (run_id={run.info.run_id})."
        )


def run_study() -> optuna.Study:
    """Run Optuna study and print a concise summary of best trial metrics."""
    study = optuna.create_study(direction="minimize", study_name="demandcast_rf_tuning")
    study.optimize(objective, n_trials=N_TRIALS)

    best_trial = study.best_trial
    best_val_mae = best_trial.user_attrs.get("val_mae")

    print("\nStudy complete.")
    print(f"Best trial: {best_trial.number}")
    print(f"Best mean_cv_mae: {best_trial.value:.4f}")
    if best_val_mae is not None:
        print(f"Best trial val_mae: {best_val_mae:.4f}")
    print(f"Best params: {best_trial.params}")
    return study


def main() -> None:
    study = run_study()

    best_params = {
        **study.best_trial.params,
        "random_state": 42,
        "n_jobs": -1,
    }
    retrain_and_register(best_params=best_params, stage="Production")


if __name__ == "__main__":
    main()
