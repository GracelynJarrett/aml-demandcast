"""
Retrain with best hyperparameters and register to Production.
Best parameters from run d4bb5dd4b0eb42f6b0e84558efcd3699
"""
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Config
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "DemandCast_RandomSplits"
MODEL_REGISTRY_NAME = "DemandCast"
DATA_PATH = Path(__file__).resolve().parent / "data" / "features.parquet"
VAL_CUTOFF = "2025-01-22"
TEST_CUTOFF = "2025-02-01"

# Best parameters from drift-corrected search
BEST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 17,
    "max_features": "log2",
    "min_samples_leaf": 3,
    "min_samples_split": 8,
    "random_state": 42,
    "n_jobs": -1,
}

def load_and_split():
    """Load data, engineer features, and split by date cutoffs."""
    df = pd.read_parquet(DATA_PATH)
    df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
    
    # Engineer missing features if not present
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["hour"].dt.dayofweek
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    if "is_rush_hour" not in df.columns:
        is_weekday = df["day_of_week"] < 5
        is_rush = df["hour"].dt.hour.isin([7, 8, 17, 18])
        df["is_rush_hour"] = (is_weekday & is_rush).astype(int)
    
    split_ts = df["hour"]
    val_cutoff_ts = pd.to_datetime(VAL_CUTOFF)
    test_cutoff_ts = pd.to_datetime(TEST_CUTOFF)
    
    train = df[split_ts < val_cutoff_ts].copy()
    val = df[(split_ts >= val_cutoff_ts) & (split_ts < test_cutoff_ts)].copy()
    test = df[split_ts >= test_cutoff_ts].copy()
    
    return train, val, test

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Load data
    train, val, test = load_and_split()
    
    try:
        from src.features_skeleton import FEATURE_COLS
    except:
        from features_skeleton import FEATURE_COLS
    
    TARGET = "demand"
    
    # Convert datetime hour to numeric hour (0-23)
    for dset in [train, val, test]:
        if "hour" in dset.columns and pd.api.types.is_datetime64_any_dtype(dset["hour"]):
            dset["hour"] = dset["hour"].dt.hour
    
    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_val, y_val = val[FEATURE_COLS], val[TARGET]
    X_test, y_test = test[FEATURE_COLS], test[TARGET]
    
    print("=" * 60)
    print("RETRAINING WITH BEST PARAMETERS")
    print("=" * 60)
    print(f"Train size: {len(X_train)}")
    print(f"Val size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")
    print()
    print("Parameters:", BEST_PARAMS)
    print()
    
    # Train model
    model = RandomForestRegressor(**BEST_PARAMS)
    model.fit(X_train, y_train)
    
    # Validation metrics
    val_preds = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_preds)
    val_rmse = root_mean_squared_error(y_val, val_preds)
    val_r2 = r2_score(y_val, val_preds)
    
    # Test metrics
    test_preds = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_rmse = root_mean_squared_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)
    
    print("Validation Metrics:")
    print(f"  MAE:  {val_mae:.4f}")
    print(f"  RMSE: {val_rmse:.4f}")
    print(f"  R²:   {val_r2:.6f}")
    print()
    print("Test Metrics:")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R²:   {test_r2:.6f}")
    print()
    
    # Log to MLflow and register
    with mlflow.start_run(run_name="best_random_split_retrained"):
        mlflow.log_params(BEST_PARAMS)
        mlflow.log_param("split_method", "date")
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_r2", val_r2)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_r2", test_r2)
        
        mlflow.sklearn.log_model(model, "model", pip_requirements=[])
        
        run_id = mlflow.active_run().info.run_id
    
    print(f"✓ Logged to MLflow (Run ID: {run_id})")
    print()
    
    # Register to Production
    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=MODEL_REGISTRY_NAME)
    
    print(f"✓ Registered as Version {registered.version}")
    
    # Transition to Production
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_REGISTRY_NAME,
        version=registered.version,
        stage="Production",
        archive_existing_versions=True
    )
    
    print(f"✓ Transitioned to Production (archived previous)")
    print()
    print("=" * 60)
    print("SUCCESS!")
    print(f"Streamlit will now use Version {registered.version}")
    print("=" * 60)

if __name__ == "__main__":
    main()
