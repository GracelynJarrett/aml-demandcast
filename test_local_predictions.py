import pandas as pd
from pathlib import Path
import os
import sys

# Check if the local model is available instead of using MLflow registry
LOCAL_MODEL_ROOT = Path("mlartifacts")
local_model_candidates = sorted(
    LOCAL_MODEL_ROOT.rglob("MLmodel"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)

if local_model_candidates:
    import mlflow
    local_model_path = local_model_candidates[0].parent
    print(f"Loading local model from: {local_model_path}")
    model = mlflow.pyfunc.load_model(str(local_model_path))
else:
    print("No local model found")
    sys.exit(1)

FEATURE_COLS = [
    "PULocationID",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_rush_hour",
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_lag_168h",
]

# Test cases with different lag values
test_cases = [
    ("Zone 48, Fri 5 PM (HIGH)", 48, 17, 4, 0, 1, 103.5, 137.5, 111.5),
    ("Zone 42, Fri 8 AM (MED)", 42, 8, 4, 0, 1, 14.0, 11.0, 6.0),
    ("Zone 265, Mon noon (LOW)", 265, 12, 0, 0, 0, 1.0, 1.0, 1.0),
    ("Zone 3, Wed 1 PM (LOW)", 3, 13, 2, 0, 0, 1.0, 1.0, 1.0),
]

print("Testing model predictions with contextual lag values:\n")
for name, pu, hr, dow, we, rush, lag1, lag24, lag168 in test_cases:
    X = pd.DataFrame([{
        "PULocationID": pu,
        "hour": hr,
        "day_of_week": dow,
        "is_weekend": we,
        "is_rush_hour": rush,
        "demand_lag_1h": lag1,
        "demand_lag_24h": lag24,
        "demand_lag_168h": lag168,
    }], columns=FEATURE_COLS)
    
    pred = model.predict(X)[0]
    print(f"{name}: {pred:.2f} trips")
