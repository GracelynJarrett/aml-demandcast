import pandas as pd
import mlflow
import os

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
mlflow.set_tracking_uri("http://localhost:5000")

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

try:
    model = mlflow.pyfunc.load_model("models:/DemandCast/Production")
    print("✅ Model loaded")
except Exception as e:
    print(f"Error: {e}")
    exit(1)

# Test cases with high and low lag values
test_cases = [
    # Zone 48, Friday 5 PM (high demand time), with high lags
    {
        "name": "Zone 48, Friday 5 PM",
        "PULocationID": 48,
        "hour": 17,
        "day_of_week": 4,
        "is_weekend": 0,
        "is_rush_hour": 1,
        "demand_lag_1h": 103.5,
        "demand_lag_24h": 137.5,
        "demand_lag_168h": 111.5,
    },
    # Zone 42, Friday 8 AM
    {
        "name": "Zone 42, Friday 8 AM",
        "PULocationID": 42,
        "hour": 8,
        "day_of_week": 4,
        "is_weekend": 0,
        "is_rush_hour": 1,
        "demand_lag_1h": 14.0,
        "demand_lag_24h": 11.0,
        "demand_lag_168h": 6.0,
    },
    # Zone 265, Monday noon (low demand)
    {
        "name": "Zone 265, Monday noon",
        "PULocationID": 265,
        "hour": 12,
        "day_of_week": 0,
        "is_weekend": 0,
        "is_rush_hour": 0,
        "demand_lag_1h": 1.0,
        "demand_lag_24h": 1.0,
        "demand_lag_168h": 1.0,
    },
]

for test in test_cases:
    name = test.pop("name")
    X = pd.DataFrame([test], columns=FEATURE_COLS)
    pred = model.predict(X)[0]
    print(f"{name}: {pred:.2f} trips")
