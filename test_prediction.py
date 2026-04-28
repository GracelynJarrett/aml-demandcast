import os
import pandas as pd
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
mlflow.set_tracking_uri("http://localhost:5000")

# Load model
try:
    model = mlflow.pyfunc.load_model("models:/DemandCast/Production")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Create test input with contextual lag defaults
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

# Test with contextual lag values (from zone 48, weekday rush hour)
test_input = pd.DataFrame([{
    "PULocationID": 48,
    "hour": 17,
    "day_of_week": 4,
    "is_weekend": 0,
    "is_rush_hour": 1,
    "demand_lag_1h": 103.5,
    "demand_lag_24h": 137.5,
    "demand_lag_168h": 111.5,
}], columns=FEATURE_COLS)

print("\nTest input:")
print(test_input)

try:
    pred = model.predict(test_input)
    print(f"\nModel prediction output: {pred}")
    print(f"Type: {type(pred)}")
    print(f"First element: {pred[0]}")
except Exception as e:
    print(f"Error during prediction: {e}")
    import traceback
    traceback.print_exc()
