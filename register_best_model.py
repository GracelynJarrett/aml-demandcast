import mlflow
import pickle
from pathlib import Path

mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.tracking.MlflowClient()

best_run_id = "d4bb5dd4b0eb42f6b0e84558efcd3699"

# Get the run details
run = mlflow.get_run(best_run_id)
print(f"Best run: {best_run_id}")
print(f"Parameters: {run.data.params}")
print()

# Try to load the model from the run
try:
    model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
    print("✓ Found model artifact in the run")
    print()
    
    # Register it
    model_uri = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(model_uri=model_uri, name="DemandCast")
    print(f"✓ Registered as Version {registered.version}")
    print()
    
    # Transition to Production
    client.transition_model_version_stage(
        name="DemandCast",
        version=registered.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"✓ Transitioned Version {registered.version} to Production")
    print(f"✓ Archived previous Production version")
    print()
    print("Your Streamlit app will now use the best model!")
    print(f"\nBest model metrics:")
    print(f"  val_mae: {run.data.metrics.get('val_mae', 'N/A')}")
    print(f"  val_rmse: {run.data.metrics.get('val_rmse', 'N/A')}")
    print(f"  val_r2: {run.data.metrics.get('val_r2', 'N/A')}")
    
except Exception as e:
    print(f"ERROR: Could not find or register model from run.")
    print(f"Details: {e}")
    print()
    print("The best run was logged during tuning but the model artifact")
    print("was not saved. We need to retrain with those parameters.")
    print()
    print("Best parameters were:")
    print(f"  n_estimators: 300")
    print(f"  max_depth: 17")
    print(f"  max_features: log2")
    print(f"  min_samples_leaf: 3")
    print(f"  min_samples_split: 8")
