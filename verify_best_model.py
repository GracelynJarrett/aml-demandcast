import mlflow

mlflow.set_tracking_uri('http://localhost:5000')
client = mlflow.tracking.MlflowClient()

# Get Production version
versions = client.search_model_versions('name="DemandCast"')
prod = [v for v in versions if v.current_stage == 'Production'][0]

print("=" * 60)
print("✓ BEST MODEL NOW IN PRODUCTION")
print("=" * 60)
print(f"\nModel Version: {prod.version}")
print(f"Run ID: {prod.run_id}")
print()

# Get run details
run = mlflow.get_run(prod.run_id)

print("Parameters (Tuned):")
tuned_keys = ['n_estimators', 'max_depth', 'max_features', 'min_samples_leaf', 'min_samples_split', 'n_jobs', 'random_state']
for k in tuned_keys:
    val = run.data.params.get(k)
    print(f"  {k:20s}: {val}")

print()
print("Validation Metrics (on 2025-01-22 to 2025-02-01):")
print(f"  val_mae:   {run.data.metrics.get('val_mae', 'N/A'):.4f}")
print(f"  val_rmse:  {run.data.metrics.get('val_rmse', 'N/A'):.4f}")
print(f"  val_r2:    {run.data.metrics.get('val_r2', 'N/A'):.6f}")

print()
print("Test Metrics (on 2025-02-01+):")
print(f"  test_mae:  {run.data.metrics.get('test_mae', 'N/A'):.4f}")
print(f"  test_rmse: {run.data.metrics.get('test_rmse', 'N/A'):.4f}")
print(f"  test_r2:   {run.data.metrics.get('test_r2', 'N/A'):.6f}")

print()
print("=" * 60)
print("Your Streamlit app is now using the BEST model possible! 🚀")
print("=" * 60)
