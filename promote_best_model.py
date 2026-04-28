import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.tracking.MlflowClient()

# Search all DemandCast model versions
versions = client.search_model_versions('name="DemandCast"')

# Find the best run ID
best_run_id = "d4bb5dd4b0eb42f6b0e84558efcd3699"
matching = [v for v in versions if v.run_id == best_run_id]

print(f"Searching for run {best_run_id}...")
print(f"Found {len(matching)} matching versions")
print()

if matching:
    v = matching[0]
    print(f"Found: Version {v.version}, Run ID {v.run_id}")
    print(f"Current Stage: {v.current_stage}")
    print()
    
    # Transition to Production and archive the old one
    client.transition_model_version_stage(
        name="DemandCast",
        version=v.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"✓ Transitioned Version {v.version} to Production")
    print(f"✓ Archived previous Production version")
    print()
    print("Your Streamlit app will now use the best model!")
else:
    print("ERROR: Best run not found in registry.")
    print("This run may not have been registered yet.")
    print("\nAll registered versions:")
    for v in versions:
        print(f"  Version {v.version}: Run ID {v.run_id}, Stage: {v.current_stage}")
