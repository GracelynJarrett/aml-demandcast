import mlflow
import sys

try:
    mlflow.set_tracking_uri("http://localhost:5000")
    client = mlflow.tracking.MlflowClient()
    
    print("===== CURRENT PRODUCTION MODEL =====")
    versions = client.search_model_versions('name="DemandCast"')
    
    for v in sorted(versions, key=lambda x: int(x.version), reverse=True)[:3]:
        stage = v.current_stage if v.current_stage else "None"
        print(f"Version {v.version}: Run {v.run_id[:8]}... | Stage: {stage}")
        
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
