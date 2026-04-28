import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

try:
    mlflow.pyfunc.load_model("runs:/d4bb5dd4b0eb42f6b0e84558efcd3699/model")
    print("LOAD_OK")
except Exception as exc:
    print("LOAD_FAIL")
    print(type(exc).__name__)
    print(str(exc)[:500])
