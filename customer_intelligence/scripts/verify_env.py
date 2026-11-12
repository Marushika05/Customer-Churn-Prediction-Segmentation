"""Confirms the interpreter running this script has every dependency.   python -m scripts.verify_env"""
import importlib
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version.split()[0])
if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
    print("WARNING: not running inside the project virtual environment")
MODULES = ["numpy", "pandas", "scipy", "sklearn", "mlxtend", "joblib", "openpyxl", "fastapi", "uvicorn",
           "multipart", "pydantic", "streamlit", "plotly", "requests", "httpx"]
missing = []
for m in MODULES:
    try:
        importlib.import_module(m)
    except Exception as e:  # noqa: BLE001
        missing.append(f"{m} ({type(e).__name__}: {e})")
try:
    import pandas
    print("Pandas import: SUCCESS")
    print("Pandas version:", pandas.__version__)
except Exception as e:  # noqa: BLE001
    print("Pandas import: FAILED -", e)
if missing:
    print("Dependency verification: FAILED ->", "; ".join(missing))
    sys.exit(1)
print("Dependency verification: PASSED")
