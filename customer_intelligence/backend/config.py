"""Central configuration. Values can be overridden via environment variables or a .env file."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env_file() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env_file()

STORAGE_DIR = ROOT / "storage"
DB_PATH = Path(os.getenv("CIP_DB_PATH", STORAGE_DIR / "cip.db"))
ARTIFACT_DIR = Path(os.getenv("CIP_ARTIFACT_DIR", STORAGE_DIR / "artifacts"))
KEYSTORE_PATH = STORAGE_DIR / "tenant_keys.json"
RAW_DATA_PATH = ROOT / "data" / "raw" / "online_retail.csv"

ADMIN_TOKEN = os.getenv("CIP_ADMIN_TOKEN", "admin-demo-token")
API_URL = os.getenv("CIP_API_URL", "http://127.0.0.1:8000")
MAX_UPLOAD_MB = 200

# ---- ML settings ----
RANDOM_STATE = 42
HORIZON_DAYS = 90            # churn inactivity window AND CLV spend horizon
SNAPSHOT_STEP_DAYS = 30      # spacing between chronological training snapshots
MIN_HISTORY_DAYS = 60        # history required before the first snapshot
MIN_CUSTOMERS = 150          # below this, customer-level models are skipped
RISK_LOW, RISK_HIGH = 0.30, 0.60   # churn-probability bands: <0.30 Low, <0.60 Medium, else High
APRIORI_MIN_SUPPORT = 0.01
APRIORI_MIN_CONFIDENCE = 0.25
APRIORI_MIN_LIFT = 1.2
CURRENCY = "£"
