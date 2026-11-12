"""SQLite persistence. Every tenant-owned table carries tenant_id and every query is scoped by it."""
import contextlib
import hashlib
import hmac
import json
import re
import secrets
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone

import pandas as pd

from . import config
from .features import FEATURES

TENANT_ID_RE = r"^[a-f0-9]{12}$"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _\-\.]{1,38}[A-Za-z0-9]$")


class NotFound(Exception):
    pass


class Conflict(Exception):
    pass


_feature_cols_sql = ", ".join(f"{c} REAL" for c in FEATURES)
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS tenants(
  tenant_id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE COLLATE NOCASE,
  api_key_hash TEXT NOT NULL,
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS transactions(
  tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
  invoice_no TEXT, stock_code TEXT, description TEXT, quantity INTEGER,
  invoice_date TEXT, unit_price REAL, customer_id INTEGER, country TEXT,
  is_cancel INTEGER, line_total REAL);
CREATE INDEX IF NOT EXISTS idx_tx_cust ON transactions(tenant_id, customer_id);
CREATE INDEX IF NOT EXISTS idx_tx_date ON transactions(tenant_id, invoice_date);
CREATE TABLE IF NOT EXISTS customer_scores(
  tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
  customer_id INTEGER NOT NULL,
  {_feature_cols_sql},
  first_purchase TEXT, last_purchase TEXT, country TEXT,
  churn_prob REAL, risk_category TEXT,
  segment_id INTEGER, segment_name TEXT, pred_spend REAL,
  PRIMARY KEY(tenant_id, customer_id));
CREATE TABLE IF NOT EXISTS jobs(
  tenant_id TEXT PRIMARY KEY REFERENCES tenants(tenant_id) ON DELETE CASCADE,
  status TEXT, stage TEXT, message TEXT, started_at TEXT, finished_at TEXT);
CREATE TABLE IF NOT EXISTS tenant_meta(
  tenant_id TEXT PRIMARY KEY REFERENCES tenants(tenant_id) ON DELETE CASCADE,
  ingest_json TEXT, trained_at TEXT);
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@contextlib.contextmanager
def connect():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with connect() as c:
        c.executescript(SCHEMA)
        c.execute("UPDATE jobs SET status='failed', message='Interrupted by server restart' WHERE status='running'")


def tq(conn, sql: str, tenant_id: str, params: tuple = ()):
    """Tenant-scoped query guard: SQL must reference tenant_id; tenant id is always the first bound parameter."""
    if "tenant_id" not in sql:
        raise RuntimeError("Unscoped query rejected: SQL must filter on tenant_id")
    return conn.execute(sql, (tenant_id, *params))


def tdf(conn, sql: str, tenant_id: str, params: tuple = ()) -> pd.DataFrame:
    if "tenant_id" not in sql:
        raise RuntimeError("Unscoped query rejected: SQL must filter on tenant_id")
    return pd.read_sql_query(sql, conn, params=(tenant_id, *params))


# ---------------- tenants ----------------
def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create_tenant(name: str) -> tuple[str, str]:
    name = (name or "").strip()
    if not NAME_RE.match(name):
        raise ValueError("Tenant name must be 3-40 characters: letters, digits, spaces, '-', '_' or '.'")
    tid, key = uuid.uuid4().hex[:12], "cip_" + secrets.token_urlsafe(24)
    with connect() as c:
        try:
            c.execute("INSERT INTO tenants VALUES (?,?,?,?)", (tid, name, _hash(key), _now()))
        except sqlite3.IntegrityError:
            raise Conflict(f"A tenant named '{name}' already exists")
    return tid, key


def verify_key(tenant_id: str, key: str | None) -> bool:
    if not key:
        return False
    with connect() as c:
        row = tq(c, "SELECT api_key_hash FROM tenants WHERE tenant_id=?", tenant_id).fetchone()
    return bool(row) and hmac.compare_digest(row["api_key_hash"], _hash(key))


def tenant_exists(tenant_id: str) -> bool:
    with connect() as c:
        return tq(c, "SELECT 1 FROM tenants WHERE tenant_id=?", tenant_id).fetchone() is not None


def get_tenant(tenant_id: str) -> dict:
    with connect() as c:
        row = tq(c, "SELECT tenant_id, name, created_at FROM tenants WHERE tenant_id=?", tenant_id).fetchone()
    if not row:
        raise NotFound("Tenant not found")
    return dict(row)


def rotate_key(tenant_id: str) -> str:
    get_tenant(tenant_id)
    key = "cip_" + secrets.token_urlsafe(24)
    with connect() as c:
        c.execute("UPDATE tenants SET api_key_hash=? WHERE tenant_id=?", (_hash(key), tenant_id))
    return key


def list_tenants() -> list[dict]:
    with connect() as c:
        rows = [dict(r) for r in c.execute("SELECT tenant_id, name, created_at FROM tenants ORDER BY created_at")]
        for r in rows:
            t = r["tenant_id"]
            r["transactions"] = tq(c, "SELECT COUNT(*) FROM transactions WHERE tenant_id=?", t).fetchone()[0]
            r["customers"] = tq(c, "SELECT COUNT(DISTINCT customer_id) FROM transactions WHERE tenant_id=?", t).fetchone()[0]
            m = tq(c, "SELECT trained_at FROM tenant_meta WHERE tenant_id=?", t).fetchone()
            r["trained_at"] = m["trained_at"] if m else None
            j = tq(c, "SELECT status FROM jobs WHERE tenant_id=?", t).fetchone()
            r["job_status"] = j["status"] if j else None
    return rows


def delete_tenant(tenant_id: str) -> None:
    get_tenant(tenant_id)
    with connect() as c:
        tq(c, "DELETE FROM tenants WHERE tenant_id=?", tenant_id)
    shutil.rmtree(config.ARTIFACT_DIR / tenant_id, ignore_errors=True)


# ---------------- transactions ----------------
TX_COLS = ["invoice_no", "stock_code", "description", "quantity", "invoice_date", "unit_price",
           "customer_id", "country", "is_cancel", "line_total"]


def replace_transactions(tenant_id: str, df: pd.DataFrame, ingest_report: dict) -> None:
    """Replace ALL data of one tenant (transactions, derived scores, model artifacts)."""
    get_tenant(tenant_id)
    out = df[TX_COLS].copy()
    out["invoice_date"] = out["invoice_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out["is_cancel"] = out["is_cancel"].astype(int)
    out["customer_id"] = out["customer_id"].astype("object").where(out["customer_id"].notna(), None)
    rows = [(tenant_id, *r) for r in out.itertuples(index=False, name=None)]
    with connect() as c:
        for t in ("transactions", "customer_scores", "tenant_meta", "jobs"):
            tq(c, f"DELETE FROM {t} WHERE tenant_id=?", tenant_id)
        c.executemany(f"INSERT INTO transactions VALUES (?,{','.join('?' * len(TX_COLS))})", rows)
        tq(c, "INSERT INTO tenant_meta(tenant_id, ingest_json, trained_at) VALUES (?,?,NULL)", tenant_id,
           (json.dumps(ingest_report),))
    shutil.rmtree(config.ARTIFACT_DIR / tenant_id, ignore_errors=True)


def load_transactions(tenant_id: str) -> pd.DataFrame:
    with connect() as c:
        df = tdf(c, "SELECT invoice_no, stock_code, description, quantity, invoice_date, unit_price, customer_id, "
                    "country, is_cancel, line_total FROM transactions WHERE tenant_id=?", tenant_id)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["is_cancel"] = df["is_cancel"].astype(bool)
    return df


def get_ingest_report(tenant_id: str) -> dict | None:
    with connect() as c:
        r = tq(c, "SELECT ingest_json, trained_at FROM tenant_meta WHERE tenant_id=?", tenant_id).fetchone()
    if not r or not r["ingest_json"]:
        return None
    d = json.loads(r["ingest_json"])
    d["trained_at"] = r["trained_at"]
    return d


def save_scores(tenant_id: str, scores: pd.DataFrame) -> None:
    cols = ["customer_id", *FEATURES, "first_purchase", "last_purchase", "country", "churn_prob",
            "risk_category", "segment_id", "segment_name", "pred_spend"]
    out = scores.reindex(columns=cols).astype(object)
    out = out.where(out.notna(), None)
    rows = [(tenant_id, *r) for r in out.itertuples(index=False, name=None)]
    with connect() as c:
        tq(c, "DELETE FROM customer_scores WHERE tenant_id=?", tenant_id)
        c.executemany(f"INSERT INTO customer_scores VALUES (?,{','.join('?' * len(cols))})", rows)
        c.execute("UPDATE tenant_meta SET trained_at=? WHERE tenant_id=?", (_now(), tenant_id))


# ---------------- jobs ----------------
def try_start_job(tenant_id: str) -> bool:
    with connect() as c:
        row = tq(c, "SELECT status FROM jobs WHERE tenant_id=?", tenant_id).fetchone()
        if row and row["status"] == "running":
            return False
        c.execute("INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,NULL)", (tenant_id, "running", "starting", "", _now()))
    return True


def update_job(tenant_id: str, status: str | None = None, stage: str | None = None, message: str | None = None) -> None:
    with connect() as c:
        row = tq(c, "SELECT status, stage, message FROM jobs WHERE tenant_id=?", tenant_id).fetchone()
        if not row:
            return
        st = status or row["status"]
        fin = _now() if st in ("done", "failed") else None
        c.execute("UPDATE jobs SET status=?, stage=?, message=?, finished_at=? WHERE tenant_id=?",
                  (st, stage or row["stage"], message if message is not None else row["message"], fin, tenant_id))


def get_job(tenant_id: str) -> dict:
    with connect() as c:
        r = tq(c, "SELECT status, stage, message, started_at, finished_at FROM jobs WHERE tenant_id=?", tenant_id).fetchone()
    return dict(r) if r else {"status": "idle", "stage": None, "message": "", "started_at": None, "finished_at": None}
