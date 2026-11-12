"""Tenant-scoped routes. Every route depends on tenant_auth and every SQL statement filters on tenant_id."""
import tempfile
from pathlib import Path as FsPath

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Path, Query, UploadFile
from pydantic import BaseModel, Field

from .. import artifacts, config, db, ingestion, inference, training
from .deps import tenant_auth

router = APIRouter(prefix="/tenants/{tenant_id}", tags=["tenant"], dependencies=[Depends(tenant_auth)])
TID = Path(..., pattern=db.TENANT_ID_RE)


def _clean(v):
    """JSON-safe conversion (NaN -> None)."""
    return None if (isinstance(v, float) and v != v) else v


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in r.items()} for r in df.astype(object).where(df.notna(), None).to_dict("records")]


def _has_data(tid: str) -> bool:
    with db.connect() as c:
        return db.tq(c, "SELECT 1 FROM transactions WHERE tenant_id=? LIMIT 1", tid).fetchone() is not None


def _need_data(tid: str):
    if not _has_data(tid):
        raise HTTPException(409, "This tenant has no data yet. Load a dataset from Tenant Management.")


# ------------------------------------------------------------------ status / data / training
@router.get("/status")
def status(tenant_id: str = TID):
    return {"tenant": db.get_tenant(tenant_id), "has_data": _has_data(tenant_id), "trained": artifacts.has_metrics(tenant_id),
            "job": db.get_job(tenant_id), "ingest": db.get_ingest_report(tenant_id)}


def _store(tid: str, df: pd.DataFrame, report: dict):
    db.replace_transactions(tid, df, report)
    return {"tenant_id": tid, "report": report, "next": "Train models to build analytics."}


@router.post("/data/upload")
def upload(tenant_id: str = TID, file: UploadFile = File(...)):
    suffix = FsPath(file.filename or "").suffix.lower()
    if suffix not in (".csv", ".xlsx", ".xls", ".txt"):
        raise ingestion.IngestError("Unsupported file type. Upload a .csv or .xlsx file.")
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        size = 0
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                tmp.close()
                FsPath(tmp.name).unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {config.MAX_UPLOAD_MB} MB")
            tmp.write(chunk)
    try:
        raw = ingestion.read_file(FsPath(tmp.name), file.filename)
        df, report = ingestion.clean(raw)
    finally:
        FsPath(tmp.name).unlink(missing_ok=True)
    return _store(tenant_id, df, report)


@router.post("/data/demo")
def load_demo(tenant_id: str = TID, part: int = Query(0, ge=0, le=9), of: int = Query(1, ge=1, le=10)):
    """Load a deterministic customer-level partition of the bundled UCI Online Retail dataset."""
    if not config.RAW_DATA_PATH.exists():
        raise HTTPException(409, f"Demo dataset not found at {config.RAW_DATA_PATH}. Run: python -m scripts.download_data")
    df, full = ingestion.clean(ingestion.read_file(config.RAW_DATA_PATH))
    df = ingestion.demo_partition(df, part, of)
    if df.empty:
        raise ingestion.IngestError("Partition is empty")
    report = ingestion.summarize(df, full["rows_in"] if of == 1 else len(df), full["dropped"] if of == 1 else {})
    report["source"] = f"UCI Online Retail, partition {part + 1} of {of}"
    return _store(tenant_id, df, report)


@router.post("/train", status_code=202)
def train(background: BackgroundTasks, tenant_id: str = TID):
    _need_data(tenant_id)
    if not db.try_start_job(tenant_id):
        raise HTTPException(409, "Training is already running for this tenant")
    background.add_task(training.train_tenant, tenant_id)
    return {"status": "started"}


@router.get("/train/status")
def train_status(tenant_id: str = TID):
    return db.get_job(tenant_id)


@router.get("/metrics")
def metrics(tenant_id: str = TID):
    return inference.require_trained(tenant_id)


# ------------------------------------------------------------------ dashboard
@router.get("/overview")
def overview(tenant_id: str = TID):
    _need_data(tenant_id)
    with db.connect() as c:
        k = dict(db.tq(c, """SELECT
            COALESCE(SUM(CASE WHEN is_cancel=0 THEN line_total END),0) AS revenue,
            COALESCE(SUM(CASE WHEN is_cancel=1 THEN -line_total END),0) AS returns_value,
            COUNT(DISTINCT CASE WHEN is_cancel=0 THEN invoice_no END) AS transactions,
            COUNT(DISTINCT customer_id) AS customers,
            MIN(invoice_date) AS date_min, MAX(invoice_date) AS date_max
            FROM transactions WHERE tenant_id=?""", tenant_id).fetchone())
        scores = db.tdf(c, "SELECT risk_category, segment_name, monetary, churn_prob, pred_spend FROM customer_scores WHERE tenant_id=?",
                        tenant_id)
        monthly = db.tdf(c, """SELECT strftime('%Y-%m', invoice_date) AS month, SUM(line_total) AS revenue,
            COUNT(DISTINCT invoice_no) AS orders FROM transactions WHERE tenant_id=? AND is_cancel=0 GROUP BY month ORDER BY month""",
                         tenant_id)
    k["avg_order_value"] = k["revenue"] / k["transactions"] if k["transactions"] else 0
    out = {"kpis": k, "monthly": _records(monthly), "trained": artifacts.has_metrics(tenant_id),
           "churn_distribution": [], "segment_distribution": [], "insights": [], "currency": config.CURRENCY}
    cur = config.CURRENCY
    if not scores.empty and scores["risk_category"].notna().any():
        g = scores.groupby("risk_category").agg(customers=("monetary", "size"), historic_spend=("monetary", "sum"),
                                                 avg_churn_prob=("churn_prob", "mean")).reset_index()
        out["churn_distribution"] = _records(g)
        n = len(scores)
        hi = scores[scores["risk_category"] == "High"]
        out["insights"].append(
            f"{len(hi)} customers ({len(hi) / n:.0%}) are High churn risk and account for {cur}{hi['monetary'].sum():,.0f} "
            f"({hi['monetary'].sum() / scores['monetary'].sum():.0%}) of historic customer spend.")
        top = scores.sort_values("churn_prob", ascending=False).head(1)
        out["insights"].append(f"Expected share of customers lapsing within {config.HORIZON_DAYS} days (mean predicted risk): "
                               f"{scores['churn_prob'].mean():.0%}.")
    if not scores.empty and scores["segment_name"].notna().any():
        g = scores.groupby("segment_name").agg(customers=("monetary", "size"), revenue=("monetary", "sum")).reset_index()
        g["revenue_share"] = g["revenue"] / g["revenue"].sum()
        out["segment_distribution"] = _records(g.sort_values("revenue", ascending=False))
        best = g.sort_values("revenue", ascending=False).iloc[0]
        out["insights"].append(f"Segment '{best['segment_name']}' holds {best['revenue_share']:.0%} of customer revenue "
                               f"with {int(best['customers'])} customers.")
    if not scores.empty and scores["pred_spend"].notna().any():
        out["insights"].append(f"Predicted spend from identified customers over the next {config.HORIZON_DAYS} days: "
                               f"{cur}{scores['pred_spend'].sum():,.0f} (a model estimate, not a guarantee).")
    if k["returns_value"] and k["revenue"]:
        out["insights"].append(f"Cancellations/returns equal {k['returns_value'] / k['revenue']:.1%} of gross revenue.")
    return out


# ------------------------------------------------------------------ customers
SORTS = {"churn_prob": "churn_prob", "monetary": "monetary", "recency_days": "recency_days",
         "frequency": "frequency", "pred_spend": "pred_spend", "customer_id": "customer_id"}


@router.get("/customers")
def customers(tenant_id: str = TID, search: str | None = Query(None, max_length=20, pattern=r"^[0-9]*$"),
              risk: str | None = Query(None, pattern="^(Low|Medium|High)$"), segment_id: int | None = None,
              sort: str = Query("churn_prob", pattern="^(" + "|".join(SORTS) + ")$"), desc: bool = True,
              limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    inference.require_trained(tenant_id)
    where, params = ["tenant_id = ?"], []
    if search:
        where.append("CAST(customer_id AS TEXT) LIKE ?")
        params.append(f"{search}%")
    if risk:
        where.append("risk_category = ?")
        params.append(risk)
    if segment_id is not None:
        where.append("segment_id = ?")
        params.append(segment_id)
    w = " AND ".join(where)
    cols = ("customer_id, country, recency_days, frequency, monetary, avg_order_value, churn_prob, risk_category, "
            "segment_id, segment_name, pred_spend, last_purchase")
    with db.connect() as c:
        total = db.tq(c, f"SELECT COUNT(*) FROM customer_scores WHERE {w}", tenant_id, tuple(params)).fetchone()[0]
        df = db.tdf(c, f"SELECT {cols} FROM customer_scores WHERE {w} ORDER BY {SORTS[sort]} {'DESC' if desc else 'ASC'} "
                       f"NULLS LAST LIMIT ? OFFSET ?", tenant_id, (*params, limit, offset))
    return {"total": total, "items": _records(df)}


@router.get("/customers/{customer_id}")
def customer_detail(customer_id: int = Path(..., ge=0), tenant_id: str = TID):
    inference.require_trained(tenant_id)
    row = inference.customer_row(tenant_id, customer_id)
    if row is None:
        raise HTTPException(404, "Customer not found in this tenant")
    rec = _records(row.drop(columns=["tenant_id"]))[0]
    with db.connect() as c:
        inv = db.tdf(c, """SELECT invoice_no, MIN(invoice_date) AS date, SUM(line_total) AS amount, SUM(quantity) AS items,
                  MAX(is_cancel) AS is_cancel FROM transactions WHERE tenant_id=? AND customer_id=?
                  GROUP BY invoice_no ORDER BY date DESC LIMIT 15""", tenant_id, (customer_id,))
        prod = db.tdf(c, """SELECT stock_code, MAX(description) AS description, SUM(quantity) AS quantity, SUM(line_total) AS spend
                  FROM transactions WHERE tenant_id=? AND customer_id=? AND is_cancel=0
                  GROUP BY stock_code ORDER BY spend DESC LIMIT 8""", tenant_id, (customer_id,))
    return {"customer": rec, "churn_factors": inference.churn_factors(tenant_id, row),
            "recent_invoices": _records(inv), "top_products": _records(prod),
            "notes": {"pred_spend": f"Predicted spend over the next {config.HORIZON_DAYS} days - NOT lifetime value.",
                      "churn": f"Probability of no purchase in the next {config.HORIZON_DAYS} days (inactivity-derived label)."}}


# ------------------------------------------------------------------ segmentation
@router.get("/segments")
def segments(tenant_id: str = TID):
    m = inference.require_module(tenant_id, "segmentation")
    pts = artifacts.load(tenant_id, "segment_points.joblib")
    return {"metrics": m, "points": _records(pts.round(4))}


# ------------------------------------------------------------------ recommendations
class RecIn(BaseModel):
    stock_codes: list[str] = Field(min_length=1, max_length=10)
    top_n: int = Field(10, ge=1, le=50)
    min_lift: float = Field(1.0, ge=0, le=100)


@router.get("/recommendations/products")
def rec_products(tenant_id: str = TID):
    inference.require_module(tenant_id, "recommendations")
    from ..ml import recommendations as R
    return R.antecedent_products(artifacts.load(tenant_id, "rules.joblib"))


@router.post("/recommendations")
def recommendations(body: RecIn, tenant_id: str = TID):
    return {"items": inference.recommend(tenant_id, [s.strip().upper() for s in body.stock_codes], body.top_n, body.min_lift)}


@router.get("/recommendations/rules")
def rules(tenant_id: str = TID, limit: int = Query(200, ge=1, le=2000)):
    inference.require_module(tenant_id, "recommendations")
    art = artifacts.load(tenant_id, "rules.joblib")
    r = art["rules"].head(limit).copy()
    r["antecedents"] = r["antecedents"].map(lambda t: [art["names"].get(x, x) for x in t])
    r["consequents"] = r["consequents"].map(lambda t: [art["names"].get(x, x) for x in t])
    return _records(r)


# ------------------------------------------------------------------ data explorer
@router.get("/explorer/stats")
def explorer_stats(tenant_id: str = TID):
    _need_data(tenant_id)
    with db.connect() as c:
        s = dict(db.tq(c, """SELECT COUNT(*) AS lines, COUNT(DISTINCT invoice_no) AS invoices, COUNT(DISTINCT customer_id) AS customers,
            COUNT(DISTINCT stock_code) AS products, COUNT(DISTINCT country) AS countries,
            AVG(CASE WHEN customer_id IS NULL THEN 1.0 ELSE 0 END) AS anonymous_line_share,
            AVG(is_cancel) AS cancellation_line_share, MIN(invoice_date) AS date_min, MAX(invoice_date) AS date_max,
            AVG(CASE WHEN is_cancel=0 THEN unit_price END) AS avg_unit_price, AVG(CASE WHEN is_cancel=0 THEN quantity END) AS avg_quantity
            FROM transactions WHERE tenant_id=?""", tenant_id).fetchone())
        inv = db.tdf(c, "SELECT SUM(line_total) AS v FROM transactions WHERE tenant_id=? AND is_cancel=0 GROUP BY invoice_no", tenant_id)["v"]
    s.update({"median_order_value": float(inv.median()) if len(inv) else None, "mean_order_value": float(inv.mean()) if len(inv) else None,
              "p95_order_value": float(inv.quantile(0.95)) if len(inv) else None, "ingest": db.get_ingest_report(tenant_id)})
    return s


@router.get("/explorer/trends")
def explorer_trends(tenant_id: str = TID, freq: str = Query("month", pattern="^(day|week|month)$")):
    _need_data(tenant_id)
    fmt = {"day": "%Y-%m-%d", "week": "%Y-%W", "month": "%Y-%m"}[freq]
    with db.connect() as c:
        df = db.tdf(c, f"""SELECT strftime('{fmt}', invoice_date) AS period, SUM(line_total) AS revenue,
            COUNT(DISTINCT invoice_no) AS orders, COUNT(DISTINCT customer_id) AS customers
            FROM transactions WHERE tenant_id=? AND is_cancel=0 GROUP BY period ORDER BY period""", tenant_id)
    return _records(df)


@router.get("/explorer/products")
def explorer_products(tenant_id: str = TID, top: int = Query(15, ge=1, le=100),
                      by: str = Query("revenue", pattern="^(revenue|quantity|orders)$"),
                      q: str | None = Query(None, max_length=50)):
    _need_data(tenant_id)
    order = {"revenue": "revenue", "quantity": "quantity", "orders": "orders"}[by]
    where, params = "tenant_id=? AND is_cancel=0", []
    if q:
        where += " AND (UPPER(description) LIKE ? OR stock_code LIKE ?)"
        like = f"%{q.upper().replace('%', '').replace('_', '')}%"
        params += [like, like]
    with db.connect() as c:
        df = db.tdf(c, f"""SELECT stock_code, MAX(description) AS description, SUM(line_total) AS revenue,
            SUM(quantity) AS quantity, COUNT(DISTINCT invoice_no) AS orders FROM transactions WHERE {where}
            GROUP BY stock_code ORDER BY {order} DESC LIMIT ?""", tenant_id, (*params, top))
    return _records(df)


@router.get("/explorer/countries")
def explorer_countries(tenant_id: str = TID, top: int = Query(10, ge=1, le=50)):
    _need_data(tenant_id)
    with db.connect() as c:
        df = db.tdf(c, """SELECT country, SUM(line_total) AS revenue, COUNT(DISTINCT customer_id) AS customers
            FROM transactions WHERE tenant_id=? AND is_cancel=0 GROUP BY country ORDER BY revenue DESC LIMIT ?""", tenant_id, (top,))
    return _records(df)
