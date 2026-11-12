"""Feature engineering shared by training AND inference (single source of truth for preprocessing inputs).

Leakage control: features at a cutoff T use only transactions with invoice_date < T.
Labels/targets use only the window [T, T + HORIZON). Snapshots are split chronologically with purging.
"""
import numpy as np
import pandas as pd

from . import config

FEATURES = [
    "recency_days", "frequency", "monetary", "avg_order_value", "tenure_days", "avg_gap_days",
    "gap_std_days", "recency_gap_ratio", "orders_30d", "orders_90d", "spend_30d", "spend_90d",
    "spend_trend", "n_products", "avg_basket_items", "return_rate", "single_order",
]
# Heavy right-skewed columns that get log1p for linear models
SKEWED = ["recency_days", "frequency", "monetary", "avg_order_value", "tenure_days", "avg_gap_days",
          "gap_std_days", "orders_30d", "orders_90d", "spend_30d", "spend_90d", "n_products", "avg_basket_items"]

FEATURE_LABELS = {
    "recency_days": "Days since last purchase", "frequency": "Number of orders", "monetary": "Total spend",
    "avg_order_value": "Average order value", "tenure_days": "Customer age (days)",
    "avg_gap_days": "Avg days between orders", "gap_std_days": "Variability of order gaps",
    "recency_gap_ratio": "Recency vs usual gap", "orders_30d": "Orders in last 30d",
    "orders_90d": "Orders in last 90d", "spend_30d": "Spend in last 30d", "spend_90d": "Spend in last 90d",
    "spend_trend": "Share of spend in last 90d", "n_products": "Distinct products bought",
    "avg_basket_items": "Avg items per order", "return_rate": "Cancellation rate",
    "single_order": "One-time buyer",
}


class InsufficientData(Exception):
    """Raised when a model cannot be trained honestly from the available data."""


def build_features(tx: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Customer-level features from transactions strictly before `as_of`. `tx` must have non-null customer_id."""
    d = tx[tx["invoice_date"] < as_of]
    sales = d[~d["is_cancel"]]
    if sales.empty:
        return pd.DataFrame(columns=FEATURES)
    inv = (sales.groupby(["customer_id", "invoice_no"])
           .agg(date=("invoice_date", "min"), amount=("line_total", "sum"), items=("quantity", "sum"))
           .reset_index().sort_values(["customer_id", "date"]))
    inv["gap"] = inv.groupby("customer_id")["date"].diff().dt.total_seconds() / 86400
    g = inv.groupby("customer_id")
    out = pd.DataFrame({
        "frequency": g.size().astype(float),
        "monetary": g["amount"].sum(),
        "avg_basket_items": g["items"].mean(),
        "first_purchase": g["date"].min(),
        "last_purchase": g["date"].max(),
        "avg_gap_days": g["gap"].mean(),
        "gap_std_days": g["gap"].std(),
    })
    out["recency_days"] = (as_of - out["last_purchase"]).dt.total_seconds() / 86400
    out["tenure_days"] = (as_of - out["first_purchase"]).dt.total_seconds() / 86400
    out["avg_order_value"] = out["monetary"] / out["frequency"]
    out["recency_gap_ratio"] = out["recency_days"] / out["avg_gap_days"].where(out["avg_gap_days"] > 0.5)
    for days in (30, 90):
        w = inv[inv["date"] >= as_of - pd.Timedelta(days=days)].groupby("customer_id")
        out[f"orders_{days}d"] = w.size().reindex(out.index).fillna(0.0)
        out[f"spend_{days}d"] = w["amount"].sum().reindex(out.index).fillna(0.0)
    out["spend_trend"] = out["spend_90d"] / out["monetary"].clip(lower=1e-9)
    out["n_products"] = sales.groupby("customer_id")["stock_code"].nunique().reindex(out.index).astype(float)
    canc = d[d["is_cancel"]].groupby("customer_id")["invoice_no"].nunique().reindex(out.index).fillna(0.0)
    out["return_rate"] = (canc / out["frequency"]).clip(upper=1.0)
    out["single_order"] = (out["frequency"] <= 1).astype(float)
    last_country = sales.sort_values("invoice_date").groupby("customer_id")["country"].last()
    out["country"] = last_country.reindex(out.index)
    out = out.reset_index()
    out["customer_id"] = out["customer_id"].astype("int64")
    return out.set_index("customer_id")


def make_cutoffs(min_date: pd.Timestamp, max_date: pd.Timestamp, horizon: int | None = None) -> list[pd.Timestamp]:
    """Cutoffs whose full label window [T, T+H) lies inside the observed data."""
    h = pd.Timedelta(days=horizon or config.HORIZON_DAYS)
    end = max_date.normalize() + pd.Timedelta(days=1)
    t = min_date.normalize() + pd.Timedelta(days=config.MIN_HISTORY_DAYS)
    cuts = []
    while t + h <= end:
        cuts.append(t)
        t += pd.Timedelta(days=config.SNAPSHOT_STEP_DAYS)
    return cuts


def build_snapshots(tx: pd.DataFrame, cutoffs: list[pd.Timestamp]) -> pd.DataFrame:
    """Stack customer snapshots. Adds `cutoff`, `churned` (no purchase in [T,T+H)) and `future_spend` (sales in [T,T+H))."""
    h = pd.Timedelta(days=config.HORIZON_DAYS)
    sales = tx[~tx["is_cancel"]]
    parts = []
    for t in cutoffs:
        f = build_features(tx, t)
        if f.empty:
            continue
        fut = sales[(sales["invoice_date"] >= t) & (sales["invoice_date"] < t + h)]
        spend = fut.groupby("customer_id")["line_total"].sum()
        f = f.copy()
        f["future_spend"] = spend.reindex(f.index).fillna(0.0)
        f["churned"] = (~f.index.isin(spend.index)).astype(int)
        f["cutoff"] = t
        parts.append(f.reset_index())
    if not parts:
        raise InsufficientData("No customer snapshots could be built")
    return pd.concat(parts, ignore_index=True)


def chronological_split(snaps: pd.DataFrame):
    """train / validation / test by cutoff date, with purging: every training label window ends before the
    validation cutoff, and test is the latest cutoff (never used for fitting or model selection)."""
    h = pd.Timedelta(days=config.HORIZON_DAYS)
    cuts = sorted(snaps["cutoff"].unique())
    if len(cuts) < 3:
        raise InsufficientData(f"Need at least 3 chronological snapshots (have {len(cuts)}); "
                               "the data span is too short for honest time-based validation.")
    test_c, val_c = cuts[-1], cuts[-2]
    train_cuts = [c for c in cuts if c + h <= val_c]
    if not train_cuts:
        raise InsufficientData("No training snapshots remain after purging overlapping label windows.")
    tr = snaps[snaps["cutoff"].isin(train_cuts)]
    va = snaps[snaps["cutoff"] == val_c]
    te = snaps[snaps["cutoff"] == test_c]
    if min(len(tr), len(va), len(te)) < 50:
        raise InsufficientData("Too few customers per chronological split.")
    info = {"train_cutoffs": [str(c)[:10] for c in train_cuts], "val_cutoff": str(val_c)[:10],
            "test_cutoff": str(test_c)[:10], "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "horizon_days": config.HORIZON_DAYS}
    return tr, va, te, info


def numeric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES].astype(float)
