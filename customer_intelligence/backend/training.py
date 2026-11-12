"""Training orchestration for one tenant: features -> 4 models -> scores -> persisted artifacts."""
import logging
import traceback

import numpy as np
import pandas as pd

from . import artifacts, config, db
from .features import (InsufficientData, build_features, build_snapshots, make_cutoffs)
from .ml import churn, clv, recommendations, segmentation

log = logging.getLogger("cip.training")


def train_tenant(tenant_id: str) -> None:
    """Runs in a background thread. Each module fails independently with an honest reason."""
    try:
        db.update_job(tenant_id, stage="loading data")
        tx_all = db.load_transactions(tenant_id)
        if tx_all.empty:
            raise ValueError("This tenant has no transactions. Load data first.")
        tx = tx_all[tx_all["customer_id"].notna()].copy()
        tx["customer_id"] = tx["customer_id"].astype("int64")
        metrics: dict = {"trained_on": {"transactions": int(len(tx_all)), "customer_transactions": int(len(tx)),
                                        "horizon_days": config.HORIZON_DAYS}}
        as_of = tx_all["invoice_date"].max().normalize() + pd.Timedelta(days=1)
        metrics["as_of"] = str(as_of)[:10]

        feats = pd.DataFrame()
        if tx["customer_id"].nunique() >= config.MIN_CUSTOMERS:
            db.update_job(tenant_id, stage="feature engineering")
            feats = build_features(tx, as_of)
        n_cust = len(feats)
        reason = f"Need at least {config.MIN_CUSTOMERS} identified customers (have {tx['customer_id'].nunique()})."

        snaps = None
        if n_cust >= config.MIN_CUSTOMERS:
            try:
                cuts = make_cutoffs(tx["invoice_date"].min(), tx["invoice_date"].max())
                snaps = build_snapshots(tx, cuts)
            except InsufficientData as e:
                reason = str(e)

        scores = feats.copy()
        # ---- A. churn
        db.update_job(tenant_id, stage="churn model")
        try:
            if snaps is None:
                raise InsufficientData(reason)
            art, m = churn.train(snaps)
            artifacts.save(tenant_id, "churn.joblib", art)
            metrics["churn"] = m
            p = churn.predict(art, feats)
            scores["churn_prob"], scores["risk_category"] = p, churn.risk_category(p)
        except InsufficientData as e:
            metrics["churn"] = {"status": "unavailable", "reason": str(e)}

        # ---- D. CLV
        db.update_job(tenant_id, stage="Ridge/Lasso future-spend models")
        try:
            if snaps is None:
                raise InsufficientData(reason)
            art, m = clv.train(snaps)
            artifacts.save(tenant_id, "clv.joblib", art)
            metrics["clv"] = m
            scores["pred_spend"] = clv.predict(art, feats)
        except InsufficientData as e:
            metrics["clv"] = {"status": "unavailable", "reason": str(e)}

        # ---- B. segmentation
        db.update_job(tenant_id, stage="DBSCAN segmentation")
        coords = None
        try:
            if n_cust < config.MIN_CUSTOMERS:
                raise InsufficientData(reason)
            art, m, coords = segmentation.train(feats)
            artifacts.save(tenant_id, "segmentation.joblib", art)
            metrics["segmentation"] = m
            scores = scores.join(coords[["segment_id", "segment_name"]])
            artifacts.save(tenant_id, "segment_points.joblib",
                           coords.join(feats[["monetary", "frequency", "recency_days"]]).reset_index())
        except InsufficientData as e:
            metrics["segmentation"] = {"status": "unavailable", "reason": str(e)}

        # ---- C. recommendations
        db.update_job(tenant_id, stage="Apriori rules")
        try:
            art, m = recommendations.train(tx_all)
            artifacts.save(tenant_id, "rules.joblib", art)
            metrics["recommendations"] = m
        except ValueError as e:
            metrics["recommendations"] = {"status": "unavailable", "reason": str(e)}

        db.update_job(tenant_id, stage="saving scores")
        if n_cust:
            scores = scores.reset_index()
            scores["first_purchase"] = scores["first_purchase"].astype(str).str[:19]
            scores["last_purchase"] = scores["last_purchase"].astype(str).str[:19]
            db.save_scores(tenant_id, scores)
        artifacts.save_json(tenant_id, "metrics.json", metrics)
        ok = [k for k in ("churn", "segmentation", "recommendations", "clv") if metrics.get(k, {}).get("status") == "ok"]
        db.update_job(tenant_id, status="done", stage="finished", message=f"Trained modules: {', '.join(ok) or 'none'}")
    except Exception as e:  # noqa: BLE001 - surfaced to the UI via job status
        log.error("Training failed: %s", traceback.format_exc())
        db.update_job(tenant_id, status="failed", stage="failed", message=f"{type(e).__name__}: {e}")
