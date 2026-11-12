"""Inference helpers: load persisted artifacts (cached) and score / explain / recommend. Never retrains."""
import pandas as pd

from . import artifacts, db
from .features import FEATURES
from .ml import churn, recommendations


class ModelsUnavailable(Exception):
    pass


def require_trained(tenant_id: str) -> dict:
    m = artifacts.load_json(tenant_id, "metrics.json")
    if m is None:
        raise ModelsUnavailable("Models have not been trained for this tenant yet. Train them from Tenant Management.")
    return m


def require_module(tenant_id: str, module: str) -> dict:
    m = require_trained(tenant_id).get(module, {})
    if m.get("status") != "ok":
        raise ModelsUnavailable(f"The {module} module is unavailable: {m.get('reason', 'not trained')}")
    return m


def customer_row(tenant_id: str, customer_id: int):
    with db.connect() as c:
        df = db.tdf(c, "SELECT * FROM customer_scores WHERE tenant_id=? AND customer_id=?", tenant_id, (customer_id,))
    return None if df.empty else df


def churn_factors(tenant_id: str, row: pd.DataFrame) -> list[dict]:
    art = artifacts.load(tenant_id, "churn.joblib")
    return churn.explain(art, row[FEATURES]) if art else []


def recommend(tenant_id: str, basket: list[str], top_n: int, min_lift: float) -> list[dict]:
    require_module(tenant_id, "recommendations")
    return recommendations.recommend(artifacts.load(tenant_id, "rules.joblib"), basket, top_n, min_lift)
