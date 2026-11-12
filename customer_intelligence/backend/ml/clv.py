"""Future-spend (CLV-style) estimation with Ridge and Lasso.

Target = a customer's actual net-of-returns-free sales spend in the next HORIZON days after a snapshot date
(zero if they do not buy). This is *predicted future spend over a fixed horizon*, NOT true lifetime value.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .. import config
from ..features import FEATURES, FEATURE_LABELS, chronological_split, numeric_matrix
from .transformers import linear_preprocessor

GRIDS = {"Ridge": np.logspace(-2, 3, 11), "Lasso": np.logspace(-4, 0, 11)}


def _make(kind: str, alpha: float) -> Pipeline:
    est = Ridge(alpha=alpha) if kind == "Ridge" else Lasso(alpha=alpha, max_iter=20000, random_state=config.RANDOM_STATE)
    return Pipeline([("pre", linear_preprocessor()), ("m", est)])


def _to_money(pred_log: np.ndarray) -> np.ndarray:
    """Back-transform log1p predictions (a median-type estimate; it tends to under-predict the mean of big spenders)."""
    return np.clip(np.expm1(pred_log), 0, None)


def _m(y_true, pred, y_log, pred_log) -> dict:
    return {"mae": float(mean_absolute_error(y_true, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
            "r2": float(r2_score(y_true, pred)),
            "log_rmse": float(np.sqrt(mean_squared_error(y_log, pred_log))),
            "log_r2": float(r2_score(y_log, pred_log))}


def train(snaps: pd.DataFrame) -> tuple[dict, dict]:
    tr, va, te, split = chronological_split(snaps)
    Xtr, Xva, Xte = numeric_matrix(tr), numeric_matrix(va), numeric_matrix(te)
    ytr, yva, yte = (d["future_spend"].clip(lower=0).values for d in (tr, va, te))
    ltr, lva, lte = np.log1p(ytr), np.log1p(yva), np.log1p(yte)

    models, chosen = {}, {}
    for kind, grid in GRIDS.items():
        best = None
        for a in grid:  # alpha picked on the LATER validation snapshot, not by shuffled CV
            m = _make(kind, a).fit(Xtr, ltr)
            score = np.sqrt(mean_squared_error(lva, m.predict(Xva)))
            if best is None or score < best[0]:
                best = (score, a, m)
        _, a, m = best
        pl_va, pl_te = m.predict(Xva), m.predict(Xte)
        coefs = m.named_steps["m"].coef_
        models[kind] = {"alpha": float(a),
                        "validation": _m(yva, _to_money(pl_va), lva, pl_va),
                        "test": _m(yte, _to_money(pl_te), lte, pl_te),
                        "nonzero_coefficients": int(np.sum(np.abs(coefs) > 1e-9)), "n_features": len(coefs),
                        "coefficients": sorted(({"feature": FEATURE_LABELS[f], "coef": float(c)} for f, c in zip(FEATURES, coefs)),
                                               key=lambda d: -abs(d["coef"]))}
        chosen[kind] = m
    mean_pred = np.full_like(yte, ytr.mean())
    baselines = {"Train mean": _m(yte, mean_pred, lte, np.full_like(lte, ltr.mean())),
                 "Naive: last-90d spend": _m(yte, te["spend_90d"].values, lte, np.log1p(te["spend_90d"].clip(lower=0).values))}

    selected = min(models, key=lambda k: models[k]["validation"]["log_rmse"])
    pl = chosen[selected].predict(Xte)
    rng = np.random.RandomState(config.RANDOM_STATE)
    idx = rng.choice(len(yte), size=min(400, len(yte)), replace=False)
    scatter = {"actual": yte[idx].round(2).tolist(), "predicted": _to_money(pl)[idx].round(2).tolist()}

    labelled = pd.concat([tr, va, te])
    Xl, ll = numeric_matrix(labelled), np.log1p(labelled["future_spend"].clip(lower=0).values)
    finals = {}
    for kind in GRIDS:
        f = _make(kind, models[kind]["alpha"]).fit(Xl, ll)
        finals[kind] = {"model": f}
    artifact = {"models": finals, "selected": selected}
    metrics = {"status": "ok", "selected_model": selected,
               "target_definition": f"Actual sales spend in the {config.HORIZON_DAYS} days following each snapshot date "
                                    "(0 if the customer did not buy). This is predicted FUTURE SPEND over a fixed horizon, not "
                                    "true lifetime value.",
               "selection_rule": "Alpha and model chosen on the validation snapshot by log-scale RMSE; metrics below are on the test snapshot.",
               "split": split, "models": models, "baselines": baselines, "scatter": scatter,
               "note": "Models are trained on log(1+spend) with log-scaled features; money-scale predictions are back-transformed "
                       "(median-type, so large spenders are under-predicted). Money-scale RMSE/R² are dominated by a few very large "
                       "customers - compare with the naive last-90d-spend baseline before relying on the numbers."}
    return artifact, metrics


def predict(art: dict, feats: pd.DataFrame, kind: str | None = None) -> np.ndarray:
    f = art["models"][kind or art["selected"]]
    X = numeric_matrix(feats)
    return _to_money(f["model"].predict(X))
