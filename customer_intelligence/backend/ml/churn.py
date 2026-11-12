"""Churn prediction: inactivity-derived labels, chronological validation, model comparison."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline

from .. import config
from ..features import FEATURES, FEATURE_LABELS, InsufficientData, chronological_split, numeric_matrix
from .transformers import linear_preprocessor, tree_preprocessor


def _candidates() -> dict:
    rs = config.RANDOM_STATE
    return {
        "Baseline (prior)": Pipeline([("pre", tree_preprocessor()), ("m", DummyClassifier(strategy="prior"))]),
        "Logistic Regression": Pipeline([("pre", linear_preprocessor()), ("m", LogisticRegression(max_iter=2000, C=0.5))]),
        "Random Forest": Pipeline([("pre", tree_preprocessor()),
                                   ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=10, n_jobs=1, random_state=rs))]),
        "Gradient Boosting": Pipeline([("pre", tree_preprocessor()),
                                       ("m", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                                                            l2_regularization=1.0, early_stopping=False,
                                                                            random_state=rs))]),
    }


def _proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def _metrics(y, p, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    out = {"roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
           "pr_auc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)),
           "accuracy": float(accuracy_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)),
           "threshold": float(thr)}
    return out


def _best_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[int(np.argmax(f1))]) if len(thr) else 0.5


def train(snaps: pd.DataFrame) -> tuple[dict, dict]:
    tr, va, te, split = chronological_split(snaps)
    for name, part in (("train", tr), ("validation", va), ("test", te)):
        if part["churned"].nunique() < 2:
            raise InsufficientData(f"Churn labels in the {name} split contain a single class.")
    Xtr, Xva, Xte = numeric_matrix(tr), numeric_matrix(va), numeric_matrix(te)
    ytr, yva, yte = tr["churned"].values, va["churned"].values, te["churned"].values

    results, fitted = {}, {}
    for name, pipe in _candidates().items():
        pipe.fit(Xtr, ytr)
        pv, pt = _proba(pipe, Xva), _proba(pipe, Xte)
        thr = _best_threshold(yva, pv) if name != "Baseline (prior)" else 0.5
        results[name] = {"validation": _metrics(yva, pv, thr), "test": _metrics(yte, pt, thr)}
        fitted[name] = (pipe, pt)
    real = [n for n in results if n != "Baseline (prior)"]
    best = max(real, key=lambda n: (results[n]["validation"]["roc_auc"], results[n]["validation"]["pr_auc"]))
    pipe, pt = fitted[best]
    thr = results[best]["test"]["threshold"]

    fpr, tpr, _ = roc_curve(yte, pt)
    sel = np.linspace(0, len(fpr) - 1, min(len(fpr), 100)).astype(int)
    pr_p, pr_r, _ = precision_recall_curve(yte, pt)
    sel2 = np.linspace(0, len(pr_p) - 1, min(len(pr_p), 100)).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, (pt >= thr).astype(int), labels=[0, 1]).ravel()
    imp = permutation_importance(pipe, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=config.RANDOM_STATE, n_jobs=1)
    importance = sorted(({"feature": FEATURE_LABELS[f], "importance": float(m)} for f, m in zip(FEATURES, imp.importances_mean)),
                        key=lambda d: -d["importance"])

    # Production model: evaluation is finished, so refit the chosen model on ALL labelled snapshots.
    labelled = pd.concat([tr, va, te])
    final = _candidates()[best]
    final.fit(numeric_matrix(labelled), labelled["churned"].values)
    baseline = numeric_matrix(labelled).median()
    artifact = {"model": final, "name": best, "baseline": baseline.to_dict(), "threshold": thr}
    metrics = {
        "status": "ok", "selected_model": best, "label_definition":
            f"Churned = no purchase in the {config.HORIZON_DAYS} days after the snapshot date (derived from inactivity; "
            "the dataset has no real churn labels).",
        "split": split, "churn_rate": {"train": float(ytr.mean()), "validation": float(yva.mean()), "test": float(yte.mean())},
        "models": results, "roc_curve": {"fpr": fpr[sel].tolist(), "tpr": tpr[sel].tolist()},
        "pr_curve": {"precision": pr_p[sel2].tolist(), "recall": pr_r[sel2].tolist()},
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "feature_importance": importance,
        "note": "Metrics are on the held-out latest snapshot. The deployed model is refit on all labelled snapshots.",
    }
    return artifact, metrics


def predict(art: dict, feats: pd.DataFrame) -> np.ndarray:
    return _proba(art["model"], numeric_matrix(feats))


def risk_category(p: np.ndarray | float):
    return np.where(np.asarray(p) < config.RISK_LOW, "Low", np.where(np.asarray(p) < config.RISK_HIGH, "Medium", "High"))


def explain(art: dict, row: pd.DataFrame, top_k: int = 6) -> list[dict]:
    """Occlusion-based contributions: change in churn probability when a feature is replaced by the typical
    (median) customer's value. Positive = pushes risk up. Same pipeline as scoring, so always consistent."""
    x = numeric_matrix(row).iloc[[0]]
    variants = pd.concat([x] + [x.assign(**{f: art["baseline"][f]}) for f in FEATURES], ignore_index=True)
    p = _proba(art["model"], variants)
    contrib = p[0] - p[1:]
    items = [{"feature": FEATURE_LABELS[f], "value": float(x.iloc[0][f]) if pd.notna(x.iloc[0][f]) else None,
              "typical": float(art["baseline"][f]), "contribution": float(c)} for f, c in zip(FEATURES, contrib)]
    items.sort(key=lambda d: -abs(d["contribution"]))
    return items[:top_k]
