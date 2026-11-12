"""The same historical feature preprocessing and validation protocol for all six experiments."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, roc_auc_score
from .features import FEATURES, SKEWED


def log_features(x):
    out = np.array(x, dtype=float, copy=True)
    out[~np.isfinite(out)] = np.nan
    for col in SKEWED:
        i = FEATURES.index(col)
        out[:, i] = np.log1p(np.clip(out[:, i], 0, None))
    return out


def preprocessor():
    return Pipeline([('log', FunctionTransformer(log_features)),
                     ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('scale', StandardScaler())])


def encoder(labels):
    # A dedicated noise category exists even when training happens to contain no noise.
    categories = np.unique(np.r_[labels, -1]).astype(int)
    enc = OneHotEncoder(categories=[categories], handle_unknown='ignore', sparse_output=False)
    return enc.fit(labels.reshape(-1, 1))


def estimator(name):
    if name == 'Logistic Regression':
        return LogisticRegression(C=.5, max_iter=2000, random_state=42)
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=42, n_jobs=1)


def threshold(y, probabilities):
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    score = 2*precision[:-1]*recall[:-1] / np.maximum(precision[:-1]+recall[:-1], 1e-12)
    return float(thresholds[np.argmax(score)]) if len(thresholds) else .5


def metrics(y, p, cutoff):
    pred = p >= cutoff
    return {'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None}
