"""Shared preprocessing so training and inference are identical (pickled inside each model pipeline)."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..features import FEATURES, SKEWED


class Log1pCols(BaseEstimator, TransformerMixin):
    """log1p on selected column indices (negatives clipped to 0)."""

    def __init__(self, idx=()):
        self.idx = idx

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        for i in self.idx:
            X[:, i] = np.log1p(np.clip(X[:, i], 0, None))
        return X


def linear_preprocessor() -> Pipeline:
    idx = [FEATURES.index(c) for c in SKEWED]
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("log", Log1pCols(idx)),
                     ("scale", StandardScaler())])


def tree_preprocessor() -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median"))])
