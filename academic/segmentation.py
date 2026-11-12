"""Training-only RFM clustering with explicit, frozen inductive assignment rules."""
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

RFM = ['recency_days', 'frequency', 'monetary']


def quality(z, labels, sample_size=1500):
    """Exclude DBSCAN noise; bound silhouette work, retaining all rows for fitting and DBI."""
    mask = labels != -1
    x, y = z[mask], labels[mask]
    count = len(np.unique(y))
    out = {'n_clusters': count, 'noise_share': float(np.mean(~mask)),
           'silhouette': None, 'davies_bouldin': None,
           'metric_population': 'non-noise training observations', 'silhouette_n': 0}
    if not 2 <= count < len(y):
        return out
    # Stratification ensures small clusters survive the deterministic metric sample.
    if len(y) > sample_size:
        rng = np.random.default_rng(42)
        required = np.array([np.flatnonzero(y == c)[0] for c in np.unique(y)])
        remaining = np.setdiff1d(np.arange(len(y)), required)
        idx = np.r_[required, rng.choice(remaining, sample_size-len(required), replace=False)]
        xs, ys = x[idx], y[idx]
    else:
        xs, ys = x, y
    out.update(silhouette=float(silhouette_score(xs, ys)),
               davies_bouldin=float(davies_bouldin_score(x, y)), silhouette_n=len(ys))
    return out


class RFMSegmenter:
    def __init__(self, algorithm):
        if algorithm not in ('agglomerative', 'dbscan'):
            raise ValueError('Unknown clustering algorithm')
        self.algorithm = algorithm

    def _raw(self, frame):
        x = frame[RFM].to_numpy(dtype=float)
        if len(x) == 0 or not np.isfinite(x).all() or (x < 0).any():
            raise ValueError('RFM must contain finite nonnegative values and at least one row')
        return np.log1p(x)

    def fit(self, frame):
        self.scaler = StandardScaler().fit(self._raw(frame))
        z = self.scaler.transform(self._raw(frame))
        self.search = []
        unique = len(np.unique(z, axis=0))
        if len(z) < 3 or unique < 2:
            self.labels_ = np.zeros(len(z), dtype=int)
            self.eps, self.min_samples = 1e-6, 2
            self.core, self.core_labels = z, self.labels_
        elif self.algorithm == 'agglomerative':
            best = None
            for k in range(2, min(6, len(z)-1, unique)+1):
                labels = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(z)
                q = quality(z, labels)
                self.search.append({'k': k, **q})
                if best is None or q['silhouette'] > best[0]:
                    best = (q['silhouette'], labels, k)
            self.labels_, self.n_clusters = best[1], best[2]
        else:
            best = None
            for k in sorted(set(min(x, len(z)-1) for x in (5, 10, 15))):
                # k includes self, matching DBSCAN's min_samples convention.
                distances = NearestNeighbors(n_neighbors=k).fit(z).kneighbors(z)[0][:, -1]
                for quantile in (.60, .75, .85, .92):
                    eps = max(float(np.quantile(distances, quantile)), 1e-6)
                    model = DBSCAN(eps=eps, min_samples=k).fit(z)
                    q = quality(z, model.labels_)
                    self.search.append({'eps': eps, 'min_samples': k, **q})
                    valid = q['silhouette'] is not None and q['noise_share'] <= .35 and q['n_clusters'] <= 8
                    score = q['silhouette'] if valid else -np.inf
                    if best is None or score > best[0]:
                        best = (score, model, eps, k)
            _, model, self.eps, self.min_samples = best
            self.labels_ = model.labels_
            self.core = z[model.core_sample_indices_]
            self.core_labels = self.labels_[model.core_sample_indices_]
        self.cluster_ids = np.unique(self.labels_[self.labels_ >= 0])
        self.centroids = np.array([z[self.labels_ == c].mean(axis=0) for c in self.cluster_ids])
        if self.algorithm == 'dbscan' and len(self.core):
            self.neighbors = NearestNeighbors(n_neighbors=1).fit(self.core)
        self.metrics = quality(z, self.labels_)
        self.metrics['algorithm'] = self.algorithm
        self.metrics['status'] = 'valid' if self.metrics['silhouette'] is not None else 'degenerate; metrics undefined'
        self.metrics['assignment'] = ('nearest training centroid in scaled log-RFM; Ward approximation'
                                      if self.algorithm == 'agglomerative' else
                                      'nearest training core within fitted eps; otherwise noise (-1)')
        if self.algorithm == 'dbscan':
            self.metrics.update(eps=self.eps, min_samples=self.min_samples)
        return self

    def assign(self, frame):
        if frame.empty:
            return np.array([], dtype=int)
        z = self.scaler.transform(self._raw(frame))
        if self.algorithm == 'agglomerative':
            return self.cluster_ids[((z[:, None, :]-self.centroids[None, :, :])**2).sum(axis=2).argmin(axis=1)]
        if not len(self.core):
            return np.full(len(z), -1, dtype=int)
        d, i = self.neighbors.kneighbors(z)
        return np.where(d[:, 0] <= self.eps, self.core_labels[i[:, 0]], -1)


def profiles(frame, labels, algorithm, population, probabilities=None):
    df = frame.copy()
    df['segment'] = labels
    if probabilities is not None:
        df['predicted_risk'] = probabilities
        df['retention_priority'] = probabilities * df['monetary']
    out = df.groupby('segment').agg(customers=('customer_id', 'size'),
                                    median_recency=('recency_days', 'median'),
                                    median_frequency=('frequency', 'median'),
                                    median_monetary=('monetary', 'median'),
                                    observed_churn_rate=('churned', 'mean'))
    if probabilities is not None:
        out['mean_predicted_risk'] = df.groupby('segment')['predicted_risk'].mean()
        out['total_retention_priority'] = df.groupby('segment')['retention_priority'].sum()
    global_med = df[RFM].median()
    def name(row):
        if row.name == -1:
            return 'Noise / unusual purchasing patterns'
        active = row.median_recency <= global_med.recency_days
        valuable = row.median_monetary >= global_med.monetary
        frequent = row.median_frequency >= global_med.frequency
        return f"{'Recent' if active else 'Lapsed'} / {'high' if valuable else 'low'} spend / {'frequent' if frequent else 'occasional'}"
    out['interpretation'] = out.apply(name, axis=1)
    out['algorithm'], out['population'] = algorithm, population
    return out.reset_index()
