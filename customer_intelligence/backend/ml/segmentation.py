"""Customer segmentation with DBSCAN on log-scaled, standardised RFM."""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .. import config

RFM = ["recency_days", "frequency", "monetary"]


def _transform(df: pd.DataFrame) -> np.ndarray:
    X = df[RFM].astype(float).clip(lower=0).to_numpy()
    return np.log1p(X)


def _knee_eps(X: np.ndarray, k: int) -> float:
    d = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[0][:, -1]
    d = np.sort(d)
    x = np.linspace(0, 1, len(d))
    yn = (d - d[0]) / max(d[-1] - d[0], 1e-9)
    return float(d[int(np.argmax(x - yn))]) if len(d) else 0.5


def _search(Z: np.ndarray) -> tuple[float, int, list[dict]]:
    n = len(Z)
    tried, best = [], None
    for k in (5, 10, 15, 20):
        if k >= n:
            continue
        kd = np.sort(NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)[0][:, -1])
        for q in np.linspace(0.55, 0.97, 15):
            eps = float(np.quantile(kd, q))
            lab = DBSCAN(eps=eps, min_samples=k).fit_predict(Z)
            ncl, noise = len(set(lab)) - (1 if -1 in lab else 0), float(np.mean(lab == -1))
            sizes = pd.Series(lab[lab >= 0]).value_counts()
            if ncl < 2 or ncl > 8 or noise > 0.20 or sizes.min() < max(10, 0.01 * n):
                continue
            sil = float(silhouette_score(Z[lab >= 0], lab[lab >= 0]))
            tried.append({"eps": round(eps, 4), "min_samples": k, "clusters": ncl, "noise_share": round(noise, 4), "silhouette": round(sil, 4)})
            if best is None or sil > best[0]:
                best = (sil, eps, k)
    if best is None:  # fall back to the k-distance knee heuristic
        k = min(10, max(2, n // 20))
        return _knee_eps(Z, k), k, tried
    return best[1], best[2], tried


def _name(med: pd.Series, glob: pd.Series) -> str:
    recent, freq, money = med["recency_days"] <= glob["recency_days"], med["frequency"] >= glob["frequency"], med["monetary"] >= glob["monetary"]
    if recent and freq and money:
        return "Loyal high-value"
    if recent and not freq and not money:
        return "Recent / low-spend"
    if recent:
        return "Active mid-value"
    if freq and money:
        return "Lapsed high-value"
    return "Lapsed / low-value"


def train(feats: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    """Returns (artifact, metrics, per-customer frame with segment_id/segment_name/PCA coords)."""
    df = feats[RFM].dropna()
    df = df[df["monetary"] > 0]
    X = _transform(df)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    eps, k, tried = _search(Z)
    db = DBSCAN(eps=eps, min_samples=k).fit(Z)
    lab = db.labels_
    clusters = sorted(set(lab) - {-1})
    core = Z[db.core_sample_indices_]
    core_labels = lab[db.core_sample_indices_]

    glob = df.median()
    total_rev = df["monetary"].sum()
    profiles, names = [], {-1: "Outliers (noise)"}
    for c in clusters:
        sub = df[lab == c]
        names[c] = f"S{c} · {_name(sub.median(), glob)}"
    for c in [*clusters, -1]:
        sub = df[lab == c]
        if sub.empty:
            continue
        med = sub.median()
        profiles.append({
            "segment_id": int(c), "segment_name": names[c], "customers": int(len(sub)), "share": float(len(sub) / len(df)),
            "median_recency_days": float(med["recency_days"]), "median_frequency": float(med["frequency"]),
            "median_monetary": float(med["monetary"]), "mean_monetary": float(sub["monetary"].mean()),
            "revenue_share": float(sub["monetary"].sum() / total_rev),
            "description": (f"{len(sub)} customers ({len(sub)/len(df):.0%}); typical customer last bought {med['recency_days']:.0f} days ago, "
                            f"placed {med['frequency']:.0f} orders and spent {config.CURRENCY}{med['monetary']:,.0f}. "
                            f"Holds {sub['monetary'].sum()/total_rev:.0%} of revenue."
                            + (" DBSCAN marks these as noise: they do not sit in any dense region (often very large or unusual buyers) - "
                               "they are kept visible, not forced into a cluster." if c == -1 else "")),
        })
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE).fit(Z)
    P = pca.transform(Z)
    out = pd.DataFrame({"segment_id": lab, "segment_name": [names[c] for c in lab], "pc1": P[:, 0], "pc2": P[:, 1],
                        "log_recency": X[:, 0], "log_frequency": X[:, 1], "log_monetary": X[:, 2]}, index=df.index)
    m = {"status": "ok", "algorithm": "DBSCAN", "features": "log1p(Recency, Frequency, Monetary), standardised",
         "eps": float(eps), "min_samples": int(k), "n_customers": int(len(df)), "n_clusters": len(clusters),
         "noise_points": int((lab == -1).sum()), "noise_share": float(np.mean(lab == -1)),
         "silhouette_non_noise": float(silhouette_score(Z[lab >= 0], lab[lab >= 0])) if len(clusters) >= 2 else None,
         "davies_bouldin_non_noise": float(davies_bouldin_score(Z[lab >= 0], lab[lab >= 0])) if len(clusters) >= 2 else None,
         "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_],
         "profiles": profiles, "search_grid": tried[:60],
         "note": "eps/min_samples selected by silhouette (non-noise points) subject to 2-8 clusters and <=20% noise; "
                 "falls back to the k-distance knee if no configuration qualifies. Silhouette is unsupervised and indicative only."}
    art = {"scaler": scaler, "core": core, "core_labels": core_labels, "eps": float(eps), "names": names}
    return art, m, out


def assign(art: dict, feats: pd.DataFrame) -> pd.DataFrame:
    """DBSCAN has no predict(): a new customer joins the cluster of the nearest core point if within eps, else noise."""
    X = feats[RFM].astype(float).clip(lower=0).fillna(0).to_numpy()
    Z = art["scaler"].transform(np.log1p(X))
    if len(art["core"]) == 0:
        lab = np.full(len(Z), -1)
    else:
        d, i = NearestNeighbors(n_neighbors=1).fit(art["core"]).kneighbors(Z)
        lab = np.where(d[:, 0] <= art["eps"], art["core_labels"][i[:, 0]], -1)
    return pd.DataFrame({"segment_id": lab, "segment_name": [art["names"][int(c)] for c in lab]}, index=feats.index)
