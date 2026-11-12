# Executed experiment results

UCI Online Retail II (complete workbook)

| model | features | precision | recall | f1 | roc_auc | delta_f1 | delta_roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | baseline | 0.6552 | 0.9582 | 0.7783 | 0.7922 | 0.0000 | 0.0000 |
| Random Forest | baseline | 0.6605 | 0.9562 | 0.7813 | 0.7915 | 0.0000 | 0.0000 |
| Logistic Regression | agglomerative | 0.6560 | 0.9572 | 0.7785 | 0.7921 | 0.0002 | -0.0001 |
| Random Forest | agglomerative | 0.6739 | 0.9353 | 0.7834 | 0.7919 | 0.0020 | 0.0004 |
| Logistic Regression | dbscan | 0.6542 | 0.9596 | 0.7780 | 0.7925 | -0.0003 | 0.0003 |
| Random Forest | dbscan | 0.6608 | 0.9586 | 0.7823 | 0.7924 | 0.0010 | 0.0009 |

## Interpretation

Differences are descriptive results on one held-out cohort, not statistical proof or guaranteed improvement.
The threshold for each experiment maximises validation F1. ROC-AUC uses probability scores.
Clusters are fitted on training RFM only. Test cluster labels are frozen inductive approximations.

## Chronology

```json
{
  "train_cutoff": "2011-03-14 00:00:00",
  "validation_cutoff": "2011-06-12 00:00:00",
  "test_cutoff": "2011-09-10 00:00:00",
  "horizon_days": 90,
  "train_label_end": "2011-06-12 00:00:00",
  "validation_label_end": "2011-09-10 00:00:00",
  "test_label_end": "2011-12-09 00:00:00",
  "rows": {
    "train": 4583,
    "validation": 4951,
    "test": 5256
  }
}
```

## Clustering metrics

```json
[
  {
    "n_clusters": 2,
    "noise_share": 0.0,
    "silhouette": 0.4401049052876322,
    "davies_bouldin": 0.8582994135768567,
    "metric_population": "non-noise training observations",
    "silhouette_n": 1500,
    "algorithm": "agglomerative",
    "status": "valid",
    "assignment": "nearest training centroid in scaled log-RFM; Ward approximation"
  },
  {
    "n_clusters": 2,
    "noise_share": 0.039930176740126555,
    "silhouette": 0.2975061256797653,
    "davies_bouldin": 1.0536535119824437,
    "metric_population": "non-noise training observations",
    "silhouette_n": 1500,
    "algorithm": "dbscan",
    "status": "valid",
    "assignment": "nearest training core within fitted eps; otherwise noise (-1)",
    "eps": 0.4151986557015725,
    "min_samples": 10
  }
]
```

Silhouette uses at most 1,500 training non-noise points with a fixed seed; DBI uses all training non-noise points.
All eligible customer observations are retained for clustering fitting, classifiers and test evaluation.

## Limitations

- Churn means no positive purchase in 90 days, not proven permanent customer departure.
- The 90-day horizon is a declared study assumption, not an optimised business threshold.
- Gross monetary spend excludes refunds; cancellations contribute a separate historical return feature.
- All historically observed buyers are eligible, including new and already inactive customers.
- Customers may recur across dates; this estimates later behaviour of a retailer customer population, not customer-disjoint transfer.
- Segmentation cannot add new raw information beyond RFM; gains reflect representation and model effects.
- Cluster selection uses training-only internal metrics; these scores are descriptive and selection-biased.
- Nearest-centroid Ward assignment does not reproduce refitted hierarchical clustering; nearest-core DBSCAN cannot create new clusters.
- One temporal holdout and seasonal changes limit generalisation; no causal retention effectiveness or significance is established.
- Final partial transaction day is excluded; complete recorded coverage before that boundary is assumed.
- No dashboard, API or unrelated models are required by this workflow.
