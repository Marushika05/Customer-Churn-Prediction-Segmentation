# Integrated Customer Churn Prediction and Behavioral Segmentation

University machine learning experiment using **UCI Online Retail II**, both workbook years. The academic workflow is independent of the preserved `customer_intelligence/` legacy dashboard. No API, database, tenant keys or old setup script is needed.

## Quick start (Windows PowerShell)

Run from the directory containing this README and `run_experiment.py`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_dataset.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run_experiment.py
```

The current workspace already has the environment and downloaded workbook. To rerun the finished experiment:

```powershell
.\.venv\Scripts\python.exe run_experiment.py
```

If the Python launcher cannot discover your installation, substitute its full `python.exe` path for `py -3.12`. Activation is unnecessary. Tested interpreter: Python 3.12.10. No GPU is required.

An offline Windows/Python 3.12 wheel set is retained in `wheels/` in this workspace:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-index --find-links wheels -r requirements.txt
```

## Dataset and source

- Authoritative source: https://archive.ics.uci.edu/dataset/502/online+retail+ii
- Archive: https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
- File: `data/raw/online_retail_II.xlsx`; all sheets are read and normalised before combining.
- Expected raw records: 1,067,371, December 2009–December 2011.
- Citation: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D. CC BY 4.0.
- Original one-year demo data and the old application remain under `customer_intelligence/` and are not overwritten.

Input SHA-256, worksheet sizes, cleaning counts, split dates, class prevalence and package versions are written to `results/full/run_manifest.json`.

For development with another source:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --data customer_intelligence/data/raw/online_retail.csv --output results/provisional
```

Custom inputs are labelled provisional. Compatible information is required; this is not a universal retail-dataset importer. CSV is comma-delimited, numeric integer customer IDs and quantities are required, and prices must use a single known currency. `.xls` is explicitly unsupported. Common UCI aliases are accepted. Dates support Excel datetimes, ISO dates/times and month/day/year timestamps; ambiguous day/month CSV formats require conversion first.

## Methodology

1. Clean missing/invalid fields, duplicate transaction signatures and non-product adjustment codes. Preserve valid cancellation lines separately. Anonymous transactions are excluded from customer models. Positive sales drive RFM and the churn target.
2. Ignore the final partial transaction day. Define three cutoffs 90 days apart, with at least 60 days' prior dataset history before training.
3. Features at cutoff T use only purchases/cancellations strictly before T. Churn is **no positive purchase in [T, T+90 days)**. This is a declared inactivity proxy.
4. Training labels mature by the validation cutoff; validation labels mature by the test cutoff. One observation per historical customer per cutoff is retained, without sampling customers or transactions for model fitting.
5. Use the same 17 historical customer features for every baseline: RFM, order values, purchase gaps, recent order/spend activity, product breadth, return rate and single-order indicator. IDs, dates, future spending and labels are never predictors.
6. Fit log transforms, median imputation (preserving entirely empty columns) and scaling on training only. Both classifiers receive the same numeric representation.
7. Fit Ward Agglomerative Clustering and DBSCAN separately on training log1p-RFM with training-fitted scaling. Select parameters using training internal metrics only. Ward evaluates 2–6 clusters; DBSCAN evaluates 12 k-distance configurations and prefers valid 2–8 cluster solutions with at most 35% noise. If none qualify, preserve a safe fallback and report undefined metrics explicitly.
8. Assign held-out customers to the nearest training Ward centroid; this is an inductive approximation, not native hierarchical prediction. DBSCAN uses nearest training core point within eps, otherwise noise (-1). Neither algorithm is refitted on validation or test.
9. One-hot encode cluster membership, including a dedicated noise category. Compare LR and RF with original features, original + Agglomerative membership, and original + DBSCAN membership: **six experiments**.
10. Choose each classification threshold by validation F1. Evaluate Precision, Recall, F1 and probability-based ROC-AUC on identical held-out observations. Report deltas from each classifier's own original-feature baseline. All models use fixed hyperparameters and seed 42.

Customers may recur across dates. This is a prospective time-separated customer-population evaluation, not a customer-disjoint new-customer experiment. New customers arriving after training can be evaluated because customer ID is never a feature.

## Outputs

`results/full/` contains:

- `classification_comparison.csv`: all six test results, validation results, thresholds and baseline deltas.
- `clustering_evaluation.csv`: Silhouette, Davies–Bouldin, noise coverage and assignment policy.
- `segment_profiles.csv`: training and held-out RFM summaries, observed churn and RF risk/priority. Test outcomes are used only for post-evaluation descriptions.
- `held_out_predictions.csv`: shared IDs, targets, segment assignments and six probability columns.
- `train_observations.csv`, `validation_observations.csv`, `test_observations.csv`: exact cohort features and labels for reproducibility.
- `*_parameter_search.csv`: training clustering search history.
- `classification_metrics.png`, `performance_differences.png`, `roc_curves.png`, `customer_segments.png`: exportable figures.
- `models.joblib`: all six frozen evaluated models, feature preprocessing, cluster representations and encoders. Load only trusted model files.
- `run_manifest.json`: data provenance and reproducibility metadata.
- `RESULTS.md`: executed performance and interpretation/limitations.

Silhouette uses a deterministic sample of at most 1,500 non-noise training customers to bound pairwise distance memory. DBI uses all non-noise training customers. Clustering itself, classifier training and held-out evaluation use all eligible observations. Computation is single-worker to limit RAM usage.

## Assumptions and limitations

- Ninety-day inactivity does not establish permanent churn; no performance improvement is assumed.
- Monetary value is gross positive purchase spend in GBP, not net spend after refunds. Returns contribute a separate feature.
- Coverage is assumed complete before the final-day boundary; missing transaction periods cannot be detected reliably.
- Customer eligibility includes old, newly acquired and single-order buyers; no minimum individual tenure is imposed.
- Cluster assignment approximations freeze the training structure and cannot discover new test clusters.
- Cluster labels add a representation of existing RFM, not new underlying information.
- Internal clustering scores were used for parameter selection and are descriptive, not unbiased generalisation estimates.
- A single temporal holdout cannot establish statistical significance, causal retention impact or performance across all seasons/datasets.
- Duplicate removal follows a transaction signature; true repeated identical invoice lines cannot be distinguished without line IDs.
- Dense Ward clustering is quadratic in customer count. This is appropriate for this dataset's customer-level cohort, not transaction-level clustering or arbitrarily large retailers.
- No post-test refit occurs: the saved models are the evaluated models.

## Structure and preservation

`academic/` reuses and corrects the original feature engineering and cleaning logic. `run_experiment.py` is the only experiment entry point. `tests/` uses standard-library unittest. `data/processed/` caches snapshots keyed by input and implementation hashes; `--no-cache` rebuilds them. `results/` contains newly generated output. Original files are preserved under `customer_intelligence/`; `legacy_manifest.json` records their initial SHA-256 values.

The old Gradient Boosting, Ridge/Lasso, Apriori and tenant infrastructure remain isolated in that preserved folder and are never executed or imported by the academic workflow.
