# Execution and verification report

Executed on 12 November 2026 with Python 3.12.10 on Windows.

## Dataset and experiment

- Downloaded UCI dataset 502 from the authoritative UCI archive.
- Loaded both sheets: `Year 2009-2010` and `Year 2010-2011`.
- Verified 1,067,371 original rows; 1,021,252 transaction lines remain after the declared cleaning policy.
- Workbook SHA-256: `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`.
- All eligible customer observations used: 4,583 train, 4,951 validation, 5,256 test.
- Training cutoff 14 March 2011; validation cutoff 12 June 2011; test cutoff 10 September 2011.
- Each 90-day label window ends by the next prediction cutoff. Test outcomes end at 9 December 2011 00:00; the final partial day is excluded.
- All six required model/feature combinations executed successfully. No original application/database operations were run.

## Checks performed

`python -m unittest discover -s tests -v` — **20 tests passed**.

The suite verifies hand-calculated RFM and cancellation rate, cutoff exclusion, future-transaction feature invariance, half-open label windows, invalid/anonymous/duplicate cleaning, mixed supported date strings, multi-sheet loading with different aliases, temporal purging, insufficient history, all-missing feature dimensions, one-cluster and all-noise clustering, unseen-point noise assignment, noise encoding, score-based AUC, full-dataset identity, shared held-out IDs and labels, saved-model probability reproduction, recomputed metrics and validation thresholds, training-only imputation/scaling, and preservation of every original file against its SHA-256 fingerprint.

`python -m pip check` — **No broken requirements found**.

The complete six-model experiment was executed more than once. Classification tables, clustering tables, held-out predictions and segment profiles were byte-for-byte identical between the first full execution and the independent cached-data rerun. Input parsing and feature engineering were then rerun from the original workbook after a schema-normalisation correction; final outputs were checked against the earlier execution.

The first full-data run completed in approximately 66 seconds; a rerun using the snapshot cache completed in approximately 12 seconds in this environment. These are observed timings, not a hardware-independent performance guarantee. No peak-memory benchmark was performed.

## Result interpretation

Results are in `results/full/RESULTS.md` and the CSV tables. The observed clustering gains are very small and mixed across algorithms/metrics. They do not establish a substantial or statistically significant improvement. Clustering quality scores exclude DBSCAN noise, with silhouette calculated on a deterministic sample of at most 1,500 training customers. Fitting and held-out classification evaluation are not sampled.

## Preserved scope

Original files, data, saved models and SQLite database remain unchanged in `customer_intelligence/`. The academic workflow contains only the two required classifiers, two required clustering methods and their evaluation. The old dashboard and unrelated modules are isolated rather than deleted.

No core functionality remains unimplemented. A new dashboard, deployment, statistical significance study and business validation of the inactivity horizon are outside this completed experiment.
