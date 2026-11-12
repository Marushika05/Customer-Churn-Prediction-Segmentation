# Multi-Tenant Customer Intelligence & Risk Microservice

Churn prediction, DBSCAN segmentation, Apriori recommendations and Ridge/Lasso future-spend estimation on the
UCI Online Retail dataset, behind a FastAPI backend with a Streamlit dashboard. Each tenant (business) has isolated data, models and API key.

## Windows setup (Python 3.11+)
1. Unzip to a normal folder (e.g. `C:\cip`), double-click `setup.bat` (venv, dependencies, dataset check, 2 demo tenants trained, ~2 min). Full output is saved to `setup.log`.
2. `start_api.bat` (http://127.0.0.1:8000/docs) and, in a second window, `start_ui.bat` (http://localhost:8501).
3. Optional check with the API running: `.venv\Scripts\python -m scripts.smoke_test`

Re-train from scratch: delete `storage\` and run `python -m scripts.seed_demo`. Config: copy `.env.example` to `.env`.

## Design notes
- **Churn**: label = no purchase in the 90 days after a snapshot date (the data has no real churn flag). Monthly snapshots; features use only data before the snapshot. Train < validation < test by date, with training label windows purged so they end before the validation date. Compared: logistic regression, random forest, gradient boosting vs prior baseline; model chosen on validation ROC-AUC, reported on the test snapshot, then refit on all labelled snapshots for scoring. Per-customer risk drivers use occlusion against the typical customer.
- **Segmentation**: log1p + standardised RFM, DBSCAN with eps/min_samples picked by silhouette under constraints. Noise points stay visible as "Outliers". New customers join the nearest core point's cluster within eps, otherwise noise.
- **Recommendations**: Apriori on valid invoices (cancellations and non-product codes removed), top 250 items by frequency, rules filtered by confidence and lift; chronological holdout check reported.
- **Future spend**: Ridge and Lasso on log(1+spend in the next 90 days), alpha chosen on the later validation snapshot. This is predicted 90-day spend, **not** true lifetime value.
- **Tenancy**: every tenant route needs `X-API-Key` matching the tenant in the URL; all SQL goes through a guard that requires `tenant_id`; artifacts live in `storage/artifacts/<tenant_id>/`. Admin actions use `X-Admin-Token` (`CIP_ADMIN_TOKEN`). Keys are stored hashed; the UI keeps them in `storage/tenant_keys.json` on your machine.

## Limitations
Demo tenants are customer-disjoint halves of one dataset (not independent businesses). Single-process prototype: training runs in a background thread, SQLite is the only store, and the admin token is a shared secret - not production auth.
