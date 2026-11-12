# Integrated Customer Churn Prediction and Behavioral Segmentation

**An end-to-end Customer Intelligence and Machine Learning platform using UCI Online Retail II, Python, scikit-learn, FastAPI, and Streamlit.**

This project combines customer churn prediction, behavioral segmentation, retail analytics, and product recommendations within an interactive application.

The academic research component investigates whether **unsupervised customer segmentation can improve supervised churn prediction**.

Using approximately **1.07 million retail transactions**, the project evaluates Logistic Regression and Random Forest classifiers with baseline customer features and clustering-enhanced features derived from Agglomerative Hierarchical Clustering and DBSCAN.

The platform also includes an existing multi-tenant customer analytics application powered by FastAPI and Streamlit.

---

## 1. Project Overview

Customer retention is an important challenge for retail businesses. Understanding purchasing behavior, identifying customers at risk of inactivity, and grouping customers with similar characteristics can support more informed business decisions.

Traditional churn prediction models use historical customer information to estimate future purchasing activity.

This project extends that approach by investigating whether customer segments discovered through unsupervised learning provide useful additional information for predicting churn.

The project consists of two complementary components.

### Academic Machine Learning Pipeline

A reproducible research pipeline that:

- Processes the UCI Online Retail II dataset.
- Performs customer-level feature engineering.
- Constructs historical customer observations.
- Defines churn using a 90-day inactivity window.
- Trains Logistic Regression and Random Forest classifiers.
- Performs Agglomerative Hierarchical Clustering and DBSCAN.
- Incorporates clustering-derived features into classification models.
- Compares six experimental configurations.
- Evaluates classification and clustering performance.
- Generates reproducible results, graphs, and saved models.

### Customer Intelligence Application

An interactive application that provides:

- Customer analytics and visualization.
- Customer behavior analysis.
- DBSCAN-based customer segmentation.
- Product recommendations using Apriori association rules.
- Customer spending predictions.
- Model performance visualization.
- Dataset exploration.
- Multi-tenant management.
- A dedicated Research Experiments dashboard displaying academic ML results.

The academic experiment runs independently of the application backend. However, its generated results are integrated into the Streamlit interface through the **Research Experiments** section.

---

## 2. Research Objective

The central research question is:

**Can customer segments generated using Agglomerative Hierarchical Clustering and DBSCAN improve customer churn prediction compared with conventional customer behavior features alone?**

The project compares two supervised learning algorithms:

1. Logistic Regression
2. Random Forest

With three feature configurations:

1. Baseline customer features
2. Baseline features + Agglomerative cluster membership
3. Baseline features + DBSCAN cluster membership

This produces six experiments.

| Experiment | Classification Model | Feature Configuration |
|---|---|---|
| 1 | Logistic Regression | Baseline |
| 2 | Random Forest | Baseline |
| 3 | Logistic Regression | Baseline + Agglomerative |
| 4 | Random Forest | Baseline + Agglomerative |
| 5 | Logistic Regression | Baseline + DBSCAN |
| 6 | Random Forest | Baseline + DBSCAN |

The experiments assess whether adding clustering-derived features improves predictive performance on a common held-out test set.

---

## 3. Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python 3.12 |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn |
| Classification | Logistic Regression, Random Forest |
| Clustering | Agglomerative Hierarchical Clustering, DBSCAN |
| Feature Engineering | RFM and historical purchasing behavior |
| Association Rule Mining | Apriori |
| Backend | FastAPI |
| Frontend | Streamlit |
| Data Visualization | Matplotlib and existing dashboard visualizations |
| Model Serialization | Joblib |
| Testing | Python unittest |
| Version Control | Git and GitHub |

The academic workflow is designed to run on a CPU without requiring a GPU.

---

## 4. Dataset

### UCI Online Retail II

The academic experiments use the **UCI Online Retail II dataset**, containing transaction records from a UK-based online retailer.

**Dataset information**

| Attribute | Description |
|---|---|
| Source | UCI Machine Learning Repository |
| Dataset | Online Retail II |
| Period | December 2009 – December 2011 |
| Raw Records | 1,067,371 |
| Format | Excel workbook |
| Scope | Both workbook years |
| Currency | GBP |

### Dataset Sources

- Official dataset: https://archive.ics.uci.edu/dataset/502/online+retail+ii
- Download archive: https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip

The original workbook is stored locally at:

`data/raw/online_retail_II.xlsx`

Both workbook sheets are loaded, normalized, and combined before preprocessing.

### Dataset Citation

Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository.

DOI: https://doi.org/10.24432/C5CG6D

License: CC BY 4.0.

### Dataset Attributes

The source includes transactional information such as:

- Invoice number
- Product or stock code
- Product description
- Quantity
- Invoice date
- Unit price
- Customer identifier
- Country

These records are transformed into customer-level behavioral features for churn prediction and segmentation.

---

## 5. System Architecture

The system separates the academic ML workflow from the existing customer analytics application.

### Academic Research Workflow

```text
UCI Online Retail II
        |
        v
Data Cleaning and Validation
        |
        v
Customer-Level Feature Engineering
        |
        v
Chronological Train / Validation / Test Splits
        |
        +-------------------------------+
        |                               |
        v                               v
Supervised Classification       Unsupervised Clustering
        |                               |
        |                      +--------+--------+
        |                      |                 |
        |                      v                 v
        |                Agglomerative         DBSCAN
        |                      |                 |
        +----------------------+-----------------+
                               |
                               v
                  Clustering-Enhanced Features
                               |
                               v
                      Six ML Experiments
                               |
                               v
                  Classification Evaluation
                               |
                               v
                       results/full/
                               |
                               v
                 Streamlit Research Experiments
```

### Application Architecture

```text
                 Streamlit Frontend
                         |
          +--------------+---------------+
          |                              |
          v                              v
 Existing Application Pages      Research Experiments
          |                              |
          v                              v
     FastAPI Backend              results/full/
          |                              |
          v                              v
 Tenant Data and Models         Saved CSVs and Figures
```

The Research Experiments page reads saved academic outputs directly.

It does not retrain models, modify experiment results, or require the FastAPI backend.

The other application pages continue using the existing tenant-based infrastructure.

---

## 6. Current Application Features

### 6.1 Dashboard

Provides an overview of customer-related analytics and business information.

### 6.2 Customer Intelligence

Allows exploration of customer purchasing behavior, customer-level metrics, and spending patterns.

### 6.3 Customer Segmentation

Provides DBSCAN-based customer segmentation using RFM-related features.

Customers can be analyzed according to similarities in purchasing behavior.

This page belongs to the original application and is separate from the academic clustering experiments.

### 6.4 Product Recommendations

Uses Apriori association rule mining to identify product relationships and support recommendation analysis.

### 6.5 Model Performance

Displays evaluation information for models used by the existing customer analytics application.

These results should not be confused with the separate academic experimental results.

### 6.6 Data Explorer

Provides interactive exploration of available customer and transaction data.

### 6.7 Tenant Management

Supports multiple tenants with tenant-specific customer information and analytics.

The existing FastAPI backend manages the tenant-based application functionality.

### 6.8 Research Experiments

The Research Experiments section was added to connect the corrected academic research pipeline with the existing dashboard.

It provides:

- Dataset overview and research objective.
- Comparison of all six academic experiments.
- Precision, Recall, F1-score, and ROC-AUC.
- Silhouette Score and Davies–Bouldin Index.
- Classification performance comparisons.
- ROC curves.
- Customer segmentation visualizations.
- Research findings and limitations.

The section reads directly from `results/full/`.

It works without an active tenant and does not require an active FastAPI connection.

---

## 7. Machine Learning Methodology

### 7.1 Data Cleaning

The academic pipeline performs preprocessing to improve data consistency and avoid invalid observations.

Major operations include:

- Handling missing and invalid values.
- Removing duplicate transaction signatures.
- Filtering non-product adjustment records.
- Separating valid cancellation transactions.
- Excluding anonymous transactions from customer modeling.
- Using positive purchases for customer spending features and churn labeling.

The final partial transaction day is excluded to avoid treating incomplete observation coverage as a complete day.

### 7.2 Customer Feature Engineering

The project uses 17 historical customer features.

These include:

**Recency:** Time elapsed since the customer's most recent purchase.

**Frequency:** Customer purchasing activity.

**Monetary Value:** Historical purchasing expenditure.

Additional features capture:

- Order values.
- Purchase intervals.
- Recent purchasing activity.
- Recent spending.
- Product diversity.
- Return behavior.
- Single-order customer status.

Customer identifiers, future purchasing information, and target labels are excluded from predictors.

### 7.3 Churn Definition

Churn is defined as **no positive purchase within the next 90 days**.

For a historical cutoff date T:

```text
Churn = 1:
No positive purchase in [T, T + 90 days)

Churn = 0:
At least one positive purchase in [T, T + 90 days)
```

This is an operational inactivity proxy rather than proof that a customer has permanently stopped purchasing.

### 7.4 Chronological Data Splitting

The project uses time-based separation rather than randomly splitting customer observations.

Three historical cutoffs are constructed 90 days apart.

Training, validation, and test observations are separated chronologically.

Training labels are allowed to mature before validation, and validation labels mature before testing.

This design reduces future-information leakage.

A customer may appear at multiple historical cutoffs, so the evaluation measures prospective performance on a time-separated customer population rather than performance on entirely unseen customer identities.

### 7.5 Feature Preprocessing

The pipeline applies:

- Logarithmic transformations.
- Median imputation.
- Feature scaling.
- Consistent numeric representations.

Preprocessing is fitted using training data only.

The same baseline feature representation is used by both classifiers.

### 7.6 Logistic Regression

Logistic Regression provides a supervised classification baseline.

It estimates the probability that a customer will become inactive during the defined prediction window.

### 7.7 Random Forest

Random Forest combines multiple decision trees to capture nonlinear relationships between customer behavior and churn.

It is evaluated using the same observations and baseline features as Logistic Regression.

### 7.8 Agglomerative Hierarchical Clustering

Ward Agglomerative Clustering groups customers according to their RFM characteristics.

The academic pipeline:

- Uses log-transformed RFM features.
- Fits scaling on training data.
- Evaluates 2–6 clusters.
- Selects clustering parameters using training-only internal metrics.

For validation and test customers, cluster membership is approximated using the nearest training cluster centroid.

The hierarchical model is not refitted on held-out data.

### 7.9 DBSCAN

DBSCAN identifies density-based customer groups and noise observations.

The pipeline:

- Uses training RFM features.
- Evaluates multiple neighborhood configurations.
- Considers cluster quality and noise coverage.
- Assigns held-out customers using nearby training core points.
- Labels customers without a qualifying core point as noise.

DBSCAN is not refitted on validation or test data.

### 7.10 Clustering-Enhanced Classification

Cluster memberships are converted into encoded features and combined with the original customer features.

The project then trains and evaluates:

- Logistic Regression + Agglomerative features.
- Random Forest + Agglomerative features.
- Logistic Regression + DBSCAN features.
- Random Forest + DBSCAN features.

These are compared with the original baseline models.

### 7.11 Model Evaluation

Classification performance is evaluated using:

- Precision
- Recall
- F1-score
- ROC-AUC

Classification thresholds are selected using validation F1-score.

Final metrics are calculated on held-out test observations.

Clustering performance is evaluated using:

- Silhouette Score
- Davies–Bouldin Index
- Cluster counts
- Noise coverage where applicable

Clustering parameter selection is performed using training data only.

---

## 8. Experimental Results

The academic pipeline evaluates six configurations on the same held-out test observations.

### Classification Performance

| Model | Features | Precision | Recall | F1-score | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Logistic Regression | Baseline | 0.6552 | 0.9582 | 0.7783 | 0.7922 |
| Random Forest | Baseline | 0.6605 | 0.9562 | 0.7813 | 0.7915 |
| Logistic Regression | Agglomerative | 0.6560 | 0.9572 | 0.7785 | 0.7921 |
| Random Forest | Agglomerative | 0.6739 | 0.9353 | **0.7834** | 0.7919 |
| Logistic Regression | DBSCAN | 0.6542 | 0.9596 | 0.7780 | **0.7925** |
| Random Forest | DBSCAN | 0.6608 | 0.9586 | 0.7823 | 0.7924 |

### Key Findings

**Best F1-score**

Random Forest with Agglomerative clustering features achieved an F1-score of approximately **0.7834**.

**Best ROC-AUC**

Logistic Regression with DBSCAN features achieved a ROC-AUC of approximately **0.7925**.

**Effect of Clustering**

Clustering-derived features produced small improvements in some configurations.

For example, Random Forest improved from approximately 0.7813 baseline F1-score to 0.7834 with Agglomerative features.

However, improvements were not consistent across every model and metric.

### Research Conclusion

The experiments suggest that behavioral clustering can provide a modest improvement in churn prediction under certain configurations.

However, clustering does not consistently outperform baseline customer features.

Because clustering is derived from existing RFM information, it provides an alternative representation of customer behavior rather than entirely new underlying information.

The observed improvements should not be interpreted as statistically significant without additional testing.

The results are specific to the dataset, experimental design, and selected evaluation period.

---

## 9. Research Visualizations

The academic pipeline generates four primary visualizations.

| Visualization | File |
|---|---|
| Classification metrics | `classification_metrics.png` |
| Performance differences | `performance_differences.png` |
| ROC curves | `roc_curves.png` |
| Customer segments | `customer_segments.png` |

These files are generated under:

`results/full/`

They are displayed in the Streamlit Research Experiments section.

Additional clustering and classification information is available through the generated CSV files.

---

## 10. Installation and Setup

### Requirements

- Windows 10/11 or a compatible Python environment.
- Python 3.12.
- pip.
- Git (optional for local execution).
- Sufficient memory and storage for the dataset and generated outputs.

Tested Python version: **3.12.10**.

A GPU is not required.

### Step 1: Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd customer_intelligence
```

Replace `<YOUR_REPOSITORY_URL>` with the actual GitHub repository URL.

### Step 2: Create a Virtual Environment

Run from the outer project directory in Windows PowerShell:

```powershell
py -3.12 -m venv .venv
```

### Step 3: Install Dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Step 4: Download the Dataset

```powershell
.\.venv\Scripts\python.exe download_dataset.py
```

### Step 5: Run the Academic Pipeline

```powershell
.\.venv\Scripts\python.exe run_experiment.py
```

The experiment loads the dataset, processes historical observations, evaluates the six configurations, and writes results to:

`results/full/`

### Step 6: Run Automated Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The test suite checks the academic pipeline's implementation and expected behavior.

### Rerunning Existing Experiments

If the environment and dataset are already available:

```powershell
.\.venv\Scripts\python.exe run_experiment.py
```

### Offline Dependency Installation

If a compatible offline wheel directory is available locally:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-index --find-links wheels -r requirements.txt
```

The `wheels/` directory may be a local development resource and is not guaranteed to be included in the GitHub repository.

---

## 11. Running the Streamlit Application

The existing application is located inside:

`customer_intelligence/`

It maintains a separate Python environment and application setup.

### Step 1: Open the Application Directory

From the outer project directory:

```cmd
cd customer_intelligence
```

### Step 2: Prepare the Application Environment

Follow the existing application setup instructions or run its provided setup script:

```cmd
setup.bat
```

This is separate from the academic environment.

### Step 3: Start FastAPI

Open a terminal in the nested application directory:

```cmd
start_api.bat
```

The API is served at:

http://127.0.0.1:8000

Interactive API documentation is available at:

http://127.0.0.1:8000/docs

### Step 4: Start Streamlit

Open a second terminal in the same directory:

```cmd
start_ui.bat
```

The dashboard is available at:

http://localhost:8501

### Step 5: Explore the Application

The sidebar provides access to:

- Dashboard
- Customer Intelligence
- Segmentation
- Recommendations
- Model Performance
- Data Explorer
- Tenant Management
- Research Experiments

The original tenant-based pages use the FastAPI backend.

**Research Experiments can display saved academic results independently of FastAPI and tenant selection**, provided the required result files are available.

The application environment may require compatible versions of native dependencies, including PyArrow, depending on the operating system.

---

## 12. Running with a Custom Dataset

The academic pipeline also supports compatible custom input files.

Example:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --data customer_intelligence/data/raw/online_retail.csv --output results/provisional
```

Custom runs are labelled provisional.

Input requirements include:

- Compatible transaction columns.
- Numeric customer identifiers.
- Valid quantities.
- Consistent currency.
- Supported date formats.

The pipeline is designed around Online Retail II-style transaction data and is not a universal retail dataset importer.

`.xls` files are not supported by the academic input workflow.

---

## 13. Generated Outputs

The academic pipeline stores its results under:

`results/full/`

### Classification Outputs

- `classification_comparison.csv` — Six experiment results, validation information, thresholds, and baseline deltas.
- `held_out_predictions.csv` — Held-out targets, cluster assignments, and model probabilities.

### Clustering Outputs

- `clustering_evaluation.csv` — Clustering quality and assignment information.
- `segment_profiles.csv` — Segment-level customer characteristics.
- `agglomerative_parameter_search.csv` — Agglomerative parameter search history.
- `dbscan_parameter_search.csv` — DBSCAN parameter search history.

### Reproducibility Outputs

- `train_observations.csv`
- `validation_observations.csv`
- `test_observations.csv`
- `run_manifest.json`

### Models and Reports

- `models.joblib`
- `RESULTS.md`

### Figures

- `classification_metrics.png`
- `performance_differences.png`
- `roc_curves.png`
- `customer_segments.png`

The saved model artifact contains the evaluated model components and preprocessing information.

Only trusted Joblib files should be loaded.

---

## 14. Repository Organization

The repository contains the academic research workflow alongside the preserved application.

```text
customer_intelligence/
|
|-- academic/
|   |-- Academic ML pipeline modules
|
|-- customer_intelligence/
|   |-- backend/
|   |-- frontend/
|   |   |-- app.py
|   |   |-- research_experiments.py
|   |-- Existing application files
|
|-- data/
|   |-- raw/
|   |-- processed/
|
|-- results/
|   |-- full/
|
|-- tests/
|
|-- run_experiment.py
|-- download_dataset.py
|-- requirements.txt
|-- README.md
```

The outer directory contains the academic experiment entry point and shared project documentation.

The nested `customer_intelligence/` directory contains the existing FastAPI and Streamlit application.

The academic and application components use separate execution workflows.

---

## 15. Development and Improvements

The project evolved from an initial customer analytics application into a more rigorous academic ML research platform.

### Initial Implementation

The original application provided:

- FastAPI services.
- Streamlit dashboard.
- Tenant management.
- Customer analytics.
- DBSCAN segmentation.
- Apriori recommendations.
- Customer spending predictions.
- Model performance visualization.

### Academic Pipeline Improvements

The academic workflow was developed to address the approved research objectives.

Major improvements included:

1. Using both years of the UCI Online Retail II dataset.
2. Implementing the required Logistic Regression and Random Forest classifiers.
3. Adding Agglomerative Hierarchical Clustering alongside DBSCAN.
4. Defining an explicit 90-day churn target.
5. Introducing chronological training, validation, and testing.
6. Preventing future-information leakage during feature engineering.
7. Fitting preprocessing and clustering using training data only.
8. Creating held-out cluster assignment policies.
9. Evaluating six classification configurations.
10. Generating reproducible experiment outputs and visualizations.
11. Adding automated tests and execution metadata.

### Dashboard Integration

A dedicated `research_experiments.py` module was added to the existing Streamlit frontend.

The integration:

- Adds Research Experiments to sidebar navigation.
- Reads saved academic outputs directly.
- Displays six model comparisons.
- Shows clustering evaluation metrics.
- Loads four saved visualizations.
- Presents evidence-based conclusions.
- Works without tenant selection or an active FastAPI connection.

The integration was designed to preserve existing application functionality without retraining or modifying the academic models.

---

## 16. Testing and Reproducibility

The academic pipeline includes automated tests using Python's standard `unittest` framework.

The experiments use a fixed random seed of **42** and controlled preprocessing and evaluation procedures.

Reproducibility information is recorded in:

`results/full/run_manifest.json`

This includes available information about:

- Dataset provenance.
- Input file hashes.
- Cleaning counts.
- Split dates.
- Class prevalence.
- Software package versions.
- Execution configuration.

Processed observations can be cached using input- and implementation-specific hashes.

The `--no-cache` option can be used to rebuild processed snapshots.

### Validation Performed

The development workflow has included:

- Successful execution of all six academic experiments.
- Passing automated academic tests.
- Successful FastAPI startup.
- Successful Streamlit startup.
- Resolution of environment dependency issues.
- Automated validation of the Research Experiments page.
- Validation of saved figures and result loading.
- Checks for missing-results handling.
- Checks for alternate working directories.
- Mocked rendering checks for existing dashboard pages.

The new Research Experiments integration was validated automatically, while a final live end-to-end regression test of all original application pages after integration should be performed separately.

---

## 17. Assumptions and Limitations

### Churn Definition

Ninety-day inactivity is used as a proxy for churn.

It does not establish permanent customer loss.

### Monetary Values

Monetary features represent gross positive purchase spending in GBP.

Returns are represented separately rather than directly subtracted from all monetary features.

### Dataset Coverage

The analysis assumes historical transaction coverage is sufficiently complete.

Missing transaction periods cannot always be identified reliably.

### Customer Eligibility

The pipeline includes eligible historical customers, including new and single-order buyers.

### Cluster Assignment

Agglomerative clustering does not provide native out-of-sample prediction.

Held-out assignments therefore use nearest training centroids.

DBSCAN assigns held-out customers using training core points and the selected neighborhood radius.

These methods preserve the training structure but cannot discover entirely new clusters during evaluation.

### Information Content

Clustering features are derived from existing RFM information.

They represent customer behavior differently rather than introducing independent new information.

### Clustering Evaluation

Silhouette scores use a deterministic sample of at most 1,500 non-noise training customers to control memory usage.

Davies–Bouldin scores use the available non-noise training observations.

Internal clustering scores are descriptive and are also used for parameter selection; they should not be interpreted as unbiased generalization estimates.

### Statistical Significance

The project uses a single chronological held-out evaluation.

Small observed performance improvements do not establish statistical significance.

### Generalization

Results are specific to the Online Retail II dataset and selected experimental design.

Performance may differ across retailers, seasons, and customer populations.

### Computational Complexity

Ward Agglomerative Clustering has quadratic computational and memory considerations with respect to customer count.

The implementation is suitable for the customer-level cohort used in this project but is not intended for unrestricted large-scale clustering.

### Deployment

The project has been validated in a local development environment.

A publicly hosted deployment is not currently documented.

---

## 18. Future Improvements

Potential extensions include:

- Deploying the application for public demonstration.
- Adding automated CI checks.
- Performing repeated temporal validation.
- Evaluating uncertainty around model performance differences.
- Introducing explainability techniques such as SHAP.
- Improving customer-level risk interpretation.
- Supporting larger datasets through scalable clustering methods.
- Expanding tenant-level reporting and visualization.
- Adding production monitoring and model drift detection.

These are future enhancements and are not claimed as completed functionality.

---

## 19. Conclusion

This project demonstrates how supervised and unsupervised machine learning can be combined to study customer churn and purchasing behavior.

Using approximately 1.07 million retail transactions, the academic pipeline evaluates Logistic Regression and Random Forest with and without clustering-derived features.

The results show that customer segmentation can provide modest improvements in certain configurations, although the gains are not consistent across every model.

Beyond the experiments, the project includes a working FastAPI and Streamlit customer analytics application with segmentation, recommendations, tenant management, and research visualization.

The resulting system combines **machine learning experimentation, reproducible evaluation, backend development, and interactive data visualization**.

---

## 20. References

1. Chen, D. (2012). *Online Retail II*. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
2. scikit-learn Documentation — https://scikit-learn.org/stable/
3. FastAPI Documentation — https://fastapi.tiangolo.com/
4. Streamlit Documentation — https://docs.streamlit.io/
5. pandas Documentation — https://pandas.pydata.org/docs/

---

**Project Type:** University Machine Learning Project / Customer Intelligence Application

**Primary Focus:** Customer Churn Prediction, Behavioral Segmentation, and Comparative Machine Learning Evaluation

**Core Technologies:** Python, scikit-learn, FastAPI, Streamlit, pandas, NumPy
