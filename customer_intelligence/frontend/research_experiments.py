"""Read-only academic results page; independent of tenants, APIs and training."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "full"
METRICS = ["precision", "recall", "f1", "roc_auc"]
LABELS = {"baseline": "Original features", "agglomerative": "+ Agglomerative", "dbscan": "+ DBSCAN"}
DISPLAY = {"model": "Classifier", "features": "Feature set", "precision": "Precision",
           "recall": "Recall", "f1": "F1-score", "roc_auc": "ROC-AUC"}


def load_results(directory=RESULTS_DIR):
    """Read current outputs each rerun without fitting or loading any models."""
    directory = Path(directory)
    table = pd.read_csv(directory / "classification_comparison.csv")
    clusters = pd.read_csv(directory / "clustering_evaluation.csv")
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    if not {"model", "features", *METRICS}.issubset(table.columns):
        raise ValueError("Classification results are missing required columns.")
    if not {"algorithm", "silhouette", "davies_bouldin"}.issubset(clusters.columns):
        raise ValueError("Clustering results are missing required columns.")
    expected = {(m, f) for m in ("Logistic Regression", "Random Forest") for f in LABELS}
    if len(table) != 6 or set(zip(table.model, table.features)) != expected:
        raise ValueError("Expected exactly six classifier/feature combinations.")
    for col in METRICS:
        table[col] = pd.to_numeric(table[col], errors="raise")
        if table[col].isna().any() or not table[col].between(0, 1).all():
            raise ValueError(f"Invalid or missing {col} scores.")
    if "test_observations_sha256" in table:
        if table.test_observations_sha256.isna().any() or table.test_observations_sha256.nunique() != 1:
            raise ValueError("Experiments do not identify an identical held-out cohort.")
    baseline = table[table.features == "baseline"].set_index("model")
    for metric in METRICS:
        table["delta_" + metric] = table[metric] - table.model.map(baseline[metric])
    return table, clusters, manifest


def render():
    st.title("Research Experiments")
    st.caption("Integrated Customer Churn Prediction and Behavioral Segmentation · saved experimental evidence")
    st.write("Research objective: determine whether clustering-derived RFM segment membership improves churn "
             "prediction compared with the same historical customer features alone.")
    try:
        table, clusters, manifest = load_results()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Research results could not be loaded: {exc}")
        st.info(f"Expected saved files in: {RESULTS_DIR}. This page never retrains models.")
        return

    st.subheader("Dataset and evaluation")
    st.write(manifest.get("dataset", "Dataset identity unavailable"))
    if manifest.get("provisional", True):
        st.warning("Provisional/custom input: do not present these as complete Online Retail II results.")
    split = manifest.get("split", {})
    rows = split.get("rows", {})
    cards = st.columns(4)
    cards[0].metric("Raw transaction rows", f"{manifest.get('raw_rows', 0):,}")
    for card, label, key in zip(cards[1:], ("Training customers", "Validation customers", "Test customers"),
                               ("train", "validation", "test")):
        card.metric(label, f"{rows.get(key, 0):,}")
    st.caption("Churn = no positive purchase in the next 90 days. Historical features exclude transactions at/after the cutoff. "
               "Preprocessing and clusters are fitted on training only; thresholds are selected on validation.")
    with st.expander("Temporal protocol and provenance"):
        st.dataframe(pd.DataFrame([
            {"Cohort": name.title(), "Prediction cutoff": split.get(name + "_cutoff", "Unavailable"),
             "Outcomes complete by": split.get(name + "_label_end", "Unavailable")}
            for name in ("train", "validation", "test")]), hide_index=True)
        st.write("Training outcomes mature before validation; validation outcomes mature before test predictions.")
        if manifest.get("sheets"):
            st.dataframe(pd.DataFrame(list(manifest["sheets"].items()), columns=["Worksheet", "Raw rows"]), hide_index=True)
        st.caption("Workbook SHA-256: " + manifest.get("sha256", "Unavailable"))
        st.caption("Customers may recur across dates; customer ID is never a predictor. Test results describe one temporal cohort.")

    st.subheader("Six-way classification comparison")
    view = table[["model", "features", *METRICS]].copy()
    view["features"] = view.features.map(LABELS)
    st.dataframe(view.rename(columns=DISPLAY).style.format({DISPLAY[m]: "{:.4f}" for m in METRICS}), hide_index=True)
    st.caption("All experiments use the same held-out observations and labels. ROC-AUC uses probabilities. "
               "Each threshold is tuned on the same temporally valid validation cohort.")
    metric = st.selectbox("Compare a metric", METRICS, format_func=lambda x: DISPLAY[x], key="research_metric")
    st.bar_chart(table.pivot(index="model", columns="features", values=metric)[list(LABELS)].rename(columns=LABELS))

    st.subheader("Changes from each classifier's baseline")
    integrated = table[table.features != "baseline"].copy()
    changes = integrated[["model", "features", *["delta_" + m for m in METRICS]]].copy()
    changes["features"] = changes.features.map(LABELS)
    changes = changes.rename(columns={**DISPLAY, **{"delta_" + m: "Δ " + DISPLAY[m] for m in METRICS}})
    st.dataframe(changes.style.format({"Δ " + DISPLAY[m]: "{:+.6f}" for m in METRICS}), hide_index=True)
    st.caption("Changes are absolute score differences, not relative percentages.")

    st.subheader("Clustering quality")
    cols = [c for c in ("algorithm", "n_clusters", "silhouette", "davies_bouldin", "noise_share", "silhouette_n") if c in clusters]
    cluster_view = clusters[cols].rename(columns={"algorithm": "Algorithm", "n_clusters": "Clusters",
        "silhouette": "Silhouette Score", "davies_bouldin": "Davies–Bouldin Index",
        "noise_share": "Noise share", "silhouette_n": "Silhouette sample"})
    formats = {"Silhouette Score": "{:.4f}", "Davies–Bouldin Index": "{:.4f}"}
    if "noise_share" in clusters:
        formats["Noise share"] = "{:.2%}"
    st.dataframe(cluster_view.style.format(formats, na_rep="Undefined"), hide_index=True)
    st.caption("Higher silhouette and lower Davies–Bouldin indicate better internal separation. Metrics exclude DBSCAN noise. "
               "Silhouette uses up to 1,500 training customers; DBI uses all non-noise training customers. "
               "Model fitting and test evaluation retain all eligible observations.")
    with st.expander("Assignment of held-out customers"):
        st.write("Agglomerative: nearest training centroid in scaled log-RFM, an approximation to Ward clustering. "
                 "DBSCAN: nearest training core within fitted eps, otherwise noise (−1). "
                 "Neither algorithm is refitted on validation or test customers.")

    st.subheader("Experimental conclusions")
    best_auc = table.loc[table.roc_auc.idxmax()]
    best_f1 = table.loc[table.f1.idxmax()]
    cards = st.columns(2)
    cards[0].metric("Highest observed test ROC-AUC", f"{best_auc.roc_auc:.4f}")
    cards[0].caption(f"{best_auc.model} · {LABELS[best_auc.features]}")
    cards[1].metric("Highest observed test F1", f"{best_f1.f1:.4f}")
    cards[1].caption(f"{best_f1.model} · {LABELS[best_f1.features]}")
    for row in integrated.itertuples():
        st.write(f"• {row.model} {LABELS[row.features]}: ROC-AUC change {row.delta_roc_auc:+.6f}; F1 change {row.delta_f1:+.6f}.")
    auc_changes = integrated.delta_roc_auc
    if (auc_changes > 0).all():
        st.info("All integrated variants increased observed ROC-AUC on this cohort; this alone does not establish significance or generalisation.")
    elif (auc_changes <= 0).all():
        st.info("Clustering did not improve observed ROC-AUC over the corresponding baselines on this cohort.")
    else:
        st.info("ROC-AUC changes are mixed across variants. These results do not support a consistent improvement from clustering.")
    st.caption("Best observed test scores are descriptive, not a model-selection rule. No significance test or causal retention benefit "
               "is established. Ninety-day inactivity is a churn proxy; segments derive from RFM already available to the baseline.")

    st.subheader("Research figures")
    for filename, caption in [("classification_metrics.png", "Classification metrics"),
                              ("performance_differences.png", "Integrated minus baseline performance"),
                              ("roc_curves.png", "ROC curves on the shared held-out cohort"),
                              ("customer_segments.png", "Training customer segments")]:
        path = RESULTS_DIR / filename
        if path.is_file():
            st.image(str(path), caption=caption)
        else:
            st.warning(f"Figure unavailable: {filename}")
