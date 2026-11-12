"""Streamlit dashboard:  streamlit run frontend/app.py"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend import config
from frontend import api_client as api

CUR = config.CURRENCY
RISK_COLORS = {"Low": "#2e9e5b", "Medium": "#e0a31b", "High": "#d64545"}
PAGES = ["Dashboard", "Customer Intelligence", "Segmentation", "Recommendations", "Model Performance",
         "Data Explorer", "Tenant Management"]

st.set_page_config(page_title="Customer Intelligence", page_icon="📊", layout="wide")


# ------------------------------------------------------------------ helpers
def money(v, d=0):
    return "-" if v is None or pd.isna(v) else f"{CUR}{v:,.{d}f}"


def pct(v, d=1):
    return "-" if v is None or pd.isna(v) else f"{v:.{d}%}"


def num(v, d=3):
    return "-" if v is None or pd.isna(v) else f"{v:.{d}f}"


@st.cache_data(ttl=30, show_spinner=False)
def tget(tid: str, path: str, params: tuple = ()):
    return api.tenant(tid, "GET", path, params=dict(params))


def get(path: str, **params):
    return tget(st.session_state.tid, path, tuple(sorted((k, v) for k, v in params.items() if v is not None)))


def guard(fn):
    """Run a page; show backend errors as friendly messages instead of stack traces."""
    try:
        fn()
    except api.ApiError as e:
        st.error(str(e))


def tenants_list():
    try:
        return api.admin("GET", "/tenants")
    except api.ApiError as e:
        st.sidebar.error(str(e))
        return []


def need_trained() -> bool:
    s = tget(st.session_state.tid, "/status")
    if not s["has_data"]:
        st.info("This tenant has no data yet. Go to **Tenant Management** to load a dataset.")
        return False
    if not s["trained"]:
        st.info("Models are not trained for this tenant yet. Go to **Tenant Management** and click **Train models**.")
        return False
    return True


def unavailable(m: dict, name: str) -> bool:
    if m.get("status") != "ok":
        st.warning(f"{name} is unavailable for this tenant: {m.get('reason', 'not trained')}")
        return True
    return False


# ------------------------------------------------------------------ pages
def page_dashboard():
    st.header("Dashboard")
    o = get("/overview")
    k = o["kpis"]
    c = st.columns(5)
    c[0].metric("Revenue (gross)", money(k["revenue"]))
    c[1].metric("Transactions", f"{k['transactions']:,}")
    c[2].metric("Customers", f"{k['customers']:,}")
    c[3].metric("Avg order value", money(k["avg_order_value"], 2))
    c[4].metric("Returns / cancellations", money(k["returns_value"]))
    st.caption(f"Data from {k['date_min'][:10]} to {k['date_max'][:10]}. Revenue excludes cancellations; customers are identified buyers only.")
    m = pd.DataFrame(o["monthly"])
    if not m.empty:
        fig = go.Figure()
        fig.add_bar(x=m["month"], y=m["revenue"], name="Revenue", marker_color="#4c78a8")
        fig.add_scatter(x=m["month"], y=m["orders"], name="Orders", yaxis="y2", line=dict(color="#f58518"))
        fig.update_layout(title="Monthly revenue and orders", yaxis=dict(title=f"Revenue ({CUR})"),
                          yaxis2=dict(title="Orders", overlaying="y", side="right"), height=340, legend=dict(orientation="h"),
                          margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    if not o["trained"]:
        st.info("Train models in **Tenant Management** to unlock churn, segment and spend analytics.")
        return
    l, r = st.columns(2)
    cd = pd.DataFrame(o["churn_distribution"])
    if not cd.empty:
        cd["risk_category"] = pd.Categorical(cd["risk_category"], ["Low", "Medium", "High"], ordered=True)
        f = px.bar(cd.sort_values("risk_category"), x="risk_category", y="customers", color="risk_category", color_discrete_map=RISK_COLORS,
                   text="customers", title="Churn-risk distribution", labels={"risk_category": "Risk"})
        f.update_layout(showlegend=False, height=330, margin=dict(t=50, b=10))
        l.plotly_chart(f, width="stretch")
    sd = pd.DataFrame(o["segment_distribution"])
    if not sd.empty:
        f = px.bar(sd, x="segment_name", y="revenue", text=sd["revenue_share"].map(lambda v: f"{v:.0%}"), title="Revenue by segment",
                   labels={"segment_name": "Segment", "revenue": f"Customer revenue ({CUR})"})
        f.update_layout(height=330, margin=dict(t=50, b=10))
        r.plotly_chart(f, width="stretch")
    st.subheader("Business insights")
    for i in o["insights"]:
        st.markdown(f"- {i}")
    st.subheader("Highest-value customers at high churn risk")
    hr = get("/customers", risk="High", sort="monetary", limit=10)["items"]
    if hr:
        df = pd.DataFrame(hr)[["customer_id", "monetary", "recency_days", "frequency", "churn_prob", "pred_spend", "segment_name"]]
        st.dataframe(df, width="stretch", hide_index=True, column_config={
            "monetary": st.column_config.NumberColumn("Total spend", format=f"{CUR}%.0f"),
            "recency_days": st.column_config.NumberColumn("Days since last order", format="%.0f"),
            "churn_prob": st.column_config.ProgressColumn("Churn risk", min_value=0, max_value=1, format="%.2f"),
            "pred_spend": st.column_config.NumberColumn(f"Predicted {config.HORIZON_DAYS}d spend", format=f"{CUR}%.0f")})
    else:
        st.write("No high-risk customers.")


def page_customers():
    st.header("Customer Intelligence")
    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
    search = f1.text_input("Customer ID (prefix search)", placeholder="e.g. 1234", max_chars=12)
    risk = f2.selectbox("Risk", ["All", "High", "Medium", "Low"])
    sort = f3.selectbox("Sort by", ["churn_prob", "monetary", "pred_spend", "recency_days", "frequency", "customer_id"])
    limit = f4.selectbox("Rows", [25, 50, 100, 200], index=1)
    if search and not search.isdigit():
        st.error("Customer ID must be numeric.")
        return
    res = get("/customers", search=search or None, risk=None if risk == "All" else risk, sort=sort, limit=limit)
    st.caption(f"{res['total']:,} matching customers (showing {len(res['items'])}). Select a row for details.")
    if not res["items"]:
        st.info("No customers match these filters.")
        return
    df = pd.DataFrame(res["items"])[["customer_id", "country", "recency_days", "frequency", "monetary", "churn_prob",
                                     "risk_category", "segment_name", "pred_spend"]]
    sel = st.dataframe(df, width="stretch", hide_index=True, on_select="rerun", selection_mode="single-row", column_config={
        "recency_days": st.column_config.NumberColumn("Recency (days)", format="%.0f"),
        "frequency": st.column_config.NumberColumn("Orders", format="%d"),
        "monetary": st.column_config.NumberColumn("Total spend", format=f"{CUR}%.0f"),
        "churn_prob": st.column_config.ProgressColumn("Churn risk", min_value=0, max_value=1, format="%.2f"),
        "risk_category": "Risk", "segment_name": "Segment",
        "pred_spend": st.column_config.NumberColumn(f"Predicted {config.HORIZON_DAYS}d spend", format=f"{CUR}%.0f")})
    rows = sel.selection.rows if sel and sel.selection else []
    cid = int(df.iloc[rows[0]]["customer_id"]) if rows else None
    if cid is None:
        return
    d = get(f"/customers/{cid}")
    cu = d["customer"]
    st.subheader(f"Customer {cid}")
    c = st.columns(6)
    c[0].metric("Recency", f"{cu['recency_days']:.0f} d")
    c[1].metric("Frequency", f"{cu['frequency']:.0f} orders")
    c[2].metric("Monetary", money(cu["monetary"]))
    c[3].metric("Churn risk", pct(cu["churn_prob"], 0), cu["risk_category"], delta_color="off")
    c[4].metric("Segment", cu["segment_name"] or "-")
    c[5].metric(f"Predicted {config.HORIZON_DAYS}d spend", money(cu["pred_spend"]))
    st.caption(d["notes"]["pred_spend"] + " " + d["notes"]["churn"])
    l, r = st.columns(2)
    fac = pd.DataFrame(d["churn_factors"])
    if not fac.empty:
        fac = fac.sort_values("contribution")
        fig = px.bar(fac, x="contribution", y="feature", orientation="h", title="What drives this customer's churn risk",
                     color=fac["contribution"] > 0, color_discrete_map={True: "#d64545", False: "#2e9e5b"},
                     labels={"contribution": "Change in churn probability vs a typical customer", "feature": ""},
                     hover_data={"value": ":.2f", "typical": ":.2f"})
        fig.update_layout(showlegend=False, height=330, margin=dict(t=50, b=10))
        l.plotly_chart(fig, width="stretch")
        l.caption("Occlusion method: each feature is replaced by the typical customer's value and the drop in predicted risk is shown. "
                  "Red raises risk, green lowers it.")
    r.markdown("**Top products**")
    r.dataframe(pd.DataFrame(d["top_products"]), width="stretch", hide_index=True)
    st.markdown("**Recent invoices**")
    inv = pd.DataFrame(d["recent_invoices"])
    inv["type"] = inv["is_cancel"].map({0: "Order", 1: "Cancellation"})
    st.dataframe(inv.drop(columns=["is_cancel"]), width="stretch", hide_index=True)


def page_segmentation():
    st.header("Segmentation (DBSCAN on RFM)")
    mt = get("/metrics")
    if unavailable(mt["segmentation"], "Segmentation"):
        return
    s = get("/segments")
    m, pts = s["metrics"], pd.DataFrame(s["points"])
    c = st.columns(5)
    c[0].metric("Clusters", m["n_clusters"])
    c[1].metric("Noise points", f"{m['noise_points']:,}", pct(m["noise_share"]), delta_color="off")
    c[2].metric("eps / min_samples", f"{m['eps']:.2f} / {m['min_samples']}")
    c[3].metric("Silhouette (non-noise)", num(m["silhouette_non_noise"]))
    c[4].metric("Davies-Bouldin", num(m["davies_bouldin_non_noise"]))
    view = st.radio("View", ["3D (log R, F, M)", "2D (PCA)", "2D Frequency vs Monetary"], horizontal=True)
    pts["Segment"] = pts["segment_name"]
    cmap = {n: ("#9aa0a6" if sid == -1 else None) for n, sid in zip(pts["segment_name"], pts["segment_id"])}
    cmap = {k: v for k, v in cmap.items() if v}
    order = {"Segment": sorted(pts["Segment"].unique())}
    if view.startswith("3D"):
        fig = px.scatter_3d(pts, x="log_recency", y="log_frequency", z="log_monetary", color="Segment", color_discrete_map=cmap,
                            category_orders=order, hover_data=["customer_id"], opacity=0.75, height=620,
                            labels={"log_recency": "log(1+Recency)", "log_frequency": "log(1+Frequency)", "log_monetary": "log(1+Monetary)"})
        fig.update_traces(marker_size=3)
    elif view.startswith("2D (PCA"):
        fig = px.scatter(pts, x="pc1", y="pc2", color="Segment", color_discrete_map=cmap, category_orders=order, hover_data=["customer_id"],
                         opacity=0.75, height=560, labels={"pc1": f"PC1 ({m['pca_explained_variance'][0]:.0%})", "pc2": f"PC2 ({m['pca_explained_variance'][1]:.0%})"})
    else:
        fig = px.scatter(pts, x="frequency", y="monetary", color="Segment", color_discrete_map=cmap, category_orders=order,
                         hover_data=["customer_id", "recency_days"], log_x=True, log_y=True, opacity=0.75, height=560)
    st.plotly_chart(fig, width="stretch")
    st.caption("Grey points are DBSCAN noise: customers outside every dense region. They are kept and shown, not forced into a cluster.")
    st.subheader("Segment profiles")
    prof = pd.DataFrame(m["profiles"])
    st.dataframe(prof[["segment_name", "customers", "share", "median_recency_days", "median_frequency", "median_monetary", "revenue_share"]],
                 width="stretch", hide_index=True, column_config={
        "segment_name": "Segment", "customers": "Customers", "share": st.column_config.NumberColumn("Share", format="percent"),
        "median_recency_days": st.column_config.NumberColumn("Median recency (d)", format="%.0f"),
        "median_frequency": st.column_config.NumberColumn("Median orders", format="%.0f"),
        "median_monetary": st.column_config.NumberColumn("Median spend", format=f"{CUR}%.0f"),
        "revenue_share": st.column_config.NumberColumn("Revenue share", format="percent")})
    for p in m["profiles"]:
        with st.expander(p["segment_name"], expanded=p["segment_id"] == -1):
            st.write(p["description"])
    with st.expander("How the parameters were chosen"):
        st.write(m["note"])
        if m["search_grid"]:
            st.dataframe(pd.DataFrame(m["search_grid"]).sort_values("silhouette", ascending=False).head(10), hide_index=True)


def page_recs():
    st.header("Product Recommendations (Apriori)")
    mt = get("/metrics")
    if unavailable(mt["recommendations"], "Recommendations"):
        return
    prods = get("/recommendations/products")
    label = {p["stock_code"]: f"{p['description']} ({p['stock_code']})" for p in prods}
    c1, c2, c3 = st.columns([3, 1, 1])
    main = c1.selectbox("Product", list(label), format_func=label.get, help="Products with at least one single-item rule")
    top_n = c2.slider("Max results", 3, 30, 10)
    min_lift = c3.slider("Min lift", 1.0, 20.0, 1.5, 0.5)
    extra = st.multiselect("Optionally add more products to the basket", [k for k in label if k != main], format_func=label.get, max_selections=5)
    res = api.tenant(st.session_state.tid, "POST", "/recommendations", json={"stock_codes": [main, *extra], "top_n": top_n, "min_lift": min_lift})["items"]
    if not res:
        st.info("No rules meet these thresholds. Lower the minimum lift.")
    else:
        df = pd.DataFrame(res)
        df["because_of"] = df["because_of"].map(lambda x: " + ".join(x))
        st.dataframe(df[["description", "stock_code", "support", "confidence", "lift", "because_of"]], width="stretch", hide_index=True,
                     column_config={"support": st.column_config.NumberColumn("Support", format="%.3f"),
                                    "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
                                    "lift": st.column_config.NumberColumn("Lift", format="%.1f"), "because_of": "Because basket has"})
        fig = px.bar(df.sort_values("lift"), x="lift", y="description", orientation="h", color="confidence", title="Recommended products by lift",
                     color_continuous_scale="Blues", labels={"description": ""})
        fig.update_layout(height=max(300, 32 * len(df) + 100), margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    st.caption("Support = share of invoices containing the items together; confidence = P(recommended | basket); lift = confidence vs. its base rate (>1 means positively associated).")
    with st.expander("Top association rules in this tenant"):
        r = pd.DataFrame(get("/recommendations/rules", limit=300))
        r["antecedents"] = r["antecedents"].map(" + ".join)
        r["consequents"] = r["consequents"].map(" + ".join)
        st.dataframe(r, width="stretch", hide_index=True)
        st.plotly_chart(px.scatter(r, x="support", y="confidence", color="lift", color_continuous_scale="Viridis", hover_data=["antecedents", "consequents"], height=380), width="stretch")


def metric_table(models: dict, split: str, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{"Model": n, **{c: v[split][c] for c in cols}} for n, v in models.items()])


def page_perf():
    st.header("Model Performance")
    st.caption("All numbers are computed during training on held-out, later-in-time data. Nothing here is hard-coded.")
    mt = get("/metrics")
    t1, t2, t3, t4 = st.tabs(["Churn", "Future spend (Ridge / Lasso)", "Segmentation", "Recommendations"])
    with t1:
        m = mt["churn"]
        if not unavailable(m, "Churn model"):
            st.write(m["label_definition"])
            sp = m["split"]
            st.caption(f"Train snapshots: {', '.join(sp['train_cutoffs'])} | validation: {sp['val_cutoff']} | test: {sp['test_cutoff']} "
                       f"(rows {sp['n_train']:,}/{sp['n_val']:,}/{sp['n_test']:,}). Training label windows end before the validation date (purged). "
                       f"Churn rate train/val/test: {pct(m['churn_rate']['train'],0)}/{pct(m['churn_rate']['validation'],0)}/{pct(m['churn_rate']['test'],0)}.")
            st.success(f"Selected model (best validation ROC-AUC): **{m['selected_model']}**")
            df = metric_table(m["models"], "test", ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy", "brier"])
            st.dataframe(df, width="stretch", hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.3f") for c in df.columns[1:]})
            a, b = st.columns(2)
            fig = go.Figure(go.Scatter(x=m["roc_curve"]["fpr"], y=m["roc_curve"]["tpr"], name="ROC"))
            fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="Chance")
            fig.update_layout(title="ROC curve (test)", xaxis_title="False positive rate", yaxis_title="True positive rate", height=330, margin=dict(t=50, b=10))
            a.plotly_chart(fig, width="stretch")
            cm = m["confusion_matrix"]
            fig = px.imshow([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], text_auto=True, x=["Pred active", "Pred churn"], y=["Actual active", "Actual churn"],
                            color_continuous_scale="Blues", title=f"Confusion matrix (test, threshold {m['models'][m['selected_model']]['test']['threshold']:.2f})")
            fig.update_layout(height=330, margin=dict(t=50, b=10))
            b.plotly_chart(fig, width="stretch")
            fi = pd.DataFrame(m["feature_importance"]).head(10).sort_values("importance")
            st.plotly_chart(px.bar(fi, x="importance", y="feature", orientation="h", title="Permutation importance (drop in test ROC-AUC)", height=360), width="stretch")
            st.caption(m["note"])
    with t2:
        m = mt["clv"]
        if not unavailable(m, "Future-spend model"):
            st.warning(m["target_definition"])
            st.write(m["selection_rule"])
            rows = [{"Model": n, "alpha": v["alpha"], **v["test"], "Non-zero coefs": f"{v['nonzero_coefficients']}/{v['n_features']}"} for n, v in m["models"].items()]
            rows += [{"Model": f"Baseline: {n}", **v} for n, v in m["baselines"].items()]
            df = pd.DataFrame(rows).rename(columns={"mae": f"MAE ({CUR})", "rmse": f"RMSE ({CUR})", "r2": "R² (money)", "log_rmse": "RMSE (log)", "log_r2": "R² (log)"})
            st.dataframe(df, width="stretch", hide_index=True)
            st.success(f"Selected: **{m['selected_model']}**")
            st.caption(m["note"])
            a, b = st.columns(2)
            sc = pd.DataFrame(m["scatter"])
            fig = px.scatter(sc, x="actual", y="predicted", opacity=0.5, title="Predicted vs actual spend (test sample)", log_x=True, log_y=True)
            fig.add_scatter(x=[1, sc["actual"].max()], y=[1, sc["actual"].max()], mode="lines", line=dict(dash="dash"), name="Perfect")
            fig.update_layout(height=360, margin=dict(t=50, b=10))
            a.plotly_chart(fig, width="stretch")
            kind = b.radio("Coefficients of", list(m["models"]), horizontal=True, index=list(m["models"]).index(m["selected_model"]))
            cf = pd.DataFrame(m["models"][kind]["coefficients"]).head(12).sort_values("coef")
            b.plotly_chart(px.bar(cf, x="coef", y="feature", orientation="h", title=f"{kind} coefficients (standardised features)", height=330), width="stretch")
    with t3:
        m = mt["segmentation"]
        if not unavailable(m, "Segmentation"):
            st.write(f"{m['algorithm']} on {m['features']}.")
            st.json({k: m[k] for k in ["eps", "min_samples", "n_customers", "n_clusters", "noise_points", "noise_share", "silhouette_non_noise", "davies_bouldin_non_noise"]})
            st.caption(m["note"])
    with t4:
        m = mt["recommendations"]
        if not unavailable(m, "Recommendations"):
            c = st.columns(5)
            c[0].metric("Baskets", f"{m['baskets']:,}")
            c[1].metric("Frequent items", m["frequent_items"])
            c[2].metric("Frequent itemsets", f"{m['frequent_itemsets']:,}")
            c[3].metric("Rules", f"{m['rules']:,}")
            c[4].metric("Median lift", num(m["lift_median"], 1))
            h = m["holdout"]
            if h.get("evaluated_rules"):
                st.write(f"**Chronological holdout** ({h['evaluated_rules']} top rules): mean confidence {num(h['mean_train_confidence'], 2)} on the "
                         f"mining period vs {num(h['mean_holdout_confidence'], 2)} on later invoices; "
                         f"{pct(h['share_rules_lift_gt_1'], 0)} of rules still have lift > 1.")
            st.caption(f"min_support={m['min_support']:.3f}, min_confidence={m['min_confidence']}, min_lift={m['min_lift']}. {m['note']}")


def page_explorer():
    st.header("Data Explorer")
    s = get("/explorer/stats")
    c = st.columns(6)
    c[0].metric("Lines", f"{s['lines']:,}")
    c[1].metric("Invoices", f"{s['invoices']:,}")
    c[2].metric("Customers", f"{s['customers']:,}")
    c[3].metric("Products", f"{s['products']:,}")
    c[4].metric("Median order", money(s["median_order_value"], 2))
    c[5].metric("95th pct order", money(s["p95_order_value"], 0))
    st.caption(f"{pct(s['anonymous_line_share'])} of lines have no CustomerID (excluded from customer models, included in revenue and Apriori); "
               f"{pct(s['cancellation_line_share'])} of lines are cancellations. {s['countries']} countries, {s['date_min'][:10]} to {s['date_max'][:10]}.")
    if s.get("ingest"):
        with st.expander("Ingestion / cleaning report"):
            st.json(s["ingest"])
    freq = st.radio("Trend granularity", ["month", "week", "day"], horizontal=True)
    tr = pd.DataFrame(get("/explorer/trends", freq=freq))
    fig = px.line(tr, x="period", y=["revenue", "orders", "customers"], title=f"Trend by {freq}")
    fig.update_layout(height=340, margin=dict(t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    a, b = st.columns([1, 1])
    by = a.selectbox("Rank products by", ["revenue", "quantity", "orders"])
    q = b.text_input("Search product (description or code)", max_chars=50)
    top = st.slider("Top N", 5, 50, 15)
    pr = pd.DataFrame(get("/explorer/products", by=by, top=top, q=q or None))
    if pr.empty:
        st.info("No products match.")
    else:
        st.plotly_chart(px.bar(pr.sort_values(by), x=by, y="description", orientation="h", height=max(320, 26 * len(pr) + 80)), width="stretch")
        st.dataframe(pr, width="stretch", hide_index=True)
    co = pd.DataFrame(get("/explorer/countries", top=10))
    st.plotly_chart(px.bar(co, x="country", y="revenue", title="Revenue by country (top 10)", height=320), width="stretch")


def page_tenants():
    st.header("Tenant Management")
    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash[0])
        if flash[1]:
            st.caption("API key (shown once; also saved locally):")
            st.code(flash[1], language=None)
    ts = tenants_list()
    if ts:
        df = pd.DataFrame(ts)[["name", "tenant_id", "transactions", "customers", "trained_at", "created_at"]]
        df["trained"] = df["trained_at"].notna().map({True: "Yes", False: "No"})
        st.dataframe(df.drop(columns=["trained_at"]), width="stretch", hide_index=True)
    st.subheader("Create a tenant")
    with st.form("create"):
        name = st.text_input("Business name", placeholder="e.g. Acme Gifts", max_chars=40)
        src = st.selectbox("Initial data", ["None (load later)", "Demo data - full UCI Online Retail", "Demo data - half 1 of 2", "Demo data - half 2 of 2"])
        go_ = st.form_submit_button("Create tenant")
    if go_:
        try:
            t = api.admin("POST", "/tenants", json={"name": name})
            api.save_key(t["tenant_id"], t["name"], t["api_key"])
            st.session_state.tid = t["tenant_id"]
            st.cache_data.clear()
            if src != "None (load later)":
                part, of = {"Demo data - full UCI Online Retail": (0, 1), "Demo data - half 1 of 2": (0, 2), "Demo data - half 2 of 2": (1, 2)}[src]
                with st.spinner("Loading and cleaning demo data ..."):
                    api.tenant(t["tenant_id"], "POST", "/data/demo", params={"part": part, "of": of})
            st.session_state.flash = (f"Tenant '{t['name']}' created and selected. Its API key was saved to {config.KEYSTORE_PATH.name}.", t["api_key"])
            st.rerun()
        except api.ApiError as e:
            st.error(str(e))
    if not ts:
        return
    st.divider()
    cur = next((t for t in ts if t["tenant_id"] == st.session_state.tid), ts[0])
    st.subheader(f"Manage: {cur['name']}")
    tid = cur["tenant_id"]
    if tid not in api.load_keys():
        st.warning("No API key stored locally for this tenant.")
        if st.button("Regenerate API key"):
            k = api.admin("POST", f"/tenants/{tid}/rotate-key")["api_key"]
            api.save_key(tid, cur["name"], k)
            st.success("New key saved.")
            st.rerun()
        return
    s = api.tenant(tid, "GET", "/status")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Load data**")
        choice = st.selectbox("Demo partition", ["Full dataset", "Half 1 of 2", "Half 2 of 2", "Third 1 of 3", "Third 2 of 3", "Third 3 of 3"], key="demo_choice")
        if st.button("Load demo data (replaces existing data)"):
            part, of = {"Full dataset": (0, 1), "Half 1 of 2": (0, 2), "Half 2 of 2": (1, 2), "Third 1 of 3": (0, 3), "Third 2 of 3": (1, 3), "Third 3 of 3": (2, 3)}[choice]
            try:
                with st.spinner("Loading ..."):
                    r = api.tenant(tid, "POST", "/data/demo", params={"part": part, "of": of})
                st.cache_data.clear()
                st.success(f"Loaded {r['report']['rows_kept']:,} lines / {r['report']['customers']:,} customers. Now train the models.")
            except api.ApiError as e:
                st.error(str(e))
    with d2:
        up = st.file_uploader("Upload CSV / XLSX (UCI Online Retail columns)", type=["csv", "xlsx"])
        if up and st.button("Ingest uploaded file (replaces existing data)"):
            try:
                with st.spinner("Validating and cleaning ..."):
                    r = api.tenant(tid, "POST", "/data/upload", files={"file": (up.name, up.getvalue())})
                st.cache_data.clear()
                st.success(f"Ingested {r['report']['rows_kept']:,} valid lines from {r['report']['rows_in']:,}. Dropped: {r['report']['dropped']}")
            except api.ApiError as e:
                st.error(str(e))
    st.markdown("**Models**")
    job = s["job"]
    if job["status"] == "running":
        st.info(f"Training in progress - stage: {job['stage']}")
        time.sleep(2)
        st.rerun()
    if job["status"] == "failed":
        st.error(f"Last training failed: {job['message']}")
    if job["status"] == "done":
        st.success(job["message"])
    if st.button("Train models", type="primary", disabled=not s["has_data"] or job["status"] == "running"):
        try:
            api.tenant(tid, "POST", "/train")
            st.cache_data.clear()
            st.rerun()
        except api.ApiError as e:
            st.error(str(e))
    if not s["has_data"]:
        st.caption("Load data before training.")
    with st.expander("Danger zone"):
        if st.button("Regenerate API key"):
            k = api.admin("POST", f"/tenants/{tid}/rotate-key")["api_key"]
            api.save_key(tid, cur["name"], k)
            st.success("New key saved; the old key no longer works.")
        sure = st.checkbox(f"I understand this permanently deletes '{cur['name']}' and all its data and models")
        if st.button("Delete tenant", disabled=not sure):
            api.admin("DELETE", f"/tenants/{tid}")
            st.session_state.pop("tid", None)
            st.cache_data.clear()
            st.rerun()


# ------------------------------------------------------------------ shell
def main():
    st.sidebar.title("📊 Customer Intelligence")
    h = api.health()
    if not h:
        st.sidebar.error("API offline")
        st.error(f"Cannot reach the API at {config.API_URL}. Start it with start_api.bat and refresh.")
        return
    st.sidebar.success(f"API online · {h['tenants']} tenant(s)")
    ts = tenants_list()
    ids = [t["tenant_id"] for t in ts]
    if ids and st.session_state.get("tid") not in ids:
        st.session_state.tid = ids[0]
    if ids:
        by_name = {t["name"]: t["tenant_id"] for t in ts}
        cur_name = next(n for n, i in by_name.items() if i == st.session_state.tid)
        pick = st.sidebar.selectbox("Active tenant", list(by_name), index=list(by_name).index(cur_name))
        st.session_state.tid = by_name[pick]
    page = st.sidebar.radio("Navigate", PAGES if ids else ["Tenant Management"], key="page")
    st.sidebar.caption("Churn, segments, recommendations and 90-day spend predictions. Each tenant's data is isolated by API key.")
    if page == "Tenant Management":
        guard(page_tenants)
        return
    if "tid" not in st.session_state:
        return
    fn = {"Dashboard": page_dashboard, "Customer Intelligence": page_customers, "Segmentation": page_segmentation,
          "Recommendations": page_recs, "Model Performance": page_perf, "Data Explorer": page_explorer}[page]

    def run():
        if page in ("Dashboard", "Data Explorer"):
            if tget(st.session_state.tid, "/status")["has_data"]:
                fn()
            else:
                st.info("This tenant has no data yet. Go to **Tenant Management** to load a dataset.")
        elif need_trained():
            fn()
    guard(run)


main()
