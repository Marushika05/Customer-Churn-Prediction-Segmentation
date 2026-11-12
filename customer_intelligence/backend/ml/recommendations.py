"""Frequently-bought-together recommendations via Apriori association rules (mlxtend)."""
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from .. import config

MAX_ITEMS = 250


def _basket_matrix(sales: pd.DataFrame, items: list[str] | None = None, min_support: float = 0.0):
    """Boolean invoice x product matrix. Rows = invoices (one basket each), duplicates within a basket collapse."""
    pairs = sales[["invoice_no", "stock_code"]].drop_duplicates()
    if items is None:
        n = pairs["invoice_no"].nunique()
        counts = pairs["stock_code"].value_counts()
        items = counts[counts / n >= min_support].index.tolist()[:MAX_ITEMS]  # cap keeps Apriori memory bounded
    pairs = pairs[pairs["stock_code"].isin(items)]
    mat = pd.crosstab(pairs["invoice_no"], pairs["stock_code"]).astype(bool)
    return mat.reindex(columns=items, fill_value=False), items


def _mine(sales: pd.DataFrame, min_support: float):
    mat, items = _basket_matrix(sales, min_support=min_support)
    if mat.shape[1] < 2:
        return pd.DataFrame(), mat, 0
    fi = apriori(mat, min_support=min_support, use_colnames=True, max_len=3, low_memory=True)
    if fi.empty:
        return pd.DataFrame(), mat, 0
    rules = association_rules(fi, metric="confidence", min_threshold=config.APRIORI_MIN_CONFIDENCE, num_itemsets=len(mat))
    rules = rules[rules["lift"] >= config.APRIORI_MIN_LIFT]
    return rules, mat, len(fi)


def train(tx: pd.DataFrame) -> tuple[dict, dict]:
    sales = tx[~tx["is_cancel"]]
    inv = sales.groupby("invoice_no")["invoice_date"].min().sort_values()
    n_baskets = len(inv)
    if n_baskets < 200:
        raise ValueError(f"Only {n_baskets} valid invoices; need at least 200 for association rules.")
    min_support = max(config.APRIORI_MIN_SUPPORT, 8.0 / n_baskets)  # items capped at MAX_ITEMS in addition
    names = (sales.dropna(subset=["description"]).groupby("stock_code")["description"]
             .agg(lambda s: s.mode().iloc[0]).to_dict())

    # chronological holdout check: mine on first 80% of invoices, test rules on the last 20%
    cut = inv.index[int(n_baskets * 0.8)]
    train_inv = set(inv.index[: int(n_baskets * 0.8)])
    s_tr = sales[sales["invoice_no"].isin(train_inv)]
    s_te = sales[~sales["invoice_no"].isin(train_inv)]
    r_tr, mat_tr, _ = _mine(s_tr, min_support)
    holdout = {"evaluated_rules": 0}
    if not r_tr.empty and len(s_te):
        mat_te, _ = _basket_matrix(s_te, items=list(mat_tr.columns))
        top = r_tr.sort_values("lift", ascending=False).head(500)
        hit, base, lifts = [], [], []
        for a, c in zip(top["antecedents"], top["consequents"]):
            a, c = list(a), list(c)
            has_a = mat_te[a].all(axis=1)
            if has_a.sum() < 10:
                continue
            conf = float(mat_te.loc[has_a, c].all(axis=1).mean())
            pc = float(mat_te[c].all(axis=1).mean())
            hit.append(conf)
            base.append(pc)
            lifts.append(conf / pc if pc > 0 else np.nan)
        if hit:
            holdout = {"evaluated_rules": len(hit), "mean_holdout_confidence": float(np.mean(hit)),
                       "mean_train_confidence": float(top["confidence"].head(len(hit)).mean()),
                       "mean_holdout_lift": float(np.nanmean(lifts)), "share_rules_lift_gt_1": float(np.mean(np.array(lifts) > 1))}

    rules, mat, n_itemsets = _mine(sales, min_support)  # final rules use all data
    if rules.empty:
        raise ValueError("No association rules met the support/confidence/lift thresholds.")
    rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    rules["antecedents"] = rules["antecedents"].map(lambda s: tuple(sorted(s)))
    rules["consequents"] = rules["consequents"].map(lambda s: tuple(sorted(s)))
    rules = rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
    art = {"rules": rules, "names": names}
    m = {"status": "ok", "algorithm": "Apriori", "baskets": int(n_baskets), "frequent_items": int(mat.shape[1]),
         "frequent_itemsets": int(n_itemsets), "rules": int(len(rules)), "min_support": float(min_support),
         "min_confidence": config.APRIORI_MIN_CONFIDENCE, "min_lift": config.APRIORI_MIN_LIFT,
         "support_median": float(rules["support"].median()), "confidence_median": float(rules["confidence"].median()),
         "lift_median": float(rules["lift"].median()), "lift_max": float(rules["lift"].max()),
         "holdout": holdout,
         "note": "Baskets are valid (non-cancelled) invoices. Holdout: rules mined on the first 80% of invoices, "
                 "evaluated on the last 20%."}
    return art, m


def recommend(art: dict, basket: list[str], top_n: int = 10, min_lift: float = 1.0) -> list[dict]:
    basket_set = set(basket)
    r = art["rules"]
    sub = r[r["antecedents"].map(lambda a: set(a) <= basket_set) & (r["lift"] >= min_lift)]
    best: dict[str, dict] = {}
    for a, c, s, cf, l in zip(sub["antecedents"], sub["consequents"], sub["support"], sub["confidence"], sub["lift"]):
        if set(c) & basket_set:
            continue
        for item in c:
            cur = best.get(item)
            if cur is None or (l, cf) > (cur["lift"], cur["confidence"]):
                best[item] = {"stock_code": item, "description": art["names"].get(item, item),
                              "support": float(s), "confidence": float(cf), "lift": float(l),
                              "because_of": [art["names"].get(x, x) for x in a]}
    return sorted(best.values(), key=lambda d: (-d["lift"], -d["confidence"]))[:top_n]


def antecedent_products(art: dict) -> list[dict]:
    """Single products that appear as a standalone antecedent (usable as a one-item query)."""
    r = art["rules"]
    single = r[r["antecedents"].map(len) == 1]
    counts = single.groupby(single["antecedents"].map(lambda a: a[0])).size()
    return [{"stock_code": k, "description": art["names"].get(k, k), "rules": int(v)}
            for k, v in counts.sort_values(ascending=False).items()]
