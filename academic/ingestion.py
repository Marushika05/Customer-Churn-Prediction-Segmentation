"""Reading, validating and cleaning raw transaction files (UCI Online Retail schema)."""
import re
from pathlib import Path

import pandas as pd
import numpy as np

ALIASES = {  # keys are normalised: lower-case, no spaces/underscores
    "invoiceno": "invoice_no", "invoice": "invoice_no",
    "stockcode": "stock_code", "sku": "stock_code",
    "description": "description",
    "quantity": "quantity", "qty": "quantity",
    "invoicedate": "invoice_date", "date": "invoice_date",
    "unitprice": "unit_price", "price": "unit_price",
    "customerid": "customer_id", "country": "country",
}
REQUIRED = ["invoice_no", "stock_code", "quantity", "invoice_date", "unit_price", "customer_id"]
# Non-product service/adjustment codes present in the Online Retail data
NON_PRODUCT = re.compile(r"^(POST|D|M|DOT|C2|BANK CHARGES|AMAZONFEE|S|B|CRUK|PADS|TEST\d*|ADJUST\d*|GIFT_.*)$", re.I)


class IngestError(ValueError):
    pass


def read_file(path: Path, filename: str | None = None, sheets=None) -> pd.DataFrame:
    """Read all Excel sheets by default; normalise aliases per sheet before combining."""
    name = (filename or str(path)).lower()
    if name.endswith(".xlsx"):
        frames = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        chosen = list(frames) if sheets is None else sheets
        return pd.concat([normalise_columns(frames[k]) for k in chosen], ignore_index=True)
    if name.endswith(".xls"):
        raise IngestError("Legacy .xls is unsupported; export as .xlsx or CSV")
    if name.endswith((".csv", ".txt")):
        try:
            return pd.read_csv(path, dtype=str, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, dtype=str, encoding="latin-1")
    raise IngestError("Use .xlsx or comma-separated .csv")


def normalise_columns(raw):
    out = raw.rename(columns={c: ALIASES.get(re.sub(r"[\s_]+", "", str(c).strip().lower()), str(c).strip().lower())
                              for c in raw.columns})
    if out.columns.duplicated().any():
        raise IngestError("Ambiguous duplicate columns after alias mapping")
    return out


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate schema and clean. Returns (clean_df, report). Raises IngestError on unusable input."""
    if raw is None or raw.empty:
        raise IngestError("The file contains no rows.")
    df = normalise_columns(raw)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise IngestError(f"Missing required column(s): {', '.join(missing)}. "
                          "Expected: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country")
    for opt in ("description", "country"):
        if opt not in df.columns:
            df[opt] = None
    df = df[["invoice_no", "stock_code", "description", "quantity", "invoice_date", "unit_price", "customer_id", "country"]].copy()
    n_in = len(df)
    drops: dict[str, int] = {}

    df["invoice_no"] = df["invoice_no"].astype("string").str.strip()
    df["stock_code"] = df["stock_code"].astype("string").str.strip().str.upper()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["invoice_date"]):
        s = df["invoice_date"].astype("string")
        # Explicit supported formats avoid ambiguous day/month guessing and partial mixed-format loss.
        parsed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        for fmt in ("%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            mask = parsed.isna()
            parsed.loc[mask] = pd.to_datetime(s.loc[mask], format=fmt, errors="coerce")
        df["invoice_date"] = parsed

    def drop(mask: pd.Series, reason: str):
        nonlocal df
        n = int(mask.sum())
        if n:
            drops[reason] = n
            df = df[~mask]

    drop(df[["invoice_no", "stock_code", "quantity", "unit_price", "invoice_date"]].isna().any(axis=1), "missing_or_unparseable_fields")
    df = df.copy()
    for c in ("invoice_no", "stock_code"):
        drop(df[c].eq("").fillna(False), "blank_" + c)
    drop(~np.isfinite(df["quantity"]) | ~np.isfinite(df["unit_price"]), "nonfinite_values")
    drop(df["quantity"].mod(1).ne(0), "fractional_quantity")
    bad_id = df["customer_id"].notna() & (~np.isfinite(df["customer_id"]) | df["customer_id"].mod(1).ne(0))
    df.loc[bad_id, "customer_id"] = np.nan
    df["is_cancel"] = df["invoice_no"].str.upper().str.startswith("C")
    drop(df["stock_code"].str.match(NON_PRODUCT), "non_product_codes")
    ok_sale = (~df["is_cancel"]) & (df["quantity"] > 0) & (df["unit_price"] > 0)
    ok_cancel = df["is_cancel"] & (df["quantity"] < 0) & (df["unit_price"] > 0)
    drop(~(ok_sale | ok_cancel), "invalid_quantity_or_price")
    before = len(df)
    df = df.drop_duplicates(subset=["invoice_no", "stock_code", "quantity", "invoice_date", "unit_price", "customer_id"])
    if before - len(df):
        drops["exact_duplicates"] = before - len(df)
    if df.empty:
        raise IngestError("No valid transactions remain after cleaning.")

    df["description"] = df["description"].astype("string").str.strip().str.upper()
    df["country"] = df["country"].astype("string").str.strip().fillna("Unknown")
    df["quantity"] = df["quantity"].astype(int)
    df["line_total"] = (df["quantity"] * df["unit_price"]).round(4)
    df["customer_id"] = df["customer_id"].astype("Int64")
    df["invoice_no"] = df["invoice_no"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    df = df.sort_values("invoice_date", kind="stable").reset_index(drop=True)

    report = summarize(df, n_in, drops)
    return df, report


def summarize(df: pd.DataFrame, rows_in: int, dropped: dict) -> dict:
    return {
        "rows_in": rows_in, "rows_kept": int(len(df)), "dropped": dropped,
        "sale_lines": int((~df["is_cancel"]).sum()), "cancellation_lines": int(df["is_cancel"].sum()),
        "customers": int(df["customer_id"].nunique()),
        "anonymous_line_share": float(df["customer_id"].isna().mean()),
        "date_min": str(df["invoice_date"].min()), "date_max": str(df["invoice_date"].max()),
    }


def demo_partition(df: pd.DataFrame, part: int, of: int) -> pd.DataFrame:
    """Deterministic split by customer so demo tenants hold disjoint customers (anonymous rows split by invoice hash)."""
    if of < 1 or not (0 <= part < of):
        raise IngestError("Invalid demo partition")
    if of == 1:
        return df
    cust = df["customer_id"]
    inv_hash = pd.util.hash_array(df["invoice_no"].to_numpy(dtype=object)) % of
    key = cust.astype("float64").fillna(-1).astype("int64") % of
    key = key.where(cust.notna(), pd.Series(inv_hash.astype("int64"), index=df.index))
    return df[key == part].reset_index(drop=True)
