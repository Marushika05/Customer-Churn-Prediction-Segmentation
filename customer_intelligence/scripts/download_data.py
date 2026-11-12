"""Download the UCI Online Retail dataset as CSV into data/raw/.   python -m scripts.download_data"""
import sys
import zipfile
from io import BytesIO

import pandas as pd
import requests

from backend import config

SOURCES = [  # same 541,909-row dataset
    ("csv", "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/all/online-retail-dataset.csv"),
    ("zip", "https://archive.ics.uci.edu/static/public/352/online+retail.zip"),
]


def main():
    config.RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if config.RAW_DATA_PATH.exists():
        print("Dataset already present:", config.RAW_DATA_PATH)
        return
    for kind, url in SOURCES:
        try:
            print("Downloading", url)
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            if kind == "csv":
                config.RAW_DATA_PATH.write_bytes(r.content)
            else:
                z = zipfile.ZipFile(BytesIO(r.content))
                name = next(n for n in z.namelist() if n.lower().endswith(".xlsx"))
                pd.read_excel(z.open(name), dtype={"InvoiceNo": str, "StockCode": str}).to_csv(config.RAW_DATA_PATH, index=False)
            print("Saved", config.RAW_DATA_PATH)
            return
        except Exception as e:  # noqa: BLE001
            print("  failed:", e)
    sys.exit("Could not download automatically. Download 'Online Retail' from https://archive.ics.uci.edu/dataset/352/online+retail "
             f"and save it as {config.RAW_DATA_PATH} (CSV) - or upload the .xlsx in the app's Tenant Management page.")


if __name__ == "__main__":
    main()
