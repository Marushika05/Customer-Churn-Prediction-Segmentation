"""Create demo tenants from the UCI Online Retail data, train all models and store keys for the UI.

    python -m scripts.seed_demo            # 2 demo businesses (customer-disjoint halves of the data)
    python -m scripts.seed_demo --parts 3
"""
import argparse
import json
import time

from backend import config, db, ingestion, training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, default=2)
    args = ap.parse_args()
    if not config.RAW_DATA_PATH.exists():
        raise SystemExit(f"Dataset missing: run  python -m scripts.download_data")
    db.init_db()
    keys = json.loads(config.KEYSTORE_PATH.read_text()) if config.KEYSTORE_PATH.exists() else {}
    existing = {t["name"].lower(): t for t in db.list_tenants()}
    print("Cleaning dataset ...")
    full, _ = ingestion.clean(ingestion.read_file(config.RAW_DATA_PATH))
    for i in range(args.parts):
        name = f"Demo Retailer {chr(65 + i)}"
        if name.lower() in existing:
            print(f"{name}: exists, skipping")
            continue
        tid, key = db.create_tenant(name)
        keys[tid] = {"name": name, "key": key}
        df = ingestion.demo_partition(full, i, args.parts)
        rep = ingestion.summarize(df, len(df), {})
        rep["source"] = f"UCI Online Retail, partition {i + 1} of {args.parts}"
        db.replace_transactions(tid, df, rep)
        print(f"{name} ({tid}): {len(df):,} lines, {rep['customers']} customers. Training ...")
        t = time.time()
        db.try_start_job(tid)
        training.train_tenant(tid)
        print("   ", db.get_job(tid)["message"], f"({time.time() - t:.0f}s)")
    config.KEYSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.KEYSTORE_PATH.write_text(json.dumps(keys, indent=1))
    print("Done. Keys saved to", config.KEYSTORE_PATH)


if __name__ == "__main__":
    main()
