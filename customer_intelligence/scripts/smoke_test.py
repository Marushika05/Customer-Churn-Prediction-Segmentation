"""End-to-end API check incl. tenant isolation.  Usage: python -m scripts.smoke_test  (API must be running)"""
import json
import sys

import requests

from backend import config

B = config.API_URL
ADMIN = {"X-Admin-Token": config.ADMIN_TOKEN}
fails = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name, extra if not cond else "")
    if not cond:
        fails.append(name)


def get(path, key=None, **kw):
    return requests.get(B + path, headers={"X-API-Key": key} if key else ADMIN, timeout=120, **kw)


keys = json.loads(config.KEYSTORE_PATH.read_text())
tids = list(keys)
A, Bt = tids[0], tids[1]
kA, kB = keys[A]["key"], keys[Bt]["key"]

check("health", get("/health").json()["status"] == "ok")
check("admin list needs token", requests.get(B + "/tenants").status_code == 401)
check("list tenants", len(get("/tenants").json()) >= 2)
for p in ["status", "overview", "metrics", "customers?limit=5", "segments", "recommendations/products", "recommendations/rules?limit=5",
          "explorer/stats", "explorer/trends?freq=week", "explorer/products?by=quantity", "explorer/countries", "train/status"]:
    r = get(f"/tenants/{A}/{p}", kA)
    check(f"GET {p}", r.status_code == 200, r.text[:200])
cust = get(f"/tenants/{A}/customers?limit=3", kA).json()["items"]
cid = cust[0]["customer_id"]
d = get(f"/tenants/{A}/customers/{cid}", kA)
check("customer detail + factors", d.status_code == 200 and len(d.json()["churn_factors"]) > 0, d.text[:200])
prods = get(f"/tenants/{A}/recommendations/products", kA).json()
r = requests.post(f"{B}/tenants/{A}/recommendations", headers={"X-API-Key": kA}, json={"stock_codes": [prods[0]["stock_code"]], "top_n": 5})
check("recommendations", r.status_code == 200 and len(r.json()["items"]) > 0, r.text[:200])
# ---- isolation
check("A key on B path -> 403", get(f"/tenants/{Bt}/overview", kA).status_code == 403)
check("no key -> 403", requests.get(f"{B}/tenants/{A}/overview").status_code == 403)
check("A customer invisible to B", get(f"/tenants/{Bt}/customers/{cid}", kB).status_code == 404)
check("search scoped", all(True for _ in [0]) and get(f"/tenants/{Bt}/customers?search={cid}", kB).json()["total"] == 0)
ra, rb = get(f"/tenants/{A}/overview", kA).json(), get(f"/tenants/{Bt}/overview", kB).json()
check("tenant revenue differs", ra["kpis"]["revenue"] != rb["kpis"]["revenue"])
# ---- validation / errors
check("bad customer search -> 422", get(f"/tenants/{A}/customers?search=abc", kA).status_code == 422)
check("bad tenant id -> 422", get("/tenants/NOT-VALID/overview", kA).status_code == 422)
check("empty basket -> 422", requests.post(f"{B}/tenants/{A}/recommendations", headers={"X-API-Key": kA}, json={"stock_codes": []}).status_code == 422)
check("bad upload type -> 422", requests.post(f"{B}/tenants/{A}/data/upload", headers={"X-API-Key": kA}, files={"file": ("x.exe", b"abc")}).status_code == 422)
check("bad csv schema -> 422", requests.post(f"{B}/tenants/{A}/data/upload", headers={"X-API-Key": kA}, files={"file": ("x.csv", b"a,b\n1,2\n")}).status_code == 422)
# ---- lifecycle of a fresh empty tenant
t = requests.post(B + "/tenants", headers=ADMIN, json={"name": "Smoke Test Co"})
check("create tenant", t.status_code == 201, t.text)
if t.status_code == 201:
    tid, k = t.json()["tenant_id"], t.json()["api_key"]
    check("dup name -> 409", requests.post(B + "/tenants", headers=ADMIN, json={"name": "smoke test co"}).status_code == 409)
    check("empty tenant overview -> 409", get(f"/tenants/{tid}/overview", k).status_code == 409)
    check("empty tenant metrics -> 409", get(f"/tenants/{tid}/metrics", k).status_code == 409)
    check("train without data -> 409", requests.post(f"{B}/tenants/{tid}/train", headers={"X-API-Key": k}).status_code == 409)
    csv = b"InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country\n1,A,X,2,12/1/2010 8:26,2.5,1,UK\n"
    up = requests.post(f"{B}/tenants/{tid}/data/upload", headers={"X-API-Key": k}, files={"file": ("tiny.csv", csv)})
    check("upload tiny csv", up.status_code == 200, up.text)
    import time
    requests.post(f"{B}/tenants/{tid}/train", headers={"X-API-Key": k})
    for _ in range(30):
        j = get(f"/tenants/{tid}/train/status", k).json()
        if j["status"] != "running":
            break
        time.sleep(1)
    m = get(f"/tenants/{tid}/metrics", k).json()
    check("tiny tenant degrades gracefully", j["status"] == "done" and m["churn"]["status"] == "unavailable", str(j))
    check("rotate key invalidates old", requests.post(f"{B}/tenants/{tid}/rotate-key", headers=ADMIN).status_code == 200 and get(f"/tenants/{tid}/status", k).status_code == 403)
    check("delete tenant", requests.delete(f"{B}/tenants/{tid}", headers=ADMIN).status_code == 200)
print("\nFAILED:" if fails else "\nALL CHECKS PASSED", fails or "")
sys.exit(1 if fails else 0)
