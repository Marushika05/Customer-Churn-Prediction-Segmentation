"""Per-tenant model artifact storage (artifacts/<tenant_id>/...)."""
import json
import os
import re
import threading
from pathlib import Path

import joblib

from . import config

_TID = re.compile(r"^[a-f0-9]{12}$")
_cache: dict[str, tuple[float, object]] = {}
_lock = threading.Lock()


def tenant_dir(tenant_id: str) -> Path:
    if not _TID.match(tenant_id):
        raise ValueError("invalid tenant id")
    d = config.ARTIFACT_DIR / tenant_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(tenant_id: str, name: str, obj) -> None:
    p = tenant_dir(tenant_id) / name
    tmp = p.with_suffix(p.suffix + ".tmp")
    joblib.dump(obj, tmp)
    os.replace(tmp, p)


def save_json(tenant_id: str, name: str, obj) -> None:
    p = tenant_dir(tenant_id) / name
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)


def load(tenant_id: str, name: str):
    """Cached load; re-reads only when the file changes (models are never retrained per request)."""
    p = tenant_dir(tenant_id) / name
    if not p.exists():
        return None
    m = p.stat().st_mtime
    key = str(p)
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] == m:
            return hit[1]
        obj = joblib.load(p)
        _cache[key] = (m, obj)
        return obj


def load_json(tenant_id: str, name: str):
    p = tenant_dir(tenant_id) / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def has_metrics(tenant_id: str) -> bool:
    return (config.ARTIFACT_DIR / tenant_id / "metrics.json").exists()
