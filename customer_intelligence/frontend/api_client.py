"""Thin HTTP client for the FastAPI backend + local key store (tenant API keys live on this machine only)."""
import json

import requests

from backend import config


class ApiError(Exception):
    pass


def _req(method: str, path: str, headers: dict, **kw):
    try:
        r = requests.request(method, config.API_URL + path, headers=headers, timeout=kw.pop("timeout", 180), **kw)
    except requests.ConnectionError:
        raise ApiError(f"Cannot reach the API at {config.API_URL}. Start it with start_api.bat")
    except requests.Timeout:
        raise ApiError("The API request timed out")
    if r.status_code >= 400:
        try:
            d = r.json().get("detail", r.text)
        except ValueError:
            d = r.text
        if isinstance(d, list):  # pydantic validation errors
            d = "; ".join(f"{'.'.join(map(str, e.get('loc', [])[1:]))}: {e.get('msg')}" for e in d)
        raise ApiError(str(d))
    return r.json()


def load_keys() -> dict:
    return json.loads(config.KEYSTORE_PATH.read_text()) if config.KEYSTORE_PATH.exists() else {}


def save_key(tenant_id: str, name: str, key: str) -> None:
    keys = load_keys()
    keys[tenant_id] = {"name": name, "key": key}
    config.KEYSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.KEYSTORE_PATH.write_text(json.dumps(keys, indent=1))


def admin(method: str, path: str, **kw):
    return _req(method, path, {"X-Admin-Token": config.ADMIN_TOKEN}, **kw)


def tenant(tid: str, method: str, path: str, **kw):
    key = load_keys().get(tid, {}).get("key")
    if not key:
        raise ApiError("No API key stored for this tenant. Open Tenant Management and regenerate its key.")
    return _req(method, f"/tenants/{tid}{path}", {"X-API-Key": key}, **kw)


def health() -> dict | None:
    try:
        return requests.get(config.API_URL + "/health", timeout=3).json()
    except Exception:  # noqa: BLE001
        return None
