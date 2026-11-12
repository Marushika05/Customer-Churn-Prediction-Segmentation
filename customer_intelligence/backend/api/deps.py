"""Authentication & tenant-boundary enforcement. Every tenant route depends on `tenant_auth`."""
import hmac

from fastapi import Header, HTTPException, Path

from .. import config, db


def admin_auth(x_admin_token: str | None = Header(default=None)) -> None:
    if not x_admin_token or not hmac.compare_digest(x_admin_token, config.ADMIN_TOKEN):
        raise HTTPException(401, "Missing or invalid admin token (X-Admin-Token)")


def tenant_auth(tenant_id: str = Path(..., pattern=db.TENANT_ID_RE), x_api_key: str | None = Header(default=None)) -> str:
    """The API key must belong to the tenant in the URL. Same 403 for unknown tenant / wrong key (no enumeration)."""
    if not db.verify_key(tenant_id, x_api_key):
        raise HTTPException(403, "Invalid tenant or API key (X-API-Key)")
    return tenant_id
