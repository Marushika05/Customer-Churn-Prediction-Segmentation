"""Tenant management (admin-token protected)."""
from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field

from .. import db
from .deps import admin_auth

router = APIRouter(prefix="/tenants", tags=["tenants"], dependencies=[Depends(admin_auth)])


class TenantIn(BaseModel):
    name: str = Field(min_length=3, max_length=40)


@router.post("", status_code=201)
def create(body: TenantIn):
    tid, key = db.create_tenant(body.name)
    return {"tenant_id": tid, "name": body.name.strip(), "api_key": key,
            "note": "Store this key now; only its hash is kept."}


@router.get("")
def list_all():
    return db.list_tenants()


@router.delete("/{tenant_id}")
def delete(tenant_id: str = Path(..., pattern=db.TENANT_ID_RE)):
    db.delete_tenant(tenant_id)
    return {"deleted": tenant_id}


@router.post("/{tenant_id}/rotate-key")
def rotate(tenant_id: str = Path(..., pattern=db.TENANT_ID_RE)):
    return {"tenant_id": tenant_id, "api_key": db.rotate_key(tenant_id)}
