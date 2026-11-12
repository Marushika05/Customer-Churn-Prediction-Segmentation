"""FastAPI application entry point:  uvicorn backend.api.main:app"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .. import artifacts, config, db, ingestion, inference
from . import routes_admin, routes_tenant


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="Multi-Tenant Customer Intelligence & Risk Service", version="1.0", lifespan=lifespan)
app.include_router(routes_admin.router)
app.include_router(routes_tenant.router)


def _err(code: int):
    async def handler(_: Request, exc: Exception):
        return JSONResponse(status_code=code, content={"detail": str(exc.args[0]) if exc.args else str(exc)})
    return handler


app.add_exception_handler(ingestion.IngestError, _err(422))
app.add_exception_handler(db.NotFound, _err(404))
app.add_exception_handler(db.Conflict, _err(409))
app.add_exception_handler(inference.ModelsUnavailable, _err(409))
app.add_exception_handler(ValueError, _err(422))


@app.get("/health", tags=["health"])
def health():
    try:
        with db.connect() as c:
            n = c.execute("SELECT COUNT(*) FROM tenants").fetchone()[0]
        return {"status": "ok", "database": "ok", "tenants": n, "raw_demo_data_present": config.RAW_DATA_PATH.exists()}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=503, content={"status": "error", "database": str(e)})
