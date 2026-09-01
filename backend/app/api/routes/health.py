from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.database.session import SessionLocal
from app.schemas.common import HealthResponse
from app.core.config import get_settings
from app.services.runtime import last_model_check, verify_models

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    database = "ok"
    try:
        async with SessionLocal() as db:
            await db.execute(text("SELECT 1"))
    except Exception:
        database = "unavailable"
    return HealthResponse(status="ok", service="retina-nexus-api", version="0.1.0", database=database)


@router.get("/ready", tags=["health"])
async def readiness_check() -> JSONResponse:
    """Return whether required runtime functionality can accept screening work."""
    database = "ok"
    try:
        async with SessionLocal() as db:
            await db.execute(text("SELECT 1"))
    except Exception:
        database = "unavailable"
    model_check = last_model_check()
    if model_check is None:
        try:
            model_check = verify_models(get_settings(), load_models=True)
        except Exception:
            model_check = {"status": "NOT_READY", "required_models_available": False, "models": {}, "optional_capabilities": {}}
    report_generator = "AVAILABLE"
    try:
        from app.api.routes.reports import _pdf_bytes
        if not callable(_pdf_bytes):
            report_generator = "UNAVAILABLE"
    except Exception:
        report_generator = "UNAVAILABLE"
    ready = database == "ok" and model_check.get("required_models_available") is True and report_generator == "AVAILABLE"
    payload = {
        "status": "READY" if ready else "NOT_READY",
        "backend_ready": True,
        "database": database,
        "classifier": model_check.get("models", {}).get("classifier", {"status": "REQUIRED_MODEL_UNAVAILABLE"}),
        "optional_models": {name: value for name, value in model_check.get("models", {}).items() if name != "classifier"},
        "report_generator": {"status": report_generator},
        "note": "Readiness verifies deployment availability and loadability only; it is not a clinical validation signal.",
    }
    return JSONResponse(status_code=200 if ready else 503, content=payload)
