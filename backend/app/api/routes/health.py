from fastapi import APIRouter
from sqlalchemy import text

from app.database.session import SessionLocal
from app.schemas.common import HealthResponse

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
