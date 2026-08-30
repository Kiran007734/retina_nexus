from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.models.model_version import ModelVersion
from app.schemas.common import ModelVersionResponse
from app.services.model_registry import model_response_payload

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelVersionResponse])
async def registered_models(db: AsyncSession = Depends(get_db)) -> list[ModelVersionResponse]:
    items = (await db.execute(select(ModelVersion).order_by(ModelVersion.created_at.desc()))).scalars().all()
    return [ModelVersionResponse.model_validate(model_response_payload(item)) for item in items]
