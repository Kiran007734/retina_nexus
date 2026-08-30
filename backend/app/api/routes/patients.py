from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.repositories.patients import create_patient, get_patient, list_patients
from app.schemas.common import PatientCreate, PatientResponse

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("", response_model=list[PatientResponse])
async def patients(db: AsyncSession = Depends(get_db)) -> list[PatientResponse]:
    return [PatientResponse.model_validate(patient) for patient in await list_patients(db)]


@router.post("", response_model=PatientResponse, status_code=201)
async def add_patient(payload: PatientCreate, db: AsyncSession = Depends(get_db)) -> PatientResponse:
    try:
        patient = await create_patient(db, payload.anonymized_identifier, payload.age_group, payload.clinical_metadata)
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Anonymized identifier already exists") from exc
    return PatientResponse.model_validate(patient)


@router.get("/{patient_id}", response_model=PatientResponse)
async def patient(patient_id: UUID, db: AsyncSession = Depends(get_db)) -> PatientResponse:
    item = await get_patient(db, patient_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return PatientResponse.model_validate(item)
