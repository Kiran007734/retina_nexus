from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.patient import Patient


async def create_patient(db: AsyncSession, anonymized_identifier: str, age_group: str | None, clinical_metadata: str | None) -> Patient:
    patient = Patient(id=uuid4(), anonymized_identifier=anonymized_identifier, age_group=age_group, clinical_metadata=clinical_metadata)
    db.add(patient)
    await db.commit()
    await db.refresh(patient)
    return patient


async def list_patients(db: AsyncSession) -> list[Patient]:
    result = await db.execute(select(Patient).order_by(Patient.created_at.desc()))
    return list(result.scalars().all())


async def get_patient(db: AsyncSession, patient_id: UUID) -> Patient | None:
    return await db.get(Patient, patient_id)
