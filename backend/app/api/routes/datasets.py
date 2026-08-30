from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.models.dataset import Dataset, DatasetVersion
from app.models.dataset_statistics import DatasetStatistics
from app.models.dataset_validation import DatasetValidationRun
from app.schemas.datasets import DatasetResponse, DatasetStatisticsResponse, DatasetValidationResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _availability_status(dataset: Dataset, version: DatasetVersion | None, stats: DatasetStatistics | None) -> str:
    """Expose the four-state data availability contract without changing lifecycle state."""
    if dataset.status.value == "blocked":
        return "INVALID"
    if version is None or (version.image_count or 0) == 0:
        return "MISSING"
    if stats is None:
        return "PARTIALLY AVAILABLE"
    if stats.corrupted_files > 0 or stats.readable_files < stats.total_files:
        return "PARTIALLY AVAILABLE"
    return "AVAILABLE"


async def _get_dataset_or_404(dataset_id: UUID, db: AsyncSession) -> Dataset:
    dataset = await db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


async def _latest_version(dataset_id: UUID, db: AsyncSession) -> DatasetVersion | None:
    return (await db.execute(select(DatasetVersion).where(DatasetVersion.dataset_id == dataset_id).order_by(desc(DatasetVersion.created_at)).limit(1))).scalar_one_or_none()


async def _to_response(dataset: Dataset, db: AsyncSession) -> DatasetResponse:
    version = await _latest_version(dataset.id, db)
    stats = None
    if version:
        stats = (await db.execute(select(DatasetStatistics).where(DatasetStatistics.dataset_version_id == version.id).order_by(desc(DatasetStatistics.created_at)).limit(1))).scalar_one_or_none()
    return DatasetResponse(
        id=dataset.id, slug=dataset.slug, name=dataset.name, purpose=dataset.purpose,
        status=dataset.status.value, raw_path=dataset.raw_path,
        availability_status=_availability_status(dataset, version, stats),
        latest_version=version.version if version else None,
        image_count=(stats.total_files if stats else version.image_count if version else None),
        readiness_score=stats.readiness_score if stats else None,
        created_at=dataset.created_at,
    )


@router.get("", response_model=list[DatasetResponse])
async def list_datasets(db: AsyncSession = Depends(get_db)) -> list[DatasetResponse]:
    datasets = (await db.execute(select(Dataset).order_by(Dataset.name))).scalars().all()
    return [await _to_response(dataset, db) for dataset in datasets]


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: UUID, db: AsyncSession = Depends(get_db)) -> DatasetResponse:
    return await _to_response(await _get_dataset_or_404(dataset_id, db), db)


@router.get("/{dataset_id}/statistics", response_model=DatasetStatisticsResponse)
async def get_statistics(dataset_id: UUID, db: AsyncSession = Depends(get_db)) -> DatasetStatisticsResponse:
    await _get_dataset_or_404(dataset_id, db)
    version = await _latest_version(dataset_id, db)
    stats = None
    if version:
        stats = (await db.execute(select(DatasetStatistics).where(DatasetStatistics.dataset_version_id == version.id).order_by(desc(DatasetStatistics.created_at)).limit(1))).scalar_one_or_none()
    if stats is None:
        return DatasetStatisticsResponse(dataset_id=dataset_id, dataset_version=version.version if version else None, total_files=0, readable_files=0, corrupted_files=0, duplicate_exact_count=0, duplicate_perceptual_count=0)
    return DatasetStatisticsResponse(
        dataset_id=dataset_id, dataset_version=version.version if version else None,
        total_files=stats.total_files, readable_files=stats.readable_files,
        corrupted_files=stats.corrupted_files, duplicate_exact_count=stats.duplicate_exact_count,
        duplicate_perceptual_count=stats.duplicate_perceptual_count, class_distribution=stats.class_distribution,
        resolution_statistics=stats.resolution_statistics, metadata_completeness=stats.metadata_completeness,
        readiness_score=stats.readiness_score, created_at=stats.created_at,
    )


@router.get("/{dataset_id}/validation", response_model=DatasetValidationResponse)
async def get_validation(dataset_id: UUID, db: AsyncSession = Depends(get_db)) -> DatasetValidationResponse:
    await _get_dataset_or_404(dataset_id, db)
    version = await _latest_version(dataset_id, db)
    run = None
    if version:
        run = (await db.execute(select(DatasetValidationRun).where(DatasetValidationRun.dataset_version_id == version.id).order_by(desc(DatasetValidationRun.created_at)).limit(1))).scalar_one_or_none()
    if run is None:
        return DatasetValidationResponse(dataset_id=dataset_id, dataset_version=version.version if version else None, status="not_run")
    return DatasetValidationResponse(
        dataset_id=dataset_id, dataset_version=version.version if version else None,
        validation_run_id=run.id, status=run.status, report_path=run.report_path,
        leakage_report_path=run.leakage_report_path, summary=run.summary,
        readiness_score=run.readiness_score, started_at=run.started_at, completed_at=run.completed_at,
    )
