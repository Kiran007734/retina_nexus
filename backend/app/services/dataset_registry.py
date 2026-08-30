import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetStatus, DatasetVersion
from app.models.dataset_source import DatasetSource
from app.models.dataset_statistics import DatasetStatistics
from app.models.dataset_validation import DatasetValidationRun


def registry_path() -> Path:
    # The source checkout keeps `backend/app` beside `ml`, while the backend
    # image copies `backend/app` to `/app/app` and `ml` to `/app/ml`. Resolve
    # both layouts without relying on a host-specific absolute path.
    candidates = (
        Path(__file__).resolve().parents[3] / "ml" / "datasets" / "metadata" / "dataset_registry.json",
        Path(__file__).resolve().parents[2] / "ml" / "datasets" / "metadata" / "dataset_registry.json",
        Path.cwd() / "ml" / "datasets" / "metadata" / "dataset_registry.json",
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def load_registry() -> dict:
    path = registry_path()
    if not path.exists():
        return {"registry_version": "unavailable", "datasets": []}
    return json.loads(path.read_text(encoding="utf-8"))


async def seed_registry(db: AsyncSession) -> None:
    """Seed registry metadata and synchronize verified local reports.

    Raw datasets are never committed to Git. When governance scripts have
    produced reports in the local checkout, their measured counts are exposed
    through the API; otherwise the dataset remains explicitly not acquired.
    """
    registry = load_registry()
    changed = False
    root = Path(__file__).resolve().parents[3]
    for definition in registry.get("datasets", []):
        existing = (await db.execute(select(Dataset).where(Dataset.slug == definition["slug"]))).scalar_one_or_none()
        if existing is None:
            dataset = Dataset(
                slug=definition["slug"], name=definition["name"], purpose=definition["purpose"],
                status=DatasetStatus.NOT_ACQUIRED, raw_path=definition["raw_path"],
                registry_metadata=definition,
            )
            db.add(dataset)
            await db.flush()
            source_type = "kaggle" if definition.get("kaggle_competition") else "manual"
            db.add(DatasetSource(
                dataset_id=dataset.id, source_type=source_type,
                source_uri=definition.get("kaggle_competition"),
                access_notes=definition.get("manual_setup"), acquisition_status="not_started",
            ))
            changed = True
        else:
            dataset = existing

        raw_path = root / definition["raw_path"]
        report_dir = root / "ml" / "datasets" / "metadata" / "reports" / definition["slug"]
        validation_path = report_dir / "dataset_validation_report.json"
        statistics_path = report_dir / "dataset_statistics.json"
        if not raw_path.exists() or not validation_path.is_file() or not statistics_path.is_file():
            continue
        try:
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        files = validation.get("files", {})
        labels = validation.get("labels", {})
        fingerprint = files.get("scan_fingerprint") or "unversioned"
        version_name = f"{definition['slug']}-inventory-{fingerprint[:12]}"
        dataset.status = DatasetStatus.AVAILABLE if validation.get("status") == "pass" else DatasetStatus.BLOCKED
        dataset.registry_metadata = {**(dataset.registry_metadata or definition), "last_verified_version": version_name, "last_verified_report": str(validation_path.relative_to(root))}
        source = (await db.execute(select(DatasetSource).where(DatasetSource.dataset_id == dataset.id).limit(1))).scalar_one_or_none()
        if source is not None:
            source.acquisition_status = "available"
        version = (await db.execute(select(DatasetVersion).where(DatasetVersion.dataset_id == dataset.id, DatasetVersion.version == version_name))).scalar_one_or_none()
        if version is None:
            version = DatasetVersion(dataset_id=dataset.id, version=version_name)
            db.add(version)
            await db.flush()
        version.file_count = int(files.get("total_files", 0))
        version.image_count = int(files.get("total_files", 0))
        version.label_count = int(labels.get("annotation_rows", 0))
        version.manifest_path = f"ml/datasets/metadata/splits/{definition['slug']}/splits.json"
        version.acquired_at = version.acquired_at or datetime.now(timezone.utc)
        stats = (await db.execute(select(DatasetStatistics).where(DatasetStatistics.dataset_version_id == version.id).order_by(desc(DatasetStatistics.created_at)).limit(1))).scalar_one_or_none()
        if stats is None:
            stats = DatasetStatistics(dataset_version_id=version.id)
            db.add(stats)
        stats.total_files = int(files.get("total_files", 0))
        stats.readable_files = int(files.get("readable_files", 0))
        stats.corrupted_files = int(files.get("corrupted_files", 0))
        stats.duplicate_exact_count = int(files.get("duplicate_exact_count", statistics.get("duplicate_exact_count", 0)))
        stats.duplicate_perceptual_count = int(files.get("duplicate_perceptual_count", statistics.get("duplicate_perceptual_count", 0)))
        stats.class_distribution = labels.get("class_distribution")
        stats.resolution_statistics = files.get("resolution_statistics", statistics.get("resolution_statistics"))
        stats.metadata_completeness = labels.get("metadata_completeness", statistics.get("metadata_completeness"))
        stats.readiness_score = (validation.get("readiness") or {}).get("score")
        run = (await db.execute(select(DatasetValidationRun).where(DatasetValidationRun.dataset_version_id == version.id).order_by(desc(DatasetValidationRun.created_at)).limit(1))).scalar_one_or_none()
        if run is None:
            run = DatasetValidationRun(dataset_version_id=version.id)
            db.add(run)
        run.status = validation.get("status", "unknown")
        run.report_path = str(validation_path.relative_to(root))
        run.leakage_report_path = "ml/datasets/metadata/data_leakage_report.json"
        run.summary = {"labels": labels, "duplicates": {"exact": files.get("duplicate_exact_count", 0), "perceptual": files.get("duplicate_perceptual_count", 0), "conflicting_groups": len(validation.get("duplicate_label_conflicts", []))}, "training_eligibility": validation.get("training_eligibility"), "leakage": validation.get("leakage")}
        run.readiness_score = (validation.get("readiness") or {}).get("score")
        run.started_at = run.started_at or datetime.now(timezone.utc)
        run.completed_at = datetime.now(timezone.utc)
        changed = True
    if changed:
        await db.commit()
