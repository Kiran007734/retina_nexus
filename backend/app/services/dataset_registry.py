import json
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetStatus
from app.models.dataset_source import DatasetSource


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
    """Seed metadata only; this never marks a dataset as acquired."""
    registry = load_registry()
    changed = False
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
    if changed:
        await db.commit()
