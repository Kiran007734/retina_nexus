from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import analytics, auth, datasets, demo, health, images, models, monitoring, patients, reports, reviews, screening
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.database.base import Base
from app.database.session import engine
from app.database.session import SessionLocal
from app.services.dataset_registry import seed_registry
from app.services.model_registry import seed_model_registry
import app.models  # noqa: F401 - registers all SQLAlchemy models

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.environment in {"development", "test"}:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with SessionLocal() as db:
            await seed_registry(db)
            await seed_model_registry(db)
    logger.info("retina_nexus.startup", extra={"environment": settings.environment})
    yield
    logger.info("retina_nexus.shutdown")


app = FastAPI(
    title="RETINA-NEXUS API",
    description="Privacy-conscious foundation for explainable diabetic retinopathy screening.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_prefix = "/api/v1"
app.include_router(health.router, prefix=api_prefix)
app.include_router(auth.router, prefix=api_prefix)
app.include_router(images.router, prefix=api_prefix)
app.include_router(screening.router, prefix=api_prefix)
app.include_router(patients.router, prefix=api_prefix)
app.include_router(reports.router, prefix=api_prefix)
app.include_router(reviews.router, prefix=api_prefix)
app.include_router(models.router, prefix=api_prefix)
app.include_router(monitoring.router, prefix=api_prefix)
app.include_router(datasets.router, prefix=api_prefix)
app.include_router(analytics.router, prefix=api_prefix)
app.include_router(demo.router, prefix=api_prefix)


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    return {"name": "RETINA-NEXUS API", "status": "online", "docs": "/docs"}
