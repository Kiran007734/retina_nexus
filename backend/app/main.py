from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import inspect

from app.api.routes import analytics, auth, datasets, demo, health, images, models, monitoring, patients, reports, reviews, screening
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.safe_errors import safe_error_message
from app.database.base import Base
from app.database.session import engine
from app.database.session import SessionLocal
from app.services.dataset_registry import seed_registry
from app.services.model_registry import seed_model_registry
from app.services.runtime import ensure_runtime_directories, verify_models
from app.core.logging import request_id_context
import app.models  # noqa: F401 - registers all SQLAlchemy models

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


def _repair_legacy_sqlite_schema(connection) -> None:
    """Add nullable/defaulted columns to old local databases.

    Development databases were historically created with ``create_all`` and
    therefore may not have an Alembic version table.  Fresh and production
    databases still use the normal migration path; this small idempotent
    compatibility repair keeps an existing local prototype database from
    failing during registry seeding after a schema addition.
    """
    if connection.dialect.name != "sqlite":
        return
    inspector = inspect(connection)
    for table in Base.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue
        existing = {column["name"] for column in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing or column.primary_key:
                continue
            if not column.nullable and column.server_default is None:
                logger.warning(
                    "retina_nexus.schema_column_skipped",
                    extra={"event": "retina_nexus.schema_column_skipped", "table": table.name, "column": column.name},
                )
                continue
            column_type = column.type.compile(dialect=connection.dialect)
            statement = f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {column_type}'
            if column.server_default is not None:
                default_text = getattr(column.server_default.arg, "text", str(column.server_default.arg))
                if default_text.upper() == "CURRENT_TIMESTAMP":
                    statement += " DEFAULT CURRENT_TIMESTAMP"
                else:
                    escaped = default_text.replace("'", "''")
                    statement += f" DEFAULT '{escaped}'"
            elif column.nullable:
                statement += " NULL"
            else:
                statement += " NOT NULL"
            connection.exec_driver_sql(statement)
            logger.info(
                "retina_nexus.schema_column_repaired",
                extra={"event": "retina_nexus.schema_column_repaired", "table": table.name, "column": column.name},
            )


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_runtime_directories(settings)
    if settings.environment in {"development", "test"}:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            await connection.run_sync(_repair_legacy_sqlite_schema)
        async with SessionLocal() as db:
            await seed_registry(db)
            await seed_model_registry(db)
    if settings.verify_models_on_startup:
        try:
            model_check = verify_models(settings, load_models=True, load_optional_models=False, verify_optional_checksums=False)
            logger.info("retina_nexus.models_verified", extra={"event": "retina_nexus.models_verified", "status_code": 200 if model_check["required_models_available"] else 503})
        except Exception:
            logger.exception("retina_nexus.models_verification_failed", extra={"event": "retina_nexus.models_verification_failed"})
    logger.info("retina_nexus.startup", extra={"event": "retina_nexus.startup"})
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


@app.middleware("http")
async def request_tracing_middleware(request: Request, call_next):
    supplied = request.headers.get("X-Request-ID", "")
    request_id = supplied[:80] if supplied and supplied.replace("-", "").replace("_", "").isalnum() else str(uuid4())
    token = request_id_context.set(request_id)
    request.state.request_id = request_id
    started = perf_counter()
    try:
        response = await call_next(request)
    finally:
        duration_ms = round((perf_counter() - started) * 1000, 3)
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        logger.info("api.request", extra={"event": "api.request", "method": request.method, "path": route_path, "status_code": getattr(locals().get("response"), "status_code", 500), "duration_ms": duration_ms, "request_id": request_id})
        request_id_context.reset(token)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def request_validation_error(_: Request, __: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": "Request validation failed", "error_code": "REQUEST_VALIDATION_ERROR"})


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict):
        message = safe_error_message(detail.get("message"), "Request could not be processed")
        error_code = str(detail.get("code", "HTTP_ERROR"))
    else:
        message = safe_error_message(detail, "Request could not be processed")
        error_code = {404: "NOT_FOUND", 409: "CONFLICT", 413: "PAYLOAD_TOO_LARGE", 415: "UNSUPPORTED_MEDIA_TYPE", 422: "INVALID_REQUEST", 503: "SERVICE_UNAVAILABLE"}.get(exc.status_code, "HTTP_ERROR")
    return JSONResponse(status_code=exc.status_code, content={"detail": message, "error_code": error_code})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("api.unhandled_exception", extra={"event": "api.unhandled_exception", "error_code": "INTERNAL_ERROR"})
    return JSONResponse(status_code=500, content={"detail": "An internal server error occurred", "error_code": "INTERNAL_ERROR", "request_id": getattr(request.state, "request_id", "-")})

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
