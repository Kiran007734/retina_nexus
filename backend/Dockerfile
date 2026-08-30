FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY backend/requirements.txt backend/requirements-ml.txt ./
# The image can serve a registered classifier when an authorized checkpoint is
# mounted; no checkpoint is bundled or assumed by this build.
RUN pip install --no-cache-dir -r requirements-ml.txt
COPY backend/app ./app
COPY backend/alembic.ini ./alembic.ini
COPY backend/alembic ./alembic
# Include the metadata registry and empty directory layout in the image. Real
# datasets remain external and are never bundled by this repository.
COPY ml ./ml

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
