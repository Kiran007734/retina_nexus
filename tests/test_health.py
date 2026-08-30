import os

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_retina_nexus.db")
os.environ.setdefault("ENVIRONMENT", "test")

from fastapi.testclient import TestClient

from app.main import app


def test_root_endpoint():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "RETINA-NEXUS API"


def test_health_endpoint():
    response = TestClient(app).get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_dataset_registry_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/v1/datasets")
    assert response.status_code == 200
    assert {item["slug"] for item in response.json()} == {"aptos2019", "idrid", "drive", "messidor"}


def test_cors_allows_local_frontend():
    with TestClient(app) as client:
        response = client.get("/api/v1/health", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
