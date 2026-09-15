from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.database.base import Base  # noqa: E402
from app.database.session import get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models.user import User  # noqa: E402
from app.core.security import UserRole, verify_password  # noqa: E402


def test_complete_authentication_contract() -> None:
    asyncio.run(_run_authentication_contract())


async def _run_authentication_contract() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async def override_db():
        async with session_factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            weak = await client.post("/api/v1/auth/signup", json={
                "full_name": "Test Clinician",
                "email": "clinician@example.org",
                "organization": "Retina Clinic",
                "professional_role": "clinician",
                "password": "weak-password",
                "terms_accepted": True,
            })
            assert weak.status_code == 422

            signup = await client.post("/api/v1/auth/signup", json={
                "full_name": "Test Clinician",
                "email": "Clinician@Example.org",
                "organization": "Retina Clinic",
                "professional_role": "administrator",
                "password": "SecurePrototype1!",
                "terms_accepted": True,
            })
            assert signup.status_code == 201
            assert signup.json()["user"]["email"] == "clinician@example.org"
            assert signup.json()["user"]["professional_role"] == "administrator"
            assert signup.json()["user"]["role"] == "healthcare_worker"

            async with session_factory() as db:
                stored = (await db.execute(select(User).where(User.email == "clinician@example.org"))).scalar_one()
                assert stored.password_hash != "SecurePrototype1!"
                assert verify_password("SecurePrototype1!", stored.password_hash)
                assert stored.organization == "Retina Clinic"
                assert stored.role == UserRole.HEALTHCARE_WORKER

            duplicate = await client.post("/api/v1/auth/signup", json={
                "full_name": "Duplicate User",
                "email": "CLINICIAN@example.org",
                "organization": "Retina Clinic",
                "professional_role": "researcher",
                "password": "AnotherSecure1!",
                "terms_accepted": True,
            })
            assert duplicate.status_code == 409
            assert duplicate.json()["detail"] == "An account with this email already exists"

            invalid_login = await client.post("/api/v1/auth/login", json={"email": "clinician@example.org", "password": "WrongPassword1!"})
            assert invalid_login.status_code == 401

            valid_login = await client.post("/api/v1/auth/login", json={"email": "CLINICIAN@example.org", "password": "SecurePrototype1!"})
            assert valid_login.status_code == 200
            token = valid_login.json()["access_token"]
            current = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
            assert current.status_code == 200
            assert current.json()["organization"] == "Retina Clinic"

            malformed = await client.get("/api/v1/auth/me", headers={"Authorization": "Bearer invalid-token"})
            assert malformed.status_code == 401

            known_reset = await client.post("/api/v1/auth/forgot-password", json={"email": "clinician@example.org"})
            unknown_reset = await client.post("/api/v1/auth/forgot-password", json={"email": "unknown@example.org"})
            assert known_reset.status_code == unknown_reset.status_code == 202
            assert known_reset.json() == unknown_reset.json()
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()
