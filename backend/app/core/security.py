from datetime import datetime, timedelta, timezone
from enum import StrEnum

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import get_settings

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserRole(StrEnum):
    ADMIN = "admin"
    CLINICIAN = "clinician"
    HEALTHCARE_WORKER = "healthcare_worker"


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject: str, role: UserRole) -> str:
    expires = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    return jwt.encode({"sub": subject, "role": role.value, "exp": expires}, settings.secret_key, algorithm="HS256")


def decode_access_token(token: str) -> dict[str, str] | None:
    try:
        return jwt.decode(token, settings.secret_key, algorithms=["HS256"])
    except JWTError:
        return None
