from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.core.security import UserRole


class LoginRequest(BaseModel):
    # Login remains compatible with already-persisted development accounts
    # that use reserved domains; public registration still uses EmailStr.
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_login_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized.count("@") != 1 or "." not in normalized.rsplit("@", 1)[1]:
            raise ValueError("Enter a valid email address")
        return normalized


class ProfessionalRole(StrEnum):
    OPHTHALMOLOGIST = "ophthalmologist"
    OPTOMETRIST = "optometrist"
    CLINICIAN = "clinician"
    SCREENING_OPERATOR = "screening_operator"
    RESEARCHER = "researcher"
    ADMINISTRATOR = "administrator"


class SignupRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=160)
    email: EmailStr
    organization: str = Field(min_length=2, max_length=200)
    professional_role: ProfessionalRole
    password: str = Field(min_length=12, max_length=72)
    terms_accepted: bool

    @field_validator("full_name", "organization")
    @classmethod
    def strip_text(cls, value: str) -> str:
        stripped = value.strip()
        if len(stripped) < 2:
            raise ValueError("Value must contain at least 2 non-whitespace characters")
        return stripped

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must not exceed 72 UTF-8 bytes")
        checks = (
            any(character.islower() for character in value),
            any(character.isupper() for character in value),
            any(character.isdigit() for character in value),
            any(not character.isalnum() for character in value),
        )
        if not all(checks):
            raise ValueError("Password must include uppercase, lowercase, number, and symbol")
        return value

    @field_validator("terms_accepted")
    @classmethod
    def require_terms(cls, value: bool) -> bool:
        if not value:
            raise ValueError("Prototype terms and privacy notice must be accepted")
        return value


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ForgotPasswordResponse(BaseModel):
    message: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class CurrentUser(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    # Existing development databases may contain reserved-domain QA accounts.
    # New login/signup requests remain EmailStr-validated; response rendering
    # must remain backward-compatible with those already-persisted accounts.
    email: str
    full_name: str
    organization: str | None = None
    professional_role: ProfessionalRole | None = None
    role: UserRole


class SignupResponse(BaseModel):
    user: CurrentUser
    message: str
