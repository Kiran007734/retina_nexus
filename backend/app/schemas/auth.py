from uuid import UUID

from pydantic import BaseModel, EmailStr, Field

from app.core.security import UserRole


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class CurrentUser(BaseModel):
    id: UUID
    email: EmailStr
    full_name: str
    role: UserRole
