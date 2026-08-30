from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_claims
from app.core.security import create_access_token, verify_password
from app.database.session import get_db
from app.models.user import User
from app.schemas.auth import CurrentUser, LoginRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)) -> TokenResponse:
    user = (await db.execute(select(User).where(User.email == payload.email))).scalar_one_or_none()
    if user is None or not user.is_active or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return TokenResponse(access_token=create_access_token(str(user.id), user.role))


@router.get("/me", response_model=CurrentUser)
async def current_user(claims: dict = Depends(get_current_claims), db: AsyncSession = Depends(get_db)) -> CurrentUser:
    user = await db.get(User, claims.get("sub"))
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return CurrentUser.model_validate(user)
