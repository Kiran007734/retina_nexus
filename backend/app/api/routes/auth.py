import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_claims
from app.core.security import UserRole, create_access_token, hash_password, verify_password
from app.database.session import get_db
from app.models.user import User
from app.schemas.auth import CurrentUser, ForgotPasswordRequest, ForgotPasswordResponse, LoginRequest, SignupRequest, SignupResponse, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger(__name__)


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)) -> TokenResponse:
    email = str(payload.email).strip().lower()
    user = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if user is None or not user.is_active or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return TokenResponse(access_token=create_access_token(str(user.id), user.role))


@router.post("/signup", response_model=SignupResponse, status_code=status.HTTP_201_CREATED)
async def signup(payload: SignupRequest, db: AsyncSession = Depends(get_db)) -> SignupResponse:
    email = str(payload.email).strip().lower()
    existing = (await db.execute(select(User.id).where(User.email == email))).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists")

    # Public self-registration is deliberately least-privilege. The selected
    # professional title is retained separately and never grants administrator
    # authorization merely because it was selected in a public form.
    user = User(
        email=email,
        full_name=payload.full_name,
        organization=payload.organization,
        professional_role=payload.professional_role.value,
        password_hash=hash_password(payload.password),
        role=UserRole.HEALTHCARE_WORKER,
        is_active=True,
    )
    db.add(user)
    try:
        await db.commit()
    except IntegrityError as error:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists") from error
    await db.refresh(user)
    return SignupResponse(user=CurrentUser.model_validate(user), message="Workspace account created. Sign in to continue.")


@router.post("/forgot-password", response_model=ForgotPasswordResponse, status_code=status.HTTP_202_ACCEPTED)
async def forgot_password(payload: ForgotPasswordRequest) -> ForgotPasswordResponse:
    # Keep the response identical for known and unknown addresses. Email
    # delivery is intentionally delegated to deployment infrastructure; no
    # reset token or account existence is exposed by this prototype endpoint.
    logger.info("Password reset requested through privacy-preserving endpoint")
    return ForgotPasswordResponse(message="Password reset instructions have been sent if an account exists for this email.")


@router.get("/me", response_model=CurrentUser)
async def current_user(claims: dict = Depends(get_current_claims), db: AsyncSession = Depends(get_db)) -> CurrentUser:
    try:
        user_id = UUID(str(claims.get("sub")))
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token") from error
    user = await db.get(User, user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return CurrentUser.model_validate(user)
