from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from core.database import get_db
from model.models import User

router = APIRouter(prefix="/auth", tags=["Authentication"])

# ------------------------------
# Request Schemas
# ------------------------------
class SignupRequest(BaseModel):
    password: str
    name: str | None = None
    email: EmailStr
    affiliation: str | None = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# ------------------------------
# Routes
# ------------------------------
@router.post("/signup")
def signup(request: SignupRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == request.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already exists",
        )

    user = User(
        password_hash=request.password,  # ⚠️ Plain-text for now
        name=request.name,
        email=request.email,
        affiliation=request.affiliation,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return {"message": "✅ Signup successful", "user_id": user.user_id}


@router.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    # 🔍 Find user by email
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # ⚠️ Compare plain-text passwords (for now)
    if user.password_hash != request.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    return {
        "message": "✅ Login successful",
        "user": {
            "user_id": user.user_id,
            "name": user.name,
            "email": user.email,
            "affiliation": user.affiliation,
        },
    }
