from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from arq import create_pool
from arq.connections import RedisSettings

from ...core.security import hash_password, verify_password, create_access_token
from ...config import settings
from ...db.session import get_db
from ...db.models.user import User
from ...db.enums import PrivilegioEnum
from ...schemas.user import UserCreate, UserResponse, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(body: UserCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        privilegio=PrivilegioEnum.admin,
        nome=body.nome,
        matricula=body.matricula,
        data_nascimento=body.data_nascimento,
        telefone=body.telefone,
        departamento=body.departamento,
        curso_programa=body.curso_programa,
        curriculo_lattes=body.curriculo_lattes,
        titulo=body.titulo,
        descricao=body.descricao,
        areas_atuacao=body.areas_atuacao,
        projetos_anteriores=body.projetos_anteriores,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    redis_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    await redis_pool.enqueue_job("run_user_embedding_task", str(user.id))
    await redis_pool.aclose()

    return user


@router.post("/login", response_model=TokenResponse)
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == form.username))
    user = result.scalar_one_or_none()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return TokenResponse(access_token=create_access_token(user.id))
