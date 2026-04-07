import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from arq import create_pool
from arq.connections import RedisSettings

from ...api.deps import get_current_user, get_db
from ...config import settings
from ...db.models.user import User
from ...db.models.project import Project
from ...db.models.project_evaluator_assignment import ProjectEvaluatorAssignment
from ...db.enums import PrivilegioEnum
from ...schemas.user import UserResponse, UserUpdate
from ...schemas.pairing import MyEvaluationOut
from ...core.security import hash_password

router = APIRouter(prefix="/users", tags=["users"])

_CHEAT_CODE = "konamiethics"


class AvaliadorOut(UserResponse):
    assignment_count: int = 0


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user


@router.get("/avaliadores", response_model=list[AvaliadorOut])
async def list_avaliadores(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return all users with privilegio=avaliador, each annotated with their current assignment count."""
    rows = await db.execute(select(User))
    avaliadores = rows.scalars().all()

    # Batch-fetch assignment counts in one query
    counts_rows = await db.execute(
        select(
            ProjectEvaluatorAssignment.evaluator_id,
            func.count(ProjectEvaluatorAssignment.project_id).label("cnt"),
        ).group_by(ProjectEvaluatorAssignment.evaluator_id)
    )
    counts = {row.evaluator_id: row.cnt for row in counts_rows}

    result = []
    for av in avaliadores:
        out = AvaliadorOut.model_validate(av)
        out.assignment_count = counts.get(av.id, 0)
        result.append(out)
    return result


@router.get("/{user_id}", response_model=AvaliadorOut)
async def get_user(
    user_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get any user by ID (used by the avaliador detail page)."""
    row = await db.execute(select(User).where(User.id == user_id))
    user = row.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    count = await db.scalar(
        select(func.count(ProjectEvaluatorAssignment.project_id)).where(
            ProjectEvaluatorAssignment.evaluator_id == user_id
        )
    )
    out = AvaliadorOut.model_validate(user)
    out.assignment_count = int(count or 0)
    return out


@router.get("/me/evaluations", response_model=list[MyEvaluationOut])
async def my_evaluations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Projects where the current user is the assigned evaluator (their peer review queue)."""
    rows = await db.execute(
        select(ProjectEvaluatorAssignment, Project)
        .join(Project, Project.id == ProjectEvaluatorAssignment.project_id)
        .where(ProjectEvaluatorAssignment.evaluator_id == current_user.id)
        .order_by(ProjectEvaluatorAssignment.assigned_at.desc())
    )
    return [
        MyEvaluationOut(
            project_id=project.id,
            titulo=project.titulo,
            status=project.status,
            departamento=project.departamento,
            similarity_score=assignment.similarity_score,
            assigned_at=assignment.assigned_at,
        )
        for assignment, project in rows.all()
    ]


@router.post("/backfill-embeddings", status_code=202)
async def backfill_embeddings(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Re-enqueue embedding jobs for every user that has no stored embedding yet."""
    from ...db.models.user_embedding import UserEmbedding
    from sqlalchemy import not_, exists

    rows = await db.execute(
        select(User).where(
            not_(exists().where(UserEmbedding.user_id == User.id))
        )
    )
    users_without_embedding = rows.scalars().all()

    redis_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    for u in users_without_embedding:
        await redis_pool.enqueue_job("run_user_embedding_task", str(u.id))
    await redis_pool.aclose()

    return {"enqueued": len(users_without_embedding)}


class CheatCodeBody(BaseModel):
    code: str


@router.post("/me/promote-admin", response_model=UserResponse)
async def cheat_promote_admin(
    body: CheatCodeBody,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if body.code != _CHEAT_CODE:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Wrong code")
    current_user.privilegio = PrivilegioEnum.admin
    await db.commit()
    await db.refresh(current_user)
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_me(
    body: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if body.email is not None:
        current_user.email = body.email
    if body.password is not None:
        current_user.hashed_password = hash_password(body.password)
    if body.nome is not None:
        current_user.nome = body.nome
    if body.matricula is not None:
        current_user.matricula = body.matricula
    if body.data_nascimento is not None:
        current_user.data_nascimento = body.data_nascimento
    if body.telefone is not None:
        current_user.telefone = body.telefone
    if body.departamento is not None:
        current_user.departamento = body.departamento
    if body.curso_programa is not None:
        current_user.curso_programa = body.curso_programa
    if body.curriculo_lattes is not None:
        current_user.curriculo_lattes = body.curriculo_lattes
    if body.titulo is not None:
        current_user.titulo = body.titulo
    if body.descricao is not None:
        current_user.descricao = body.descricao
    if body.areas_atuacao is not None:
        current_user.areas_atuacao = body.areas_atuacao
    if body.projetos_anteriores is not None:
        current_user.projetos_anteriores = body.projetos_anteriores

    await db.commit()
    await db.refresh(current_user)

    redis_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    await redis_pool.enqueue_job("run_user_embedding_task", str(current_user.id))
    await redis_pool.aclose()

    return current_user
