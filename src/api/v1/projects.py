import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, or_
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_current_user, require_admin, get_db
from ...db.models.user import User
from ...db.models.project import Project, project_assistants, project_team
from ...db.enums import PrivilegioEnum
from ...schemas.project import (
    ProjectCreate,
    ProjectUpdate,
    ProjectResponse,
    ProjectStatusUpdate,
    ProjectMemberAdd,
)

router = APIRouter(prefix="/projects", tags=["projects"])


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_project_or_404(project_id: uuid.UUID, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project


def _can_read(project: Project, user: User) -> bool:
    if user.privilegio in (PrivilegioEnum.admin, PrivilegioEnum.avaliador):
        return True
    if project.criador_id == user.id or project.orientador_id == user.id:
        return True
    return False


def _can_edit(project: Project, user: User) -> bool:
    if user.privilegio == PrivilegioEnum.admin:
        return True
    return project.criador_id == user.id or project.orientador_id == user.id


def _assert_can_read(project: Project, user: User) -> None:
    if not _can_read(project, user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")


def _assert_can_edit(project: Project, user: User) -> None:
    if not _can_edit(project, user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Edit access denied")


def _assert_is_criador(project: Project, user: User) -> None:
    if user.privilegio == PrivilegioEnum.admin:
        return
    if project.criador_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the creator can do this")


# ── CRUD ──────────────────────────────────────────────────────────────────────

@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    body: ProjectCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = Project(
        criador_id=current_user.id,
        **body.model_dump(),
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return project


@router.get("", response_model=list[ProjectResponse])
async def list_projects(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.privilegio in (PrivilegioEnum.admin, PrivilegioEnum.avaliador):
        result = await db.execute(select(Project))
    else:
        # Projects where the user is criador, orientador, team member, or assistant
        result = await db.execute(
            select(Project).where(
                or_(
                    Project.criador_id == current_user.id,
                    Project.orientador_id == current_user.id,
                    Project.id.in_(
                        select(project_team.c.project_id).where(
                            project_team.c.user_id == current_user.id
                        )
                    ),
                    Project.id.in_(
                        select(project_assistants.c.project_id).where(
                            project_assistants.c.user_id == current_user.id
                        )
                    ),
                )
            )
        )
    return result.scalars().all()


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _get_project_or_404(project_id, db)
    _assert_can_read(project, current_user)
    return project


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: uuid.UUID,
    body: ProjectUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _get_project_or_404(project_id, db)
    _assert_can_edit(project, current_user)
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(project, field, value)
    await db.commit()
    await db.refresh(project)
    return project


@router.patch("/{project_id}/status", response_model=ProjectResponse)
async def update_project_status(
    project_id: uuid.UUID,
    body: ProjectStatusUpdate,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    project = await _get_project_or_404(project_id, db)
    project.status = body.status
    await db.commit()
    await db.refresh(project)
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _get_project_or_404(project_id, db)
    _assert_is_criador(project, current_user)
    await db.delete(project)
    await db.commit()


# ── Team management ───────────────────────────────────────────────────────────

@router.post("/{project_id}/team", status_code=status.HTTP_204_NO_CONTENT)
async def add_team_member(
    project_id: uuid.UUID,
    body: ProjectMemberAdd,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _get_project_or_404(project_id, db)
    _assert_is_criador(project, current_user)

    user_result = await db.execute(select(User).where(User.id == body.user_id))
    user = user_result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    await db.execute(
        project_team.insert().values(project_id=project_id, user_id=body.user_id)
        .on_conflict_do_nothing()
    )
    await db.commit()


@router.delete("/{project_id}/team/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_team_member(
    project_id: uuid.UUID,
    user_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _get_project_or_404(project_id, db)
    _assert_is_criador(project, current_user)
    await db.execute(
        project_team.delete().where(
            project_team.c.project_id == project_id,
            project_team.c.user_id == user_id,
        )
    )
    await db.commit()


@router.post("/{project_id}/assistants", status_code=status.HTTP_204_NO_CONTENT)
async def add_assistant(
    project_id: uuid.UUID,
    body: ProjectMemberAdd,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _get_project_or_404(project_id, db)
    _assert_is_criador(project, current_user)

    user_result = await db.execute(select(User).where(User.id == body.user_id))
    user = user_result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    await db.execute(
        project_assistants.insert().values(project_id=project_id, user_id=body.user_id)
        .on_conflict_do_nothing()
    )
    await db.commit()


@router.delete("/{project_id}/assistants/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_assistant(
    project_id: uuid.UUID,
    user_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _get_project_or_404(project_id, db)
    _assert_is_criador(project, current_user)
    await db.execute(
        project_assistants.delete().where(
            project_assistants.c.project_id == project_id,
            project_assistants.c.user_id == user_id,
        )
    )
    await db.commit()
