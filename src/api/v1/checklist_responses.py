import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_current_user, get_db
from ...db.models.user import User
from ...db.models.project import Project
from ...db.models.checklist_response import ChecklistResponse
from ...db.enums import PrivilegioEnum
from ...schemas.checklist_response import (
    ChecklistResponseCreate,
    ChecklistResponseUpdate,
    ChecklistResponseOut,
)

router = APIRouter(tags=["checklist-responses"])


@router.post(
    "/projects/{project_id}/responses",
    response_model=ChecklistResponseOut,
    status_code=status.HTTP_201_CREATED,
)
async def submit_response(
    project_id: uuid.UUID,
    body: ChecklistResponseCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_project_or_404(project_id, db)

    # Upsert: if the user already answered this question for this project, update it
    stmt = (
        pg_insert(ChecklistResponse)
        .values(
            id=uuid.uuid4(),
            checklist_id=body.checklist_id,
            project_id=project_id,
            question_id=body.question_id,
            user_id=current_user.id,
            resposta=body.resposta,
        )
        .on_conflict_do_update(
            constraint="uq_response_per_user_project_question",
            set_={"resposta": body.resposta},
        )
        .returning(ChecklistResponse)
    )
    result = await db.execute(stmt)
    await db.commit()
    return result.scalar_one()


@router.get("/projects/{project_id}/responses", response_model=list[ChecklistResponseOut])
async def list_responses(
    project_id: uuid.UUID,
    checklist_id: uuid.UUID | None = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_project_or_404(project_id, db)

    query = select(ChecklistResponse).where(ChecklistResponse.project_id == project_id)

    # Non-admin users only see their own responses
    if current_user.privilegio != PrivilegioEnum.admin:
        query = query.where(ChecklistResponse.user_id == current_user.id)

    if checklist_id is not None:
        query = query.where(ChecklistResponse.checklist_id == checklist_id)

    result = await db.execute(query)
    return result.scalars().all()


@router.patch("/responses/{response_id}", response_model=ChecklistResponseOut)
async def update_response(
    response_id: uuid.UUID,
    body: ChecklistResponseUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    response = await _get_response_or_404(response_id, db)
    if response.user_id != current_user.id and current_user.privilegio != PrivilegioEnum.admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(response, field, value)
    await db.commit()
    await db.refresh(response)
    return response


@router.delete("/responses/{response_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_response(
    response_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    response = await _get_response_or_404(response_id, db)
    if response.user_id != current_user.id and current_user.privilegio != PrivilegioEnum.admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    await db.delete(response)
    await db.commit()


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_project_or_404(project_id: uuid.UUID, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project


async def _get_response_or_404(response_id: uuid.UUID, db: AsyncSession) -> ChecklistResponse:
    result = await db.execute(
        select(ChecklistResponse).where(ChecklistResponse.id == response_id)
    )
    response = result.scalar_one_or_none()
    if response is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Response not found")
    return response
