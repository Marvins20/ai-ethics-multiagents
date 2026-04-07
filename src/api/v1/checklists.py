import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_current_user, require_admin, get_db
from ...db.models.user import User
from ...db.models.project_type import ProjectType
from ...db.models.checklist import Checklist, QuestionGroup, Question
from ...schemas.checklist import (
    ChecklistCreate,
    ChecklistUpdate,
    ChecklistResponse,
    QuestionGroupCreate,
    QuestionGroupUpdate,
    QuestionGroupResponse,
    QuestionCreate,
    QuestionUpdate,
    QuestionResponse,
)

router = APIRouter(tags=["checklists"])


# ── Checklist (1:1 under project-type) ───────────────────────────────────────

@router.post(
    "/project-types/{project_type_id}/checklist",
    response_model=ChecklistResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_checklist(
    project_type_id: uuid.UUID,
    body: ChecklistCreate,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    pt_result = await db.execute(select(ProjectType).where(ProjectType.id == project_type_id))
    if pt_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project type not found")

    checklist = Checklist(project_type_id=project_type_id, **body.model_dump())
    db.add(checklist)
    await db.commit()
    await db.refresh(checklist)
    from sqlalchemy.orm import attributes
    attributes.set_committed_value(checklist, "groups", [])
    attributes.set_committed_value(checklist, "responses", [])
    return checklist


@router.get("/project-types/{project_type_id}/checklist", response_model=ChecklistResponse)
async def get_checklist(
    project_type_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Checklist)
        .where(Checklist.project_type_id == project_type_id)
        .options(
            selectinload(Checklist.groups).selectinload(QuestionGroup.questions)
        )
    )
    checklist = result.scalar_one_or_none()
    if checklist is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Checklist not found")
    return checklist


@router.patch("/checklists/{checklist_id}", response_model=ChecklistResponse)
async def update_checklist(
    checklist_id: uuid.UUID,
    body: ChecklistUpdate,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Checklist).where(Checklist.id == checklist_id))
    checklist = result.scalar_one_or_none()
    if checklist is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Checklist not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(checklist, field, value)
    await db.commit()
    await db.refresh(checklist)
    return checklist


@router.delete("/checklists/{checklist_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_checklist(
    checklist_id: uuid.UUID,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Checklist).where(Checklist.id == checklist_id))
    checklist = result.scalar_one_or_none()
    if checklist is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Checklist not found")
    await db.delete(checklist)
    await db.commit()


# ── Question groups ───────────────────────────────────────────────────────────

@router.post(
    "/checklists/{checklist_id}/groups",
    response_model=QuestionGroupResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_group(
    checklist_id: uuid.UUID,
    body: QuestionGroupCreate,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    cl_result = await db.execute(select(Checklist).where(Checklist.id == checklist_id))
    if cl_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Checklist not found")

    group = QuestionGroup(checklist_id=checklist_id, **body.model_dump())
    db.add(group)
    await db.commit()
    await db.refresh(group)
    from sqlalchemy.orm import attributes
    attributes.set_committed_value(group, "questions", [])
    return group


@router.patch("/question-groups/{group_id}", response_model=QuestionGroupResponse)
async def update_group(
    group_id: uuid.UUID,
    body: QuestionGroupUpdate,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(QuestionGroup).where(QuestionGroup.id == group_id))
    group = result.scalar_one_or_none()
    if group is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question group not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(group, field, value)
    await db.commit()
    await db.refresh(group)
    return group


@router.delete("/question-groups/{group_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_group(
    group_id: uuid.UUID,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(QuestionGroup).where(QuestionGroup.id == group_id))
    group = result.scalar_one_or_none()
    if group is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question group not found")
    await db.delete(group)
    await db.commit()


# ── Questions ─────────────────────────────────────────────────────────────────

@router.post(
    "/question-groups/{group_id}/questions",
    response_model=QuestionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_question(
    group_id: uuid.UUID,
    body: QuestionCreate,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    grp_result = await db.execute(select(QuestionGroup).where(QuestionGroup.id == group_id))
    if grp_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question group not found")

    question = Question(group_id=group_id, **body.model_dump())
    db.add(question)
    await db.commit()
    await db.refresh(question)
    return question


@router.patch("/questions/{question_id}", response_model=QuestionResponse)
async def update_question(
    question_id: uuid.UUID,
    body: QuestionUpdate,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Question).where(Question.id == question_id))
    question = result.scalar_one_or_none()
    if question is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(question, field, value)
    await db.commit()
    await db.refresh(question)
    return question


@router.delete("/questions/{question_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_question(
    question_id: uuid.UUID,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Question).where(Question.id == question_id))
    question = result.scalar_one_or_none()
    if question is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found")
    await db.delete(question)
    await db.commit()
