import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, and_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from arq import create_pool
from arq.connections import RedisSettings

from ...api.deps import get_current_user, get_db
from ...config import settings
from ...db.models.user import User
from ...db.models.project import Project
from ...db.models.questionnaire_job import QuestionnaireJob
from ...db.models.questionnaire_response import QuestionnaireResponse
from ...schemas.questionnaire import (
    QuestionnaireJobResponse,
    QuestionnaireResponseCreate,
    QuestionnaireResponseOut,
)

router = APIRouter(tags=["questionnaires"])


async def _get_project(project_id: uuid.UUID, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.post(
    "/projects/{project_id}/questionnaire",
    response_model=QuestionnaireJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_questionnaire(
    project_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_project(project_id, db)

    # If there's already a pending/running job, return it
    existing = await db.execute(
        select(QuestionnaireJob)
        .where(
            QuestionnaireJob.project_id == project_id,
            QuestionnaireJob.status.in_(["pending", "running"]),
        )
        .order_by(QuestionnaireJob.created_at.desc())
        .limit(1)
    )
    active_job = existing.scalar_one_or_none()
    if active_job:
        return active_job

    job = QuestionnaireJob(project_id=project_id, status="pending")
    db.add(job)
    await db.commit()
    await db.refresh(job)
    from sqlalchemy.orm import attributes
    attributes.set_committed_value(job, "responses", [])

    redis_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    await redis_pool.enqueue_job("run_questionnaire_task", str(job.id), str(project_id))
    await redis_pool.aclose()

    return job


@router.get(
    "/projects/{project_id}/questionnaire",
    response_model=QuestionnaireJobResponse | None,
)
async def get_questionnaire(
    project_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_project(project_id, db)
    result = await db.execute(
        select(QuestionnaireJob)
        .where(QuestionnaireJob.project_id == project_id)
        .order_by(QuestionnaireJob.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


@router.post(
    "/projects/{project_id}/questionnaire/responses",
    response_model=QuestionnaireResponseOut,
)
async def upsert_questionnaire_response(
    project_id: uuid.UUID,
    body: QuestionnaireResponseCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_project(project_id, db)

    # Check job belongs to project
    job_result = await db.execute(
        select(QuestionnaireJob).where(
            QuestionnaireJob.id == body.questionnaire_job_id,
            QuestionnaireJob.project_id == project_id,
        )
    )
    if job_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Questionnaire job not found")

    now = datetime.now(timezone.utc)
    stmt = (
        insert(QuestionnaireResponse)
        .values(
            questionnaire_job_id=body.questionnaire_job_id,
            project_id=project_id,
            question_idx=body.question_idx,
            user_id=current_user.id,
            resposta=body.resposta,
            created_at=now,
            updated_at=now,
        )
        .on_conflict_do_update(
            constraint="uq_questionnaire_response",
            set_={"resposta": body.resposta, "updated_at": now},
        )
        .returning(QuestionnaireResponse)
    )
    result = await db.execute(stmt)
    await db.commit()
    row = result.scalar_one()
    return row


@router.get(
    "/projects/{project_id}/questionnaire/responses",
    response_model=list[QuestionnaireResponseOut],
)
async def list_questionnaire_responses(
    project_id: uuid.UUID,
    questionnaire_job_id: uuid.UUID | None = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_project(project_id, db)

    filters = [QuestionnaireResponse.project_id == project_id]
    if questionnaire_job_id:
        filters.append(QuestionnaireResponse.questionnaire_job_id == questionnaire_job_id)

    result = await db.execute(
        select(QuestionnaireResponse).where(and_(*filters))
    )
    return result.scalars().all()
