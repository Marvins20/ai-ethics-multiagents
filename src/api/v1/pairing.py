import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from arq import create_pool
from arq.connections import RedisSettings

from ...api.deps import get_current_user, get_db
from ...config import settings
from ...db.models.user import User
from ...db.models.project import Project
from ...db.models.checklist import Checklist, Question, QuestionGroup
from ...db.models.checklist_response import ChecklistResponse
from ...db.models.questionnaire_job import QuestionnaireJob
from ...db.models.questionnaire_response import QuestionnaireResponse
from ...db.models.pairing_job import PairingJob
from ...db.models.project_evaluator_assignment import ProjectEvaluatorAssignment
from ...schemas.pairing import PairingJobResponse, EvaluatorAssignmentOut, AutoavaliacaoCompleteResponse

router = APIRouter(tags=["pairing"])


async def _get_project_or_404(project_id: uuid.UUID, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


async def _check_all_answered(
    project_id: uuid.UUID,
    project_type_id: uuid.UUID | None,
    user_id: uuid.UUID,
    db: AsyncSession,
) -> bool:
    checklist_complete = False
    if project_type_id is not None:
        checklist_row = await db.execute(
            select(Checklist).where(Checklist.project_type_id == project_type_id)
        )
        checklist = checklist_row.scalar_one_or_none()
        if checklist is not None:
            total_q = await db.scalar(
                select(func.count(Question.id))
                .join(QuestionGroup, Question.group_id == QuestionGroup.id)
                .where(QuestionGroup.checklist_id == checklist.id)
            )
            answered_q = await db.scalar(
                select(func.count(ChecklistResponse.id)).where(
                    ChecklistResponse.project_id == project_id,
                    ChecklistResponse.checklist_id == checklist.id,
                    ChecklistResponse.user_id == user_id,
                    ChecklistResponse.resposta.isnot(None),
                )
            )
            checklist_complete = (total_q or 0) > 0 and answered_q == total_q

    q_job_row = await db.execute(
        select(QuestionnaireJob)
        .where(QuestionnaireJob.project_id == project_id, QuestionnaireJob.status == "completed")
        .order_by(QuestionnaireJob.created_at.desc())
        .limit(1)
    )
    q_job = q_job_row.scalar_one_or_none()

    questionnaire_complete = False
    if q_job is not None and q_job.result:
        total_qitems = len(q_job.result)
        answered_qitems = await db.scalar(
            select(func.count(QuestionnaireResponse.id)).where(
                QuestionnaireResponse.project_id == project_id,
                QuestionnaireResponse.questionnaire_job_id == q_job.id,
                QuestionnaireResponse.user_id == user_id,
                QuestionnaireResponse.resposta.isnot(None),
            )
        )
        questionnaire_complete = total_qitems > 0 and answered_qitems == total_qitems

    return checklist_complete and questionnaire_complete


@router.get("/projects/{project_id}/autoavaliacao-complete", response_model=AutoavaliacaoCompleteResponse)
async def check_autoavaliacao_complete(
    project_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _get_project_or_404(project_id, db)
    if project.criador_id is None:
        return AutoavaliacaoCompleteResponse(complete=False)
    complete = await _check_all_answered(project_id, project.project_type_id, project.criador_id, db)
    return AutoavaliacaoCompleteResponse(complete=complete)


@router.get("/projects/{project_id}/par-complete", response_model=AutoavaliacaoCompleteResponse)
async def check_par_complete(
    project_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _get_project_or_404(project_id, db)
    assignment_row = await db.execute(
        select(ProjectEvaluatorAssignment)
        .where(ProjectEvaluatorAssignment.project_id == project_id)
    )
    assignment = assignment_row.scalar_one_or_none()
    if assignment is None:
        return AutoavaliacaoCompleteResponse(complete=False)
    complete = await _check_all_answered(project_id, project.project_type_id, assignment.evaluator_id, db)
    return AutoavaliacaoCompleteResponse(complete=complete)


@router.post(
    "/projects/{project_id}/pairing",
    response_model=PairingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_pairing(
    project_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_project_or_404(project_id, db)

    existing = await db.execute(
        select(PairingJob).where(
            PairingJob.project_id == project_id,
            PairingJob.status.in_(["pending", "running"]),
        )
    )
    active = existing.scalar_one_or_none()
    if active:
        return active

    job = PairingJob(project_id=project_id, status="pending")
    db.add(job)
    await db.commit()
    await db.refresh(job)

    redis_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    await redis_pool.enqueue_job("run_pairing_task", str(job.id), str(project_id))
    await redis_pool.aclose()

    return job


@router.get(
    "/projects/{project_id}/pairing",
    response_model=PairingJobResponse | None,
)
async def get_pairing_job(
    project_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the latest pairing job for this project."""
    await _get_project_or_404(project_id, db)
    result = await db.execute(
        select(PairingJob)
        .where(PairingJob.project_id == project_id)
        .order_by(PairingJob.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


@router.get(
    "/projects/{project_id}/pairing/assignments",
    response_model=list[EvaluatorAssignmentOut],
)
async def get_pairing_assignments(
    project_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get evaluator assignments for this project, including evaluator name/email."""
    await _get_project_or_404(project_id, db)
    result = await db.execute(
        select(ProjectEvaluatorAssignment, User)
        .join(User, User.id == ProjectEvaluatorAssignment.evaluator_id)
        .where(ProjectEvaluatorAssignment.project_id == project_id)
    )
    rows = result.all()

    out = []
    for assignment, evaluator in rows:
        out.append(
            EvaluatorAssignmentOut(
                project_id=assignment.project_id,
                evaluator_id=assignment.evaluator_id,
                similarity_score=assignment.similarity_score,
                load_at_assignment=assignment.load_at_assignment,
                assigned_at=assignment.assigned_at,
                evaluator_nome=evaluator.nome,
                evaluator_email=evaluator.email,
            )
        )
    return out
