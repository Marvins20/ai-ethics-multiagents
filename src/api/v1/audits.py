import uuid
import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from arq import create_pool
from arq.connections import RedisSettings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis

from ...api.deps import get_current_user, get_db
from ...config import settings
from ...db.models.user import User
from ...db.models.project import Project
from ...db.models.audit_job import AuditJob
from ...schemas.audit import AuditSubmit, AuditJobResponse, AuditJobSummary

router = APIRouter(tags=["audits"])


@router.post("/projects/{project_id}/decompose", response_model=AuditJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_decompose(
    project_id: uuid.UUID,
    body: AuditSubmit,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _get_owned_project(project_id, current_user.id, db)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    thread_id = str(uuid.uuid4())
    job = AuditJob(
        project_id=project_id,
        status="pending",
        input_text=body.project_description,
        langgraph_thread_id=thread_id,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    redis_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    await redis_pool.enqueue_job("run_decompose_task", str(job.id), body.project_description, thread_id)
    await redis_pool.aclose()

    return job


@router.post("/projects/{project_id}/audits", response_model=AuditJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_audit(
    project_id: uuid.UUID,
    body: AuditSubmit,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _get_owned_project(project_id, current_user.id, db)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    thread_id = str(uuid.uuid4())
    job = AuditJob(
        project_id=project_id,
        status="pending",
        input_text=body.project_description,
        langgraph_thread_id=thread_id,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    redis_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    await redis_pool.enqueue_job(
        "run_audit_task",
        str(job.id),
        body.project_description,
        thread_id,
        body.pre_analysis,
    )
    await redis_pool.aclose()

    return job


@router.get("/projects/{project_id}/audits", response_model=list[AuditJobSummary])
async def list_audits(
    project_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _get_owned_project(project_id, current_user.id, db)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    jobs_result = await db.execute(
        select(AuditJob)
        .where(AuditJob.project_id == project_id)
        .order_by(AuditJob.created_at.desc())
    )
    return jobs_result.scalars().all()


@router.get("/audits/{job_id}", response_model=AuditJobResponse)
async def get_audit(
    job_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await _get_owned_job(job_id, current_user.id, db)
    return job


@router.get("/audits/{job_id}/stream")
async def stream_audit_progress(
    job_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Stream agent progress via Server-Sent Events."""
    job = await _get_owned_job(job_id, current_user.id, db)

    if job.status == "completed":
        async def completed_stream():
            yield f"data: {json.dumps({'status': 'completed', 'result': job.result})}\n\n"
        return StreamingResponse(completed_stream(), media_type="text/event-stream")

    if job.status == "failed":
        async def failed_stream():
            yield f"data: {json.dumps({'status': 'failed', 'error': job.error_message})}\n\n"
        return StreamingResponse(failed_stream(), media_type="text/event-stream")

    channel = f"audit:{job_id}:progress"

    async def event_generator():
        redis_client = aioredis.from_url(settings.redis_url)
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel)
        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode()
                yield f"data: {data}\n\n"
                parsed = json.loads(data)
                if parsed.get("status") in ("completed", "failed"):
                    break
        finally:
            await pubsub.unsubscribe(channel)
            await redis_client.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _get_owned_project(project_id: uuid.UUID, user_id: uuid.UUID, db: AsyncSession) -> Project | None:
    result = await db.execute(
        select(Project).where(Project.id == project_id, Project.criador_id == user_id)
    )
    return result.scalar_one_or_none()


async def _get_owned_job(job_id: uuid.UUID, user_id: uuid.UUID, db: AsyncSession) -> AuditJob:
    result = await db.execute(
        select(AuditJob)
        .join(Project, AuditJob.project_id == Project.id)
        .where(AuditJob.id == job_id, Project.criador_id == user_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audit job not found")
    return job
