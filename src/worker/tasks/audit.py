"""ARQ task: run the full multi-agent audit pipeline (Phase 2)."""
import json
import uuid
from datetime import datetime, timezone

from arq import ArqRedis
from langchain_core.messages import HumanMessage
from sqlalchemy import update, select

from ...db.session import AsyncSessionLocal
from ...db.models.audit_job import AuditJob
from ...db.models.questionnaire_job import QuestionnaireJob
from ._utils import _serialize_event

# Node sequence for the full audit (project_analyst is instant when pre_analysis provided)
_NODE_SEQUENCE = [
    ("project_analyst",            "Analisando o projeto..."),
    ("supervisor",                 "Preparando análise de riscos..."),
    ("risk_agent",                 "Avaliando riscos éticos..."),
    ("incident_agent",             "Buscando incidentes relacionados..."),
    ("proprietary_framework_agent","Verificando conformidade normativa..."),
    ("final_classifier_agent",     "Classificando risco final..."),
]


async def run_audit_task(
    ctx: dict,
    job_id: str,
    input_text: str,
    thread_id: str,
    pre_analysis: dict | None = None,
) -> None:
    """
    Phase 2: runs the full multi-agent pipeline.
    If pre_analysis (approved analysis_result from Phase 1) is provided,
    the project_analyst node is skipped and the pipeline starts from supervisor.
    """
    redis: ArqRedis = ctx["redis"]
    channel = f"audit:{job_id}:progress"

    async def publish(payload: dict) -> None:
        await redis.publish(channel, json.dumps(payload, default=str))

    try:
        async with AsyncSessionLocal() as db:
            await db.execute(
                update(AuditJob).where(AuditJob.id == uuid.UUID(job_id)).values(status="running")
            )
            await db.commit()

        await publish({"status": "running", "message": "Audit started"})

        from ...graphs import build_audit_graph

        graph = build_audit_graph()

        initial_state: dict = {
            "messages": [HumanMessage(content=input_text)],
            "llm_calls": 0,
            "thread_id": thread_id,
        }
        if pre_analysis:
            initial_state["analysis_result"] = pre_analysis

        final_state: dict = {}
        step_index = 0

        async for event in graph.astream(
            initial_state,
            config={},
            stream_mode="values",
        ):
            final_state = event
            if step_index < len(_NODE_SEQUENCE):
                node_name, label = _NODE_SEQUENCE[step_index]
                await publish({"status": "running", "node": node_name, "label": label})
                step_index += 1

        serialized = _serialize_event(final_state)

        async with AsyncSessionLocal() as db:
            await db.execute(
                update(AuditJob)
                .where(AuditJob.id == uuid.UUID(job_id))
                .values(status="completed", result=serialized, completed_at=datetime.now(timezone.utc))
            )
            await db.commit()

        await publish({"status": "completed", "result": serialized})

        # Auto-trigger questionnaire generation once the full audit completes
        async with AsyncSessionLocal() as db:
            audit_row = await db.execute(
                select(AuditJob).where(AuditJob.id == uuid.UUID(job_id))
            )
            audit = audit_row.scalar_one_or_none()
            if audit:
                existing_q = await db.execute(
                    select(QuestionnaireJob)
                    .where(QuestionnaireJob.project_id == audit.project_id)
                    .where(QuestionnaireJob.status.in_(["pending", "running"]))
                    .limit(1)
                )
                if not existing_q.scalar_one_or_none():
                    q_job = QuestionnaireJob(project_id=audit.project_id, status="pending")
                    db.add(q_job)
                    await db.commit()
                    await db.refresh(q_job)
                    await redis.enqueue_job(
                        "run_questionnaire_task", str(q_job.id), str(audit.project_id)
                    )

    except Exception as exc:
        error_msg = str(exc)
        async with AsyncSessionLocal() as db:
            await db.execute(
                update(AuditJob)
                .where(AuditJob.id == uuid.UUID(job_id))
                .values(status="failed", error_message=error_msg, completed_at=datetime.now(timezone.utc))
            )
            await db.commit()

        await publish({"status": "failed", "error": error_msg})
        raise
