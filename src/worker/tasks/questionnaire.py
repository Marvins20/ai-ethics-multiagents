"""ARQ task: generate the ethical self-assessment questionnaire."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import update, select

from ...db.session import AsyncSessionLocal
from ...db.models.audit_job import AuditJob
from ...db.models.questionnaire_job import QuestionnaireJob
from ...db.models.pairing_job import PairingJob


async def run_questionnaire_task(ctx: dict, job_id: str, project_id: str) -> None:
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(
                update(QuestionnaireJob)
                .where(QuestionnaireJob.id == uuid.UUID(job_id))
                .values(status="running")
            )
            await db.commit()

        # Load the latest completed full-audit job (has identified_risks etc.)
        audit_result_data: dict = {}
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(AuditJob)
                .where(
                    AuditJob.project_id == uuid.UUID(project_id),
                    AuditJob.status == "completed",
                )
                .order_by(AuditJob.created_at.desc())
                .limit(1)
            )
            audit_job = result.scalar_one_or_none()
            if audit_job is not None and audit_job.result:
                audit_result_data = dict(audit_job.result)

        if not audit_result_data:
            raise ValueError("No completed audit job found for this project")

        identified_risks: list = audit_result_data.get("identified_risks", [])
        risk_classification: str = audit_result_data.get("risk_classification", "")
        executive_summary: str = audit_result_data.get("executive_summary", "")

        from langchain_core.messages import SystemMessage, HumanMessage as HMsg
        from ...agents.questionnaire_agent import SYSTEM_PROMPT
        from ...state import QuestionnaireResult
        from ...model import model

        risks_text = (
            "\n".join(f"- {r}" for r in identified_risks)
            if identified_risks
            else "(nenhum risco específico identificado)"
        )
        human_content = (
            f"Classificação geral de risco do projeto: {risk_classification}\n\n"
            f"Sumário executivo:\n{executive_summary}\n\n"
            f"Riscos específicos identificados no projeto de IA:\n{risks_text}"
        )

        # Use ainvoke directly — avoids blocking the event loop with sync invoke()
        structured_chain = model._base.with_structured_output(QuestionnaireResult)
        q_result = await structured_chain.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HMsg(content=human_content)]
        )

        if isinstance(q_result, dict):
            q_result = QuestionnaireResult(**q_result)

        items = [item.model_dump() for item in q_result.items]  # type: ignore

        async with AsyncSessionLocal() as db:
            await db.execute(
                update(QuestionnaireJob)
                .where(QuestionnaireJob.id == uuid.UUID(job_id))
                .values(
                    status="completed",
                    result=items,
                    completed_at=datetime.now(timezone.utc),
                )
            )
            await db.commit()

        # Auto-trigger pairing as soon as questionnaire is ready (idempotent)
        try:
            redis = ctx["redis"]
            async with AsyncSessionLocal() as db:
                existing = await db.execute(
                    select(PairingJob)
                    .where(
                        PairingJob.project_id == uuid.UUID(project_id),
                        PairingJob.status.in_(["pending", "running", "completed"]),
                    )
                    .limit(1)
                )
                if not existing.scalar_one_or_none():
                    pairing_job = PairingJob(project_id=uuid.UUID(project_id), status="pending")
                    db.add(pairing_job)
                    await db.commit()
                    await db.refresh(pairing_job)
                    await redis.enqueue_job("run_pairing_task", str(pairing_job.id), project_id)
        except Exception:
            pass  # pairing trigger failure must not fail the questionnaire task

    except Exception as exc:
        error_msg = str(exc)
        async with AsyncSessionLocal() as db:
            await db.execute(
                update(QuestionnaireJob)
                .where(QuestionnaireJob.id == uuid.UUID(job_id))
                .values(
                    status="failed",
                    error_message=error_msg,
                    completed_at=datetime.now(timezone.utc),
                )
            )
            await db.commit()
        raise
