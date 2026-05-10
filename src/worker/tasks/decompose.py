"""ARQ task: decompose a project into actions (Phase 1)."""
import json
import uuid
import traceback
from datetime import datetime, timezone

from arq import ArqRedis
from langchain_core.messages import HumanMessage
from sqlalchemy import update

from ...db.session import AsyncSessionLocal
from ...db.models.audit_job import AuditJob
from ._utils import _serialize_event


async def run_decompose_task(ctx: dict, job_id: str, input_text: str, thread_id: str) -> None:
    """
    Phase 1: runs only project_analyst_agent to decompose the project into actions.
    Fast (~15s). Result stored in AuditJob.result as the serialized AgentState,
    which includes analysis_result.actions[].
    """
    redis: ArqRedis = ctx["redis"]
    channel = f"audit:{job_id}:progress"

    async def publish(payload: dict) -> None:
        try:
            await redis.publish(channel, json.dumps(payload, default=str))
        except Exception:
            pass

    try:
        async with AsyncSessionLocal() as db:
            await db.execute(
                update(AuditJob).where(AuditJob.id == uuid.UUID(job_id)).values(status="running")
            )
            await db.commit()

        await publish({"status": "running", "message": "Decomposing project into actions..."})

        from ...graphs import build_decompose_graph

        graph = build_decompose_graph()
        final_state: dict = {}

        async for event in graph.astream(
            {"messages": [HumanMessage(content=input_text)], "llm_calls": 0, "thread_id": thread_id},
            config={},
            stream_mode="values",
        ):
            final_state = event

        await publish({"status": "running", "node": "project_analyst", "label": "Ações identificadas!"})

        serialized = _serialize_event(final_state)

        async with AsyncSessionLocal() as db:
            await db.execute(
                update(AuditJob)
                .where(AuditJob.id == uuid.UUID(job_id))
                .values(status="completed", result=serialized, completed_at=datetime.now(timezone.utc))
            )
            await db.commit()

        await publish({"status": "completed"})

    except BaseException:
        error_msg = traceback.format_exc()
        try:
            async with AsyncSessionLocal() as db:
                await db.execute(
                    update(AuditJob)
                    .where(AuditJob.id == uuid.UUID(job_id))
                    .values(status="failed", error_message=error_msg, completed_at=datetime.now(timezone.utc))
                )
                await db.commit()
        except BaseException:
            pass
        await publish({"status": "failed", "error": error_msg})
