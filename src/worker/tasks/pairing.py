"""ARQ task: pair a project with the best-matched evaluator."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import update, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ...config import settings
from ...db.session import AsyncSessionLocal
from ...db.models.user import User
from ...db.models.user_embedding import UserEmbedding
from ...db.models.project import Project
from ...db.models.pairing_job import PairingJob
from ...db.models.project_evaluator_assignment import ProjectEvaluatorAssignment

_MAX_EVALUATOR_LOAD = 3


def _build_project_text(project: Project) -> str:
    parts: list[str] = []
    if project.titulo:
        parts.append(f"Título: {project.titulo}")
    if project.tema:
        parts.append(f"Tema: {project.tema}")
    if project.problema:
        parts.append(f"Problema: {project.problema}")
    if project.pergunta_de_pesquisa:
        parts.append(f"Pergunta de pesquisa: {project.pergunta_de_pesquisa}")
    if project.hipotese:
        parts.append(f"Hipótese: {project.hipotese}")
    if project.objetivo:
        parts.append(f"Objetivo: {project.objetivo}")
    if project.metodo:
        parts.append(f"Método: {project.metodo}")
    if project.riscos:
        parts.append(f"Riscos: {project.riscos}")
    if project.beneficios:
        parts.append(f"Benefícios: {project.beneficios}")
    if project.palavras_chave:
        parts.append(f"Palavras-chave: {', '.join(project.palavras_chave)}")
    return "\n".join(parts)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


async def run_pairing_task(ctx: dict, job_id: str, project_id: str) -> None:
    """
    Assign the best-matched avaliador to a project using a
    similarity-weighted, load-balanced greedy algorithm.

    ═══════════════════════════════════════════════════════════════════
    PAIRING STRATEGY — Similarity-Weighted Load-Balanced Greedy
    ═══════════════════════════════════════════════════════════════════

    Goal
    ────
    Find the avaliador whose research profile is semantically closest to
    the project, while preventing any single avaliador from accumulating
    too many assignments.

    Inputs
    ──────
    • Project text  — concatenation of: título, tema, problema,
      pergunta_de_pesquisa, hipótese, objetivo, método, riscos,
      benefícios, palavras-chave.
    • Candidate set — all users with privilegio=avaliador who have a
      stored embedding (i.e., have a non-empty research profile).

    Algorithm
    ─────────
    1. Embed the project text using the same Google Gemini model used for
       user profiles (models/gemini-embedding-001), so vectors live in the
       same semantic space.

    2. For each candidate compute cosine similarity:
           similarity = cos(project_vec, evaluator_vec)

    3. Apply a load penalty to discourage assigning to already-busy
       evaluators:
           score = similarity × (1 − load / (MAX_LOAD + 1))

       where load = number of projects currently assigned to this avaliador.
       This gives the following effective multipliers:
           load=0 → ×1.00   (no penalty)
           load=1 → ×0.75
           load=2 → ×0.50
           load=3 → excluded (at cap, MAX_LOAD=3)

       The penalty is proportional, so a highly relevant evaluator with
       load=2 (×0.50) still beats a marginally relevant one with load=0
       (×1.00) if the similarity gap is large enough.

    4. Exclude any candidate whose load ≥ MAX_LOAD.

    5. Sort remaining candidates by score descending; assign the top one.

    6. Upsert into project_evaluator_assignments (one row per project).
       Store similarity_score and load_at_assignment for auditability.

    Guarantees
    ──────────
    • No avaliador exceeds MAX_LOAD=3 active assignments.
    • Among equally loaded candidates, the most semantically similar wins.
    • If all avaliadores are at cap, the job fails with a clear message.
    ═══════════════════════════════════════════════════════════════════
    """
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(
                update(PairingJob)
                .where(PairingJob.id == uuid.UUID(job_id))
                .values(status="running")
            )
            await db.commit()

        # 1. Load project
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Project).where(Project.id == uuid.UUID(project_id)))
            project = result.scalar_one_or_none()
            if project is None:
                raise ValueError(f"Project {project_id} not found")
            project_text = _build_project_text(project)

        if not project_text.strip():
            raise ValueError("Project has no text content to embed")

        # 2. Embed the project
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from sqlalchemy import func

        embedder = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model_name,
            google_api_key=settings.google_api_key,
        )
        project_vector: list[float] = await embedder.aembed_query(project_text)

        # 3. Fetch all users who have a stored embedding (all users are potential evaluators)
        async with AsyncSessionLocal() as db:
            rows = await db.execute(
                select(User, UserEmbedding)
                .join(UserEmbedding, UserEmbedding.user_id == User.id)
            )
            candidates = rows.all()

        if not candidates:
            raise ValueError("No users with embeddings found. Users must register and have their profile processed before pairing.")

        # 4. Compute current assignment loads
        async with AsyncSessionLocal() as db:
            loads: dict[uuid.UUID, int] = {}
            for user, _ in candidates:
                count = await db.scalar(
                    select(func.count(ProjectEvaluatorAssignment.project_id)).where(
                        ProjectEvaluatorAssignment.evaluator_id == user.id
                    )
                )
                loads[user.id] = int(count or 0)

        # 5. Score each candidate (exclude those at cap)
        scored: list[tuple[uuid.UUID, float, float, int]] = []  # (id, similarity, score, load)
        for user, emb in candidates:
            load = loads.get(user.id, 0)
            if load >= _MAX_EVALUATOR_LOAD:
                continue
            similarity = _cosine_similarity(project_vector, emb.embedding)
            # Load penalty: each slot used reduces score proportionally
            score = similarity * (1.0 - load / (_MAX_EVALUATOR_LOAD + 1))
            scored.append((user.id, similarity, score, load))

        if not scored:
            raise ValueError(
                f"All avaliadores have reached the maximum load of {_MAX_EVALUATOR_LOAD} projects"
            )

        scored.sort(key=lambda x: x[2], reverse=True)
        best_id, best_similarity, _, best_load = scored[0]

        # 6. Upsert the assignment
        now = datetime.now(timezone.utc)
        result_data = [
            {
                "evaluator_id": str(best_id),
                "similarity_score": best_similarity,
                "load_at_assignment": best_load,
            }
        ]
        async with AsyncSessionLocal() as db:
            stmt = (
                pg_insert(ProjectEvaluatorAssignment)
                .values(
                    project_id=uuid.UUID(project_id),
                    evaluator_id=best_id,
                    similarity_score=best_similarity,
                    load_at_assignment=best_load,
                    assigned_at=now,
                )
                .on_conflict_do_update(
                    index_elements=["project_id", "evaluator_id"],
                    set_={
                        "similarity_score": best_similarity,
                        "load_at_assignment": best_load,
                        "assigned_at": now,
                    },
                )
            )
            await db.execute(stmt)
            await db.execute(
                update(PairingJob)
                .where(PairingJob.id == uuid.UUID(job_id))
                .values(status="completed", result=result_data, completed_at=now)
            )
            await db.commit()

    except Exception as exc:
        error_msg = str(exc)
        async with AsyncSessionLocal() as db:
            await db.execute(
                update(PairingJob)
                .where(PairingJob.id == uuid.UUID(job_id))
                .values(status="failed", error_message=error_msg, completed_at=datetime.now(timezone.utc))
            )
            await db.commit()
        raise
