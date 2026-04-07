import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ...config import settings
from ...db.session import AsyncSessionLocal
from ...db.models.user import User
from ...db.models.user_embedding import UserEmbedding


def _build_profile_text(user: User) -> str:
    """Produce a single text blob that represents a user's research profile.

    Always returns at least the user's email so every registered user gets an
    embedding even before they fill in their full research profile.
    """
    parts: list[str] = []
    if user.nome:
        parts.append(f"Nome: {user.nome}")
    if user.titulo:
        parts.append(f"Título: {user.titulo}")
    if user.departamento:
        parts.append(f"Departamento: {user.departamento}")
    if user.curso_programa:
        parts.append(f"Curso/Programa: {user.curso_programa}")
    if user.areas_atuacao:
        areas = ", ".join(user.areas_atuacao)
        parts.append(f"Áreas de atuação: {areas}")
    if user.descricao:
        parts.append(f"Descrição: {user.descricao}")
    if user.projetos_anteriores:
        parts.append(f"Projetos anteriores: {user.projetos_anteriores}")
    # Fallback: always include email so the text is never empty
    if not parts:
        parts.append(f"Pesquisador: {user.email}")
    return "\n".join(parts)


async def run_user_embedding_task(ctx: dict, user_id: str) -> None:
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
        user = result.scalar_one_or_none()
        if user is None:
            return

        profile_text = _build_profile_text(user)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embedder = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model_name,
        google_api_key=settings.google_api_key,
    )
    vector: list[float] = await embedder.aembed_query(profile_text)

    now = datetime.now(timezone.utc)
    async with AsyncSessionLocal() as db:
        stmt = (
            pg_insert(UserEmbedding)
            .values(
                user_id=uuid.UUID(user_id),
                embedding=vector,
                model_name=settings.embedding_model_name,
                updated_at=now,
            )
            .on_conflict_do_update(
                index_elements=["user_id"],
                set_={
                    "embedding": vector,
                    "model_name": settings.embedding_model_name,
                    "updated_at": now,
                },
            )
        )
        await db.execute(stmt)
        await db.commit()
