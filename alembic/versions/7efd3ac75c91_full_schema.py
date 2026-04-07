"""full_schema

Revision ID: 7efd3ac75c91
Revises: b770f68c0753
Create Date: 2026-04-12

Extends the initial schema with the full domain model:
- Adds profile fields to users (privilegio, nome, cpf, curriculo_lattes, titulo, departamento)
- Drops and recreates projects with the full research project schema
- Adds project_types, checklists, question_groups, questions, checklist_responses
- Adds norms and has_read
- Adds project_assistants and project_team association tables
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

revision: str = "7efd3ac75c91"
down_revision: Union[str, Sequence[str], None] = "b770f68c0753"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── 1. Extend users ───────────────────────────────────────────────────────
    op.add_column("users", sa.Column("privilegio", sa.String(20), nullable=False, server_default="usuario"))
    op.add_column("users", sa.Column("nome", sa.String(255), nullable=True))
    op.add_column("users", sa.Column("cpf", sa.String(14), nullable=True))
    op.add_column("users", sa.Column("curriculo_lattes", sa.String(500), nullable=True))
    op.add_column("users", sa.Column("titulo", sa.String(20), nullable=True))
    op.add_column("users", sa.Column("departamento", sa.String(20), nullable=True))

    # ── 2. Drop old simple projects (no data yet) ─────────────────────────────
    # audit_jobs has a FK on projects, so it must be dropped first
    op.drop_table("audit_jobs")
    op.drop_table("projects")

    # ── 3. Create project_types ───────────────────────────────────────────────
    op.create_table(
        "project_types",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("nome", sa.String(255), nullable=False),
    )

    # ── 4. Recreate projects WITHOUT audit_id (circular FK resolved below) ────
    op.create_table(
        "projects",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("titulo", sa.String(500), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="rascunho"),
        sa.Column("criador_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("orientador_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("project_type_id", sa.Uuid(), sa.ForeignKey("project_types.id", ondelete="SET NULL"), nullable=True),
        sa.Column("departamento", sa.String(20), nullable=True),
        sa.Column("area_primaria", sa.String(20), nullable=True),
        sa.Column("subarea_primaria", sa.String(20), nullable=True),
        sa.Column("area_secundaria", sa.String(20), nullable=True),
        sa.Column("subarea_secundaria", sa.String(20), nullable=True),
        sa.Column("desenho", sa.Text(), nullable=True),
        sa.Column("palavras_chave", ARRAY(sa.String(100)), nullable=True),
        sa.Column("envolve_humanos", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("tema", sa.Text(), nullable=True),
        sa.Column("problema", sa.Text(), nullable=True),
        sa.Column("pergunta_de_pesquisa", sa.Text(), nullable=True),
        sa.Column("hipotese", sa.Text(), nullable=True),
        sa.Column("justificativa_da_hipotese", sa.Text(), nullable=True),
        sa.Column("objetivo", sa.Text(), nullable=True),
        sa.Column("objetivo_especifico", sa.Text(), nullable=True),
        sa.Column("metodo", sa.Text(), nullable=True),
        sa.Column("classificacao_da_pesquisa", sa.Text(), nullable=True),
        sa.Column("riscos", sa.Text(), nullable=True),
        sa.Column("beneficios", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # ── 5. Recreate audit_jobs referencing the new projects table ─────────────
    op.create_table(
        "audit_jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("project_id", sa.Uuid(), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("input_text", sa.Text(), nullable=False),
        sa.Column("result", JSONB(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("langgraph_thread_id", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Now that audit_jobs exists, add the audit_id FK column to projects
    op.add_column("projects", sa.Column("audit_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_projects_audit_id",
        "projects", "audit_jobs",
        ["audit_id"], ["id"],
        ondelete="SET NULL",
    )

    # ── 7. Association tables ─────────────────────────────────────────────────
    op.create_table(
        "project_assistants",
        sa.Column("project_id", sa.Uuid(), sa.ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    )

    op.create_table(
        "project_team",
        sa.Column("project_id", sa.Uuid(), sa.ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    )

    # ── 8. Checklists ─────────────────────────────────────────────────────────
    op.create_table(
        "checklists",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("nome", sa.String(255), nullable=False),
        sa.Column("descricao", sa.Text(), nullable=True),
        sa.Column("project_type_id", sa.Uuid(), sa.ForeignKey("project_types.id", ondelete="CASCADE"), nullable=False),
        sa.UniqueConstraint("project_type_id", name="uq_checklist_project_type"),
    )

    op.create_table(
        "question_groups",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("checklist_id", sa.Uuid(), sa.ForeignKey("checklists.id", ondelete="CASCADE"), nullable=False),
        sa.Column("nome", sa.String(255), nullable=False),
        sa.Column("ordem", sa.Integer(), nullable=False, server_default="0"),
    )

    op.create_table(
        "questions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("group_id", sa.Uuid(), sa.ForeignKey("question_groups.id", ondelete="CASCADE"), nullable=False),
        sa.Column("descricao", sa.Text(), nullable=False),
        sa.Column("ordem", sa.Integer(), nullable=False, server_default="0"),
    )

    op.create_table(
        "checklist_responses",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("checklist_id", sa.Uuid(), sa.ForeignKey("checklists.id", ondelete="CASCADE"), nullable=False),
        sa.Column("project_id", sa.Uuid(), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("question_id", sa.Uuid(), sa.ForeignKey("questions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("resposta", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("project_id", "question_id", "user_id", name="uq_response_per_user_project_question"),
    )

    # ── 9. Norms & has_read ───────────────────────────────────────────────────
    op.create_table(
        "norms",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("titulo", sa.String(500), nullable=False),
        sa.Column("arquivo", sa.String(1000), nullable=True),
        sa.Column("publico", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("incluir_avaliacao", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("obrigatorio", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "has_read",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("norm_id", sa.Uuid(), sa.ForeignKey("norms.id", ondelete="CASCADE"), nullable=False),
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("user_id", "norm_id", name="uq_has_read_user_norm"),
    )


def downgrade() -> None:
    op.drop_table("has_read")
    op.drop_table("norms")
    op.drop_table("checklist_responses")
    op.drop_table("questions")
    op.drop_table("question_groups")
    op.drop_table("checklists")
    op.drop_table("project_team")
    op.drop_table("project_assistants")
    # Drop the circular FK before dropping the tables
    op.drop_constraint("fk_projects_audit_id", "projects", type_="foreignkey")
    op.drop_column("projects", "audit_id")
    op.drop_table("audit_jobs")
    op.drop_table("projects")
    op.drop_table("project_types")

    # Restore minimal projects table
    op.create_table(
        "projects",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("owner_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.drop_column("users", "departamento")
    op.drop_column("users", "titulo")
    op.drop_column("users", "curriculo_lattes")
    op.drop_column("users", "cpf")
    op.drop_column("users", "nome")
    op.drop_column("users", "privilegio")
