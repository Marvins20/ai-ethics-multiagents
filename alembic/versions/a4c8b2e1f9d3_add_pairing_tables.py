"""add pairing_jobs and project_evaluator_assignments tables

Revision ID: a4c8b2e1f9d3
Revises: f1a9b3c2d7e5
Create Date: 2026-04-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "a4c8b2e1f9d3"
down_revision: Union[str, None] = "f1a9b3c2d7e5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "pairing_jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", name="uq_pairing_job_project"),
    )

    op.create_table(
        "project_evaluator_assignments",
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("evaluator_id", sa.UUID(), nullable=False),
        sa.Column("similarity_score", sa.Float(), nullable=False),
        sa.Column("load_at_assignment", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["evaluator_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("project_id", "evaluator_id"),
    )


def downgrade() -> None:
    op.drop_table("project_evaluator_assignments")
    op.drop_table("pairing_jobs")
