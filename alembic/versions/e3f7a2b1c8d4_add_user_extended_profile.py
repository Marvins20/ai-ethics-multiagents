"""add user extended profile fields

Revision ID: e3f7a2b1c8d4
Revises: df152d8f600c
Create Date: 2026-04-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "e3f7a2b1c8d4"
down_revision: Union[str, None] = "df152d8f600c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("descricao", sa.Text(), nullable=True))
    op.add_column(
        "users",
        sa.Column(
            "areas_atuacao",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.add_column("users", sa.Column("projetos_anteriores", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "projetos_anteriores")
    op.drop_column("users", "areas_atuacao")
    op.drop_column("users", "descricao")
