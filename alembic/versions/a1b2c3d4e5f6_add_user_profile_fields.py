"""add_user_profile_fields

Revision ID: a1b2c3d4e5f6
Revises: 7efd3ac75c91
Create Date: 2026-04-14

"""
from alembic import op
import sqlalchemy as sa

revision = 'a1b2c3d4e5f6'
down_revision = '7efd3ac75c91'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('users', sa.Column('matricula', sa.String(50), nullable=True))
    op.add_column('users', sa.Column('data_nascimento', sa.Date(), nullable=True))
    op.add_column('users', sa.Column('telefone', sa.String(20), nullable=True))
    op.add_column('users', sa.Column('curso_programa', sa.String(500), nullable=True))


def downgrade() -> None:
    op.drop_column('users', 'curso_programa')
    op.drop_column('users', 'telefone')
    op.drop_column('users', 'data_nascimento')
    op.drop_column('users', 'matricula')
