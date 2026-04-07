import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    String, Text, Boolean, DateTime, ForeignKey,
    Enum as SAEnum, Table, Column, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..base import Base
from ..enums import DepartamentoEnum, ProjectStatusEnum, AreaCnpqEnum, SubAreaCnpqEnum


# ── Association tables (many-to-many) ─────────────────────────────────────────

project_assistants = Table(
    "project_assistants",
    Base.metadata,
    Column("project_id", ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True),
    Column("user_id", ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
)

project_team = Table(
    "project_team",
    Base.metadata,
    Column("project_id", ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True),
    Column("user_id", ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
)


# ── Project ───────────────────────────────────────────────────────────────────

class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)

    # ── Identity ──────────────────────────────────────────────────────────────
    titulo: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(
        SAEnum(ProjectStatusEnum, native_enum=False),
        nullable=False,
        default=ProjectStatusEnum.rascunho,
    )

    # ── Ownership / team ──────────────────────────────────────────────────────
    criador_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    orientador_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    criador: Mapped["User | None"] = relationship(  # type: ignore[name-defined]
        back_populates="projetos_criados", foreign_keys=[criador_id]
    )
    orientador: Mapped["User | None"] = relationship(  # type: ignore[name-defined]
        back_populates="projetos_orientados", foreign_keys=[orientador_id]
    )
    assistentes_de_pesquisa: Mapped[list["User"]] = relationship(  # type: ignore[name-defined]
        secondary=project_assistants, back_populates="projetos_assistidos"
    )
    equipe_de_pesquisa: Mapped[list["User"]] = relationship(  # type: ignore[name-defined]
        secondary=project_team, back_populates="projetos_equipe"
    )

    # ── Classification ────────────────────────────────────────────────────────
    project_type_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("project_types.id", ondelete="SET NULL"), nullable=True
    )
    departamento: Mapped[str | None] = mapped_column(
        SAEnum(DepartamentoEnum, native_enum=False), nullable=True
    )
    area_primaria: Mapped[str | None] = mapped_column(
        SAEnum(AreaCnpqEnum, native_enum=False), nullable=True
    )
    subarea_primaria: Mapped[str | None] = mapped_column(
        SAEnum(SubAreaCnpqEnum, native_enum=False), nullable=True
    )
    area_secundaria: Mapped[str | None] = mapped_column(
        SAEnum(AreaCnpqEnum, native_enum=False), nullable=True
    )
    subarea_secundaria: Mapped[str | None] = mapped_column(
        SAEnum(SubAreaCnpqEnum, native_enum=False), nullable=True
    )

    # ── Core content ──────────────────────────────────────────────────────────
    desenho: Mapped[str | None] = mapped_column(Text, nullable=True)
    palavras_chave: Mapped[list[str] | None] = mapped_column(ARRAY(String(100)), nullable=True)
    envolve_humanos: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    tema: Mapped[str | None] = mapped_column(Text, nullable=True)
    problema: Mapped[str | None] = mapped_column(Text, nullable=True)
    pergunta_de_pesquisa: Mapped[str | None] = mapped_column(Text, nullable=True)
    hipotese: Mapped[str | None] = mapped_column(Text, nullable=True)
    justificativa_da_hipotese: Mapped[str | None] = mapped_column(Text, nullable=True)
    objetivo: Mapped[str | None] = mapped_column(Text, nullable=True)
    objetivo_especifico: Mapped[str | None] = mapped_column(Text, nullable=True)
    metodo: Mapped[str | None] = mapped_column(Text, nullable=True)
    classificacao_da_pesquisa: Mapped[str | None] = mapped_column(Text, nullable=True)
    riscos: Mapped[str | None] = mapped_column(Text, nullable=True)
    beneficios: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Audit link ────────────────────────────────────────────────────────────
    # Nullable — most projects won't have been audited yet
    audit_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("audit_jobs.id", ondelete="SET NULL"), nullable=True
    )

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    project_type: Mapped["ProjectType | None"] = relationship(back_populates="projects")  # type: ignore[name-defined]
    audit: Mapped["AuditJob | None"] = relationship(foreign_keys=[audit_id])  # type: ignore[name-defined]
    audit_jobs: Mapped[list["AuditJob"]] = relationship(  # type: ignore[name-defined]
        back_populates="project",
        primaryjoin="AuditJob.project_id == Project.id",
        cascade="all, delete-orphan",
    )
    checklist_responses: Mapped[list["ChecklistResponse"]] = relationship(  # type: ignore[name-defined]
        back_populates="project", cascade="all, delete-orphan"
    )
    questionnaire_jobs: Mapped[list["QuestionnaireJob"]] = relationship(  # type: ignore[name-defined]
        back_populates="project",
        cascade="all, delete-orphan",
    )
    pairing_job: Mapped["PairingJob | None"] = relationship(  # type: ignore[name-defined]
        back_populates="project", cascade="all, delete-orphan", uselist=False
    )
    evaluator_assignments: Mapped[list["ProjectEvaluatorAssignment"]] = relationship(  # type: ignore[name-defined]
        back_populates="project", cascade="all, delete-orphan"
    )
