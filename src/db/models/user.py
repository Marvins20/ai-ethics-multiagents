import uuid
from datetime import datetime, date, timezone
from sqlalchemy import String, DateTime, Date, Text, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB
from ..base import Base
from ..enums import PrivilegioEnum, TituloEnum, DepartamentoEnum


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    privilegio: Mapped[str] = mapped_column(
        SAEnum(PrivilegioEnum, native_enum=False), nullable=False, default=PrivilegioEnum.usuario
    )

    nome: Mapped[str | None] = mapped_column(String(255), nullable=True)
    cpf: Mapped[str | None] = mapped_column(String(14), nullable=True)        # format: 000.000.000-00
    curriculo_lattes: Mapped[str | None] = mapped_column(String(500), nullable=True)
    matricula: Mapped[str | None] = mapped_column(String(50), nullable=True)
    data_nascimento: Mapped[date | None] = mapped_column(Date, nullable=True)
    telefone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    curso_programa: Mapped[str | None] = mapped_column(String(500), nullable=True)

    titulo: Mapped[str | None] = mapped_column(
        SAEnum(TituloEnum, native_enum=False), nullable=True
    )
    departamento: Mapped[str | None] = mapped_column(
        SAEnum(DepartamentoEnum, native_enum=False), nullable=True
    )

    descricao: Mapped[str | None] = mapped_column(Text, nullable=True)
    areas_atuacao: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    projetos_anteriores: Mapped[str | None] = mapped_column(Text, nullable=True)

    projetos_criados: Mapped[list["Project"]] = relationship(  # type: ignore[name-defined]
        back_populates="criador",
        foreign_keys="Project.criador_id",
    )

    projetos_orientados: Mapped[list["Project"]] = relationship(  # type: ignore[name-defined]
        back_populates="orientador",
        foreign_keys="Project.orientador_id",
    )
  
    projetos_assistidos: Mapped[list["Project"]] = relationship(  # type: ignore[name-defined]
        secondary="project_assistants",
        back_populates="assistentes_de_pesquisa",
    )
  
    projetos_equipe: Mapped[list["Project"]] = relationship(  # type: ignore[name-defined]
        secondary="project_team",
        back_populates="equipe_de_pesquisa",
    )

    has_reads: Mapped[list["HasRead"]] = relationship(back_populates="user", cascade="all, delete-orphan")  # type: ignore[name-defined]
    checklist_responses: Mapped[list["ChecklistResponse"]] = relationship(back_populates="user", cascade="all, delete-orphan")  # type: ignore[name-defined]
    embedding: Mapped["UserEmbedding | None"] = relationship(back_populates="user", cascade="all, delete-orphan", uselist=False)  # type: ignore[name-defined]
    evaluator_assignments: Mapped[list["ProjectEvaluatorAssignment"]] = relationship(  # type: ignore[name-defined]
        back_populates="evaluator", cascade="all, delete-orphan"
    )
